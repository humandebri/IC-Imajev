#!/usr/bin/env python3
"""Compare old finite/codec passes and checked streaming codec on one module."""
import argparse,hashlib,json,pathlib,subprocess,time,zipfile,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import encode,decode

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--canister',required=True);ap.add_argument('--directory',required=True);a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
 helper=ROOT/'artifacts/codec_scan/native-target/release/args';wasm=ROOT/'artifacts/codec_scan/target/wasm32-unknown-unknown/release/imajev_codec_scan_bench.wasm';sha=lambda b:hashlib.sha256(b).hexdigest()
 paths=list((ROOT/'scripts/codec_scan_bench/src').rglob('*.rs'))+[ROOT/p for p in ['scripts/codec_scan_bench/Cargo.toml','scripts/codec_scan_bench/Cargo.lock','crates/imajev-runtime/src/bf16_codec.rs','crates/imajev-runtime/src/block_codec.rs','crates/imajev-runtime/src/lib.rs','client/transport.py']]+[pathlib.Path(__file__)];paths+=list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml'];hashes={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths}
 def status():return json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--network','local','--identity','imajev-local','--json'],text=True,cwd=ROOT))['module_hash'].removeprefix('0x')
 module=sha(wasm.read_bytes());assert status()==module;cases=[]
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
 for label in ['prefix','617','insufficient','maximum','normal']:
  source=ROOT/'artifacts/prefix_codec/full-quantize-scan-proof'/label;sr=(source/'report.json').read_bytes();r=json.loads(sr);choices=[]
  for q in r['queries']:
   p=source/'queries'/f"{q['index']:06d}.request.bin";raw=p.read_bytes();n=int.from_bytes(raw[:4],'little');h=json.loads(raw[4:4+n])
   if h['encoding']=='bf16-block256-exact-v1':choices.append((len(raw),p,raw,n))
  _,p,raw,n=max(choices,key=lambda row:row[0]);h,x=decode(raw);assert h['model']==m['model']and h['pack_hash']==m['pack_hash']
  cases.append(dict(name=label,data=raw[4+n:-32],values=x.size,decoded_digest=sha(x.astype('<f4').tobytes()),source_path=str(p.relative_to(ROOT)),source_sha256=sha(raw),source_report_sha256=sha(sr)))
 def payload(x):
  h=dict(version=1,model=m['model'],pack_hash=m['pack_hash'],input_hash='c'*64,step=0,op='bf16',tensor='',dims=[],scalars=[],encoding='bf16-block256-exact-v1');b=encode(h,x);n=int.from_bytes(b[:4],'little');return b[4+n:-32]
 for n,kind in [(0,'bf16'),(1,'bf16'),(7,'mixed'),(8,'bf16'),(9,'mixed'),(255,'mixed'),(256,'bf16'),(257,'mixed'),(511,'mixed'),(513,'mixed'),(900000,'bf16'),(400000,'mixed')]:
  rng=np.random.default_rng(n);bits=rng.integers(0,0x7f800000,n,dtype=np.uint32);bits|=rng.integers(0,2,n,dtype=np.uint32)<<31
  if kind=='bf16':bits&=np.uint32(0xffff0000)
  elif n:bits[::256]=0x3f800001
  x=bits.view(np.float32);cases.append(dict(name=f'{kind}-{n}',data=payload(x),values=n,decoded_digest=sha(x.astype('<f4').tobytes()),scope='Synthetic finite bits and vector/block tails'))
 bad=[]
 for i,bits in enumerate([0x7f800000,0xff800000,0x7f800001,0x7fc00000,0xff800001,0xffffffff]):
  x=np.ones(513,dtype='<f4').view('<u4');x[0]=x[256]=0x3f800001;x[512]=bits;raw=(513).to_bytes(4,'little')+bytes([7])+x.tobytes();bad.append((f'nonfinite-f32-{bits:08x}',raw))
  raw=(513).to_bytes(4,'little')+bytes([0])+np.array([0x3f80]*512+[bits>>16],dtype='<u2').tobytes();bad.append((f'nonfinite-bf16-{bits:08x}',raw))
 bad.extend([('count-bound',(900001).to_bytes(4,'little')),('bitmap-padding',(1).to_bytes(4,'little')+bytes([128])+bytes([0,0])),('noncanonical',(256).to_bytes(4,'little')+bytes([1])+np.ones(256,dtype='<f4').tobytes()),('truncated',cases[-1]['data'][:-1])])
 results=[]
 for case in cases+[dict(name=name,data=raw,invalid=True)for name,raw in bad]:
  raw=case.pop('data');ip=d/f"{case['name']}.input.bin";ip.write_bytes(raw);measure={}
  for op in ([1]if case.get('invalid')else[0,1]):
   expected=json.loads(subprocess.check_output([str(helper),'native',str(ip),str(op)],text=True,cwd=ROOT));assert ('error'in expected)==bool(case.get('invalid'))
   if 'error'not in expected:assert expected['digest']==(sha(raw)if op==0 else case['decoded_digest'])
   modes=[]
   for method in [0,1]:
    arg=d/f"{case['name']}-{op}-{method}.args.bin";subprocess.run([str(helper),'query',str(ip),str(op),str(method),str(arg)],check=True,cwd=ROOT)
    begin=time.monotonic();reply=subprocess.check_output(['icp','canister','call',a.canister,'measure','--network','local','--identity','imajev-local','--query','--args-file',str(arg),'--args-format','bin','--output','hex'],text=True,cwd=ROOT);elapsed=time.monotonic()-begin
    rp=d/f"{case['name']}-{op}-{method}.reply.hex";rp.write_text(reply);got=json.loads(subprocess.check_output([str(helper),'decode',str(rp)],text=True,cwd=ROOT));assert ('error'in got)==('error'in expected)
    if 'error'not in expected:assert got['digest']==expected['digest']and got['output_bytes']==expected['output_bytes']
    got.update(wall_seconds=elapsed,request_bytes=arg.stat().st_size,reply_sha256=sha(reply.encode()));modes.append(got)
   measure[str(op)]=dict(expected=expected,modes=modes)
   if 'error'not in expected:measure[str(op)].update(saved_instructions=modes[0]['instructions']-modes[1]['instructions'],change_percent=100*(modes[1]['instructions']/modes[0]['instructions']-1))
  row=dict(**case,input_sha256=sha(raw),input_bytes=len(raw),measurements=measure);results.append(row);print(json.dumps(row),flush=True)
 assert status()==module;assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths}
 with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
 (d/'report.json').write_text(json.dumps(dict(scope=__doc__,source_hashes=hashes,wasm_sha256=module,helper_sha256=sha(helper.read_bytes()),canister=a.canister,ordinary_queries=sum(len(v['modes'])for r in results for v in r['measurements'].values()),model=m['model'],pack_hash=m['pack_hash'],cases=results),indent=2)+'\n')
if __name__=='__main__':main()
