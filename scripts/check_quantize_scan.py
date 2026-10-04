#!/usr/bin/env python3
"""Compare original two-pass and fused finite/peak scan on one local module."""
import argparse,hashlib,json,pathlib,subprocess,time,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--canister',required=True);ap.add_argument('--directory',required=True);a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
 helper=ROOT/'artifacts/quantize_scan/native-target/release/args';wasm=ROOT/'artifacts/quantize_scan/target/wasm32-unknown-unknown/release/imajev_quantize_scan_bench.wasm';sha=lambda b:hashlib.sha256(b).hexdigest()
 paths=list((ROOT/'scripts/quantize_scan_bench/src').rglob('*.rs'))+[ROOT/p for p in ['scripts/quantize_scan_bench/Cargo.toml','scripts/quantize_scan_bench/Cargo.lock','crates/imajev-runtime/src/quantize_simd.rs','crates/imajev-runtime/src/int8_kernel.rs']]+[pathlib.Path(__file__)];hashes={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths}
 def status():return json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--network','local','--identity','imajev-local','--json'],text=True,cwd=ROOT))['module_hash'].removeprefix('0x')
 module=sha(wasm.read_bytes());assert status()==module;cases=[]
 source=ROOT/'artifacts/s1_address_reuse/check';sr=(source/'report.json').read_bytes();baseline=json.loads(sr)
 for label in ['prefix','617','insufficient','maximum','normal']:
  c=next(c for c in baseline['cases']if c['label']==label);p=source/f'{label}.input.bin';raw=p.read_bytes();assert sha(raw)==c['input_sha256'];cases.append(dict(name=label,data=raw,cols=2560,source_path=str(p.relative_to(ROOT)),source_sha256=sha(raw)))
 for n,c in [(1,256),(7,512),(8,512),(32,9216),(45,9216),(88,2560),(512,256)]:
  rng=np.random.default_rng(n+c);bits=rng.integers(0,0x7f800000,n*c,dtype=np.uint32);bits|=rng.integers(0,2,n*c,dtype=np.uint32)<<31
  bits[:10]=np.array([0,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff,0x3f000000,0xbf000000,0x42fe0000,0xc2fe0000],dtype=np.uint32)
  cases.append(dict(name=f'finite-{n}-{c}',data=bits.astype('<u4').tobytes(),cols=c,scope='Synthetic finite bit patterns, RNE and padding boundaries'))
 for i,bits in enumerate([0x7f800000,0xff800000,0x7f800001,0x7fc00000,0xff800001,0xffffffff]):
  x=np.ones(1536,dtype='<f4').view('<u4');x[[0,255,256,511,1535,1024][i]]=bits;cases.append(dict(name=f'nonfinite-{bits:08x}',data=x.tobytes(),cols=512,scope='Nonfinite rejection including late block'))
 results=[]
 for case in cases:
  raw=case.pop('data');ip=d/f"{case['name']}.input.bin";ip.write_bytes(raw);expected=json.loads(subprocess.check_output([str(helper),'native',str(ip),str(case['cols'])],text=True,cwd=ROOT));measure=[]
  for method in [0,1]:
   arg=d/f"{case['name']}-{method}.args.bin";subprocess.run([str(helper),'query',str(ip),str(case['cols']),str(method),str(arg)],check=True,cwd=ROOT)
   begin=time.monotonic();reply=subprocess.check_output(['icp','canister','call',a.canister,'measure','--network','local','--identity','imajev-local','--query','--args-file',str(arg),'--args-format','bin','--output','hex'],text=True,cwd=ROOT);elapsed=time.monotonic()-begin
   rp=d/f"{case['name']}-{method}.reply.hex";rp.write_text(reply);got=json.loads(subprocess.check_output([str(helper),'decode',str(rp)],text=True,cwd=ROOT));assert ('error'in got)==('error'in expected)
   if 'error' not in expected:assert got['digest']==expected['digest']
   else:assert got['error']==expected['error']
   got.update(wall_seconds=elapsed,request_bytes=arg.stat().st_size,reply_sha256=sha(reply.encode()));measure.append(got)
  row=dict(**case,input_sha256=sha(raw),input_bytes=len(raw),expected=expected,measurements=measure)
  if 'error' not in expected:row['saved_instructions']=measure[0]['instructions']-measure[1]['instructions'];row['change_percent']=100*(measure[1]['instructions']/measure[0]['instructions']-1)
  results.append(row);print(json.dumps(row),flush=True)
 assert status()==module;assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths}
 with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
 (d/'report.json').write_text(json.dumps(dict(scope=__doc__,source_hashes=hashes,wasm_sha256=module,helper_sha256=sha(helper.read_bytes()),canister=a.canister,ordinary_queries=2*len(cases),source_report_sha256=sha(sr),model=baseline['model'],pack_hash=baseline['pack_hash'],cases=results),indent=2)+'\n')
if __name__=='__main__':main()
