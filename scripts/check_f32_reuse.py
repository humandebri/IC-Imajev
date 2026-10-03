#!/usr/bin/env python3
"""F32 LoRA B diagnostic: owner-prepared weights, ordinary queries, native bit oracle.

Saved canister-produced LoRA down-A values are used as real kernel inputs.
The normal 132-token case has no whole prepared MLP; 132 is a synthetic boundary,
not a full-inference or judgment proof. Counters exclude digest and Candid.
"""
import argparse,hashlib,json,pathlib,subprocess,sys,time,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import decode
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--canister',required=True);ap.add_argument('--directory',default='artifacts/f32_reuse/check');a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);helper=ROOT/'artifacts/f32-reuse-native/release/f32_args';wasm=ROOT/'artifacts/f32_reuse/build/diagnostic.wasm';sha=lambda b:hashlib.sha256(b).hexdigest()
paths=list((ROOT/'scripts/f32_reuse_bench/src').rglob('*.rs'))+[ROOT/p for p in ['scripts/f32_reuse_bench/Cargo.toml','scripts/f32_reuse_bench/Cargo.lock','scripts/build_f32_reuse_cached.py','scripts/generate_f32_reuse.py','scripts/wasm_patch/src/main.rs','MODEL_LOCK.json','checkpoints/full-int8.manifest.json','client/transport.py','artifacts/f32_reuse/build/kernel.wat']]+[pathlib.Path(__file__)]
hashes={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths}
def status():return json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--network','local','--identity','imajev-local','--json'],text=True,cwd=ROOT))['module_hash'].removeprefix('0x')
expected=sha(wasm.read_bytes());assert status()==expected
patch=json.loads((ROOT/'artifacts/f32_reuse/build/patch.json').read_text());build=json.loads((ROOT/'artifacts/f32_reuse/cached-build/report.json').read_text());assert patch['output_sha256']==expected and patch['original_sha256']==build['wasm_sha256'] and patch['source_sha256']==sha((ROOT/'artifacts/f32_reuse/build/kernel.wat').read_bytes())
count=0
def call(method,arg=None,query=False,kind='preparation'):
 global count
 count+=1;cmd=['icp','canister','call',a.canister,method,'--network','local','--identity','imajev-local','--output','hex'];cmd+=['--args-file',str(arg),'--args-format','bin']if arg else['()']
 if query:cmd+=['--query']
 begin=time.monotonic();raw=subprocess.check_output(cmd,text=True,cwd=ROOT);seconds=time.monotonic()-begin;p=d/f'{count:04d}-{method}.hex';p.write_text(raw);r=json.loads(subprocess.check_output([str(helper),'decode',str(p),kind],text=True,cwd=ROOT));r.update(wall_seconds=seconds,request_candid_bytes=arg.stat().st_size if arg else 6,reply_hex_sha256=sha(raw.encode()));return r
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());assert m['model']==sha((ROOT/'MODEL_LOCK.json').read_bytes());name='model.language_model.layers.0.mlp.down_proj.lora_B.weight';w=next(t for t in m['tensors']if t['name']==name);assert (w['rows'],w['cols'],w['dtype'])==(2560,64,'f32')
with (ROOT/'checkpoints/full-int8.pack').open('rb')as f:f.seek(w['offset']);raw=f.read(w['bytes'])
assert len(raw)==2560*64*4;wp=d/'weights.bin';wp.write_bytes(raw);prep=[]
for start in range(0,2560,512):
 p=d/f'weights-{start}.bin';p.write_bytes(raw[start*64*4:(start+512)*64*4]);arg=d/f'weights-{start}.args.bin';subprocess.run([str(helper),'chunk',str(start),str(p),str(arg)],check=True);prep.append(call('prepare_chunk',arg))
prep.append(call('seal'));cases=[]
def measure(label,x,scope,source=None):
 n=x.size//64;ip=d/f'{label}.input.bin';ip.write_bytes(x.astype('<f4').tobytes());native=json.loads(subprocess.check_output([str(helper),'native',str(n),'2560','64',str(wp),str(ip)],text=True,cwd=ROOT));measured={}
 for method,key in [(0,'baseline'),(1,'reuse')]:
  arg=d/f'{label}-{method}.args.bin';subprocess.run([str(helper),'query',str(method),str(ip),str(arg)],check=True);measured[key]=call('project',arg,True,'measurement');assert measured[key]['digest']==native['digest']
 r=dict(label=label,tokens=n,rows=2560,rank=64,scope=scope,input_sha256=sha(ip.read_bytes()),native=native,measurements=measured,total_change_percent=100*(measured['reuse']['total_instructions']/measured['baseline']['total_instructions']-1))
 if source:r.update(source)
 cases.append(r);print(json.dumps(r),flush=True)
for label in ['prefix','617','insufficient','maximum']:
 folder=ROOT/f'artifacts/mlp-pipeline-v1-{label}';rb=(folder/'report.json').read_bytes();report=json.loads(rb);assert report['model']==m['model'] and report['pack_hash']==m['pack_hash'];q=next(q for q in report['queries']if q['op']=='mlp_prepare_down'and '.layers.0.'in q['tensor']);p=folder/'queries'/f'{q["index"]:06d}.response.bin';h,v=decode(p.read_bytes());n=h['dims'][0];ax=v[n*(2560+9216+36):];assert ax.size==n*64
 measure(label,ax,'Actual canister-produced layer0 LoRA down-A input, paired with original down-B weights',dict(source_response=str(p.relative_to(ROOT)),source_response_sha256=sha(p.read_bytes()),source_report_sha256=sha(rb)))
for n in [1,7,8,32,64,88,132]:
 x=np.resize(np.array([-0.,0.,1.,-1.,0.1234567,-0.99999994,2**-126,-2**-126],dtype='<f4'),n*64);measure(f'boundary-{n}',x,'Synthetic signed/zero/subnormal input; no full-model inference claim')
assert status()==expected;assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths};r=dict(scope=__doc__,canister=a.canister,wasm_sha256=expected,model=m['model'],pack_hash=m['pack_hash'],tensor=name,tensor_bytes_sha256=sha(raw),helper_sha256=sha(helper.read_bytes()),source_hashes=hashes,cached_build=build,patch=patch,preparation_updates=len(prep),preparation=prep,candidate_fixed_bytes=len(raw),diagnostic_fixed_bytes=2*len(raw),ordinary_queries=2*len(cases),cases=cases)
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
 for p in paths:z.write(p,str(p.relative_to(ROOT)))
