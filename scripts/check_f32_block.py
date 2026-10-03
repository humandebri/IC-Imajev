#!/usr/bin/env python3
"""F32 LoRA A diagnostic: owner-prepared weights, ordinary queries, native bit oracle.

Saved canister-produced add_norm_bf16 output values are used as real kernel inputs.
The normal 132-token case has no whole prepared MLP; 132 is a synthetic boundary,
not a full-inference or judgment proof. Counters exclude digest and Candid.
"""
import argparse,hashlib,json,pathlib,subprocess,sys,time,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import decode,encode,Transport
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--canister',required=True);ap.add_argument('--directory',default='artifacts/f32_block/check');a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);helper=ROOT/'artifacts/f32-block-native/release/f32_args';wasm=ROOT/'artifacts/f32_block/build/diagnostic.wasm';sha=lambda b:hashlib.sha256(b).hexdigest()
paths=list((ROOT/'scripts/f32_block_bench/src').rglob('*.rs'))+[ROOT/p for p in ['scripts/f32_block_bench/Cargo.toml','scripts/f32_block_bench/Cargo.lock','scripts/build_f32_block_cached.py','scripts/generate_f32_block.py','scripts/wasm_patch/src/main.rs','MODEL_LOCK.json','checkpoints/full-int8.manifest.json','client/transport.py','artifacts/f32_block/build/kernel.wat']]+[pathlib.Path(__file__)]
hashes={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths}
def status():return json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--network','local','--identity','imajev-local','--json'],text=True,cwd=ROOT))['module_hash'].removeprefix('0x')
expected=sha(wasm.read_bytes());assert status()==expected
patch=json.loads((ROOT/'artifacts/f32_block/build/patch.json').read_text());build=json.loads((ROOT/'artifacts/f32_block/cached-build/report.json').read_text());assert patch['output_sha256']==expected and patch['original_sha256']==build['wasm_sha256'] and patch['source_sha256']==sha((ROOT/'artifacts/f32_block/build/kernel.wat').read_bytes())
count=0
def call(method,arg=None,query=False,kind='preparation'):
 global count
 count+=1;cmd=['icp','canister','call',a.canister,method,'--network','local','--identity','imajev-local','--output','hex'];cmd+=['--args-file',str(arg),'--args-format','bin']if arg else['()']
 if query:cmd+=['--query']
 begin=time.monotonic();raw=subprocess.check_output(cmd,text=True,cwd=ROOT);seconds=time.monotonic()-begin;p=d/f'{count:04d}-{method}.hex';p.write_text(raw);r=json.loads(subprocess.check_output([str(helper),'decode',str(p),kind],text=True,cwd=ROOT));r.update(wall_seconds=seconds,request_candid_bytes=arg.stat().st_size if arg else 6,reply_hex_sha256=sha(raw.encode()));return r
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());assert m['model']==sha((ROOT/'MODEL_LOCK.json').read_bytes());name='model.language_model.layers.0.mlp.gate_proj.lora_A.weight';w=next(t for t in m['tensors']if t['name']==name);assert (w['rows'],w['cols'],w['dtype'])==(64,2560,'f32')
with (ROOT/'checkpoints/full-int8.pack').open('rb')as f:f.seek(w['offset']);raw=f.read(w['bytes'])
assert len(raw)==2560*64*4;rows,cols=w['rows'],w['cols'];wp=d/'weights.bin';wp.write_bytes(raw);prep=[]
for start in range(0,rows,32):
 p=d/f'weights-{start}.bin';p.write_bytes(raw[start*cols*4:(start+32)*cols*4]);arg=d/f'weights-{start}.args.bin';subprocess.run([str(helper),'chunk',str(start),str(p),str(arg)],check=True);prep.append(call('prepare_chunk',arg))
prep.append(call('seal'));cases=[]
def measure(label,x,scope,source=None):
 n=x.size//cols;ip=d/f'{label}.input.bin';ip.write_bytes(x.astype('<f4').tobytes());native=json.loads(subprocess.check_output([str(helper),'native',str(n),str(rows),str(cols),str(wp),str(ip)],text=True,cwd=ROOT));measured={}
 for method,key in [(0,'baseline'),(1,'reuse')]:
  arg=d/f'{label}-{method}.args.bin';subprocess.run([str(helper),'query',str(method),str(ip),str(arg)],check=True);measured[key]=call('project',arg,True,'measurement');assert measured[key]['digest']==native['digest']
 r=dict(label=label,tokens=n,rows=rows,cols=cols,scope=scope,input_sha256=sha(ip.read_bytes()),native=native,measurements=measured,total_change_percent=100*(measured['reuse']['total_instructions']/measured['baseline']['total_instructions']-1))
 if source:r.update(source)
 cases.append(r);print(json.dumps(r),flush=True)
norm_transport=Transport(m['model'],'http://localhost:8001/','6eydd-o3777-77775-aaama-cai',str(ROOT/'artifacts/imajev-local.pem'),d/'norm-source',m['pack_hash'])
full_wasm=ROOT/'artifacts/prefix_codec/full-build-f32-output/full.wasm';full_hash=sha(full_wasm.read_bytes());assert norm_transport.command({'op':'module_hash'})['ok']['module_hash']==full_hash
try:
 for label in ['prefix','617','insufficient','maximum']:
  folder=ROOT/f'artifacts/mlp-pipeline-v1-{label}';rb=(folder/'report.json').read_bytes();report=json.loads(rb);assert report['model']==m['model'] and report['pack_hash']==m['pack_hash'];q=next(q for q in report['queries']if q['op']=='mlp_prepare_down'and '.layers.0.'in q['tensor']);p=folder/'queries'/f'{q["index"]:06d}.request.bin';h,v=decode(p.read_bytes());n=h['dims'][0]
  header=dict(h,op='add_norm_bf16',dims=[n,2560],scalars=[1e-6],aux=[],encoding='bf16-block256-exact-v1');arg=d/f'{label}-norm.request.bin';out=d/f'{label}-norm.response.bin';arg.write_bytes(encode(header,v));metric=norm_transport.command(dict(op='step',input=str(arg),output=str(out)));_,norm=decode(out.read_bytes());assert norm.size==2*n*cols
  measure(label,norm[n*cols:],'Actual canister-produced layer0 normalized MLP input and original gate-A weights',dict(source_request=str(p.relative_to(ROOT)),source_request_sha256=sha(p.read_bytes()),source_report_sha256=sha(rb),norm_module_hash=full_hash,norm_response_sha256=sha(out.read_bytes()),norm_metric=metric))
 assert norm_transport.command({'op':'module_hash'})['ok']['module_hash']==full_hash
finally:norm_transport.close()
for n in [1,7,8,32,64,88,132]:
 x=np.resize(np.array([-0.,0.,1.,-1.,0.1234567,-0.99999994,2**-126,-2**-126],dtype='<f4'),n*cols);measure(f'boundary-{n}',x,'Synthetic signed/zero/subnormal input; no full-model inference claim')
assert status()==expected;assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths};r=dict(scope=__doc__,canister=a.canister,wasm_sha256=expected,model=m['model'],pack_hash=m['pack_hash'],tensor=name,tensor_bytes_sha256=sha(raw),helper_sha256=sha(helper.read_bytes()),source_hashes=hashes,cached_build=build,patch=patch,preparation_updates=len(prep),preparation=prep,candidate_fixed_bytes=len(raw),diagnostic_fixed_bytes=2*len(raw),ordinary_queries=2*len(cases),cases=cases)
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
 for p in paths:z.write(p,str(p.relative_to(ROOT)))
