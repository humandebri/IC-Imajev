#!/usr/bin/env python3
"""Isolated exact fixed-weight-bounded I16 benchmark using saved full-model query inputs.

Preparation uses update, measurements use ordinary query. Never installs or
modifies the validated inference canister. Captures native/output digests and
bookend module hashes; kernel counters exclude digest and Candid encoding.
"""
import argparse,hashlib,json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'client'))
from transport import decode

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--directory',default='artifacts/bounded_i16/check');a=ap.parse_args()
 d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);helper=ROOT/'artifacts/bounded_i16/native/release/bounded_args';wasm=ROOT/'artifacts/bounded_i16/diagnostic.wasm';sha=lambda b:hashlib.sha256(b).hexdigest()
 sources=list((ROOT/'scripts/bounded_i16_bench/src').rglob('*.rs'))+[ROOT/'scripts/bounded_i16_bench/Cargo.toml',ROOT/'scripts/bounded_i16_bench/Cargo.lock',ROOT/'scripts/generate_bounded_i16.py',ROOT/'scripts/build_bounded_i16_cached.py',pathlib.Path(__file__),ROOT/'crates/imajev-runtime/Cargo.toml',ROOT/'Cargo.toml',ROOT/'Cargo.lock',ROOT/'MODEL_LOCK.json',ROOT/'checkpoints/full-int8.manifest.json',ROOT/'client/transport.py']+list((ROOT/'crates/imajev-runtime/src').glob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']
 hashes={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sources}
 def status():return json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--network','local','--identity','imajev-local','--json'],text=True,cwd=ROOT))['module_hash'].removeprefix('0x')
 expected=sha(wasm.read_bytes());assert status()==expected
 def call(method,path=None,query=False,decode_as='preparation'):
  cmd=['icp','canister','call',a.canister,method,'--network','local','--identity','imajev-local','--output','hex']
  if path:cmd+=['--args-file',str(path),'--args-format','bin']
  if query:cmd+=['--query']
  if path is None:cmd+=['()']
  start=time.monotonic();raw=subprocess.check_output(cmd,text=True,cwd=ROOT);elapsed=time.monotonic()-start
  reply=d/f'{method}-{len(list(d.glob("*.hex"))):04d}.hex';reply.write_text(raw)
  result=json.loads(subprocess.check_output([str(helper),'decode',str(reply),decode_as],text=True,cwd=ROOT));result.update(wall_seconds=elapsed,reply_hex_sha256=sha(raw.encode()),request_candid_bytes=path.stat().st_size if path else 6);return result
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());tensor='model.language_model.layers.3.self_attn.q_proj.weight';t=next(t for t in m['tensors'] if t['name']==tensor);assert(t['rows'],t['cols'],t['dtype'])==(8192,2560,'int8')
 with (ROOT/'checkpoints/full-int8.pack').open('rb') as f:f.seek(t['offset']);weights=f.read(t['bytes'])
 assert len(weights)==8192*2564
 assert m['model']==sha((ROOT/'MODEL_LOCK.json').read_bytes())
 assert sha(weights)==json.loads((ROOT/'artifacts/column32/check/report.json').read_text())['tensor_bytes_sha256']
 preparations=[]
 for start in range(0,8192,128):
  raw=weights[start*2560:(start+128)*2560]+weights[8192*2560+start*4:8192*2560+(start+128)*4];src=d/f'weights-{start:04d}.bin';src.write_bytes(raw);arg=d/f'weights-{start:04d}.args.bin';subprocess.run([str(helper),'chunk',str(start),str(src),str(arg)],check=True)
  preparations.append(call('prepare_chunk',arg))
 preparations.append(call('seal'))
 cases=[]
 for label in ['prefix','617','insufficient','maximum','normal']:
  source=ROOT/f'artifacts/output-pairs-v2-{label}';report_raw=(source/'report.json').read_bytes();report=json.loads(report_raw);query=next(q for q in report['queries'] if q['op'] in ('attention_q_gqa_integer','attention_full_integer') and q['tensor']==tensor);request=source/'queries'/f'{query["index"]:06d}.request.bin';header,values=decode(request.read_bytes());n=header['dims'][0];assert 1<=n<=132;rows=4096 if n>109 else 8192
  input_path=d/f'{label}.input.bin';input_path.write_bytes(values[:n*2560].astype('<f4').tobytes());wpath=d/f'weights-{rows}-native.bin';wpath.write_bytes(weights[:rows*2560]+weights[8192*2560:8192*2560+rows*4]);native=json.loads(subprocess.check_output([str(helper),'native',str(n),str(rows),'2560',str(wpath),str(input_path)],text=True,cwd=ROOT))
  measured={}
  for paired in [False,True]:
   arg=d/f'{label}-{paired}.args.bin';subprocess.run([str(helper),'query',str(paired).lower(),str(input_path),str(arg)],check=True);measured['candidate' if paired else 'baseline']=call('project',arg,True,'measurement');assert measured['candidate' if paired else 'baseline']['digest']==native['digest']
  row=dict(label=label,tokens=n,rows=rows,cols=2560,source_request=str(request.relative_to(ROOT)),source_request_sha256=sha(request.read_bytes()),source_report_sha256=sha(report_raw),input_sha256=sha(input_path.read_bytes()),native=native,measurements=measured)
  row['total_change_percent']=100*(measured['candidate']['total_instructions']/measured['baseline']['total_instructions']-1);cases.append(row);print(json.dumps(row),flush=True)
 import numpy as np
 for n in [1,7,8,32,64,88]:
  label=f'boundary-{n}';rows=8192
  x=np.resize(np.array([-127.,127.,0.,-0.,-1.,1.,0.5,-0.5,2**-126,-2**-126],dtype='<f4'),n*2560)
  ip=d/f'{label}.input.bin';ip.write_bytes(x.tobytes());wp=d/f'weights-{rows}-native.bin';wp.write_bytes(weights[:rows*2560]+weights[8192*2560:8192*2560+rows*4])
  native=json.loads(subprocess.check_output([str(helper),'native',str(n),str(rows),'2560',str(wp),str(ip)],text=True,cwd=ROOT))
  measured={}
  for paired in [False,True]:
   arg=d/f'{label}-{paired}.args.bin';subprocess.run([str(helper),'query',str(paired).lower(),str(ip),str(arg)],check=True);key='candidate' if paired else 'baseline';measured[key]=call('project',arg,True,'measurement');assert measured[key]['digest']==native['digest']
  row=dict(label=label,tokens=n,rows=rows,cols=2560,input_sha256=sha(ip.read_bytes()),scope='Synthetic signed extremes, zero and tiny inputs with the same real fixed weights; kernel boundary evidence only',native=native,measurements=measured,total_change_percent=100*(measured['candidate']['total_instructions']/measured['baseline']['total_instructions']-1));cases.append(row);print(json.dumps(row),flush=True)
 assert status()==expected
 assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sources}
 result=dict(candidate_kind='exact-weight-bounded-i16-window16-32',helper_sha256=sha(helper.read_bytes()),baseline_kind='adopted-prepared-output-pairs-real-token-tiles',scope='Pure INT8 base Q projection only. Original, adopted pair layout and candidate INT8 paired weights and fixed overflow modes are retained in this diagnostic; full-model 4GiB placement is NOT proven. Query-dependent operand preparation is charged per query. Excludes LoRA, full model, digest/Candid and judgment evaluation.',prepared_pair_bytes=8192*2564,candidate_fixed_bytes=8192*2560+8192//2*10,raw_weight_and_scale_bytes=len(weights),diagnostic_fixed_payload_bytes=2*len(weights)+8192*2560+8192//2*10,model=m['model'],pack_hash=m['pack_hash'],tensor=tensor,tensor_bytes_sha256=sha(weights),canister=a.canister,wasm_sha256=expected,cached_build=json.loads((ROOT/'artifacts/bounded_i16/cached-build/report.json').read_text()),source_hashes=hashes,preparation_updates=len(preparations),preparation=preparations,ordinary_queries=2*len(cases),module_status_update_reads=2,cases=cases)
 (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
