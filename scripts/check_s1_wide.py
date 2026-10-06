#!/usr/bin/env python3
"""Isolated exact all-real-token weight sharing benchmark using saved full-model query inputs.

Preparation uses update, measurements use ordinary query. Never installs or
modifies the validated inference canister. Captures native/output digests and
bookend module hashes; kernel counters exclude digest and Candid encoding.
"""
import argparse,hashlib,json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'client'))
from transport import decode

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--tile",type=int,choices=[64,128,256],required=True);ap.add_argument('--canister',required=True);ap.add_argument('--directory',required=True);ap.add_argument('--build-directory');ap.add_argument('--extra-source',action='append',default=[]);ap.add_argument('--boundary-tokens',default='1,7,8,32,64,88');a=ap.parse_args()
 boundary_tokens=[int(v) for v in a.boundary_tokens.split(',')]
 if not boundary_tokens or len(set(boundary_tokens))!=len(boundary_tokens) or any(not 1<=n<=109 for n in boundary_tokens):raise ValueError('Unique boundary token counts must be in1..109')
 build_dir=ROOT/(a.build_directory or f'artifacts/s1_wide/build{a.tile}')
 d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
 if any(d.iterdir()):raise ValueError('Use a fresh evidence directory')
 helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args';wasm=build_dir/'diagnostic.wasm';sha=lambda b:hashlib.sha256(b).hexdigest()
 sources=list((ROOT/'scripts/s1_wide_bench/src').rglob('*.rs'))+[ROOT/'scripts/build_s1_wide.py',ROOT/'scripts/generate_wat_s1_wide.py',ROOT/f'artifacts/s1_wide/build{a.tile}/kernel.wat',ROOT/'artifacts/s1_address_reuse/build/kernel.wat',ROOT/'scripts/wasm_patch/src/main.rs',pathlib.Path(__file__),ROOT/'MODEL_LOCK.json',ROOT/'checkpoints/full-int8.manifest.json',ROOT/'client/transport.py']
 sources += [ROOT/p for p in a.extra_source]+[build_dir/'kernel.wat']
 hashes={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sources}
 def status():return json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--network','local','--identity','imajev-local','--json'],text=True,cwd=ROOT))['module_hash'].removeprefix('0x')
 expected=sha(wasm.read_bytes());patch=json.loads((build_dir/'wide.patch.json').read_text());build=json.loads((ROOT/f'artifacts/s1_wide/raw{a.tile}/report.json').read_text());control=json.loads((build_dir/'control.patch.json').read_text());assert patch['output_sha256']==expected and patch['original_sha256']==control['output_sha256'] and control['original_sha256']==build['wasm_sha256'];assert patch['source_sha256']==sha((build_dir/'kernel.wat').read_bytes());assert status()==expected
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
 methods=[2,3]
 keys={2:'raw_s1_nozero',3:'wide'}
 for label in ['prefix','617','insufficient','maximum','normal']:
  source=ROOT/f'artifacts/output-pairs-v2-{label}';report_raw=(source/'report.json').read_bytes();report=json.loads(report_raw);query=next(q for q in report['queries'] if q['op'] in ('attention_q_gqa_integer','attention_full_integer') and q['tensor']==tensor);request=source/'queries'/f'{query["index"]:06d}.request.bin';header,values=decode(request.read_bytes());n=header['dims'][0];assert 1<=n<=132;rows=4096 if n>109 else 8192
  input_path=d/f'{label}.input.bin';input_path.write_bytes(values[:n*2560].astype('<f4').tobytes());wpath=d/f'weights-{rows}-native.bin';wpath.write_bytes(weights[:rows*2560]+weights[8192*2560:8192*2560+rows*4]);native=json.loads(subprocess.check_output([str(helper),'native',str(n),str(rows),'2560',str(wpath),str(input_path)],text=True,cwd=ROOT))
  measured={}
  for paired in methods:
   arg=d/f'{label}-{paired}.args.bin';subprocess.run([str(helper),'query',str(paired),str(input_path),str(arg)],check=True);measured[keys[paired]]=call('project',arg,True,'measurement');assert measured[keys[paired]]['digest']==native['digest']
  row=dict(label=label,tokens=n,rows=rows,cols=2560,source_request=str(request.relative_to(ROOT)),source_request_sha256=sha(request.read_bytes()),source_report_sha256=sha(report_raw),input_sha256=sha(input_path.read_bytes()),native=native,measurements=measured)
  row['total_change_percent']=100*(measured[keys[methods[-1]]]['total_instructions']/measured['raw_s1_nozero']['total_instructions']-1);cases.append(row);print(json.dumps(row),flush=True)
 import numpy as np
 for n in boundary_tokens:
  label=f'boundary-{n}';rows=8192
  x=np.resize(np.array([-127.,127.,0.,-0.,-1.,1.,0.5,-0.5,2**-126,-2**-126],dtype='<f4'),n*2560)
  ip=d/f'{label}.input.bin';ip.write_bytes(x.tobytes());wp=d/f'weights-{rows}-native.bin';wp.write_bytes(weights[:rows*2560]+weights[8192*2560:8192*2560+rows*4])
  native=json.loads(subprocess.check_output([str(helper),'native',str(n),str(rows),'2560',str(wp),str(ip)],text=True,cwd=ROOT))
  measured={}
  for paired in methods:
   arg=d/f'{label}-{paired}.args.bin';subprocess.run([str(helper),'query',str(paired),str(ip),str(arg)],check=True);key=keys[paired];measured[key]=call('project',arg,True,'measurement');assert measured[key]['digest']==native['digest']
  row=dict(label=label,tokens=n,rows=rows,cols=2560,input_sha256=sha(ip.read_bytes()),scope='Synthetic signed extremes, zero and tiny inputs with the same real fixed weights; kernel boundary evidence only',native=native,measurements=measured,total_change_percent=100*(measured[keys[methods[-1]]]['total_instructions']/measured['raw_s1_nozero']['total_instructions']-1));cases.append(row);print(json.dumps(row),flush=True)
 assert status()==expected
 assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sources}
 result=dict(candidate_kind='raw-S1-nozero-with-shared-offset-and-fixed-input-pointers',helper_sha256=sha(helper.read_bytes()),baseline_kind='same-module-raw-S1-32-output-nozero',scope='Pure INT8 Q projection: identical S1 arithmetic, control computes32 outputs; candidate widens output tile and reuses input loads across more rows. Same-module counters exclude digest/Candid; not full-model inference.',prepared_pair_bytes=8192*2564,candidate_fixed_bytes=8192*2560,raw_weight_and_scale_bytes=len(weights),diagnostic_fixed_payload_bytes=2*len(weights)+8192*2560,model=m['model'],pack_hash=m['pack_hash'],tensor=tensor,tensor_bytes_sha256=sha(weights),canister=a.canister,wasm_sha256=expected,cached_build=build,patch=patch,source_hashes=hashes,preparation_updates=len(preparations),preparation=preparations,ordinary_queries=len(methods)*len(cases),module_status_update_reads=2,cases=cases)
 import zipfile
 with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in sources:z.write(p,str(p.relative_to(ROOT)))
 (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
