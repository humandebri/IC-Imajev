#!/usr/bin/env python3
"""Probe every saved integer shape, recording instruction-limit failures too."""
import argparse,hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode,encode,is_instruction_limit
import subprocess
from prefix_inference import verify_module
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--output',required=True);ap.add_argument('--source-directory',default='artifacts/layout-full-617');ap.add_argument('--op',help='Probe this operation instead of integer projections');ap.add_argument('--token-scale',action='store_true');args=ap.parse_args()
 source=ROOT/args.source_directory;m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());r=json.loads((source/('first-report.json' if (source/'first-report.json').exists() else 'report.json')).read_text());dest=ROOT/args.output;dest.mkdir(parents=True,exist_ok=True);cases=[];seen=set()
 t=Transport(m['model'],'http://localhost:8001/',args.canister,str(ROOT/'artifacts/imajev-local.pem'),dest,m['pack_hash'])
 wasm_hash=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest()
 try:
  verify_module(t,wasm_hash)
  for q in r['queries']:
   if (q['op'] != args.op if args.op else 'integer' not in q['op']):continue
   p=source/'queries'/f"{q['index']:06d}";h,_=decode(p.with_suffix('.request.bin').read_bytes());key=q['op'],tuple(h['dims'][:3])
   if key in seen:continue
   if args.token_scale and q['op'] not in ('lora_integer','linear_integer_bf16','mlp_gate_up_integer'):continue
   seen.add(key);c=dict(op=q['op'],dims=h['dims'],before_instructions=q['ok']['instructions'])
   try:
    request=p.with_suffix('.request.bin')
    if args.token_scale:
     h,x=decode(request.read_bytes());h['op']=('linear_integer_token_bf16' if q['op']=='linear_integer_bf16' else q['op']+'_token');request=dest/f"{q['index']}.request.bin";request.write_bytes(encode(h,x))
    out=dest/f"{q['index']}.response.bin";result=t.command(dict(op='step',input=str(request),output=str(out)))
    _,a=decode(out.read_bytes());_,before=decode(p.with_suffix('.response.bin').read_bytes())
    if args.token_scale:
     native=dest/f"{q['index']}.native.bin";subprocess.run([str(ROOT/'target/release/primitive'),str(request),str(native),str(ROOT/'checkpoints/full-int8.manifest.json'),str(ROOT/'checkpoints/full-int8.pack')],check=True);_,b=decode(native.read_bytes());assert np.array_equal(a.view(np.uint32),b.view(np.uint32))
     c.update(bitwise_equal_to_native=True,arithmetic_changed=True,max_error_to_block256=float(abs(a-before).max()))
    else:
     assert np.array_equal(a.view(np.uint32),before.view(np.uint32));c['bitwise_equal']=True
    c['after_instructions']=result['ok']['instructions']
   except RuntimeError as e:
    if not is_instruction_limit(e):raise
    c.update(instruction_limit_exceeded=True)
   cases.append(c);print(c,flush=True)
  verify_module(t,wasm_hash)
  (dest/'report.json').write_text(json.dumps(dict(cases=cases,wasm_sha256=wasm_hash,deployed_wasm_sha256=wasm_hash),indent=2)+'\n')
 finally:t.close()
if __name__=='__main__':main()
