#!/usr/bin/env python3
"""Measure real stable read instructions and independently verify exact coefficient bytes."""
from pathlib import Path
import argparse,json,hashlib,subprocess,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);arg=ap.parse_args();d=ROOT/'artifacts/s2-k2-prepacked-stable-probe-v1';out=d/'stable-read-check';out.mkdir(exist_ok=False)
 b=json.loads((d/'build/report.json').read_text());status=json.loads(subprocess.check_output(['icp','canister','status',arg.canister,'--network','local','--identity','imajev-local','--json'],text=True));assert status['module_hash'].removeprefix('0x')==b['wasm_sha256']
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());t=next(x for x in m['tensors']if x['name']=='model.language_model.layers.3.self_attn.q_proj.weight');assert(t['rows'],t['cols'])==(8192,2560)
 with (ROOT/'checkpoints/full-int8.pack').open('rb')as f:f.seek(t['offset']);raw=f.read(t['rows']*t['cols'])
 w=np.frombuffer(raw,dtype=np.int8).astype(np.int16).reshape(512,4,4,10,4,64).transpose(0,3,4,2,1,5).reshape(512,10,16,4,64)
 plan=ROOT/'artifacts/s2-k2-prepacked-kernels-v1/plan.py';ns={'__name__':'rank49','__file__':str(plan)};exec(compile(plan.read_text(),str(plan),'exec'),ns);_,bd,_,leaves,*_=ns['plan']();coeff=np.empty((512,10,49,32,4,2),dtype='<i2')
 for mi,(_,name)in enumerate(leaves):coeff[:,:,mi]=sum(v*w[:,:,i].astype(np.int32)for i,v in bd.symbols[name].items()).reshape(512,10,4,32,2).transpose(0,1,3,2,4)
 data=memoryview(coeff).cast('B');assert len(data)==128450560
 did=out/'read.did';did.write_text('type M=record{digest:vec nat8;quantize_instructions:nat64;input_prepare_instructions:nat64;project_instructions:nat64;total_instructions:nat64;output_values:nat64;heap_pages:nat64};service:{coefficient_read_cost:(nat32)->(M)query;}')
 helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args';cases=[]
 for count in [0,16,4096,31360,156800,313600,1048576,16777216]:
  reply=out/f'{count}.reply.hex';raw_reply=subprocess.check_output(['icp','canister','call',arg.canister,'coefficient_read_cost',f'({count}:nat32)','--network','local','--identity','imajev-local','--candid',str(did),'--query','--output','hex'],text=True);reply.write_text(raw_reply)
  result=json.loads(subprocess.check_output([str(helper),'decode',str(reply),'measurement'],text=True));expected=list(hashlib.sha256(data[:count]).digest());assert result['digest']==expected and result['output_values']==count;assert result['total_instructions']==result['project_instructions']and result['quantize_instructions']==result['input_prepare_instructions']==0
  cases.append(dict(bytes=count,instructions=result['total_instructions'],measurement=result,reply=str(reply.relative_to(ROOT)),reply_sha256=sha(reply)));print(json.dumps(dict(bytes=count,instructions=result['total_instructions'])),flush=True)
 files=[Path(__file__),d/'build/report.json',plan,did,helper]+[ROOT/c['reply']for c in cases];r=dict(module=b['wasm_sha256'],canister=arg.canister,precomputed_bytes=len(data),all_coefficient_byte_digests_equal=True,cases=cases,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Stable read interval only, allocation/digest/Candid excluded. Inference still reads heap coefficients; no stable-backed inference or full performance claim.')
 (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
