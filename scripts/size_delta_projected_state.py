#!/usr/bin/env python3
"""Exact packet sizing for a proposed projected Delta stage; no kernel/speed claim."""
import hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import encode,decode

def main():
 baseline='blake3-v1';rows=[]
 for label in ['prefix','617','insufficient','maximum','normal']:
  directory=ROOT/f'artifacts/{baseline}-{label}';r=json.loads((directory/'report.json').read_text());qs=r['queries'];prefix='model.language_model.layers.0.linear_attn'
  q=next(q for q in qs if q['tensor']==prefix+'.in_proj_qkv.weight');h,x=decode((directory/'queries'/f'{q["index"]:06d}.request.bin').read_bytes());n=h['dims'][0]
  # Capture input is tag0/plain; only input values enter packet sizing.
  assert x.size==n*2560
  keep=label=='prefix'
  if label in ('prefix','normal'):
   conv=np.zeros((3,8192),np.float32);states=np.zeros((32,128,128),np.float32)
  else:
   with np.load(ROOT/f'artifacts/{baseline}-prefix/queries/states/layer-00.npz') as f:conv=f['conv'].copy();states=f['delta'].copy()
  outq=next(q for q in qs if q['tensor']==prefix+'.out_proj.weight');_,out=decode((directory/'queries'/f'{outq["index"]:06d}.request.bin').read_bytes());out=out.reshape(n,32,128)
  with np.load(directory/'queries/states/layer-00.npz') as f:new_conv=f['conv'].copy();new_state=f['delta'].copy() if keep else None
  for first in [0,16]:
   indices=np.concatenate([np.arange(first//2*128,(first//2+8)*128),np.arange(2048+first//2*128,2048+(first//2+8)*128),np.arange(4096+first*128,4096+(first+16)*128)])
   header=dict(h,op='delta_project_stage_integer',dims=[n,16,first,int(keep)],scalars=[],aux=[],encoding='bf16-block256-exact-v1')
   packet=encode(header,np.concatenate([x,conv[:,indices].ravel(),states[first:first+16].ravel()]))
   reply=encode(header,np.concatenate([out[:,first:first+16].ravel(),new_conv[:,indices].ravel(),*([new_state[first:first+16].ravel()] if keep else [])]))
   # Conservative raw byte budget for lossless INT8 capture plus four rank64
   # A products and block scales, without actually computing/encoding them.
   capture_bytes=n*2560+4*n*(2560//256+4*64)
   projection_mac=n*((4096+2048+64)*2560+64*(4096+2048+2*2560+2*(32+2560)))
   rows.append(dict(label=label,n=n,head_start=first,heads=16,keep_state=keep,input_bytes=len(packet),reply_bytes=len(reply),capture_byte_upper_bound=capture_bytes+1024,reply_with_capture_upper_bound=len(reply)+capture_bytes+1024,projection_mac_estimate=projection_mac,source_report_sha256=hashlib.sha256((directory/'report.json').read_bytes()).hexdigest()))
 report=dict(cases=rows,all_packets_under_2M=all(max(c['input_bytes'],c['reply_with_capture_upper_bound'])<2_000_000 for c in rows),scope='Real BF16/F32 values, hypothetical op and fixed rank64 capture byte budget only; no numerical implementation, query count, instruction or accuracy evidence',script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
 dest=ROOT/'artifacts/attention-fusion/delta-projected-packet-sizes.json';dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
