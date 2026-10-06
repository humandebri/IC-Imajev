#!/usr/bin/env python3
"""Compare malformed shape rejection and legal primitive frames in ordinary queries."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,encode
from prefix_inference import verify_module

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--module',required=True);ap.add_argument('--label',required=True);a=ap.parse_args()
 d=ROOT/'artifacts/guarded-release-v2/full-proof-v1/boundaries'/a.label;d.mkdir(parents=True,exist_ok=False)
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());cases=[]
 for label,dims in [('wrap_2pow32',[262144,16384]),('wrap_square',[65536,65536]),('wrap_unsigned',[2147483648,2]),('zero_huge',[0,4294967295]),('huge_one',[1,4294967295]),('zero_width',[1,0]),('size_mismatch',[1,2560])]:cases.append((label,'rms_scaled',dims,[],[1e-6,1.],0,False))
 cases += [('add_norm_wrap','add_norm_bf16',[262144,16384],[],[1e-6],0,False),('matmul_wrap','matmul',[0,65536,65536],[],[],0,False),('matmul_work','matmul',[32768,32768,4],[],[],0,False),('progress_wrap','bf16',[],[0.5],[],2**64-1,False),('bad_norm_epsilon','rms_scaled',[1,1],[1.],[-1.,1.],0,False),('legal_norm','rms_scaled',[1,2560],np.ones(2560,dtype='<f4'),[1e-6,1.],0,True),('legal_empty_norm','rms_scaled',[0,2560],[],[1e-6,1.],0,True),('legal_large_flat','add_bf16',[450000],np.zeros(900000,dtype='<f4'),[],0,True)]
 t=Transport(m['model'],'http://localhost:8001/','6eydd-o3777-77775-aaama-cai',str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],bridge_binary=str(ROOT/'artifacts/query-packing-v3/build/imajev-client'))
 rows=[]
 try:
  verify_module(t,a.module)
  for label,op,dims,values,scalars,step,accepted in cases:
   h=dict(version=3,model=m['model'],pack_hash=m['pack_hash'],input_hash='0'*64,step=step,op=op,tensor='',dims=dims,scalars=scalars,encoding='bf16-block256-exact-v1')
   request=d/(label+'.request.bin');response=d/(label+'.response.bin');request.write_bytes(encode(h,np.asarray(values,dtype='<f4')))
   try:reply=t.command(dict(op='step',input=str(request),output=str(response)));row=dict(accepted=True,reply=reply,response_sha256=sha(response))
   except RuntimeError as error:row=dict(accepted=False,error=str(error))
   row.update(case=label,request_sha256=sha(request),expected_accepted=accepted);rows.append(row)
   (d/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
   assert row['accepted']==accepted,(label,row)
  verify_module(t,a.module)
 finally:t.close()
 report=dict(module=a.module,cases=rows,ordinary_queries=len(rows),all_acceptance_checks_passed=True,scope=__doc__);(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(module=a.module,queries=len(rows),passed=True)))
if __name__=='__main__':main()
