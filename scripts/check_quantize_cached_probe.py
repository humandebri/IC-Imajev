#!/usr/bin/env python3
"""Check all quantized values/scales against independent F32 divide/RNE/clamp oracle."""
from pathlib import Path
import hashlib,json,subprocess,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/quantize-cached-v1';CAN='zm54s-at777-77775-aaa5q-cai';HELPER=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def oracle(v):
 v=np.asarray(v,dtype='<f4').reshape(-1,256);bits=v.view('<u4')&0x7fffffff;peak=bits.max(axis=1)
 if np.any(peak>=0x7f800000):return b'ERR:integer projection input',0
 with np.errstate(all='ignore'):
  maximum=peak.view('<f4');scale=np.maximum(np.divide(maximum,np.float32(127),dtype=np.float32),np.float32(np.nextafter(np.float32(0),np.float32(1))));scale[maximum==0]=np.float32(1)
  q=np.clip(np.rint(np.divide(v,scale[:,None],dtype=np.float32)),-127,127).astype('<i2')
 return scale.astype('<f4').tobytes()+q.tobytes(),v.size
def main():
 d=D/'check';d.mkdir(exist_ok=False);b=json.loads((D/'build/report.json').read_text());assert sha(D/'build/diagnostic.wasm')==b['module']
 for m in [b['source_hashes'],b['dependency_hashes']]:assert all(sha(ROOT/p)==h for p,h in m.items())
 did=d/'diagnostic.did';did.write_text('type M=record{digest:vec nat8;quantize_instructions:nat64;input_prepare_instructions:nat64;project_instructions:nat64;total_instructions:nat64;output_values:nat64;heap_pages:nat64};service:{project:(vec nat8,nat8)->(M)query}')
 def status():return json.loads(subprocess.check_output(['icp','canister','status',CAN,'--network','local','--identity','imajev-local','--json'],text=True))['module_hash'].removeprefix('0x')
 assert status()==b['module'];cases=[]
 def add(label,values,scope='Synthetic boundary input'):cases.append((label,np.resize(np.asarray(values,dtype='<f4'),max(256,(len(values)+255)//256*256)),scope))
 add('zeros',[0.,-0.]);add('ties',[-127.,-126.5,-125.5,-64.5,-63.5,-1.5,-.5,.5,1.5,63.5,64.5,125.5,126.5,127.]);add('maximum',np.array([0x7f7fffff,0xff7fffff,0x00800000,0x00000001,0x80000001],dtype='<u4').view('<f4'))
 for m in [1,63,127,190,191,255,1024,0x7fffff]:add('subnormal-'+str(m),np.array([m,m|0x80000000,1,0,0x80000000],dtype='<u4').view('<f4'))
 threshold=np.float32(np.float32(127)*np.finfo(np.float32).tiny);tb=threshold.view('<u4').item()
 for offset in [-2,-1,0,1,2]:add('normal-scale-bound-'+str(offset),np.array([tb+offset,(tb+offset)|0x80000000,1,0],dtype='<u4').view('<f4'))
 for bad in [0x7f800000,0xff800000,0x7fc00000,0x7f800001,0xff800001]:add('nonfinite-'+hex(bad),np.array([1,0,bad],dtype='<u4').view('<f4'))
 rng=np.random.default_rng(7149)
 for n in [48,56,57]:
  values=rng.standard_normal(n*2560).astype('<f4');add('shape-'+str(n),values,'Generated normal F32, whole n*2560 quantization only')
 for label in ['prefix','617','normal']:
  p=ROOT/'artifacts/remaining-efficiency/odd-check-v2'/f'{label}.input.bin';v=np.frombuffer(p.read_bytes(),dtype='<f4');assert v.size%256==0;add('saved-'+label,v,'Saved real Q projection input: '+str(p.relative_to(ROOT)))
 results=[]
 for label,values,scope in cases:
  ip=d/f'{label}.input.bin';ip.write_bytes(values.tobytes());expected,count=oracle(values);ep=d/f'{label}.oracle.bin';ep.write_bytes(expected);measurements=[]
  for method in range(3):
   arg=d/f'{label}-{method}.args.bin';subprocess.run([str(HELPER),'query',str(method),str(ip),str(arg)],check=True)
   cmd=['icp','canister','call',CAN,'project','--network','local','--identity','imajev-local','--candid',str(did),'--args-file',str(arg),'--args-format','bin','--query','--output','hex'];start=time.monotonic();raw=subprocess.check_output(cmd,text=True);reply=d/f'{label}-{method}.hex';reply.write_text(raw)
   m=json.loads(subprocess.check_output([str(HELPER),'decode',str(reply),'measurement'],text=True));assert bytes(m.pop('digest'))==expected,(label,method);assert m['output_values']==count;assert m['total_instructions']==m['project_instructions']+m['input_prepare_instructions'];m.update(method=method,reply=str(reply.relative_to(ROOT)),reply_sha256=sha(reply),wall_seconds=time.monotonic()-start);measurements.append(m)
  results.append(dict(label=label,scope=scope,input_hash=sha(ip),oracle_hash=sha(ep),blocks=values.size//256,error=count==0,measurements=measurements));print(json.dumps(dict(label=label,body=[m['project_instructions']for m in measurements])),flush=True)
 assert status()==b['module'];paths=[Path(__file__),HELPER,D/'build/report.json',did];report=dict(module=b['module'],canister=CAN,cases=results,all_values_and_scales_equal=True,queries=len(cases)*3,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in paths},scope=__doc__);(d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
