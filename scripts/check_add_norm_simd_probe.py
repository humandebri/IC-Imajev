#!/usr/bin/env python3
"""Compare both residual and normalized bits against an independent ordered F32 oracle."""
from pathlib import Path
import hashlib,json,struct,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CANISTER='zm54s-at777-77775-aaa5q-cai';HELPER=ROOT/'artifacts/f32-block-native/release/f32_args'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bf(v):
 a=np.asarray(v,dtype='<f4');b=a.view('<u4');return ((b+np.uint32(0x7fff)+((b>>16)&1))&np.uint32(0xffff0000)).view('<f4')
def oracle(x,w,n,c,eps):
 count=n*c;res=bf(np.add(x[:count],x[count:],dtype=np.float32));out=np.empty(count*2,dtype='<f4');out[:count]=res
 for row in range(n):
  vals=res[row*c:(row+1)*c];total=np.float32(0.)
  for v in vals:total=np.float32(total+np.float32(v*v))
  scale=np.float32(np.float32(1.)/np.sqrt(np.float32(np.float32(total/np.float32(c))+eps),dtype=np.float32))
  out[count+row*c:count+(row+1)*c]=bf(np.multiply(np.multiply(vals,scale,dtype=np.float32),w,dtype=np.float32))
 return out.tobytes()
def main():
 d=ROOT/'artifacts/add-norm-simd-v1/check';d.mkdir(exist_ok=False);build=json.loads((d.parent/'build/report.json').read_text());status=json.loads(subprocess.check_output(['icp','canister','status',CANISTER,'--network','local','--identity','imajev-local','--json'],text=True));assert status['module_hash'].removeprefix('0x')==build['module']
 assert all(sha(ROOT/p)==h for g in ['source_hashes','dependency_hashes'] for p,h in build[g].items())
 bits=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f808000,0xbf818000,0x3f807fff,0xbf817fff,0x3f808001,0xbf818001,0x3e123456,0xbe123456]
 pattern=np.array(bits,dtype='<u4').view('<f4');cases=[]
 for n,c in [(1,1),(2,3),(3,4),(3,7),(5,9),(1,2560),(48,2560),(56,2560),(57,2560),(1,8192)]:
  x=np.resize(pattern,n*c*2).copy();w=np.resize(np.array([1.,-1.,0.5,1.25,-0.,2.],dtype='<f4'),c);cases.append(('boundary-'+str(n)+'x'+str(c),n,c,x,w))
 actual_path=ROOT/'artifacts/f32_block/check/617.input.bin';actual=np.frombuffer(actual_path.read_bytes(),dtype='<f4');n,c=56,2560
 x=np.resize(actual,n*c*2).copy();w=bf(np.resize(np.array([1.03,-1.03,0.8,1.2],dtype='<f4'),c));cases.append(('saved-operands-617',n,c,x,w))
 result=[];eps=np.float32(1e-6)
 for label,n,c,x,w in cases:
  inp=d/(label+'.input.bin');inp.write_bytes(struct.pack('<fff',n,c,eps)+x.tobytes()+w.tobytes());expected=d/(label+'.expected.bin');expected.write_bytes(oracle(x,w,n,c,eps));measurements=[]
  for method in [0,1]:
   arg=d/f'{label}-{method}.args.bin';subprocess.run([str(HELPER),'query',str(method),str(inp),str(arg)],check=True);reply=d/f'{label}-{method}.hex';reply.write_text(subprocess.check_output(['icp','canister','call',CANISTER,'project','--args-file',str(arg),'--args-format','bin','--query','--network','local','--identity','imajev-local','--output','hex'],text=True));m=json.loads(subprocess.check_output([str(HELPER),'decode',str(reply),'measurement'],text=True));digest=bytes(m.pop('digest'));assert digest==expected.read_bytes(),(label,method,next((i for i,(a,b) in enumerate(zip(digest,expected.read_bytes())) if a!=b),None));assert m['output_values']==2*n*c and m['total_instructions']==m['project_instructions']+m['input_prepare_instructions'];measurements.append(dict(measurement=m,reply_sha256=sha(reply)))
  r=dict(label=label,n=n,cols=c,input_sha256=sha(inp),expected_sha256=sha(expected),measurements=measurements,bitwise_equal=True,reduction_percent=100*(1-measurements[1]['measurement']['project_instructions']/measurements[0]['measurement']['project_instructions']));result.append(r);print({k:r[k] for k in ['label','reduction_percent']},flush=True)
 assert json.loads(subprocess.check_output(['icp','canister','status',CANISTER,'--network','local','--identity','imajev-local','--json'],text=True))['module_hash'].removeprefix('0x')==build['module']
 r=dict(module=build['module'],cases=result,helper_sha256=sha(HELPER),source_sha256=sha(Path(__file__)),oracle_inputs_sha256=sha(actual_path),all_bits_equal=True,scope='Isolated add_norm: both complete outputs against scalar Wasm and independent F32 ordered sum oracle. Saved activation values serve as operands; not actual full inference add_norm intermediate.')
 (d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
