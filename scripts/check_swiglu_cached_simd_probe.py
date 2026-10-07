#!/usr/bin/env python3
"""Check complete SwiGLU digests with and without tables and independent BF16/F32 arithmetic."""
from pathlib import Path
import hashlib,json,struct,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CANISTER='zm54s-at777-77775-aaa5q-cai';HELPER=ROOT/'artifacts/f32-block-native/release/f32_args'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bf(v):
 a=np.asarray(v,dtype='<f4');bits=a.view('<u4');return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view('<f4')
def oracle(seed,n):
 g,u=seed[:,0],seed[:,1]
 with np.errstate(over='ignore',under='ignore',invalid='raise'):
  e=bf(np.exp(np.abs(g),dtype=np.float32));y=bf(np.divide(np.float32(1.),bf(np.add(np.float32(1.),e,dtype=np.float32)),dtype=np.float32));sig=np.where(g<0,y,bf(np.subtract(np.float32(1.),y,dtype=np.float32)));out=bf(np.multiply(bf(np.multiply(g,sig,dtype=np.float32)),u,dtype=np.float32))
 return hashlib.sha256(np.resize(out,n).astype('<f4').tobytes()).digest()
def main():
 d=ROOT/'artifacts/swiglu-cached-simd-v1/check';d.mkdir(exist_ok=False);b=json.loads((d.parent/'build/report.json').read_text());assert json.loads(subprocess.check_output(['icp','canister','status',CANISTER,'--network','local','--identity','imajev-local','--json'],text=True))['module_hash'].removeprefix('0x')==b['module']
 assert all(sha(ROOT/p)==h for g in ['source_hashes','dependency_hashes']for p,h in b[g].items())
 pattern=np.array([-2.,-1.,-0.5,-0.,0.,0.5,1.,2.,8.,-8.,100.,-100.],dtype='<f4');seed=np.stack([pattern,np.resize(np.array([1.,-1.,0.5,1.25,-0.,2.],dtype='<f4'),len(pattern))],axis=1)
 cases=[('boundary-'+str(n),n,seed)for n in [1,3,4,5,7,8,4096,9216,48*9216,56*9216,57*9216]]
 non=np.array([1,0x80000001,0x3f808000,0xbf818000,0x3f807fff,0xbf817fff,0x3f808001,0xbf818001],dtype='<u4').view('<f4');cases.append(('non-bf16',33,np.stack([non,non[::-1]],axis=1)))
 actual_path=ROOT/'artifacts/f32_block/check/617.input.bin';actual=np.frombuffer(actual_path.read_bytes(),dtype='<f4')[:8192];cases.append(('saved-operands-617',56*9216,np.stack([actual,actual[::-1]],axis=1)))
 result=[]
 for label,n,seed in cases:
  inp=d/(label+'.input.bin');inp.write_bytes(struct.pack('<I',n)+seed.astype('<f4').tobytes());expected=oracle(seed,n);(d/(label+'.expected.bin')).write_bytes(expected);measurements=[]
  for method in range(4):
   arg=d/f'{label}-{method}.args.bin';subprocess.run([str(HELPER),'query',str(method),str(inp),str(arg)],check=True);reply=d/f'{label}-{method}.hex';reply.write_text(subprocess.check_output(['icp','canister','call',CANISTER,'project','--args-file',str(arg),'--args-format','bin','--query','--network','local','--identity','imajev-local','--output','hex'],text=True));m=json.loads(subprocess.check_output([str(HELPER),'decode',str(reply),'measurement'],text=True));assert bytes(m.pop('digest'))==expected,(label,method);assert m['output_values']==n and m['total_instructions']==m['input_prepare_instructions']+m['project_instructions'];measurements.append(dict(measurement=m,reply_sha256=sha(reply)))
  r=dict(label=label,values=n,input_sha256=sha(inp),expected_sha256=sha(d/(label+'.expected.bin')),measurements=measurements,bitwise_equal=True,reduction_percent=100*(1-measurements[1]['measurement']['project_instructions']/measurements[0]['measurement']['project_instructions']),no_table_reduction_percent=100*(1-measurements[3]['measurement']['project_instructions']/measurements[2]['measurement']['project_instructions']));result.append(r);print({k:r[k]for k in ['label','reduction_percent','no_table_reduction_percent']},flush=True)
 r=dict(module=b['module'],cases=result,helper_sha256=sha(HELPER),source_sha256=sha(Path(__file__)),source_operand_sha256=sha(actual_path),all_bits_equal=True,scope='Isolated pointwise SwiGLU. Four methods compare scalar/SIMD with prepared tables or query-local clear. Header count expands identical seed operands in input preparation, outside the measured body; saved values are arithmetic operands, not a full MLP intermediate. Both complete output digests match independent ordered BF16/F32 arithmetic.')
 (d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
