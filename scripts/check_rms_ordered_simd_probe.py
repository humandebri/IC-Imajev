#!/usr/bin/env python3
"""Check F32 RMS outputs against scalar Wasm and a fully ordered independent oracle."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/rms-ordered-simd-v1';p=ROOT/'scripts/check_add_norm_simd_probe.py';s=p.read_text().replace('artifacts/add-norm-simd-v1/check','artifacts/rms-ordered-simd-v1/check')
 before='count=n*c;res=bf(np.add(x[:count],x[count:],dtype=np.float32));out=np.empty(count*2,dtype=\'<f4\');out[:count]=res';assert s.count(before)==1
 s=s.replace(before,"count=n*c;res=x[:count];out=np.empty(count,dtype='<f4')")
 before='out[count+row*c:count+(row+1)*c]=bf(np.multiply(np.multiply(vals,scale,dtype=np.float32),w,dtype=np.float32))';assert s.count(before)==1
 s=s.replace(before,'out[row*c:(row+1)*c]=np.multiply(np.multiply(vals,scale,dtype=np.float32),w,dtype=np.float32)')
 s=s.replace("m['output_values']==2*n*c","m['output_values']==n*c")
 s=s.replace('(1,8192)]','(1,8192),(1,255),(1,256)]')
 s=s.replace("result=[];eps=np.float32(1e-6)","pattern=np.array([16.,2**-12,1000.,1e-6,-16.,-1000.],dtype='<f4');n,c=3,1025;cases.append(('sum-order',n,c,np.resize(pattern,n*c*2),np.resize(np.array([1.03,0.8,-1.2],dtype='<f4'),c)))\n result=[];eps=np.float32(1e-6)")
 s=s.replace('Isolated add_norm: both complete outputs','Isolated RMS: complete unrounded F32 outputs').replace('actual full inference add_norm intermediate','actual full inference RMS intermediate')
 (d/'frozen-check.py').write_text(s);(d/'check-entry-hashes.json').write_text(json.dumps({str(v.relative_to(ROOT)):sha(v) for v in [Path(__file__),p,d/'frozen-check.py']},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
