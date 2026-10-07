#!/usr/bin/env python3
"""Primitive per-leaf normalization and depth2 integer/modular scalar-dot proof."""
from pathlib import Path
from fractions import Fraction
from math import gcd,lcm
import json,hashlib,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 source=ROOT/'scripts/audit_rank48_primary_coefficients.py';ns=dict(__name__='parse',__file__=str(source));exec(compile(source.read_text(),str(source),'exec'),ns);old=ROOT/'artifacts/rank48-primary-coefficients-v1';audit=json.loads((old/'integer-audit.json').read_text());assert audit['all4096_original_basis_products_exact'];assert all(sha(ROOT/p)==h for p,h in audit['source_hashes'].items())
 L,R,P=[ns['parse'](old/('4x4x4_48_rational_'+x+'.sms'))for x in ['L','R','P']]
 def primitive(row):
  den=lcm(*(v.denominator for v in row));v=[int(x*den)for x in row];g=gcd(*v);assert g>0;sign=1 if next(x for x in v if x)>0 else-1;return[vv//(g*sign)for vv in v],Fraction(g*sign,den)
 aa=[primitive(row)for row in L];bb=[primitive(row)for row in R];pc=[[P[o][m]*aa[m][1]*bb[m][1]for m in range(48)]for o in range(16)];den=lcm(*(v.denominator for row in pc for v in row));assert den&(den-1)==0
 ai=np.array([x[0]for x in aa],dtype=np.int64);bi=np.array([x[0]for x in bb],dtype=np.int64);ci=np.array([[int(v*den)for v in row]for row in pc],dtype=np.int64)
 # Exact primitive tensor equality checked independently after normalization.
 tensor=np.einsum('om,ma,mb->oab',ci,ai,bi,dtype=np.int64);target=np.zeros((16,16,16),dtype=np.int64)
 for t in range(4):
  for n in range(4):
   for k in range(4):target[t*4+n,t*4+k,k*4+n]=den
 assert np.array_equal(tensor,target)
 avec=[];bvec=[];cvec=[]
 for m in range(48):
  for n in range(48):
   avec.append(np.kron(ai[m].reshape(4,4),ai[n].reshape(4,4)).ravel());bvec.append(np.kron(bi[m].reshape(4,4),bi[n].reshape(4,4)).ravel());cvec.append(np.kron(ci[:,m].reshape(4,4),ci[:,n].reshape(4,4)).ravel())
 A=np.array(avec,dtype=np.int64);B=np.array(bvec,dtype=np.int64);C=np.array(cvec,dtype=np.int64).T;apos=abs(A).sum(axis=1)*127;bpos=np.maximum(B,0).sum(axis=1);bneg=-np.minimum(B,0).sum(axis=1);blo=-128*bpos-127*bneg;bhi=127*bpos+128*bneg;assert apos.max()<=32767 and blo.min()>=-32768 and bhi.max()<=32767
 divisor=den*den;shift=divisor.bit_length()-1;finalbound=divisor*127*128*256;assert finalbound<2**31
 d=ROOT/'artifacts/rank2304-integer-lift-v1';d.mkdir(exist_ok=False);coef=d/'integer-coefficients.npz';np.savez_compressed(coef,A=A.astype('<i2'),B=B.astype('<i2'),C=C.astype('<i4'),denom=np.array([divisor],dtype='<u4'))
 rng=np.random.default_rng(230448);cases=[];files=[Path(__file__),source,old/'integer-audit.json',coef]+[old/('4x4x4_48_rational_'+x+'.sms')for x in ['L','R','P']]
 for i in range(12):
  q=rng.integers(-127,128,(16,256),dtype=np.int64);w=rng.integers(-128,128,(256,16),dtype=np.int64)
  if i==1:q.fill(127);w.fill(127)
  if i==2:q.fill(-127);w.fill(-128)
  if i==3:q[:]=np.resize([-127,127],q.shape);w[:]=np.resize([-128,127],w.shape)
  if i==4:
   q[:]=np.sign(A[np.argmax(apos)]).reshape(16,16).repeat(16,axis=1)*127
   chosen=B[np.argmin(blo)].reshape(16,16);w[:]=np.where(chosen.repeat(16,axis=0)>0,-128,127)
  qa=q.reshape(256,16);wb=w.reshape(16,16,16).transpose(0,2,1).reshape(256,16);av=A@qa;bv=B@wb;assert av.min()>=-32768 and av.max()<=32767 and bv.min()>=-32768 and bv.max()<=32767
  exact_products=(av*bv).sum(axis=1,dtype=np.int64);wrapped_products=exact_products.astype('<i4').astype(np.int64);numerator=(C@wrapped_products).astype('<i4').astype(np.int64);assert np.all(numerator%divisor==0);actual=(numerator>>shift).reshape(16,16);expected=q@w;assert np.array_equal(actual,expected)
  p=d/f'case-{i}.npz';np.savez_compressed(p,q=q.astype('<i2'),w=w.astype('<i2'),expected=expected.astype('<i4'));files.append(p);cases.append(dict(index=i,all256_integer_dots_exact=True,leaf_products_exceed_signed32=bool(np.any(exact_products>2**31-1)|np.any(exact_products<-(2**31)))))
 r=dict(complete=True,rank=2304,primitive_base_rank=48,primitive_base_divisor=den,final_divisor=divisor,exact_primitive_base_tensor_identities=4096,activation_min_bound=-int(apos.max()),activation_max_bound=int(apos.max()),weight_min_bound=int(blo.min()),weight_max_bound=int(bhi.max()),final_scaled_dot_signed32_bound=finalbound,all12_modular_depth2_dot_conditions_equal=True,cases=cases,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Exact primitive coefficients plus Kronecker depth2; twelve actual16x256 by256x16 scalar-dot conditions including modularI32 leaf/C accumulation. I16 bounds proved over fullI8 range, final division bypower2 exact. NoSIMDWasm/ICperformance/memoryarchitecture/fullpaidclaim.')
 (d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(d/'evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps({k:r[k]for k in ['rank','primitive_base_divisor','final_divisor','activation_min_bound','activation_max_bound','weight_min_bound','weight_max_bound','final_scaled_dot_signed32_bound']}))
if __name__=='__main__':main()
