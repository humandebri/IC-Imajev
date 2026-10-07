#!/usr/bin/env python3
"""Validate4096 exact tensor identities and power-of-two integer implementation bounds."""
from pathlib import Path
from fractions import Fraction
from math import lcm
import json,hashlib,re,itertools,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parse(p):
 lines=[x.strip()for x in p.read_text().splitlines()if x.strip()and not x.startswith('#')];rows,cols,kind=lines[0].split();assert kind=='R';rows=int(rows);cols=int(cols);assert rows<=64 and cols<=64;out=[[Fraction(0)for _ in range(cols)]for _ in range(rows)];seen=set();assert lines[-1]=='0 0 0'
 for line in lines[1:-1]:
  a,b,c=line.split();a=int(a)-1;b=int(b)-1;assert 0<=a<rows and 0<=b<cols and (a,b)not in seen and re.fullmatch(r'-?\d+(?:/\d+)?',c);seen.add((a,b));out[a][b]=Fraction(c)
 return out
 def_unused=0

def mm(a,b):
 assert len(a[0])==len(b)
 return [[sum((x*b[k][j]for k,x in enumerate(row)),Fraction(0))for j in range(len(b[0]))]for row in a]
def main():
 d=ROOT/'artifacts/rank48-primary-coefficients-v1';f=json.loads((d/'report.json').read_text());assert all(sha(ROOT/item['path'])==item['sha256']for item in f['files']);m={name:parse(d/('4x4x4_48_rational'+name+'.sms'))for name in ['_L','_R','_P','-ALT_L','-ALT_R','-ALT_P','-CoB_L','-CoB_R','-CoB_P']}
 L,R,P=[m['_'+name]for name in ['L','R','P']];assert [len(L),len(L[0]),len(R),len(R[0]),len(P),len(P[0])]==[48,16,48,16,16,48]
 assert mm(m['-ALT_L'],m['-CoB_L'])==L and mm(m['-ALT_R'],m['-CoB_R'])==R and mm(m['-CoB_P'],m['-ALT_P'])==P
 checked=0
 for out,ai,bi in itertools.product(range(16),repeat=3):
  actual=sum((P[out][k]*L[k][ai]*R[k][bi]for k in range(48)),Fraction(0));expected=int(ai//4==out//4 and ai%4==bi//4 and bi%4==out%4);assert actual==expected,(out,ai,bi);checked+=1
 scales=[lcm(*(v.denominator for row in mat for v in row))for mat in [L,R,P]];assert all(s>0 and s&(s-1)==0 for s in scales);il,ir,ip=[[[int(v*s)for v in row]for row in mat]for mat,s in zip([L,R,P],scales)];sa,sb,sc=scales;denom=sa*sb*sc;abound=max(sum(abs(v)for v in row)*127 for row in il);bbound=max(sum(abs(v)for v in row)*128 for row in ir);assert abound<=32767 and bbound<=32767;final_bound=denom*127*128*256;assert final_bound<2**31
 # Tensor identity implies reconstructed integer numerator=denom*original dot.
 # All C arithmetic may be modulo2^32; final numerator is within signed32 range,
 # and exactly divisible bydenom, so one final arithmetic shift recovers originaldot.
 files=[Path(__file__),d/'report.json']+[ROOT/v['path']for v in f['files']];r=dict(complete=True,rank=48,exact_tensor_identities=checked,all4096_original_basis_products_exact=True,all_alt_CoB_compositions_exact=True,alternative_basis_dimensions=[47,47,47],integer_scales=scales,final_divisor=denom,activation_int16_bound=abound,weight_int16_bound=bbound,leaf_dot_i32_bound_K64=abound*bbound*64,final_scaled_original_dot_signed32_bound_K256=final_bound,modular_i32_C_reconstruction_then_exact_shift_valid=True,source_commit=f['commit'],source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},wasm_generated=False,performance_verified=False,scope='Exact Fraction tensor validation. ALT47 is rectangular redundant encoding, not an invertible16x16basis; no asymptotic cost assumption. Integer admissibility only, not generated kernel/runtimeresource/IC/fullpaid proof.')
 (d/'integer-audit.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(d/'integer-evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'integer-audit.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps({key:r[key]for key in ['rank','exact_tensor_identities','integer_scales','final_divisor','activation_int16_bound','weight_int16_bound','final_scaled_original_dot_signed32_bound_K256']}))
if __name__=='__main__':main()
