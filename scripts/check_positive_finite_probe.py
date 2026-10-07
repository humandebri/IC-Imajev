#!/usr/bin/env python3
"""Verify positive-finite classification with an independent unsigned range oracle."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/positive-finite-v1';p=ROOT/'scripts/check_finite_simd_probe.py';s=p.read_text().replace('artifacts/finite-simd-v1','artifacts/positive-finite-v1').replace('range(4):','range(3):').replace('queries=len(result)*4','queries=len(result)*3')
 s=s.replace('(bits[i:i+stride]&0x7f800000)!=0x7f800000','(bits[i:i+stride]>0)&(bits[i:i+stride]<0x7f800000)')
 s=s.replace('[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x7f7fffff,0xff7fffff]','[1,0x007fffff,0x00800000,0x3f800000,0x7f7fffff]')
 s=s.replace("values=np.zeros((count,16),dtype='<u4')","values=np.full((count,16),0x3f800000,dtype='<u4')")
 s=s.replace('unusual=[0x7f800000','unusual=[0,0x80000000,0x80000001,0x807fffff,0x80800000,0xbf800000,0xff7fffff,0x7f800000')
 (d/'frozen-check.py').write_text(s);(d/'check-upstream.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
