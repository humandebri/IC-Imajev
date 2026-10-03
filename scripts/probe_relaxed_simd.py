#!/usr/bin/env python3
"""Emit a minimal local-only capability probe; do not alter inference modules.

The opcode remains in the Wasm binary, with no optimizing compiler to erase it.
The function discards a zero dot and sends an empty reply. Installation tests
whether the selected IC validator supports relaxed SIMD, not model accuracy.
"""
import argparse,hashlib,json,pathlib

def leb(n):
 out=bytearray()
 while True:
  b=n&127;n>>=7;out.append(b|(128 if n else 0))
  if not n:return bytes(out)
def string(s):
 b=s.encode();return leb(len(b))+b
def section(i,b):return bytes([i])+leb(len(b))+b

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',default='artifacts/relaxed-simd');a=ap.parse_args();d=pathlib.Path(a.directory);d.mkdir(parents=True,exist_ok=True)
 # type0 () -> (), imported ic0.msg_reply, function1, one-page memory.
 w=b'\0asm\1\0\0\0'+section(1,b'\1\x60\0\0')
 w+=section(2,b'\1'+string('ic0')+string('msg_reply')+b'\0\0')
 w+=section(3,b'\1\0')+section(5,b'\1\0\1')
 w+=section(7,b'\2'+string('memory')+b'\2\0'+string('canister_query relaxed_probe')+b'\0\1')
 body=b'\0'+(b'\xfd\x0c'+b'\0'*16)*3+b'\xfd'+leb(0x113)+b'\x1a\x10\0\x0b'
 w+=section(10,b'\1'+leb(len(body))+body)
 p=d/'probe.wasm';p.write_bytes(w)
 (d/'probe.json').write_text(json.dumps(dict(wasm_sha256=hashlib.sha256(w).hexdigest(),bytes=len(w),opcode='i32x4.relaxed_dot_i8x16_i7x16_add_s',subopcode=0x113,scope='Selected local validator capability only; no inference or weight data'),indent=2)+'\n')
 # Exact signed-byte decomposition: q = low7 - 128*negative_flag.
 for q in range(-127,128):
  low=q&127;negative=int(q<0);assert 0<=low<=127 and negative in (0,1)
  for weight in range(-128,128):assert weight*q==weight*low-128*weight*negative
 # Two low7 products stay in I16 even at the extrema, avoiding saturation.
 assert 2*128*127<=32768 and 2*127*127<32768
 print(p)
if __name__=='__main__':main()
