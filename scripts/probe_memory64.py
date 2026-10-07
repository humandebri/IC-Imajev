#!/usr/bin/env python3
"""One-page memory64 and i64 System API pointer capability, isolated from inference."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def leb(n):
 out=bytearray()
 while True:
  b=n&127;n>>=7;out.append(b|(128 if n else 0))
  if not n:return bytes(out)
def string(s):
 b=s.encode();return leb(len(b))+b
def section(i,b):return bytes([i])+leb(len(b))+b

def main():
 d=ROOT/'artifacts/memory64-capability-v1';d.mkdir(exist_ok=False)
 # ()->(), (i64,i64)->(), ic0.reply_data_append and reply, query function2.
 w=b'\0asm\1\0\0\0'+section(1,b'\2\x60\0\0\x60\2\x7e\x7e\0')
 w+=section(2,b'\2'+string('ic0')+string('msg_reply_data_append')+b'\0\1'+string('ic0')+string('msg_reply')+b'\0\0')
 w+=section(3,b'\1\0')+section(5,b'\1\4\1')
 w+=section(7,b'\2'+string('memory')+b'\2\0'+string('canister_query memory64_probe')+b'\0\2')
 reply=b'DIDL\0\1\x78'+(64).to_bytes(8,'little')
 body=b'\0\x42\0\x42'+leb(len(reply))+b'\x10\0\x10\1\x0b';w+=section(10,b'\1'+leb(len(body))+body)
 w+=section(11,b'\1\0\x42\0\x0b'+leb(len(reply))+reply)
 p=d/'probe.wasm';p.write_bytes(w);(d/'probe.did').write_text('service:{memory64_probe:()->(nat64)query;}')
 (d/'report.json').write_text(json.dumps(dict(module=hashlib.sha256(w).hexdigest(),bytes=len(w),memory64=True,initial_pages=1,system_api_pointer_type='i64',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Initial one-page local capability only; no large heap allocation, no inference data or performance claim.'),indent=2)+'\n');print(json.dumps(dict(module=hashlib.sha256(w).hexdigest(),bytes=len(w))))
if __name__=='__main__':main()
