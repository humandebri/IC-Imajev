#!/usr/bin/env python3
"""Check both wire versions and corrupted frames on a selected local canister."""
import argparse,hashlib,json,pathlib,struct,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,encode,decode,atomic
from prefix_inference import verify_module

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--wasm',required=True);ap.add_argument('--native',required=True);ap.add_argument('--directory',required=True);a=ap.parse_args()
 d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());sha=hashlib.sha256((ROOT/a.wasm).read_bytes()).hexdigest()
 t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']);rows=[];negative=[]
 try:
  verify_module(t,sha)
  v=np.array([0.,-0.,1.0000001,1e-40,4.25],np.float32)
  for codec in ['','bf16-exact','bf16-block256-exact-v1']:
   for version in [1,2]:
    h=dict(version=version,model=m['model'],pack_hash=m['pack_hash'],input_hash='c'*64,step=len(rows),op='bf16',tensor='',dims=[],scalars=[],encoding=codec)
    p=d/f'{len(rows):02d}.request.bin';out=d/f'{len(rows):02d}.response.bin';native=d/f'{len(rows):02d}.native.bin';atomic(p,encode(h,v))
    subprocess.run([str(ROOT/a.native),str(p),str(native)],check=True)
    measured=t.command(dict(op='step',input=str(p),output=str(out)));assert out.read_bytes()==native.read_bytes()
    rh,x=decode(out.read_bytes());assert rh['version']==version and rh['step']==h['step']+1
    rows.append(dict(version=version,encoding=codec,measurement=measured,reply_sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
  h=dict(version=2,model=m['model'],pack_hash=m['pack_hash'],input_hash='c'*64,step=0,op='bf16',tensor='',dims=[],scalars=[],encoding='')
  b=encode(h,v);n,=struct.unpack('<I',b[:4]);cases={}
  for name,pos in [('header',4),('payload',4+n),('digest',len(b)-1)]:
   raw=bytearray(b);raw[pos]^=1;cases[name]=bytes(raw)
  cases['downgrade']=b.replace(b'"version":2',b'"version":1',1)
  cases['wrong-algorithm']=b[:-32]+hashlib.sha256(b[:-32]).digest()
  cases['unknown-version']=b.replace(b'"version":2',b'"version":3',1)
  cases['oversized-header']=struct.pack('<I',16385)+b[4:]
  cases['truncated']=b[:35]
  for name,body in cases.items():
   p=d/f'bad-{name}.bin';out=d/f'bad-{name}.response.bin';atomic(p,body)
   try:t.command(dict(op='step',input=str(p),output=str(out)))
   except RuntimeError as e:negative.append(dict(name=name,error=str(e)))
   else:raise AssertionError('Invalid frame accepted: '+name)
  verify_module(t,sha)
 finally:t.close()
 sources=[pathlib.Path(__file__),ROOT/'client/transport.py',ROOT/'crates/imajev-runtime/src/lib.rs',ROOT/'Cargo.lock',ROOT/'requirements.lock']
 report=dict(canister=a.canister,wasm_sha256=sha,native_sha256=hashlib.sha256((ROOT/a.native).read_bytes()).hexdigest(),successful_ordinary_queries=len(rows),rejected_ordinary_queries=len(negative),certified_module_reads=2,rows=rows,negative=negative,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Wire protocol only; no matrix weights, speed or full-model accuracy claim')
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('rows','negative','source_hashes')}))
if __name__=='__main__':main()
