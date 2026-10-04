#!/usr/bin/env python3
"""Reject checksummed invalid block payloads through the actual step entry point."""
import argparse,hashlib,json,pathlib,sys,struct,zipfile
import blake3
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport
from prefix_inference import verify_module

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--canister',required=True);ap.add_argument('--wasm',required=True);ap.add_argument('--directory',required=True);ap.add_argument('--host-checksum',action='store_true');a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest();module=sha((ROOT/a.wasm).read_bytes())
 paths=list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'canisters/inference/src').rglob('*.rs'))+[ROOT/'client/transport.py',pathlib.Path(__file__)];hashes={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths}
 source=ROOT/'artifacts/codec_scan/check';r=json.loads((source/'report.json').read_text());cases=[]
 t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'])
 try:
  verify_module(t,module)
  for case in r['cases']:
   if not case.get('invalid'):continue
   p=source/(case['name']+'.input.bin');raw=p.read_bytes();assert sha(raw)==case['input_sha256']
   for version in ([1,2,3] if a.host_checksum else [1,2]):
    h=dict(version=version,model=m['model'],pack_hash=m['pack_hash'],input_hash='c'*64,step=0,op='bf16',tensor='',dims=[],scalars=[],encoding='bf16-block256-exact-v1')
    header=json.dumps(h,separators=(',',':')).encode();body=struct.pack('<I',len(header))+header+raw;body+=hashlib.sha256(body).digest()if version in (1,3) else blake3.blake3(body).digest()
    ip=d/f"{case['name']}-v{version}.request.bin";ip.write_bytes(body)
    try:t.command(dict(op='step',input=str(ip),output=str(d/f"{case['name']}-v{version}.reply.bin")))
    except RuntimeError as e:cases.append(dict(name=case['name'],version=version,request_sha256=sha(body),error=str(e)))
    else:raise AssertionError('Checksummed invalid activation accepted')
  verify_module(t,module)
 finally:t.close()
 assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths}
 with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
 (d/'report.json').write_text(json.dumps(dict(scope=__doc__,wasm_sha256=module,source_hashes=hashes,rejected_ordinary_queries=len(cases),cases=cases),indent=2)+'\n')
if __name__=='__main__':main()
