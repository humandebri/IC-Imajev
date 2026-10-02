#!/usr/bin/env python3
"""Whole pack digest is checked by update seal, independently of valid chunk hashes."""
import argparse,hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import Transport
ap=argparse.ArgumentParser();ap.add_argument('--url',default='http://localhost:8001/');ap.add_argument('--canister',default='4qggx-l3777-77775-aaaca-cai');args=ap.parse_args();m=json.loads((ROOT/'checkpoints/readout.manifest.json').read_text());directory=ROOT/'artifacts/upload-check';directory.mkdir(parents=True,exist_ok=True);bad=directory/'incorrect.manifest.json';bad.write_text(json.dumps({**m,'pack_hash':'0'*64}));t=Transport(m['model'],args.url,args.canister,str(ROOT/'artifacts/imajev-local.pem'),directory,m['pack_hash']);report={'url':args.url,'canister':args.canister,'wasm_sha256':hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest()}
try:t.upload(bad,ROOT/'checkpoints/readout.pack')
except RuntimeError as e:assert 'pack hash mismatch'in str(e);report['wrong_pack_hash_rejected']=str(e)
else:raise AssertionError('seal accepted incorrect pack digest')
report['correct_pack_upload']=t.upload(ROOT/'checkpoints/readout.manifest.json',ROOT/'checkpoints/readout.pack');t.close();(directory/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
