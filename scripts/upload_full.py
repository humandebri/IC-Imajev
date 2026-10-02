#!/usr/bin/env python3
"""Resume chunk upload to an explicitly selected local model canister."""
import argparse,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,atomic
ap=argparse.ArgumentParser();ap.add_argument('--url',default='http://localhost:8001/');ap.add_argument('--canister',required=True);ap.add_argument('--manifest',default='checkpoints/full-int8.manifest.json');ap.add_argument('--pack',default='checkpoints/full-int8.pack');ap.add_argument('--concurrency',type=int,default=16);ap.add_argument('--directory',default='artifacts/full-int8-upload');args=ap.parse_args()
m=json.loads((ROOT/args.manifest).read_text());directory=ROOT/args.directory;t=Transport(m['model'],args.url,args.canister,str(ROOT/'artifacts/imajev-local.pem'),directory,m['pack_hash'])
try:
 result=t.command(dict(op='upload_parallel',manifest=str(ROOT/args.manifest),pack=str(ROOT/args.pack),concurrency=args.concurrency));atomic(directory/'report.json',(json.dumps(dict(model=m['model'],pack_hash=m['pack_hash'],url=args.url,canister=args.canister,result=result),indent=2)+'\n').encode());print(json.dumps(result,indent=2))
finally:t.close()
