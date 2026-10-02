#!/usr/bin/env python3
"""Complete selected diagnostic graphs in fresh journals; ordinary queries only."""
import argparse,hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--records',type=int,nargs='+',default=[3,5,11,17,19]);ap.add_argument('--canister',default='46el7-ql777-77775-aaada-cai');ap.add_argument('--fuse-mlp',action='store_true');ap.add_argument('--fuse-add-norm',action='store_true');ap.add_argument('--directory',default='artifacts/integer-canister-diagnostics');args=ap.parse_args();directory=ROOT/args.directory;directory.mkdir(parents=True,exist_ok=True);fixture=json.loads((ROOT/'artifacts/reference-orders.json').read_text());records=[]
for index in args.records:
 d=directory/f'record-{index:02d}';cmd=[sys.executable,str(ROOT/'scripts/run_full_canister.py'),'--arithmetic','int8','--canister',args.canister,'--wire-codec','bf16-exact','--delta-head-cap','8','--attention-head-cap','8','--compact-lossless','--row-cap','8192','--work-cap','2500000000','--reference','artifacts/reference-orders.json','--record',str(index),'--directory',str(d)]
 if args.fuse_add_norm:cmd.append('--fuse-add-norm')
 if args.fuse_mlp:cmd.append('--fuse-mlp')
 # Existing completed reports are evidence, not replaced by a replay's timings.
 p=d/'first-report.json'
 if not p.exists():
  with (directory/f'record-{index:02d}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  p.write_bytes((d/'report.json').read_bytes())
 r=json.loads(p.read_text());f=fixture['records'][index];a=r['decision_query']['ok']['decision'];rec=dict(record=index,id=f['id'],offset=f['offset'],gold=f['gold'],options=f['options'],reference_value=f['result']['value'],value=a['value'],matches_official_label=a['value']==f['result']['value'],typed_output_valid=r['comparison']['typed_output_valid'],probabilities=a['probabilities'],unknown_probability=a['unknown_probability'],query_count=r['query_count'],total_instructions=r['total_instructions'],total_candid_bytes=r['total_candid_bytes'],wall_seconds=r['wall_seconds_this_run'],wasm_sha256=r['wasm_sha256'],report_sha256=hashlib.sha256(p.read_bytes()).hexdigest());records.append(rec)
 (directory/'summary.json').write_text(json.dumps(dict(scope='Selected actual canister arithmetic diagnostic cases; no general accuracy guarantee',records=records),indent=2)+'\n');print(rec,flush=True)
