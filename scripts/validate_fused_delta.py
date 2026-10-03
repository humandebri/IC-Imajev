#!/usr/bin/env python3
"""Validate lossless Delta stage fusion and widened short-token MLP on local IC."""
import argparse,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);a=ap.parse_args()
 common=['--canister',a.canister,'--arithmetic','int8','--wire-codec','bf16-exact','--compact-lossless','--fuse-add-norm','--fuse-mlp','--delta-head-cap','16','--attention-head-cap','8','--row-cap','16384','--work-cap','2500000000','--token-cap','132','--compact-heads','--fuse-delta','--wide-mlp']
 def run(script,*args):subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True)
 cache='artifacts/fused-stage-prefix/queries'
 run('run_prefix_canister.py',*common,'--prepare-prefix','--cache',cache,'--directory','artifacts/fused-stage-prefix')
 run('compare_compact_heads.py','--before','artifacts/compact-prefix','--after','artifacts/fused-stage-prefix','--output','docs/fused-stage-prefix-results.json')
 for label,record in [('617',0),('insufficient',19),('maximum',11)]:
  options=[] if label=='617' else ['--reference','artifacts/reference-serving.json','--record',str(record)]
  directory=f'artifacts/fused-stage-{label}'
  run('run_prefix_canister.py',*common,'--cache',cache,'--directory',directory,*options)
  run('compare_compact_heads.py','--before',f'artifacts/compact-{label}','--after',directory,'--output',f'docs/fused-stage-{label}-results.json')
 run('run_full_canister.py',*common,'--directory','artifacts/fused-stage-normal')
 run('compare_compact_heads.py','--before','artifacts/compact-normal','--after','artifacts/fused-stage-normal','--output','docs/fused-stage-normal-results.json')
if __name__=='__main__':main()
