#!/usr/bin/env python3
"""Run isolated compact-head sessions on an already prepared explicit local canister."""
import argparse,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);args=ap.parse_args()
 common=['--canister',args.canister,'--arithmetic','int8','--wire-codec','bf16-exact','--compact-lossless','--fuse-add-norm','--fuse-mlp','--delta-head-cap','8','--attention-head-cap','8','--row-cap','16384','--work-cap','2500000000','--token-cap','132','--compact-heads']
 def run(script,*options):subprocess.run([sys.executable,str(ROOT/'scripts'/script),*options],cwd=ROOT,check=True)
 def compare(before,after,output):run('compare_compact_heads.py','--before',before,'--after',after,'--output',output)
 run('run_full_canister.py',*common,'--directory','artifacts/compact-normal')
 compare('artifacts/layout-full-617','artifacts/compact-normal','docs/compact-normal-results.json')
 cache='artifacts/compact-prefix/queries'
 run('run_prefix_canister.py',*common,'--prepare-prefix','--cache',cache,'--directory','artifacts/compact-prefix')
 compare('artifacts/prefix-45-preparation','artifacts/compact-prefix','docs/compact-preparation-results.json')
 for label,record,before in [('617',0,'artifacts/prefix-hit-617'),('insufficient',19,'artifacts/prefix-hit-diagnostics/record-19'),('maximum',11,'artifacts/prefix-hit-diagnostics/record-11')]:
  options=[] if label=='617' else ['--reference','artifacts/reference-serving.json','--record',str(record)]
  directory=f'artifacts/compact-{label}'
  run('run_prefix_canister.py',*common,'--cache',cache,'--directory',directory,*options)
  compare(before,directory,f'docs/compact-{label}-results.json')
if __name__=='__main__':main()
