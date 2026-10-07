#!/usr/bin/env python3
"""Prepare deterministic guards; execute only after full proof restoration."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    d=ROOT/'artifacts/paid-rank7-prepare8-v1';original=ROOT/'scripts/prove_paid_common_raw_hybrid_upgrade_guards.py'
    prior=json.loads((ROOT/'artifacts/paid-common-raw-hybrid-retry-v3/summary.json').read_text())
    assert sha(original)==prior['workflow_hashes'][str(original.relative_to(ROOT))]
    code=original.read_text().replace('ROOT = Path(__file__).resolve().parents[1]','ROOT = Path('+repr(str(ROOT))+')')
    code=code.replace('artifacts/paid-common-raw-hybrid-v1','artifacts/paid-rank7-prepare8-v1')
    planned=d/'planned-upgrade-guard-driver.py';compile(code,str(planned),'exec')
    if planned.exists():assert planned.read_text()==code
    else:planned.write_text(code)
    if args.run:
        saved_argv=sys.argv;sys.argv=[str(planned),'--directory',str(d)]
        try:exec(compile(code,str(planned),'exec'),dict(__file__=str(planned),__name__='__main__'))
        finally:sys.argv=saved_argv
    else:print('Prepared guard driver; execution requires complete restored proof')
if __name__=='__main__':main()
