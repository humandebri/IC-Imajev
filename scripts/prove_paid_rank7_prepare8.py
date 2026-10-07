#!/usr/bin/env python3
"""Snapshot-protected all-three paid proof for the verified eight-lane revision."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    old=ROOT/'artifacts/paid-common-raw-hybrid-v1';d=ROOT/'artifacts/paid-rank7-prepare8-v1'
    for group in ['proof-entry-hashes.json','workflow-hashes.json']:
        for p,h in json.loads((old/group).read_text()).items():assert sha(ROOT/p)==h,p
    for p,h in json.loads((d/'workflow-hashes.json').read_text()).items():assert sha(ROOT/p)==h,p
    equal=json.loads((d/'paid-source-equivalence.json').read_text());assert equal['paid_contract_and_scheduler_and_lib_byteexact']
    build=json.loads((d/'build/report.json').read_text());assert sha(d/'build/full.wasm')==equal['module']==build['wasm_sha256']
    for key in ['source_hashes','dependency_hashes']:
        for p,h in build[key].items():assert sha(ROOT/p)==h,p
    verification=ROOT/'artifacts/paid-common-raw-hybrid-v1/proof-storage/proof/recovery/verification/verified.json'
    verified=json.loads(verification.read_text());assert verified['complete'] and verified['full_state_equal'] and verified['target']=='4caro-hl777-77775-aaaba-cai'
    reference=ROOT/'artifacts/paid-common-raw-hybrid-retry-v3/proof/report.json'
    saved=json.loads(reference.read_text());assert saved['complete'] and len(saved['results'])==3
    storage=Path.home()/'.codex/goal-artifacts/01a10ffe-1dc8-7931-b3de-a3e4d35bd910/paid-rank7-prepare8-v1/proof'
    assert not storage.exists() and not (d/'proof').exists() and not (d/'frozen-proof.py').exists()
    assert shutil.disk_usage(ROOT).free>10*1024**3 and shutil.disk_usage(storage.parent.parent).free>30*1024**3
    source=old/'frozen-proof.py';code=source.read_text()
    code=code.replace('ROOT=Path(__file__).resolve().parents[1];','ROOT=Path('+repr(str(ROOT))+');')
    code=code.replace("B=ROOT/'artifacts/paid-common-raw-hybrid-v1/build';D=ROOT/'artifacts/paid-common-raw-hybrid-v1/proof-storage/proof'","B=ROOT/'artifacts/paid-rank7-prepare8-v1/build';D=ROOT/'artifacts/paid-rank7-prepare8-v1/proof'")
    before=' D.mkdir(parents=True,exist_ok=False);';assert code.count(before)==1
    code=code.replace(before,' Path('+repr(str(storage))+').mkdir(parents=True,exist_ok=False);')
    guard=' callers=[];snapshot=None;stopped=False;results=[];checks=[]';assert code.count(guard)==1
    code=code.replace(guard," baseline=json.loads((ROOT/'artifacts/local-goal-recovery-v1/baseline-state.json').read_text())\n assert dict(module=BASELINE,cache=cache,pack=pack)==baseline,'live baseline differs before trial'\n"+guard)
    before="    assert digest(debug['final_hidden'])==digest(saved['debug']['final_hidden']),(name,'final hidden')"
    assert code.count(before)==1
    code=code.replace(before,before+"\n    latest=json.loads((ROOT/'artifacts/paid-common-raw-hybrid-retry-v3/proof/report.json').read_text());current=next(v for v in latest['results']if v['case']==name)\n    assert debug['hidden_hashes']==current['debug']['hidden_hashes'] and debug['state_hashes']==current['debug']['state_hashes'],(name,'latest best all32')\n    assert digest(debug['final_hidden'])==digest(current['debug']['final_hidden']),(name,'latest best final hidden')")
    code=code.replace("paths=[ROOT/'artifacts/paid-k2", "paths=[ROOT/'artifacts/paid-common-raw-hybrid-retry-v3/proof/report.json',ROOT/'artifacts/paid-rank7-prepare8-v1/workflow-hashes.json',ROOT/'artifacts/paid-rank7-prepare8-v1/paid-source-equivalence.json',ROOT/'artifacts/paid-k2",1)
    compile(code,str(d/'frozen-proof.py'),'exec')
    (d/'proof').symlink_to(storage,target_is_directory=True)
    frozen=d/'frozen-proof.py';frozen.write_text(code)
    files=[Path(__file__),source,old/'proof-entry-hashes.json',d/'workflow-hashes.json',d/'build/report.json',d/'paid-source-equivalence.json',verification,reference,ROOT/'artifacts/local-goal-recovery-v1/baseline-state.json',frozen]
    (d/'proof-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
    (d/'proof-entry.json').write_text(json.dumps(dict(prepared=True,run=args.run,storage=str(storage),all3_and_latest_best_all32_checks=True,snapshot_and_live_baseline_guard=True),indent=2)+'\n')
    if args.run:exec(compile(code,str(frozen),'exec'),dict(__file__=str(frozen),__name__='__main__'))
    else:print('Prepared complete proof; no canister mutation')
if __name__=='__main__':main()
