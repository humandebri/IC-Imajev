#!/usr/bin/env python3
"""Snapshot-preserved diagnostic upgrade, measurement, and exact module restoration."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/winograd2-register-v1/proof'
C='2sm34-hd777-77775-aaauq-cai'
BASE='92581eaa5824d78c84e76b96d77d76320703b5bc1b6940385bba68b8850444ac'
def main():
    D.mkdir(exist_ok=False);events=[]
    def run(*args):
        out=subprocess.check_output(['icp','canister',*args,'--network','local','--identity','imajev-local'],text=True,cwd=ROOT)
        events.append(dict(args=args,output=out));(D/'operations.json').write_text(json.dumps(events,indent=2)+'\n');return out
    def status():return json.loads(run('status',C,'--json'))
    before=status();assert before['module_hash']=='0x'+BASE
    (D/'before.json').write_text(json.dumps(before,indent=2)+'\n')
    b=ROOT/'artifacts/winograd2-register-v1/build';report=json.loads((b/'report.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(b/'diagnostic.wasm')==report['wasm_sha256']
    for key in ['source_hashes','dependency_hashes']:
        assert all(sha(ROOT/p)==h for p,h in report[key].items())
    snapshot=None;stopped=False
    try:
        run('stop',C);stopped=True
        snapshot=run('snapshot','create',C,'--quiet').strip()
        (D/'snapshot.json').write_text(json.dumps(dict(canister=C,snapshot_id=snapshot,baseline=BASE,candidate=report['wasm_sha256']),indent=2)+'\n')
        run('install',C,'--mode','upgrade','--wasm',str(b/'diagnostic.wasm'),'--yes')
        run('start',C);stopped=False
        with (D/'measurement.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'scripts/check_winograd2_register_probe.py'),'--canister',C],cwd=ROOT,check=True,stdout=log,stderr=log)
    finally:
        if snapshot:
            if not stopped:run('stop',C);stopped=True
            run('snapshot','restore',C,snapshot);run('start',C);stopped=False
            after=status();assert after['module_hash']==before['module_hash'] and after['memory_size']==before['memory_size']
            (D/'restored.json').write_text(json.dumps(dict(restored=True,module=BASE,memory_size=after['memory_size'],snapshot_id=snapshot),indent=2)+'\n')
            run('snapshot','delete',C,snapshot)
            print('Diagnostic module and state restored; snapshot deleted.',flush=True)
        elif stopped:run('start',C)
if __name__=='__main__':main()
