#!/usr/bin/env python3
"""Guard own small probe, install and measure core-only IC update methods."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CID='4zfnl-5t777-77775-aaadq-cai'
IDENTITY='imajev-local'
OLD='1bfed03d74aea873ed78e1f4568a04a49e29ad587b6effdf0b01cd0dd539d0e9'
OWNER='cibxp-okw3l-gfzvi-p6ltu-23ss3-7tlfz-nk65x-tvhvb-mg56y-b5hkf-eqe'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(args):return subprocess.check_output(['icp','canister']+args+['--network','local','--identity',IDENTITY],text=True)
def status():return json.loads(command(['status',CID,'--json']))
def main():
    d=ROOT/'artifacts/rank2304-modular-meter-v1'
    b=json.loads((d/'build.json').read_text())
    for p,h in b['source_hashes'].items(): assert sha(ROOT/p)==h,p
    check=d/'check';check.mkdir(exist_ok=False)
    s=status();(check/'pre-status.json').write_text(json.dumps(s,indent=2)+'\n')
    assert s['module_hash'].removeprefix('0x')==OLD,s
    assert s['id']==CID and s['settings']['controllers']==[OWNER],s
    assert s['status'].lower()=='stopped',s
    (check/'install.txt').write_text(command(['install',CID,'--mode','reinstall','--wasm',str(d/'probe.wasm'),'--yes']))
    s=status();assert s['module_hash'].removeprefix('0x')==sha(d/'probe.wasm'),s
    (check/'start.txt').write_text(command(['start',CID]))
    results=[]
    try:
        for m in b['methods']:
            raw=command(['call',CID,m['method'],'()','--candid',str(d/'probe.did'),'--output','hex'])
            path=check/f"{m['method']}.hex";path.write_text(raw)
            data=bytes.fromhex(raw.strip().removeprefix('0x'))
            assert len(data)==1048 and data[:10]==bytes.fromhex('4449444c016d7b027800') and data[18:20]==bytes([0x84,0x08]),(len(data),data[:20].hex())
            measured=int.from_bytes(data[10:18],'little')
            ref=ROOT/f"artifacts/rank2304-modular-simd-v1/execution/{m['index']}-universal32.expected.bin"
            config=json.loads((ROOT/'artifacts/rank2304-modular-simd-v1/execution/config.json').read_text())
            out=config['layout']['out'][0]
            assert data[20:1044]==ref.read_bytes()[out:out+1024],m
            flag=int.from_bytes(data[1044:],'little')
            if m['mode']=='universal32':assert flag==2
            elif m['mode']=='checked16':assert flag==int(m['index']!=4)
            else:assert flag==0xa5a5a5a5
            result=dict(**m,instructions=measured,all256_output_i32_exact=True,flag=flag,reply=str(path.relative_to(ROOT)),reply_sha256=sha(path))
            results.append(result);print(json.dumps(result),flush=True)
        s=status();assert s['module_hash'].removeprefix('0x')==sha(d/'probe.wasm'),s
    finally:
        (check/'stop.txt').write_text(command(['stop',CID]))
        (check/'post-status.json').write_text(json.dumps(status(),indent=2)+'\n')
    comparisons=[]
    for i in range(12):
        r={x['mode']:x['instructions'] for x in results if x['index']==i}
        comparisons.append(dict(index=i,**r,universal32_vs_direct_percent=(r['universal32']/r['direct']-1)*100,checked16_vs_direct_percent=(r['checked16']/r['direct']-1)*100))
    files=[Path(__file__),d/'build.json']+sorted(check.iterdir())
    report=dict(complete=True,ic_core_performance_verified=True,all36_update_outputs_exact=True,module=sha(d/'probe.wasm'),comparisons=comparisons,replies=results,
                source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},full_paid_goal_achieved=False,adopted=False,scope=b['scope'])
    (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(comparisons),flush=True)
if __name__=='__main__':main()
