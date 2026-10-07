#!/usr/bin/env python3
"""Real Wasm and IC metering of old/new operand encoders with partial blocks."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CID='4zfnl-5t777-77775-aaadq-cai'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cmd(args):return subprocess.check_output(['icp','canister']+args+['--network','local','--identity','imajev-local'],text=True)
def status():return json.loads(cmd(['status',CID,'--json']))
def main():
    d=ROOT/'artifacts/rank7-prepare8-probe-v1';b=json.loads((d/'build.json').read_text())
    for p,h in b['source_hashes'].items():assert sha(ROOT/p)==h,p
    check=d/'check';check.mkdir(exist_ok=False)
    runner=check/'runner.cjs';runner.write_text('''const fs=require('fs');
const d=process.argv[2],root=process.argv[3];
const b=JSON.parse(fs.readFileSync(`${d}/build.json`));
const module_=new WebAssembly.Module(fs.readFileSync(`${d}/probe.wasm`));
const results=[];
for(const c of b.cases)for(const width of [4,8]){
 const inst=new WebAssembly.Instance(module_,{ic0:{performance_counter:()=>0n,msg_reply_data_append:()=>{},msg_reply:()=>{}}});
 const input=fs.readFileSync(`${root}/${c.input}`),expected=fs.readFileSync(`${root}/${c.expected}`);
 const q=inst.exports.memory.buffer.byteLength+256,out=q+input.length+256;
 const end=out+expected.length+256;
 inst.exports.memory.grow(Math.ceil((end-inst.exports.memory.buffer.byteLength)/65536));
 const mem=new Uint8Array(inst.exports.memory.buffer);mem.fill(0xa5,q-256,end);
 mem.set(input,q);mem.fill(0x5a,out,out+expected.length);
 inst.exports[`prepare${width}`](q,c.cols,c.pairs,out,c.first,c.end);
 const actual=Buffer.from(mem.subarray(out,out+expected.length));
 fs.writeFileSync(`${d}/check/node-${width}-${c.index}.bin`,actual);
 if(!actual.equals(expected))throw Error(`${width}/${c.index}: output mismatch`);
 if(!Buffer.from(mem.subarray(q,q+input.length)).equals(input))throw Error('input modified');
 for(const [a,z]of [[q-256,q],[q+input.length,out],[out+expected.length,end]])if(mem.subarray(a,z).some(v=>v!==0xa5))throw Error('sentinel modified');
 results.push({index:c.index,width,all_operand_bytes_exact:true,input_and_sentinels_preserved:true});
}
fs.writeFileSync(`${d}/check/node-results.json`,JSON.stringify(results,null,2)+'\\n');
console.log(`PASS ${results.length} raw encoder memory checks`);
''')
    run=subprocess.run(['node',str(runner),str(d),str(ROOT)],capture_output=True,text=True)
    (check/'node-stdout.txt').write_text(run.stdout);(check/'node-stderr.txt').write_text(run.stderr);run.check_returncode()
    s=status();(check/'pre-status.json').write_text(json.dumps(s,indent=2)+'\n')
    old=json.loads((ROOT/'artifacts/rank2304-modular-meter-v2/report.json').read_text())['module']
    assert s['id']==CID and s['status']=='Stopped' and s['module_hash'].removeprefix('0x')==old,s
    assert s['settings']['controllers']==['cibxp-okw3l-gfzvi-p6ltu-23ss3-7tlfz-nk65x-tvhvb-mg56y-b5hkf-eqe'],s
    (check/'install.txt').write_text(cmd(['install',CID,'--mode','reinstall','--wasm',str(d/'probe.wasm'),'--yes']))
    assert status()['module_hash'].removeprefix('0x')==sha(d/'probe.wasm')
    (check/'start.txt').write_text(cmd(['start',CID]))
    results=[]
    try:
        for c in b['cases']:
            expected=(ROOT/c['expected']).read_bytes()
            for width in [4,8]:
                method=f'prepare{width}_{c["index"]}'
                raw=cmd(['call',CID,method,'()','--candid',str(d/'probe.did'),'--output','hex'])
                path=check/f'{method}.hex';path.write_text(raw)
                data=bytes.fromhex(raw.strip().removeprefix('0x'))
                assert data[:10]==bytes.fromhex('4449444c016d7b027800')
                count=int.from_bytes(data[10:18],'little');at=18;size=0;shift=0
                while True:
                    v=data[at];at+=1;size|=(v&127)<<shift;shift+=7
                    if v<128:break
                assert size==len(expected) and data[at:]==expected
                results.append(dict(index=c['index'],width=width,instructions=count,all_operand_bytes_exact=True,reply=str(path.relative_to(ROOT)),reply_sha256=sha(path)))
                print(json.dumps(dict(index=c['index'],width=width,instructions=count)),flush=True)
    finally:
        (check/'stop.txt').write_text(cmd(['stop',CID]))
        s=status();(check/'post-status.json').write_text(json.dumps(s,indent=2)+'\n')
    assert s['status']=='Stopped' and s['module_hash'].removeprefix('0x')==sha(d/'probe.wasm')
    comparisons=[]
    for c in b['cases']:
        values={r['width']:r['instructions'] for r in results if r['index']==c['index']}
        comparisons.append(dict(**c,prepare4=values[4],prepare8=values[8],saved=values[4]-values[8],percent=(values[8]/values[4]-1)*100))
    files=[Path(__file__),d/'build.json']+sorted(check.iterdir())
    report=dict(complete=True,all14_node_operand_buffers_exact=True,all14_ic_operand_buffers_exact=True,comparisons=comparisons,replies=results,module=sha(d/'probe.wasm'),own_probe_stopped=True,adopted=False,full_paid_goal_achieved=False,
                source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},scope=b['scope'])
    (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(comparisons),flush=True)
if __name__=='__main__':main()
