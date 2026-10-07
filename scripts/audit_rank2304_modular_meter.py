#!/usr/bin/env python3
"""Independently re-decode replies and execute installed module against raw dots."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CID='4zfnl-5t777-77775-aaadq-cai'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    d=ROOT/'artifacts/rank2304-modular-meter-v2'
    prior=ROOT/'artifacts/rank2304-modular-meter-v1'
    simd=ROOT/'artifacts/rank2304-modular-simd-v1'
    files={Path(__file__)}
    for p in [prior/'build.json',prior/'report.json',d/'entry.json',d/'checker-entry.json',d/'build.json',d/'report.json',simd/'report.json',simd/'execution-report.json']:
        value=json.loads(p.read_text());files.add(p)
        hashes=value.get('source_hashes',value if p.name.endswith('entry.json') else {})
        for name,h in hashes.items():
            q=ROOT/name;assert sha(q)==h,name;files.add(q)
    r=json.loads((d/'report.json').read_text())
    b=json.loads((d/'build.json').read_text())
    runner=d/'independent-node.cjs'
    fixtures=d/'independent-raw-fixtures';fixtures.mkdir(exist_ok=False)
    expected={}
    for i in range(12):
        p=ROOT/f'artifacts/rank2304-latest-conditional-lift-v1/case-{i}.npz';files.add(p)
        with np.load(p) as z:
            q,w=z['q'].astype(np.int64),z['w'].astype(np.int64)
            dot=q@w
            assert np.array_equal(dot,z['expected'])
        assert dot.min()>=-2**31 and dot.max()<2**31
        expected[i]=dot.astype('<i4').tobytes()
        (fixtures/f'{i}.bin').write_bytes(expected[i])
    measurements={}
    for result in r['replies']:
        p=ROOT/result['reply'];assert sha(p)==result['reply_sha256']
        data=bytes.fromhex(p.read_text().strip().removeprefix('0x'))
        assert data[:10]==bytes.fromhex('4449444c016d7b027800')
        assert len(data)==1048 and data[18:20]==bytes([0x84,0x08])
        assert data[20:1044]==expected[result['index']]
        value=int.from_bytes(data[10:18],'little');assert value==result['instructions']
        mode=result['mode'];flag=int.from_bytes(data[1044:],'little')
        assert flag==({'universal32':2,'checked16':int(result['index']!=4)}.get(mode,0xa5a5a5a5))
        measurements.setdefault(mode,[]).append(value)
    assert len(r['replies'])==48 and all(len(v)==12 for v in measurements.values())
    assert set(measurements['direct_unrolled'])=={74713}
    assert set(measurements['universal32'])=={230852}
    assert set(measurements['direct'])=={246649}
    # Verify exactly the compiled candidate functions are embedded in both IC probes.
    candidate=(simd/'kernel.wat').read_text().split('\n',1)[1].removesuffix(')\n')
    for mode in ['universal32','checked16']:
        candidate=candidate.replace(f'(func(export "{mode}")',f'(func ${mode}(export "{mode}")')
    for base in [prior,d]:assert candidate in (base/'probe.wat').read_text()
    runner.write_text('''const fs=require('fs');
const d=process.argv[2];
const module_=new WebAssembly.Module(fs.readFileSync(`${d}/probe.wasm`));
const methods=JSON.parse(fs.readFileSync(`${d}/build.json`)).methods;
const results=[];
for(const m of methods){
 let chunks=[];let instance;
 const imports={ic0:{performance_counter:()=>0n,msg_reply_data_append:(p,n)=>chunks.push(Buffer.from(new Uint8Array(instance.exports.memory.buffer,p,n))),msg_reply:()=>{}}};
 instance=new WebAssembly.Instance(module_,imports);
 instance.exports[`canister_update ${m.method}`]();
 const actual=Buffer.concat(chunks);
 const expected=fs.readFileSync(`${d}/independent-raw-fixtures/${m.index}.bin`);
 if(!actual.subarray(20,1044).equals(expected))throw Error(m.method);
 results.push({...m,output_from_raw_integer_dots_exact:true});
}
fs.writeFileSync(`${d}/independent-node-results.json`,JSON.stringify(results,null,2)+'\\n');
console.log(`PASS ${results.length} installed-module exports against independent raw dots`);
''')
    run=subprocess.run(['node',str(runner),str(d)],capture_output=True,text=True)
    (d/'independent-node-stdout.txt').write_text(run.stdout)
    (d/'independent-node-stderr.txt').write_text(run.stderr)
    run.check_returncode()
    s=json.loads(subprocess.check_output(['icp','canister','status',CID,'--network','local','--identity','imajev-local','--json'],text=True))
    assert s['status']=='Stopped' and s['module_hash'].removeprefix('0x')==r['module']
    (d/'fresh-final-status.json').write_text(json.dumps(s,indent=2)+'\n')
    for p in [runner,d/'independent-node-results.json',d/'independent-node-stdout.txt',d/'independent-node-stderr.txt',d/'fresh-final-status.json']+sorted(fixtures.iterdir()):files.add(p)
    audit=dict(complete=True,all48_replies_independently_redecoded=True,all48_installed_module_exports_match_fresh_raw_integer_dots=True,
               compiled_candidate_embedded_byteexact=True,own_probe_stopped=True,
               universal32_core_vs_unrolled_direct_percent=(230852/74713-1)*100,
               adopted=False,full_paid_goal_achieved=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},
               scope='Core-only screening evidence. The direct unrolled control is not the current best full rank7 projection. Input preparation, weight packing, F32 scales and worker orchestration were not measured. No full inference gain or regression claim.')
    out=d/'independent-audit.json';out.write_text(json.dumps(audit,indent=2)+'\n');files.add(out)
    archive=d/'frozen-evidence.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip()is None
        for p in sorted(files):assert hashlib.sha256(z.read(str(p.relative_to(ROOT)))).hexdigest()==sha(p)
    (d/'archive-identity.json').write_text(json.dumps(dict(complete=True,sha256=sha(archive),files=len(files),all_archived_bytes_verified=True),indent=2)+'\n')
    print(run.stdout,end='');print('sealed:',sha(archive))
if __name__=='__main__':main()
