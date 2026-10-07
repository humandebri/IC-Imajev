#!/usr/bin/env python3
"""Check same-module float hashes against independent hashlib bytes and counters."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
CANISTER='zm54s-at777-77775-aaa5q-cai'
HELPER=ROOT/'artifacts/f32-block-native/release/f32_args'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d=ROOT/'artifacts/float-hash-v1/check-v2'
    d.mkdir(exist_ok=False)
    build=json.loads((d.parent/'build/report.json').read_text())
    status=json.loads(subprocess.check_output(['icp','canister','status',CANISTER,'--network','local','--identity','imajev-local','--json'],text=True))
    assert status['module_hash'].removeprefix('0x')==build['module']
    for group in ['source_hashes','dependency_hashes']:
        for p,h in build[group].items():assert sha(ROOT/p)==h
    actual=(ROOT/'artifacts/f32_block/check/617.input.bin').read_bytes()
    pattern=b''.join(v.to_bytes(4,'little') for v in [0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc10000,0x3f123456])
    cases=[('real-617',actual)]+[(f'boundary-{n}',(pattern*((4*n+len(pattern)-1)//len(pattern)))[:4*n]) for n in [1,3,4096,24576,146432,196608]]
    result=[]
    for label,values in cases:
        path=d/(label+'.input.bin');path.write_bytes(values)
        expected=list(hashlib.sha256(values).digest())
        measured=[]
        for method in [0,1]:
            argument=d/f'{label}-{method}.args.bin'
            subprocess.run([str(HELPER),'query',str(method),str(path),str(argument)],check=True)
            raw=subprocess.check_output(['icp','canister','call',CANISTER,'project','--args-file',str(argument),'--args-format','bin','--query','--network','local','--identity','imajev-local','--output','hex'],text=True)
            reply=d/f'{label}-{method}.hex';reply.write_text(raw)
            decoded=json.loads(subprocess.check_output([str(HELPER),'decode',str(reply),'measurement'],text=True))
            assert decoded['digest']==expected,(label,decoded['digest'],expected)
            assert decoded['total_instructions']==decoded['input_prepare_instructions']+decoded['project_instructions']
            measured.append(dict(measurement=decoded,reply_sha256=sha(reply)))
        entry=dict(label=label,values=len(values)//4,input_sha256=sha(path),measurements=measured,
                   reduction_percent=100*(1-measured[1]['measurement']['project_instructions']/measured[0]['measurement']['project_instructions']))
        result.append(entry);print(json.dumps(dict(label=label,reduction_percent=entry['reduction_percent'])),flush=True)
    r=dict(module=build['module'],cases=result,all_hashes_equal=True,helper_sha256=sha(HELPER),source_sha256=sha(Path(__file__)),scope='Isolated hash cost only; not full worker inference.')
    (d/'report.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
