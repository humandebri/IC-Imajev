#!/usr/bin/env python3
"""Measure Laya baseline/unsigned-bit peak quantization in one disposable local canister."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
from urllib.parse import urlparse
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build',required=True)
    ap.add_argument('--directory',required=True)
    ap.add_argument('--identity',default='imajev-local')
    args=ap.parse_args()
    build=(ROOT/args.build).resolve();dest=(ROOT/args.directory).resolve()
    if not all(p.is_relative_to(ROOT) for p in [build,dest]):raise ValueError('paths must stay inside repository')
    proof=json.loads((build/'node-report.json').read_text());wasm=build/'diagnostic.wasm'
    assert hashlib.sha256(wasm.read_bytes()).hexdigest()==proof['wasmSha256']
    assert all(r['bitwiseEqual'] and r['scalarEqual'] for r in proof['rows'])
    cli=['--network','local','--identity',args.identity]
    def call(command):return subprocess.check_output(['icp',*command],cwd=ROOT,text=True,stdin=subprocess.DEVNULL).strip()
    network=json.loads(call(['network','status','--json']))
    assert urlparse(network['api_url']).hostname in ('localhost','127.0.0.1','::1')
    dest.mkdir(parents=True,exist_ok=False)
    report={'scope':'Laya row quantization with allocation/output assembly, not full model','api_url':network['api_url'],'wasm_sha256':proof['wasmSha256'],
            'node_report_sha256':hashlib.sha256((build/'node-report.json').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'canister':None,'rows':[],'cleanup':{},'error':None}
    def save():(dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        principal=call(['canister','create','--detached','--quiet',*cli])
        assert re.fullmatch(r'[a-z2-7-]+',principal) and principal.endswith('-cai')
        report['canister']=principal;save()
        call(['canister','install',principal,'--mode','install','--wasm',str(wasm),'--args','()',*cli])
        nonce=0
        for case in proof['rows']:
            # 20 realistic shapes plus SIMD-tail, finite-edge and nonfinite checks.
            if case['tokens']==3 and not (case['cols'] in [17,1024] and case['pattern']>=1):continue
            n,cols,pattern=case['tokens'],case['cols'],case['pattern']
            call(['canister','call',principal,'configure',struct.pack('<III',n,cols,pattern).hex(),'--args-format','hex','--output','hex',*cli])
            measurements={'laya':[],'unsigned_peak':[]}
            for repeat in range(3):
                for backend in ([0,1] if repeat%2==0 else [1,0]):
                    nonce+=1
                    output=call(['canister','call',principal,'measure',struct.pack('<II',backend,nonce).hex(),'--query','--args-format','hex','--output','hex',*cli])
                    raw=bytes.fromhex(output.removeprefix('0x'));assert len(raw)==24
                    instructions,digest,valid,length=struct.unpack('<QQII',raw)
                    assert bool(valid)==case['valid'] and length==n*(cols+4) and f'{digest:016x}'==case['outputFnv64']
                    measurements['laya' if backend==0 else 'unsigned_peak'].append({'instructions':instructions,'digest':f'{digest:016x}','valid':bool(valid),'nonce':nonce})
            for values in measurements.values():assert len({v['instructions'] for v in values})==1
            counts={name:values[0]['instructions'] for name,values in measurements.items()}
            row={'tokens':n,'cols':cols,'pattern':pattern,'valid':case['valid'],'instructions':counts,'candidate_over_laya':counts['unsigned_peak']/counts['laya'],'measurements':measurements}
            report['rows'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k!='measurements'}),flush=True)
    except Exception as error:
        report['error']=str(error);raise
    finally:
        if report['canister']:
            for action in ['stop','delete']:
                result=subprocess.run(['icp','canister',action,report['canister'],*cli],cwd=ROOT,text=True,stdin=subprocess.DEVNULL,capture_output=True)
                report['cleanup'][action]={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
        save()
if __name__=='__main__':main()
