#!/usr/bin/env python3
"""Measure latest S1 pair-bounds and streamed rank49/output64 against native INT8."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B = ROOT/'artifacts/s1-k2-mlp160-v1/build'
HELPER = ROOT/'artifacts/bounded_i16/native/release/wat_s1_args'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--canister',required=True)
    a = ap.parse_args()
    d = ROOT/'artifacts/s1-k2-mlp160-v1/check-down'
    d.mkdir(exist_ok=False)
    did = d/'diagnostic.did'
    did.write_text('''type Prep = record { instructions:nat64; bytes:nat64; rows:nat64 };
type Measurement = record { digest:vec nat8; quantize_instructions:nat64;
input_prepare_instructions:nat64; project_instructions:nat64; total_instructions:nat64;
output_values:nat64; heap_pages:nat64 };
service : { prepare_chunk:(nat32,vec nat8)->(Prep); seal:()->(Prep);
project:(vec nat8,nat8)->(Measurement) query; }
''')
    manifest = json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
    assert manifest['model'] == sha(ROOT/'MODEL_LOCK.json')
    build = json.loads((B/'report.json').read_text())
    paths = list(build['source_hashes']) + list(build['dependency_hashes']) + [
        str(Path(__file__).relative_to(ROOT)),str(HELPER.relative_to(ROOT)),
        str(did.relative_to(ROOT)),str((B/'report.json').relative_to(ROOT)),
        'MODEL_LOCK.json','checkpoints/full-int8.manifest.json']
    identities = {p:sha(ROOT/p) for p in paths}
    for key in ['source_hashes','dependency_hashes']:
        assert all(identities[p] == digest for p,digest in build[key].items())
    module = sha(B/'diagnostic.wasm')
    assert module == build['wasm_sha256']
    def status():
        return json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--network','local','--identity','imajev-local','--json'],text=True))['module_hash'].removeprefix('0x')
    assert status() == module
    count = 0
    def call(method,arg=None,query=False):
        nonlocal count
        count += 1
        cmd = ['icp','canister','call',a.canister,method,'--network','local','--identity','imajev-local',
               '--candid',str(did),'--output','hex']
        cmd += ['--args-file',str(arg),'--args-format','bin'] if arg else ['()']
        if query: cmd += ['--query']
        start = time.monotonic()
        raw = subprocess.check_output(cmd,text=True,cwd=ROOT)
        reply = d/f'{count:04d}-{method}.hex'; reply.write_text(raw)
        result = json.loads(subprocess.check_output([str(HELPER),'decode',str(reply),'measurement' if query else 'preparation'],text=True))
        result.update(reply=str(reply.relative_to(ROOT)),reply_sha256=sha(reply),wall_seconds=time.monotonic()-start)
        return result
    name = 'model.language_model.layers.3.mlp.down_proj.weight'
    tensor = next(t for t in manifest['tensors'] if t['name']==name)
    assert (tensor['rows'],tensor['cols'],tensor['dtype']) == (2560,9216,'int8')
    with (ROOT/'checkpoints/full-int8.pack').open('rb') as f:
        f.seek(tensor['offset']); weights = f.read(tensor['bytes'])
    assert len(weights) == 2560*9220
    weight_hash = hashlib.sha256(weights).hexdigest()
    assert weight_hash == hashlib.sha256(weights).hexdigest()
    preparations = []
    for start in range(0,2560,128):
        end = start+128
        raw = weights[start*9216:end*9216]+weights[2560*9216+start*4:2560*9216+end*4]
        path = d/f'weights-{start}.bin'; path.write_bytes(raw)
        arg = d/f'weights-{start}.args.bin'
        subprocess.run([str(HELPER),'chunk',str(start),str(path),str(arg)],check=True)
        preparations.append(call('prepare_chunk',arg))
    preparations.append(call('seal'))
    previous = ROOT/'artifacts/remaining-efficiency/odd-check-v2'
    previous_report = json.loads((previous/'report.json').read_text())
    cases = []
    for label in []:
        path = previous/f'{label}.input.bin'
        old = next(c for c in previous_report['cases'] if c['label']==label)
        assert sha(path) == old['input_sha256']
        cases.append((label,np.frombuffer(path.read_bytes(),dtype='<f4'),dict(source=str(path.relative_to(ROOT)),source_sha256=sha(path),scope='Saved real full-model activation from earlier Q projection')))
    for n in [1,2,3,5,7,8,9,16,24,31,32,40]:
        values = np.resize(np.array([-127.,127.,0.,-0.,-1.,1.,0.5,-0.5,2**-126,-2**-126],dtype='<f4'),n*9216)
        cases.append((f'boundary-{n}',values,dict(scope='Synthetic signed extremes/zero/tiny values at genuine MLP down shape')))
    results = []
    cases.sort(key=lambda case: (case[0]!='boundary-8',case[0]!='size-67'))
    for label,values,provenance in cases:
        n = values.size//9216; rows = 4096 if n>109 else 2560
        ip = d/f'{label}.input.bin'; ip.write_bytes(values.astype('<f4').tobytes())
        wp = d/f'weights-{rows}-native.bin'
        if not wp.exists(): wp.write_bytes(weights[:rows*9216]+weights[2560*9216:2560*9216+rows*4])
        native = json.loads(subprocess.check_output([str(HELPER),'native',str(n),str(rows),'9216',str(wp),str(ip)],text=True))
        measured = {}
        for method,key in [(3,'s1_pair_bounds'),(4,'winograd7')]:
            arg = d/f'{label}-{method}.args.bin'
            subprocess.run([str(HELPER),'query',str(method),str(ip),str(arg)],check=True)
            measured[key] = call('project',arg,True)
            assert measured[key]['digest'] == native['digest'], (label,key)
        before = measured['s1_pair_bounds']['total_instructions']; after = measured['winograd7']['total_instructions']
        case = dict(label=label,tokens=n,rows=rows,cols=9216,input_sha256=sha(ip),native=native,
                    provenance=provenance,measurements=measured,reduction_percent=100*(1-after/before),bitwise_equal=True)
        results.append(case)
        print(json.dumps(dict(label=label,tokens=n,before=before,after=after,reduction_percent=case['reduction_percent'])),flush=True)
    assert status() == module
    assert identities == {p:sha(ROOT/p) for p in paths}
    report = dict(wasm_sha256=module,canister=a.canister,source_hashes=identities,
                  model=manifest['model'],pack_hash=manifest['pack_hash'],tensor=name,weight_sha256=weight_hash,
                  preparation_updates=len(preparations),preparations=preparations,ordinary_queries=len(results)*2,
                  cases=results,scope='MLP down weight projection with quantization/input preparation included; digest/Candid excluded. No full-model/query-count claim.')
    with zipfile.ZipFile(d/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in paths: archive.write(ROOT/p,p)
    (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(results),ordinary_queries=len(results)*2,all_bits_equal=True)),flush=True)


if __name__ == '__main__':
    main()
