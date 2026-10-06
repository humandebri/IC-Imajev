#!/usr/bin/env python3
"""Measure output32/output128 on real LoRA weights and verify every output bit."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / 'artifacts/f32-output128-v1/build'
HELPER = ROOT / 'artifacts/f32-block-native/release/f32_args'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canister', required=True)
    parser.add_argument('--tensor', choices=['down-B'], required=True)
    args = parser.parse_args()
    directory = ROOT / 'artifacts/f32-output128-v1' / args.tensor
    directory.mkdir(exist_ok=False)
    manifest = json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
    assert manifest['model'] == sha(ROOT/'MODEL_LOCK.json')
    tensor = {'gate-A':'gate_proj.lora_A.weight', 'down-B':'down_proj.lora_B.weight',
              'down-A':'down_proj.lora_A.weight'}[args.tensor]
    name = 'model.language_model.layers.0.mlp.' + tensor
    weight = next(w for w in manifest['tensors'] if w['name'] == name)
    rows, cols = weight['rows'], weight['cols']
    assert rows%128 == 0 and cols%64 == 0 and weight['dtype'] == 'f32'
    with (ROOT/'checkpoints/full-int8.pack').open('rb') as stream:
        stream.seek(weight['offset']); raw = stream.read(weight['bytes'])
    assert len(raw) == rows*cols*4
    weights = directory/'weights.bin'; weights.write_bytes(raw)
    expected_module = sha(BUILD/'diagnostic.wasm')
    count = 0

    def status():
        return json.loads(subprocess.check_output([
            'icp','canister','status',args.canister,'--network','local','--identity','imajev-local','--json'],text=True))['module_hash'].removeprefix('0x')

    def call(method, argument=None, query=False):
        nonlocal count
        count += 1
        command = ['icp','canister','call',args.canister,method,'--network','local','--identity','imajev-local','--output','hex']
        command += ['--args-file',str(argument),'--args-format','bin'] if argument else ['()']
        if query: command += ['--query']
        start = time.monotonic()
        output = subprocess.check_output(command,text=True,cwd=ROOT)
        path = directory/f'{count:04d}-{method}.hex'; path.write_text(output)
        measurement = json.loads(subprocess.check_output([str(HELPER),'decode',str(path),'measurement' if query else 'preparation'],text=True))
        measurement.update(seconds=time.monotonic()-start,reply_sha256=sha(path))
        return measurement

    assert status() == expected_module
    preparation = []
    chunk_rows = max(1,1_000_000//(cols*4))
    for start in range(0,rows,chunk_rows):
        end = min(rows,start+chunk_rows)
        path = directory/f'weights-{start}.bin'; path.write_bytes(raw[start*cols*4:end*cols*4])
        arg = directory/f'weights-{start}.args.bin'
        subprocess.run([str(HELPER),'chunk',str(start),str(path),str(arg)],check=True)
        preparation.append(call('prepare_chunk',arg))
    preparation.append(call('seal'))
    cases = []
    if args.tensor in ['gate-A','down-B']:
        origin = ROOT/('artifacts/f32_block/check' if args.tensor=='gate-A' else 'artifacts/f32_reuse/check')
        previous = json.loads((origin/'report.json').read_text())
        for label in ['prefix','617','insufficient','maximum']:
            path = origin/f'{label}.input.bin'
            evidence = next(c for c in previous['cases'] if c['label']==label)
            assert sha(path)==evidence['input_sha256']
            cases.append((label,np.frombuffer(path.read_bytes(),dtype='<f4'),dict(source=str(path.relative_to(ROOT)),source_sha256=sha(path),scope='Previously measured real canister LoRA input')))
        # Short-path sizes use prefixes of a saved real activation tensor, not
        # a newly executed short-prompt model. This is a kernel-size probe.
        actual = np.frombuffer((origin/'617.input.bin').read_bytes(),dtype='<f4').reshape(-1,cols)
        for n in [57,59,67]:
            cases.append((f'size-{n}',actual[:n].ravel(),dict(scope='First n tokens of saved real activation; kernel probe only')))
    for n in ([1,3,7,8,32,45] if args.tensor=='down-A' else [1,3,7,8,32,64,88,132]):
        values = np.resize(np.array([-0.,0.,1.,-1.,0.1234567,-0.99999994,2**-126,-2**-126],dtype='<f4'),n*cols)
        cases.append((f'boundary-{n}',values,dict(scope='Synthetic signed/zero/subnormal input')))
    result = []
    for label, values, provenance in cases:
        n = values.size//cols
        path = directory/f'{label}.input.bin'; path.write_bytes(values.astype('<f4').tobytes())
        assert path.stat().st_size < 1_900_000
        native = json.loads(subprocess.check_output([str(HELPER),'native',str(n),str(rows),str(cols),str(weights),str(path)],text=True))
        measurements = {}
        for method, key in [(0,'output32'),(1,'output128')]:
            arg = directory/f'{label}-{method}.args.bin'
            subprocess.run([str(HELPER),'query',str(method),str(path),str(arg)],check=True)
            measurements[key] = call('project',arg,True)
            assert measurements[key]['digest']==native['digest'], (label,key)
        change = 100*(measurements['output128']['total_instructions']/measurements['output32']['total_instructions']-1)
        case = dict(label=label,tokens=n,rows=rows,cols=cols,input_sha256=sha(path),native=native,
                    provenance=provenance,measurements=measurements,change_percent=change,bitwise_equal=True)
        result.append(case)
        print(json.dumps(dict(label=label,tokens=n,change_percent=change)),flush=True)
    assert status() == expected_module
    report = dict(model=manifest['model'],pack_hash=manifest['pack_hash'],tensor=name,
                  weight_sha256=sha(weights),wasm_sha256=expected_module,canister=args.canister,
                  source_sha256=sha(pathlib.Path(__file__)),build_report_sha256=sha(BUILD/'report.json'),
                  helper_sha256=sha(HELPER),preparation_updates=len(preparation),preparation=preparation,
                  ordinary_queries=2*len(result),cases=result,
                  scope='Projection only; counters include F32 input decode, excludes digest/Candid. No whole-model or query-count claim')
    (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(tensor=args.tensor,ordinary_queries=report['ordinary_queries'],all_bits_equal=True)),flush=True)


if __name__ == '__main__':
    main()
