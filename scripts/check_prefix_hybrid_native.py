#!/usr/bin/env python3
"""Compare complete native Delta projections/recurrence/output with exact packets.

Only saved canister-produced query operands enter the runtime. This is boundary
parity evidence, not a full-model Wasm or judgment-accuracy claim.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import decode
from prefix_hybrid import NAME, encode_request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', default='617:0,617:22,617:30,insufficient:0,maximum:0')
    ap.add_argument('--directory', default='artifacts/prefix_codec/native-delta-check')
    args = ap.parse_args()
    directory = ROOT / args.directory
    directory.mkdir(parents=True, exist_ok=True)
    helper = ROOT / 'artifacts/prefix_codec/native-target/release/primitive'
    sha = lambda b: hashlib.sha256(b).hexdigest()
    sources = list((ROOT/'crates/imajev-runtime/src').rglob('*.rs')) + [ROOT/'crates/imajev-runtime/Cargo.toml', ROOT/'Cargo.lock', ROOT/'client/prefix_hybrid.py', pathlib.Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sources}
    measurements = []
    for case in args.cases.split(','):
        label, layer = case.split(':'); layer = int(layer)
        source = ROOT / f'artifacts/output-pairs-v2-{label}'
        raw = (source/'report.json').read_bytes(); report = json.loads(raw)
        query = next(q for q in report['queries'] if q['op']=='delta_full_log_integer' and f'.layers.{layer}.' in q['tensor'])
        request = source/'queries'/f'{query["index"]:06d}.request.bin'
        frame = request.read_bytes(); header, values = decode(frame)
        n, heads, prefix, keep = header['dims']
        assert heads==32 and prefix==45 and keep==0
        packet_path = ROOT/f'artifacts/prefix_codec/reuse-cache/layer-{layer:02d}.npf1'
        packet = packet_path.read_bytes()
        with np.load(ROOT/f'artifacts/output-pairs-v2-prefix/queries/states/layer-{layer:02d}.npz', allow_pickle=False) as state:
            assert values[n*2560+3*8192:].tobytes() == state['delta_log'].tobytes()
        hybrid = dict(header, op='delta_full_hybrid_integer', encoding=NAME)
        candidate_frame = encode_request(hybrid, values[:n*2560+3*8192], packet)
        prefix_path = directory/f'{label}-{layer:02d}'
        baseline_path = prefix_path.with_suffix('.baseline.request.bin')
        hybrid_path = prefix_path.with_suffix('.hybrid.request.bin')
        baseline_path.write_bytes(frame); hybrid_path.write_bytes(candidate_frame)
        results = {}
        for kind, path in [('baseline',baseline_path),('hybrid',hybrid_path)]:
            output = prefix_path.with_suffix(f'.{kind}.response.bin')
            start = time.monotonic()
            subprocess.run([str(helper),str(path),str(output),str(ROOT/'checkpoints/full-int8.manifest.json'),str(ROOT/'checkpoints/full-int8.pack')],check=True,cwd=ROOT)
            elapsed = time.monotonic()-start
            returned, out = decode(output.read_bytes())
            assert returned['step']==header['step']+1
            results[kind] = dict(values=out.size,digest=sha(out.tobytes()),wall_seconds=elapsed,output_sha256=sha(output.read_bytes()))
        assert results['baseline']['digest']==results['hybrid']['digest']
        row=dict(label=label,layer=layer,tokens=n,bitwise_equal=True,request_sha256=sha(frame),source_report_sha256=sha(raw),packet_sha256=sha(packet),hybrid_frame_sha256=sha(candidate_frame),measurements=results)
        measurements.append(row)
        print(json.dumps(row),flush=True)
    assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sources}
    result=dict(scope='Complete native Delta boundary only; no Wasm instructions/full inference/judgment claim',helper_sha256=sha(helper.read_bytes()),source_hashes=hashes,cases=measurements)
    (directory/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
