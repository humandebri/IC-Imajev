#!/usr/bin/env python3
"""Prepare pinned immutable dense weights using owner updates."""
import argparse
import hashlib
import json
import pathlib
import sys
import time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport
from prefix_inference import verify_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canister', required=True)
    ap.add_argument('--wasm', required=True)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--manifest', default='checkpoints/full-int8.manifest.json')
    ap.add_argument('--include-f32', action='store_true')
    ap.add_argument('--require-prepared-rope', action='store_true')
    ap.add_argument('--require-prepared-activation', action='store_true')
    ap.add_argument('--require-output-pairs', action='store_true')
    ap.add_argument('--require-all-output-pairs', action='store_true',
                    help='Also require prepared 32-row gate weights for paired-only builds')
    ap.add_argument('--cache-budget', type=int)
    args = ap.parse_args()
    dest = ROOT / args.directory
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / args.manifest
    m = json.loads(manifest_path.read_text())
    assert m['model'] == hashlib.sha256((ROOT / 'MODEL_LOCK.json').read_bytes()).hexdigest()
    tensors = [x for x in m['tensors'] if (x['dtype'] == 'int8' or args.include_f32 and x['dtype'] == 'f32') and 'embed_tokens' not in x['name']]
    assert all(x['bytes'] == (x['rows'] * x['cols'] + 4 * x['rows'] if x['dtype'] == 'int8' else 4 * x['rows'] * x['cols']) and 0 < x['bytes'] <= 128 * 1024 * 1024 for x in tensors)
    expected_bytes = sum(x['bytes'] for x in tensors)
    min_pair_rows=32 if args.require_all_output_pairs else 512
    expected_paired_bytes=sum(x['rows']*x['cols'] for x in tensors if x['dtype']=='int8' and x['rows']>=min_pair_rows and x['rows']%32==0 and x['cols']%256==0 and x['rows']*x['cols']<=30_000_000)
    cache_budget = args.cache_budget if args.cache_budget is not None else (4_080_000_000 if args.include_f32 else 3_600_000_000)
    if expected_bytes > cache_budget:
        raise ValueError('Declared dense cache budget')
    expected_names = {x['name'] for x in tensors}
    sha = hashlib.sha256((ROOT / args.wasm).read_bytes()).hexdigest()
    t = Transport(m['model'], 'http://localhost:8001/', args.canister,
                  str(ROOT / 'artifacts/imajev-local.pem'), dest, m['pack_hash'])
    updates, sent, received, instructions = 0, 0, 0, 0
    start = time.monotonic()
    try:
        verify_module(t, sha)
        pack = t.command(dict(op='pack_status'))['ok']
        assert pack['ready'] and pack['model'] == m['model'] and pack['pack_hash'] == m['pack_hash']
        assert pack['bytes'] == pack['received'] == pack['hashed'] == m['bytes']
        initial = t.command(dict(op='weight_cache_status'))['ok']['cache']
        assert set(initial['names']) <= expected_names
        cached = set(initial['names'])
        assert initial['bytes'] == sum(x['bytes'] for x in tensors if x['name'] in cached)
        for index, tensor in enumerate(tensors):
            if tensor['name'] in cached:
                continue
            result = t.command(dict(op='warm_weights', name=tensor['name']))
            info = result['ok']['cache']
            cached.add(tensor['name'])
            assert set(info['names']) == cached
            assert info['bytes'] == sum(x['bytes'] for x in tensors if x['name'] in cached)
            updates += 1
            sent += result['ok']['request_bytes']
            received += result['ok']['reply_bytes']
            instructions += info['preparation_instructions']
            with (dest / 'updates.jsonl').open('a') as log:
                log.write(json.dumps(dict(index=index, tensor=tensor['name'], **result)) + '\n')
            print(json.dumps(dict(index=index, cached_bytes=info['bytes'], update_instructions=info['preparation_instructions'])), flush=True)
        final = t.command(dict(op='weight_cache_status'))['ok']['cache']
        if args.require_prepared_rope: assert final.get('rope_bytes') == 131072
        if args.require_prepared_activation: assert final.get('activation_bytes') == 1048576
        if args.require_output_pairs or args.require_all_output_pairs: assert final.get('paired_weight_bytes') == expected_paired_bytes
        assert final['bytes'] == expected_bytes and set(final['names']) == expected_names
        verify_module(t, sha)
        report = dict(scope='Immutable model weight preparation only; no question-dependent inference',
            include_f32=args.include_f32, require_prepared_rope=args.require_prepared_rope, require_prepared_activation=args.require_prepared_activation, require_output_pairs=args.require_output_pairs, require_all_output_pairs=args.require_all_output_pairs, expected_paired_bytes=expected_paired_bytes, declared_cache_budget=cache_budget,
            model=m['model'], pack_hash=m['pack_hash'], wasm_sha256=sha,
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            initial=initial, final=final, expected_tensors=len(tensors), expected_bytes=expected_bytes,
            update_calls_this_run=updates, preparation_instructions_this_run=instructions,
            update_request_bytes_this_run=sent, update_reply_bytes_this_run=received,
            pack_status_queries=1, cache_status_queries=2, certified_module_reads=2,
            wall_seconds_this_run=time.monotonic()-start)
        (dest / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('initial', 'final')}), flush=True)
    finally:
        t.close()


if __name__ == '__main__':
    main()
