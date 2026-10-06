#!/usr/bin/env python3
"""Validate the full graph with immutable dense weights prepared beforehand."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport
from prefix_inference import verify_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canister', required=True)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--baseline', default='quantized-invariants-v2')
    ap.add_argument('--wasm', required=True)
    ap.add_argument('--preparation', required=True)
    ap.add_argument('--reuse-projection-inputs', action='store_true')
    ap.add_argument('--frame-checksum', choices=['sha256','blake3'], default='sha256')
    ap.add_argument('--fuse-attention', action='store_true')
    ap.add_argument('--fuse-delta-projected', action='store_true')
    ap.add_argument('--fuse-delta-finish', action='store_true')
    ap.add_argument('--fuse-mlp-pipeline', action='store_true')
    ap.add_argument('--fuse-attention-full', action='store_true')
    ap.add_argument('--fuse-mlp-full', action='store_true')
    ap.add_argument('--fuse-delta-full-log', action='store_true')
    ap.add_argument('--require-prepared-rope', action='store_true')
    ap.add_argument('--require-prepared-activation', action='store_true')
    ap.add_argument('--require-output-pairs', action='store_true')
    ap.add_argument('--mlp-full-token-cap',type=int,choices=[87,89],default=87)
    args = ap.parse_args()
    if not args.run_name.replace('-', '').isalnum():
        ap.error('Run name must be alphanumeric with hyphens')
    m = json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_text())
    prepared = json.loads((ROOT / args.preparation / 'report.json').read_text())
    sha = hashlib.sha256((ROOT / args.wasm).read_bytes()).hexdigest()
    assert prepared['wasm_sha256'] == sha and prepared['pack_hash'] == m['pack_hash']
    expected = prepared['final']
    if args.require_prepared_rope: assert expected.get('rope_bytes') == 131072 and prepared['require_prepared_rope']
    if args.require_prepared_activation: assert expected.get('activation_bytes') == 1048576 and prepared['require_prepared_activation']
    if args.require_output_pairs: assert expected.get('paired_weight_bytes') == prepared['expected_paired_bytes'] and prepared['require_output_pairs']
    directory = ROOT / f'artifacts/{args.run_name}-cache-checks'
    directory.mkdir(parents=True, exist_ok=True)
    def snapshot(label):
        t = Transport(m['model'], 'http://localhost:8001/', args.canister,
            str(ROOT / 'artifacts/imajev-local.pem'), directory / label, m['pack_hash'])
        try:
            verify_module(t, sha)
            cache = t.command(dict(op='weight_cache_status'))['ok']['cache']
            if args.require_prepared_rope: assert cache.get('rope_bytes') == 131072
            if args.require_prepared_activation: assert cache.get('activation_bytes') == 1048576
            if args.require_output_pairs: assert cache.get('paired_weight_bytes') == prepared['expected_paired_bytes']
            assert cache['bytes'] == expected['bytes'] and cache['names'] == expected['names']
            return cache
        finally:
            t.close()
    paths = [ROOT / 'Cargo.toml', ROOT / 'Cargo.lock', ROOT / 'requirements.lock', ROOT / 'canisters/inference/Cargo.toml', ROOT / 'canisters/inference/inference.did',
             ROOT / 'crates/imajev-runtime/Cargo.toml', ROOT / 'crates/imajev-client/src/main.rs',
             pathlib.Path(__file__), ROOT / 'scripts/prepare_weight_cache.py']
    paths += list((ROOT / 'client').glob('*.py'))
    paths += [ROOT / 'scripts' / name for name in ['run_full_canister.py','run_prefix_canister.py','validate_terminal_readout.py','compare_compact_heads.py']]
    paths += [ROOT/'crates/imajev-runtime/src/bin/delta_log_verify.rs']
    if args.require_output_pairs:
        paths += [ROOT/'scripts/generate_output_pairs.py',ROOT/'scripts/generate_lane_pair.py',ROOT/'scripts/lane_pair_bench/src/kernel.rs']
    paths += list((ROOT / 'canisters/inference/src').glob('*.rs'))
    paths += list((ROOT / 'crates/imajev-runtime/src').glob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = snapshot('before')
    subprocess.run([sys.executable, str(ROOT / 'scripts/validate_terminal_readout.py'),
        '--canister', args.canister, '--run-name', args.run_name, '--baseline', args.baseline,
        '--frame-checksum', args.frame_checksum, '--fuse-mlp-norm', '--fuse-norm-rope', '--wire-codec', 'bf16-block256-exact-v1', *(['--reuse-projection-inputs'] if args.reuse_projection_inputs else []), *(['--fuse-attention'] if args.fuse_attention else []), *(['--fuse-delta-projected'] if args.fuse_delta_projected else []), *(['--fuse-delta-finish'] if args.fuse_delta_finish else []), *(['--fuse-mlp-pipeline'] if args.fuse_mlp_pipeline else []), *(['--fuse-attention-full'] if args.fuse_attention_full else []), *(['--fuse-mlp-full','--mlp-full-token-cap',str(args.mlp_full_token_cap)] if args.fuse_mlp_full else []), *(['--fuse-delta-full-log'] if args.fuse_delta_full_log else [])], cwd=ROOT, check=True)
    after = snapshot('after')
    assert source_hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_kernel_unroll.py'),
        '--run-name', args.run_name, '--baseline', args.baseline], cwd=ROOT, check=True)
    (directory / 'report.json').write_text(json.dumps(dict(wasm_sha256=sha,
        pack_hash=m['pack_hash'], preparation_report_sha256=hashlib.sha256((ROOT / args.preparation / 'report.json').read_bytes()).hexdigest(),
        before=before, after=after, unchanged=True, prepared_rope_required=args.require_prepared_rope, prepared_activation_required=args.require_prepared_activation, output_pairs_required=args.require_output_pairs, implementation_hashes=source_hashes, administrative_cache_status_queries=2,
        certified_module_reads=2, scope='Cache preparation is fixed model data; all question inference remains ordinary queries'), indent=2)+'\n')


if __name__ == '__main__':
    main()
