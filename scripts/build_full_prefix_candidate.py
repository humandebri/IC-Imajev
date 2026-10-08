#!/usr/bin/env python3
"""Freeze source and build a fail-closed full-prefix experimental Wasm.

Does not install. The raw pair ABI stub must be patched and validated before
use. Dedicated target directories leave the adopted module intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--directory', required=True)
ap.add_argument('--target-directory', required=True)
ap.add_argument('--opt-level', choices=['1', '2', '3'], default='3')
ap.add_argument('--single-quad',action='store_true')
ap.add_argument('--strassen-raw',action='store_true')
ap.add_argument('--strassen-output128',action='store_true')
ap.add_argument('--f32-k-continue',action='store_true')
ap.add_argument('--attention-mlp-stream',action='store_true')
ap.add_argument('--direct-mlp-reply',action='store_true')
ap.add_argument('--mlp-half',action='store_true')
ap.add_argument('--mlp-attention-finish',action='store_true')
ap.add_argument('--mlp-stream',action='store_true')
ap.add_argument('--mlp-delta-stream',action='store_true')
ap.add_argument('--int8-k-continue',action='store_true')
ap.add_argument('--f32-output-reuse',action='store_true')
ap.add_argument('--f32-output-generic',action='store_true')
ap.add_argument('--direct-input', action='store_true',
                help='Use the matched original-I16-buffer WAT ABI')
ap.add_argument('--terminal-attention', action='store_true')
ap.add_argument('--terminal-tail', action='store_true')
ap.add_argument('--terminal-stream', action='store_true')
ap.add_argument('--host-checksum',action='store_true')
ap.add_argument('--prefix-start', action='store_true')
ap.add_argument('--delta-no-writeback', action='store_true')
ap.add_argument('--delta-state-layout', action='store_true')
ap.add_argument('--prefix-update-hoist',action='store_true')
ap.add_argument('--mlp-delta-fusion', action='store_true')
ap.add_argument('--instruction-profile', action='store_true')
ap.add_argument('--attention-views', action='store_true')
args = ap.parse_args()
directory = ROOT/args.directory
directory.mkdir(parents=True, exist_ok=True)
assert not (directory/'source.zip').exists(), 'Use a new directory for a changed build'
features = ','.join([
    'experimental-attention-full','experimental-attention-fusion',
    'experimental-blake3','experimental-byte-buffer',
    'experimental-delta-finish','experimental-delta-full-log',
    'experimental-delta-projected',
    'experimental-full-weight-cache','experimental-matrix-tail',
    'experimental-mlp-full','experimental-mlp-pipeline',
    'experimental-prepared-activation','experimental-prepared-output-pairs',
    'experimental-prepared-rope','experimental-projection-reuse',
    'experimental-prefix-hybrid','experimental-lora-input-sharing',
    'experimental-mlp-full89','experimental-pair-wat'])
features += ',experimental-paired-only'
if args.single_quad:features += ',experimental-single-quad'
if args.attention_views:features += ',experimental-attention-views'
if args.host_checksum:features += ',experimental-host-checksum'
if args.strassen_raw:features += ',experimental-strassen-raw'
if args.strassen_output128:features += ',experimental-strassen-output128'
if args.f32_k_continue:features += ',experimental-f32-k-continue'
if args.attention_mlp_stream:features += ',experimental-attention-mlp-stream'
if args.mlp_attention_finish:features += ',experimental-mlp-attention-finish'
if args.direct_mlp_reply:features += ',experimental-direct-mlp-reply'
if args.mlp_half:features += ',experimental-mlp-half'
if args.mlp_stream:features += ',experimental-mlp-stream'
if args.mlp_delta_stream:features += ',experimental-mlp-delta-stream'
if args.int8_k_continue:features += ',experimental-int8-k-continue'
if args.f32_output_reuse:features += ',experimental-f32-output-reuse'
if args.f32_output_generic:features += ',experimental-f32-output-generic'
if args.direct_input:
    features += ',experimental-pair-direct-input'
if args.terminal_attention:
    features += ',experimental-terminal-attention'
if args.instruction_profile:
    features += ',instruction-profile'
if args.mlp_delta_fusion:
    features += ',experimental-mlp-delta-fusion'
if args.prefix_update_hoist:features += ',experimental-prefix-update-hoist'
if args.delta_state_layout:
    features += ',experimental-delta-state-layout'
if args.delta_no_writeback:
    features += ',experimental-delta-no-writeback'
if args.prefix_start:
    features += ',experimental-prefix-start'
if args.terminal_stream:features += ',experimental-terminal-stream'
if args.terminal_tail:
    assert args.terminal_attention
    features += ',experimental-terminal-tail'
kernel = ROOT/('artifacts/prefix_codec/direct-input-build/direct-input.wat'
               if args.direct_input else 'artifacts/prefix_codec/full-wat/reuse.wat')
generator=ROOT/'scripts'/('generate_pair_direct_input_wat.py' if args.direct_input
                         else 'generate_pair_reuse_wat.py')
subprocess.run([sys.executable,str(generator)],cwd=ROOT,check=True)
paths = list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']
paths += list((ROOT/'canisters/inference/src').rglob('*.rs'))
paths += list((ROOT/'crates/imajev-client/src').rglob('*.rs'))
paths += list((ROOT/'client').glob('*.py'))
paths += [ROOT/p for p in ['Cargo.toml', 'Cargo.lock',
    'crates/imajev-runtime/Cargo.toml', 'canisters/inference/Cargo.toml',
    'crates/imajev-client/Cargo.toml', 'scripts/run_prefix_canister.py',
    'scripts/run_full_canister.py', 'canisters/inference/inference.did',
    'MODEL_LOCK.json', 'checkpoints/full-int8.manifest.json',
    'scripts/generate_pair_reuse_wat.py', 'scripts/check_full_prefix_hybrid.py']]
paths += [Path(__file__)]
paths += [kernel,ROOT/'scripts/build_pair_direct_input.py',generator]
paths = list(dict.fromkeys(paths))
sha = lambda b: hashlib.sha256(b).hexdigest()
hashes = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in paths}
with zipfile.ZipFile(directory/'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        z.write(p, str(p.relative_to(ROOT)))
(directory/'features.txt').write_text(features+'\n')
(directory/'source-hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
explicit_env = {'CARGO_PROFILE_RELEASE_OPT_LEVEL': args.opt_level}
env = dict(os.environ, **explicit_env)
command = ['cargo', 'build', '--offline', '--release', '-j1',
           '-p', 'imajev-inference', '--target', 'wasm32-unknown-unknown',
           '--features', features, '--target-dir', args.target_directory]
config = dict(command=command, environment=explicit_env,
    kernel_source_sha256=sha(kernel.read_bytes()),direct_input=args.direct_input,
    compiler=subprocess.check_output(['rustc', '--version', '--verbose'], text=True),
    scope='Raw fail-closed pair ABI stub. Patch and validate before installing.')
(directory/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
with (directory/'compiler.log').open('w') as log:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=log)
unchanged = hashes == {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in paths}
report = dict(exit_code=result.returncode, source_unchanged=unchanged)
if result.returncode == 0:
    assert unchanged
    module = ROOT/args.target_directory/'wasm32-unknown-unknown/release/imajev_inference.wasm'
    report.update(raw_module_sha256=sha(module.read_bytes()), raw_module_bytes=module.stat().st_size)
    (directory/'raw.wasm').write_bytes(module.read_bytes())
(directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
raise SystemExit(result.returncode)
