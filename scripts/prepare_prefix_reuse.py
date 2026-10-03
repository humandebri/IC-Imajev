#!/usr/bin/env python3
"""Prepare an exact client-held prefix once; reuse requires no preparation queries.

This is the isolated codec path, not yet connected to TextGraph inference.
The input logs must be saved canister outputs. All encoding runs on-canister.
"""
import argparse
import hashlib
import io
import json
import pathlib
import struct
import subprocess
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODEC_MODULE = '8ef15af7f8e97db0dc28d8ad7b6e32f0ed191c26bc84bdd85d7989dd104cd737'
LAYERS = tuple(i for i in range(32) if (i + 1) % 4)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_packet(packet):
    if not 12 <= len(packet) <= 1_990_000:
        raise ValueError('prefix packet size')
    magic, tokens, mask = struct.unpack('<4sII', packet[:12])
    if magic != b'NPF1' or tokens != 45 or mask.bit_count() != 18:
        raise ValueError('prefix packet header')
    if any(((mask >> (2 * p)) & 3) not in (0, 3) for p in range(16)):
        raise ValueError('prefix packet shared-K pairs')


def source_identity(directory):
    raw = (directory / 'report.json').read_bytes()
    report = json.loads(raw)
    if report.get('prefix_mode') != 'prepare' or report.get('tokens') != 45:
        raise ValueError('requires a saved 45-token canister prefix preparation')
    if report.get('fuse_delta_full_log') != 'full32-exact-prefix-innovation-v1':
        raise ValueError('requires original exact innovation logs')
    files = {str(i): sha((directory / f'queries/states/layer-{i:02d}.npz').read_bytes()) for i in LAYERS}
    return dict(version=1, codec_module=CODEC_MODULE, source_report_sha256=sha(raw),
                model=report['model'], pack_hash=report['pack_hash'],
                input_hash=report['input_hash'], source_module=report['wasm_sha256'],
                tokens=45, layers=files)


def prepare_cache(source, destination, canister, status, prepare):
    """Callbacks keep the cache policy independently testable; no host state encoder."""
    identity = source_identity(source)
    if status() != CODEC_MODULE:
        raise ValueError('prefix preparation canister module mismatch')
    manifest_path = destination / 'cache.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_bytes())
        if saved.get('identity') != identity or set(saved.get('packets', {})) != set(identity['layers']):
            raise ValueError('prefix cache identity mismatch; use a new output directory')
        for layer, entry in saved['packets'].items():
            name = f'layer-{int(layer):02d}.npf1'
            packet = (destination / name).read_bytes()
            if sha(packet) != entry['sha256'] or len(packet) != entry['bytes']:
                raise ValueError('prefix cache packet hash mismatch')
            validate_packet(packet)
        result = dict(cache_hit=True, preparation_queries=0, preparation_instructions=0,
                      preparation_candid_bytes=0, packets=saved['packets'])
    else:
        destination.mkdir(parents=True, exist_ok=True)
        packets = {}
        measurements = []
        for layer in LAYERS:
            path = source / f'queries/states/layer-{layer:02d}.npz'
            # Bookend each source: do not bind a packet to a changed source file.
            raw = path.read_bytes()
            if sha(raw) != identity['layers'][str(layer)]:
                raise ValueError('prefix source changed')
            with np.load(io.BytesIO(raw), allow_pickle=False) as state:
                log = state['delta_log'].copy()
            if log.shape != (45 * 6176,) or log.dtype != np.float32 or not np.isfinite(log).all():
                raise ValueError('prefix log shape/precision')
            packet, measured = prepare(layer, log.astype('<f4').tobytes())
            validate_packet(packet)
            name = f'layer-{layer:02d}.npf1'
            (destination / name).write_bytes(packet)
            packets[str(layer)] = dict(sha256=sha(packet), bytes=len(packet),
                                      state_digest=measured['digest'])
            measurements.append(measured)
        if source_identity(source) != identity:
            raise ValueError('prefix source changed during preparation')
        if status() != CODEC_MODULE:
            raise ValueError('prefix canister changed during preparation')
        # Publish only after all 24 packets and source/module bookends succeed.
        temporary = destination / 'cache.json.pending'
        temporary.write_text(json.dumps(dict(identity=identity, packets=packets), indent=2) + '\n')
        temporary.replace(manifest_path)
        result = dict(cache_hit=False, preparation_queries=len(measurements),
                      preparation_instructions=sum(m['instructions'] for m in measurements),
                      preparation_candid_bytes=sum(m['request_candid_bytes'] + m['reply_candid_bytes'] for m in measurements),
                      packets=packets, measurements=measurements)
    result.update(codec_canister=canister, codec_module=CODEC_MODULE,
                  cache_identity_sha256=sha(manifest_path.read_bytes()),
                  scope='Client-held prefix codec preparation/reuse only; full inference is not connected.')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix-directory', required=True, type=pathlib.Path)
    ap.add_argument('--directory', required=True, type=pathlib.Path)
    ap.add_argument('--canister', required=True)
    ap.add_argument('--run-report', required=True, type=pathlib.Path)
    args = ap.parse_args()
    helper = ROOT / 'artifacts/bounded_i16/native/release/prefix_args'
    base = ['--network', 'local', '--identity', 'imajev-local']

    def status():
        raw = subprocess.check_output(['icp', 'canister', 'status', args.canister, *base, '--json'], text=True, cwd=ROOT)
        return json.loads(raw)['module_hash'].removeprefix('0x')

    def prepare(layer, log):
        work = args.directory / 'preparation'
        work.mkdir(parents=True, exist_ok=True)
        src = work / f'layer-{layer:02d}.log.bin'
        arg = work / f'layer-{layer:02d}.args.bin'
        reply = work / f'layer-{layer:02d}.reply.hex'
        output = work / f'layer-{layer:02d}.npf1'
        src.write_bytes(log)
        subprocess.run([str(helper), 'prepare', str(src), str(arg)], check=True, cwd=ROOT)
        start = time.monotonic()
        raw = subprocess.check_output(['icp', 'canister', 'call', args.canister, 'prepare_prefix', *base,
                                       '--query', '--args-file', str(arg), '--args-format', 'bin', '--output', 'hex'], text=True, cwd=ROOT)
        elapsed = time.monotonic() - start
        reply.write_text(raw)
        measured = json.loads(subprocess.check_output([str(helper), 'decode-preparation', str(reply), str(output)], text=True, cwd=ROOT))
        measured.update(layer=layer, wall_seconds=elapsed, request_candid_bytes=arg.stat().st_size,
                        reply_candid_bytes=len(bytes.fromhex(raw.strip().removeprefix('0x'))), reply_sha256=sha(raw.encode()))
        return output.read_bytes(), measured

    start = time.monotonic()
    result = prepare_cache(args.prefix_directory, args.directory, args.canister, status, prepare)
    result.update(wall_seconds=time.monotonic() - start, helper_sha256=sha(helper.read_bytes()),
                  client_source_sha256=sha(pathlib.Path(__file__).read_bytes()))
    args.run_report.parent.mkdir(parents=True, exist_ok=True)
    args.run_report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['cache_hit', 'preparation_queries', 'preparation_instructions', 'preparation_candid_bytes']}))


if __name__ == '__main__':
    main()
