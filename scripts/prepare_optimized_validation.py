#!/usr/bin/env python3
"""Prepare a dedicated loopback validation canister with verified model/cache inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['url', 'canister', 'module', 'identity-pem']:
        p.add_argument('--' + name, required=True)
    for name in ['directory', 'bridge', 'helper', 'project-root', 'cache-reference', 'manifest', 'pack']:
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--identity', required=True)
    p.add_argument('--skip-upload', action='store_true', help='Require the existing model to be fully ready')
    p.add_argument('--install-paid-prefixes', action='store_true', help='Install the saved five-token prefixes for local paid tests')
    a = p.parse_args()
    url = urlparse(a.url)
    if url.scheme != 'http' or url.hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise SystemExit('Only explicit loopback endpoints are accepted')
    network = json.loads(subprocess.check_output(['icp', 'network', 'status', '--json', '--project-root-override', str(a.project_root.resolve())]))
    if urlparse(network['api_url']).port != url.port:
        raise SystemExit('Configured project network does not match the selected endpoint')
    model = json.loads(a.manifest.read_text())
    release = json.loads((ROOT / 'frontend/src/inference-release.json').read_text())
    if (model['model'], model['pack_hash']) != (release['model'], release['pack_hash']):
        raise SystemExit('Pinned model/pack required')
    a.directory.mkdir(parents=True, exist_ok=False)
    bridge = subprocess.Popen([str(a.bridge.resolve()), a.url, a.canister, a.identity_pem], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    def call(value):
        bridge.stdin.write(json.dumps(value) + '\n'); bridge.stdin.flush()
        row = json.loads(bridge.stdout.readline())
        if 'error' in row:
            raise RuntimeError(row['error'])
        return row['ok']
    try:
        if call(dict(op='module_hash'))['module_hash'] != a.module:
            raise RuntimeError('Validation target module mismatch')
        if not a.skip_upload:
            call(dict(op='upload_parallel', manifest=str(a.manifest.resolve()), pack=str(a.pack.resolve()), concurrency=16))
        if not call(dict(op='pack_status'))['ready']:
            raise RuntimeError('Model is not ready')
        reference = json.loads(a.cache_reference.read_text())['ok']['cache']
        cache = call(dict(op='weight_cache_status'))['cache']
        present = set(cache['names'])
        for i, name in enumerate(reference['names']):
            if name not in present:
                cache = call(dict(op='warm_weights', name=name))['cache']
            if i % 32 == 0:
                print(json.dumps(dict(warmed=i + 1, total=len(reference['names']))), flush=True)
        if cache['names'] != reference['names'] or cache['bytes'] != reference['bytes']:
            raise RuntimeError('Weight cache conditions differ')
        prefix = ROOT / 'frontend/public/inference/prefix5-v1'
        raw = (prefix / 'manifest.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != release['manifest_sha256']:
            raise RuntimeError('Prefix release mismatch')
        manifest = json.loads(raw)
        did = a.directory / 'prefix.did'
        did.write_text('service:{prepareFixedPrefixCache:(blob,blob)->(variant {Ok:record {nat32;nat64;nat64};Err:text})}')
        prepared = []
        paid_prefixes = []
        for layer in range(32):
            entry = manifest['assets'][str(layer)]
            data = (prefix / entry['file']).read_bytes()
            if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise RuntimeError('Prefix asset checksum mismatch')
            def bf16(words):
                return b''.join(struct.pack('<I', word[0] << 16) for word in struct.iter_unpack('<H', words))
            if layer % 4 == 3:
                if a.install_paid_prefixes:
                    values = a.directory / f'layer-{layer:02d}.kv.f32'
                    values.write_bytes(bf16(data))
                    paid_prefixes.append(call(dict(op='update_prefix', layer=layer, values=str(values.resolve()))))
                continue
            # Restore only the exact saved BF16 words; no host model inference.
            begin, end = 24576 * 2, (24576 + 5 * 2048) * 2
            log = bf16(data[begin:end]) + data[end:]
            if len(log) != 5 * 6176 * 4:
                raise RuntimeError('Prefix log shape')
            log_path, packet_path, args_path = [a.directory / f'layer-{layer:02d}.{suffix}' for suffix in ['f32', 'npf1', 'args.bin']]
            log_path.write_bytes(log)
            subprocess.run([str(a.helper.resolve()), 'packet', str(log_path), str(packet_path)], check=True)
            subprocess.run([str(a.helper.resolve()), 'args', str(log_path), str(packet_path), str(args_path)], check=True)
            reply = subprocess.check_output(['icp', 'canister', 'call', a.canister, 'prepareFixedPrefixCache', '--network', 'local', '--identity', a.identity, '--project-root-override', str(a.project_root.resolve()), '--candid', str(did.resolve()), '--args-file', str(args_path.resolve()), '--args-format', 'bin', '--output', 'hex'], text=True)
            reply_path = a.directory / f'layer-{layer:02d}.reply.hex'; reply_path.write_text(reply)
            result = json.loads(subprocess.check_output([str(a.helper.resolve()), 'decode', str(reply_path)]))
            if result['count'] > 24 or result['bytes'] != result['count'] * 2_097_152:
                raise RuntimeError('Fixed prefix cache result')
            prepared.append(dict(layer=layer, result=result, log_sha256=hashlib.sha256(log).hexdigest(), packet_sha256=hashlib.sha256(packet_path.read_bytes()).hexdigest()))
            if a.install_paid_prefixes:
                values = a.directory / f'layer-{layer:02d}.conv.f32'
                values.write_bytes(bf16(data[:begin]))
                paid_prefixes.append(call(dict(op='update_prefix', layer=layer, values=str(values.resolve()), packet=str(packet_path.resolve()))))
        if call(dict(op='module_hash'))['module_hash'] != a.module:
            raise RuntimeError('Validation target changed')
        report = dict(complete=True, module=a.module, model=model['model'], pack_hash=model['pack_hash'], weight_cache_bytes=cache['bytes'], names=cache['names'], prefix=prepared, paid_prefixes=paid_prefixes, scope='Dedicated local validation only; not a published release asset regeneration')
        (a.directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        bridge.stdin.close(); bridge.wait()


if __name__ == '__main__':
    main()
