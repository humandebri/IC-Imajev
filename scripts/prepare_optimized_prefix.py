#!/usr/bin/env python3
"""Compute a fresh five-token prefix on an explicit local candidate, never on the host."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from full_inference import JournalTransport, TextGraph
from prefix_inference import file_hash, graph_hash, load_cache, verify_module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['url', 'canister', 'module', 'identity-pem']:
        p.add_argument('--' + name, required=True)
    for name in ['manifest', 'bridge', 'directory']:
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    u = urlparse(a.url)
    if u.scheme != 'http' or u.hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('Explicit loopback target required')
    a.directory.mkdir(parents=True, exist_ok=False)
    m = json.loads(a.manifest.read_text())
    ids = [248045, 846, 198, 1349, 25]
    queries = a.directory / 'queries'
    t = JournalTransport(m['model'], a.url, a.canister, a.identity_pem, queries, m['pack_hash'],
                         wire_codec='bf16-block256-exact-v1', frame_version=3,
                         bridge_binary=a.bridge.resolve(), input_hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    try:
        verify_module(t, a.module)
        graph = TextGraph(t, m, arithmetic='int8', compact_lossless=True, compact_heads=True,
                          retain_terminal_state=True, fuse_add_norm=True, fuse_mlp=True,
                          fuse_mlp_norm=True, fuse_mlp_pipeline=True, fuse_mlp_full=True,
                          fuse_delta=True, fuse_delta_projected=True, fuse_delta_full_log=True,
                          fuse_norm_rope=True, fuse_attention=True, fuse_attention_full=True)
        graph.forward(ids)
        verify_module(t, a.module)
        names = [f'layer-{i:02d}.npy' for i in range(32)] + [f'states/layer-{i:02d}.npz' for i in range(32)]
        metadata = dict(version=2, model=m['model'], pack_hash=m['pack_hash'], wasm_sha256=a.module,
                        graph_sha256=graph_hash(), token_ids=ids,
                        files={name: file_hash(queries / name) for name in names})
        (queries / 'cache.json').write_text(json.dumps(metadata, indent=2) + '\n')
        load_cache(queries, m, a.module)
        report = dict(wasm_sha256=a.module, model=m['model'], pack_hash=m['pack_hash'],
                      prefix_cache_sha256=file_hash(queries / 'cache.json'),
                      queries=len(t.measurements), instructions=sum(r['ok']['instructions'] for r in t.measurements),
                      release_eligible=False, origin='Fresh candidate canister queries; no retained prefix states')
        (a.directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        t.close()


if __name__ == '__main__':
    main()
