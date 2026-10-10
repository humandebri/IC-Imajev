"""Validated five-token prefix assets shared by current paid verification tools."""
import hashlib
import json
from pathlib import Path

COMMON_PREFIX = [248045, 846, 198, 1349, 25]
MAX_TOKENS = 1024


def length_fixture(record, total):
    ids = record['token_ids']
    if ids[:5] != COMMON_PREFIX or len(ids) < 9:
        raise ValueError('requires a current five-token prompt fixture')
    if not 9 <= total <= MAX_TOKENS + 1:
        raise ValueError('fixture total token bounds')
    if total < len(ids):
        return ids[:total - 4] + ids[-4:]
    extra = ([198, 220, 16] * ((total - len(ids) + 2) // 3))[:total - len(ids)]
    return ids[:-4] + extra + ids[-4:]


def load_prefix(source, packets, manifest):
    from prefix_inference import load_cache
    from prefix_hybrid import load_packets
    source, packets = Path(source), Path(packets)
    raw = (source / 'report.json').read_bytes()
    report = json.loads(raw)
    cache = load_cache(source / 'queries', manifest, report['wasm_sha256'])
    if report['tokens'] != len(COMMON_PREFIX) or cache['metadata']['token_ids'] != COMMON_PREFIX:
        raise ValueError('requires the five-token common prefix')
    packet_manifest = json.loads((packets / 'cache.json').read_text())
    if packet_manifest['identity']['source_report_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('packet source report mismatch')
    cache['hybrid_packets'] = load_packets(packets, cache)
    return cache
