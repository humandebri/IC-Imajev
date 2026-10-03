#!/usr/bin/env python3
"""Measure fixed reads and identical-input row splits; no inference/offloading."""
import argparse
import collections
import hashlib
import json
import pathlib
import re
import struct

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    source = ROOT / args.source
    raw = (source / 'report.json').read_bytes()
    report = json.loads(raw)
    manifest = json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_text())
    sizes, layers = collections.Counter(), collections.Counter()
    for tensor in manifest['tensors']:
        name = tensor['name']
        category = ('embedding' if 'embed_tokens' in name else
                    'adapter' if '.lora_' in name else
                    'int8_dense' if tensor['dtype'] == 'int8' else 'other')
        sizes[category] += tensor['bytes']
        match = re.search(r'\.layers\.(\d+)\.', name)
        if match:
            layers[int(match[1])] += tensor['bytes']
    reads = collections.Counter()
    groups = collections.defaultdict(list)
    for query in report['queries']:
        reads[query['op']] += query['ok']['stable_read_bytes']
        if query['op'] not in ('lora_integer', 'mlp_gate_up_integer', 'mlp_add_norm_integer'):
            continue
        request = source / 'queries' / f"{query['index']:06d}.request.bin"
        blob = request.read_bytes()
        if not 36 <= len(blob) <= 2_000_000 or hashlib.sha256(blob[:-32]).digest() != blob[-32:]:
            raise ValueError('Request checksum/size')
        length, = struct.unpack('<I', blob[:4])
        if length > 16384 or 4 + length > len(blob) - 32:
            raise ValueError('Request header bounds')
        header = json.loads(blob[4:4 + length])
        if header['pack_hash'] != manifest['pack_hash'] or len(header['dims']) != 4:
            raise ValueError('Fixed pack or projection shape mismatch')
        tokens, width, cols, start = header['dims']
        # Only compare identical encoded inputs; remove variable request header
        # and checksum. Equal payloads and codec mean identical F32 operands.
        key = json.dumps(dict(op=header['op'], tensor=header['tensor'], tokens=tokens,
            cols=cols, aux=header['aux'], scalars=header['scalars'],
            encoding=header.get('encoding', ''),
            payload_sha256=hashlib.sha256(blob[4 + length:-32]).hexdigest()), sort_keys=True)
        groups[key].append(dict(index=query['index'], row_start=start, row_count=width))
    repeated = [dict(**json.loads(key), tiles=tiles) for key, tiles in groups.items() if len(tiles) > 1]
    result = dict(scope='Fixed-weight read accounting and identical-input row splits; no speed/accuracy claim',
        source=str(source.relative_to(ROOT)), source_report_sha256=hashlib.sha256(raw).hexdigest(),
        model=manifest['model'], pack_hash=manifest['pack_hash'], pack_bytes=manifest['bytes'],
        weight_bytes_by_category=dict(sizes), layer_bytes=dict(sorted(layers.items())),
        stable_read_bytes=sum(reads.values()),
        stable_read_scope='Measured ordinary step calls only; decision_fast does not expose read bytes',
        stable_read_bytes_by_op=dict(reads),
        projection_groups=len(groups), repeated_groups=len(repeated),
        redundant_projection_calls=sum(len(item['tiles']) - 1 for item in repeated), repeated=repeated)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('repeated', 'layer_bytes')}))


if __name__ == '__main__':
    main()
