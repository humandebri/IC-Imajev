#!/usr/bin/env python3
"""Export a separate experimental INT8 pack with fixed LoRA baked into weights.

Merge original BF16 projections, not already quantized projections. Copy all
unaffected tensors from the adopted INT8 pack, including embedding and readout.
Requires MLX/Metal only during preparation. Never overwrites an existing pack.
"""
import argparse
import hashlib
import json
import pathlib
import time

import mlx.core as mx
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='artifacts/merged-adapter-v2/model')
    args = ap.parse_args()
    prefix = ROOT / args.output
    destinations = [prefix.with_suffix(s) for s in ('.pack', '.manifest.json', '.provenance.json', '.pack.part')]
    if any(p.exists() for p in destinations):
        raise ValueError('Refusing to overwrite existing output')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_paths = [ROOT / 'checkpoints' / f'{p}.pack' for p in ('full', 'full-int8')]
    source_manifests = [json.loads(p.with_suffix('.manifest.json').read_text()) for p in source_paths]
    for path, manifest in zip(source_paths, source_manifests):
        if path.stat().st_size != manifest['bytes'] or sha(path) != manifest['pack_hash']:
            raise ValueError(f'Source pack identity: {path}')
    original, adopted = source_manifests
    if original['model'] != adopted['model'] or adopted['model'] != sha(ROOT / 'MODEL_LOCK.json'):
        raise ValueError('Source model identity')
    config = json.loads((ROOT / 'checkpoints/adapter/adapter_config.json').read_text())
    if config['peft_type'] != 'LORA' or config['r'] != 64 or config['lora_alpha'] != 128 or config['rank_pattern'] or config['alpha_pattern'] or config['use_dora'] or config['use_rslora']:
        raise ValueError('Only the locked rank64/scale2 LoRA is supported')
    scale = np.float32(config['lora_alpha'] / config['r'])
    original_tensors = {t['name']: t for t in original['tensors']}
    adopted_tensors = {t['name']: t for t in adopted['tensors']}
    pairs = {name.removesuffix('.lora_A.weight'): t for name, t in adopted_tensors.items() if name.endswith('.lora_A.weight')}
    if len(pairs) != 200 or sum('.lora_B.weight' in name for name in adopted_tensors) != len(pairs):
        raise ValueError('Expected exactly 200 adapter pairs')
    source = np.memmap(source_paths[0], mode='r', dtype=np.uint8)
    pinned = np.memmap(source_paths[1], mode='r', dtype=np.uint8)

    def array(mapping, t):
        data = mapping[t['offset']:t['offset'] + t['bytes']]
        if t['dtype'] == 'bf16':
            result = (data.view('<u2').astype(np.uint32) << 16).view(np.float32)
        elif t['dtype'] == 'f32':
            result = data.view('<f4')
        else:
            raise ValueError('Merge source precision')
        return mx.array(result.reshape(t['rows'], t['cols']))

    digest = hashlib.sha256()
    tensors, merged = [], []
    offset = 0
    base_macs, lora_macs = 0, 0
    with destinations[3].open('xb') as out:
        def write(data):
            nonlocal offset
            out.write(data)
            digest.update(data)
            offset += len(data)

        for t in adopted['tensors']:
            name = t['name']
            if '.lora_A.weight' in name or '.lora_B.weight' in name:
                continue
            start = offset
            key = name.removesuffix('.weight')
            if key in pairs:
                a, b = pairs[key], adopted_tensors[key + '.lora_B.weight']
                w = original_tensors[name]
                if t['dtype'] != 'int8' or (w['rows'], w['cols']) != (t['rows'], t['cols']) or (a['rows'], a['cols'], b['rows'], b['cols']) != (64, t['cols'], t['rows'], 64):
                    raise ValueError(f'Merge shape: {name}')
                weights = array(source, w) + float(scale) * (array(pinned, b) @ array(pinned, a))
                peak = mx.max(mx.abs(weights), axis=1)
                sw = mx.where(peak == 0, mx.ones_like(peak), mx.maximum(peak / 127., mx.array(np.nextafter(np.float32(0), np.float32(1)))))
                qw = mx.clip(mx.round(weights / sw[:, None]), -127, 127).astype(mx.int8)
                mx.eval(qw, sw)
                scales = np.asarray(sw, dtype='<f4')
                if not np.isfinite(scales).all() or not (scales > 0).all():
                    raise ValueError(f'Merged scale: {name}')
                write(np.asarray(qw).tobytes())
                write(scales.tobytes())
                merged.append(name)
                lora_macs += 64 * (t['rows'] + t['cols'])
                del weights, peak, sw, qw
            else:
                for start_byte in range(t['offset'], t['offset'] + t['bytes'], 1024 * 1024):
                    write(pinned[start_byte:min(start_byte + 1024 * 1024, t['offset'] + t['bytes'])].tobytes())
            if '.layers.' in name and t['dtype'] == 'int8' and name.endswith('.weight') and t['cols'] % 256 == 0:
                base_macs += t['rows'] * t['cols']
            tensors.append({**t, 'offset': start, 'bytes': offset - start})
            if len(merged) and len(merged) % 25 == 0 and key in pairs:
                print(f'Merged {len(merged)}/200 projections', flush=True)
    if len(merged) != len(pairs):
        raise ValueError('Incomplete adapter merge')
    manifest = dict(version=1, model=adopted['model'], pack_hash=digest.hexdigest(), bytes=offset, tensors=tensors)
    if sha(destinations[3]) != manifest['pack_hash']:
        raise ValueError('Written pack hash mismatch')
    destinations[3].replace(destinations[0])
    destinations[1].write_text(json.dumps(manifest, indent=2) + '\n')
    report = dict(scope='Experimental full text pack; original BF16 + F32 LoRA merged, then per-row INT8. Changed rounding; not bit-equivalent to adopted graph.',
                  model=adopted['model'], source_pack_hashes=[m['pack_hash'] for m in source_manifests], pack_hash=manifest['pack_hash'],
                  script_sha256=sha(pathlib.Path(__file__)), merged_projections=merged, adapter_scale=float(scale),
                  preparation_seconds=time.perf_counter()-started, source_bytes=adopted['bytes'], merged_bytes=offset,
                  saved_bytes=adopted['bytes']-offset, dense_base_macs_per_token=base_macs,
                  removed_lora_macs_per_token=lora_macs, dense_projection_mac_reduction=lora_macs/(base_macs+lora_macs),
                  quantization='F32 merge; symmetric per-output-row INT8; RNE [-127,127]; scale=max(abs(row))/127',
                  unchanged='All tensors without LoRA copied byte for byte from source INT8 pack; embedding/norm/readout retained')
    destinations[2].write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'merged_projections'}), flush=True)


if __name__ == '__main__':
    main()
