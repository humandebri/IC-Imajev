"""Opaque exact prefix packets; host only frames BF16 inputs and copies bytes."""
import hashlib
import json
import pathlib
import struct
import numpy as np

NAME = 'delta-hybrid-prefix-exact-v1'


def encode_request(header, values, packet):
    from transport import frame_digest
    dims = header.get('dims', [])
    if header.get('op') != 'delta_full_hybrid_integer' or header.get('encoding') != NAME or len(dims) != 4:
        raise ValueError('hybrid Delta metadata')
    n, heads, prefix, keep = dims
    if not 1 <= n <= 90 or heads != 32 or not 1 <= prefix <= 132 or keep != 0 or header.get('aux') or header.get('scalars'):
        raise ValueError('hybrid Delta bounds')
    v = np.asarray(values, dtype='<f4').ravel()
    bits = v.view('<u4')
    if v.size != n * 2560 + 3 * 8192 or not np.isfinite(v).all() or np.any(bits & 65535):
        raise ValueError('hybrid Delta BF16 input')
    if not 12 <= len(packet) <= 1_990_000 or packet[:4] != b'NPF1' or struct.unpack('<I', packet[4:8])[0] != prefix:
        raise ValueError('hybrid Delta packet identity')
    h = json.dumps(header, separators=(',', ':'), allow_nan=False).encode()
    if len(h) > 16384:
        raise ValueError('header size')
    payload = b'\x01' + struct.pack('<I', v.size) + (bits >> 16).astype('<u2').tobytes() + struct.pack('<I', len(packet)) + packet
    body = struct.pack('<I', len(h)) + h + payload
    if len(body) + 32 > 2_000_000:
        raise ValueError('hybrid Delta frame size')
    return body + frame_digest(header, body)


def load_packets(directory, cache):
    directory = pathlib.Path(directory)
    manifest = json.loads((directory / 'cache.json').read_bytes())
    identity = manifest['identity']
    metadata = cache['metadata']
    for packet_key, cache_key in [('model', 'model'), ('pack_hash', 'pack_hash'), ('source_module', 'wasm_sha256')]:
        if identity[packet_key] != metadata[cache_key]:
            raise ValueError('hybrid/primary prefix identity mismatch')
    if identity['input_hash'] != hashlib.sha256(json.dumps(metadata['token_ids']).encode()).hexdigest():
        raise ValueError('hybrid prefix token input hash')
    layers = {str(i) for i in range(32) if (i + 1) % 4}
    if identity['version'] != 1 or identity['tokens'] != len(metadata['token_ids']) or set(manifest['packets']) != layers:
        raise ValueError('hybrid prefix layer/token identity')
    packets = {}
    for layer in layers:
        # Bind to the exact primary canister-produced state file, not only tokens.
        if identity['layers'][layer] != metadata['files'][f'states/layer-{int(layer):02d}.npz']:
            raise ValueError('hybrid prefix source state mismatch')
        entry = manifest['packets'][layer]
        packet = (directory / f'layer-{int(layer):02d}.npf1').read_bytes()
        if len(packet) != entry['bytes'] or hashlib.sha256(packet).hexdigest() != entry['sha256']:
            raise ValueError('hybrid prefix packet hash')
        if packet[:4] != b'NPF1' or len(packet) < 12 or struct.unpack('<I', packet[4:8])[0] != identity['tokens']:
            raise ValueError('hybrid prefix packet tokens')
        packets[int(layer)] = packet
    return packets
