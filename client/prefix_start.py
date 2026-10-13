"""Exact IDs/conv/prefix request and two independently bounded reply blocks."""
import json
import struct
import numpy as np
from projection_codec import block_unpack

NAME = 'prefix-start-exact-v1'


def shape(h):
    d = h.get('dims', [])
    if (h.get('encoding') != NAME or h.get('op') != 'prefix_start_integer'
            or h.get('tensor') != 'model.language_model.embed_tokens.weight'
            or len(d) != 2 or any(type(v) is not int for v in d)
            or h.get('aux') or h.get('scalars')):
        raise ValueError('prefix start metadata')
    n, p = d
    if not 1 <= n <= 89 or not 1 <= p <= 132 or n+p > 512:
        raise ValueError('prefix start shape')
    return n, p


def encode_request(h, values, packet):
    from transport import frame_digest
    n, p = shape(h)
    v = np.asarray(values, dtype='<f4').ravel()
    if len(v) != n+3*8192 or not np.isfinite(v).all():
        raise ValueError('prefix start values')
    ids, conv = v[:n], v[n:]
    if np.any(ids < 0) or np.any(ids > 16777216) or np.any(ids != np.trunc(ids)):
        raise ValueError('prefix start token range')
    if np.any(conv.view('<u4') & 65535):
        raise ValueError('prefix start BF16 conv')
    if not 12 <= len(packet) <= 1990000 or packet[:4] != b'NPF1' or struct.unpack('<I', packet[4:8])[0] != p:
        raise ValueError('prefix start packet identity')
    header = json.dumps(h, separators=(',', ':'), allow_nan=False).encode()
    if len(header) > 16384:
        raise ValueError('prefix start header size')
    payload = (b'\1'+struct.pack('<I', n)+ids.astype('<u4').tobytes()
               +(conv.view('<u4') >> 16).astype('<u2').tobytes()
               +struct.pack('<I', len(packet))+packet)
    body = struct.pack('<I', len(header))+header+payload
    if len(body)+32 > 2000000:
        raise ValueError('prefix start state size')
    return body+frame_digest(h, body)


def decode_reply(h, payload):
    n, _ = shape(h)
    if payload[:1] != b'\0' or len(payload) < 5:
        raise ValueError('prefix start reply direction')
    first, = struct.unpack('<I', payload[1:5])
    if not 4 <= first <= len(payload)-5:
        raise ValueError('prefix start reply length')
    a, b = block_unpack(payload[5:5+first]), block_unpack(payload[5+first:])
    if len(a) != n*2560 or len(b) != n*2560+3*8192 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('prefix start reply count/finite')
    return np.concatenate([a, b])
