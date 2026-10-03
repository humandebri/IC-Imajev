#!/usr/bin/env python3
"""Prototype exact INT8 state serialization, without claiming Wasm savings."""
import hashlib
import json
import pathlib
import struct
import sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'client'))
from transport import decode


def pack(n,cols,rank,state):
    q_len=n*cols; sx_len=n*(cols//256)
    state=np.asarray(state,dtype='<f4').ravel()
    if not 1<=n<=512 or not 0<cols<=262144 or cols%256 or not 1<=rank<=256 or len(state)!=q_len+sx_len+n*rank or len(state)>900000:
        raise ValueError('state shape')
    q=state[:q_len];tail=state[q_len:]
    if not np.isfinite(state).all() or np.any(q<-127) or np.any(q>127) or np.any(q!=np.trunc(q)) or np.any(tail[:sx_len]<=0):
        raise ValueError('state values')
    raw=struct.pack('<4sIII',b'PRX1',n,cols,rank)+q.astype(np.int8).tobytes()+tail.tobytes()
    if len(raw)+32>2000000:raise ValueError('state bytes')
    return raw+hashlib.sha256(raw).digest()


def unpack(raw):
    if not 48<=len(raw)<=2000000 or hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:raise ValueError('state checksum')
    magic,n,cols,rank=struct.unpack('<4sIII',raw[:16])
    if magic!=b'PRX1' or not 1<=n<=512 or not 0<cols<=262144 or cols%256 or not 1<=rank<=256:raise ValueError('state shape')
    count=n*cols+n*(cols//256)+n*rank
    if count>900000 or len(raw)!=48+n*cols+4*n*(cols//256+rank):raise ValueError('state length')
    q=np.frombuffer(raw,dtype=np.int8,count=n*cols,offset=16).astype('<f4')
    tail=np.frombuffer(raw,dtype='<f4',count=n*(cols//256+rank),offset=16+n*cols)
    state=np.concatenate([q,tail])
    # Reuse the canonical value checks; reject the reserved -128 code too.
    if pack(n,cols,rank,state)!=raw:raise ValueError('state canonical values')
    return state


def main():
    source=ROOT/'artifacts/projection-reuse/simd-probe'
    dest=ROOT/'artifacts/projection-reuse/compact-prototype'
    dest.mkdir(parents=True,exist_ok=True)
    m=json.loads((ROOT/'artifacts/projection-reuse/pack/manifest.json').read_text())
    rank=m['tensors'][1]['rows'];cases=[]
    for n in [1,7,8,32,64,87,132]:
        p=source/f'warm-{n}-reuse.request.bin'
        h,state=decode(p.read_bytes());cols=h['dims'][2]
        raw=pack(n,cols,rank,state);restored=unpack(raw)
        assert np.array_equal(state.view(np.uint32),restored.view(np.uint32))
        for bad in [-128.,128.,0.5,float('nan')]:
            changed=state.copy();changed[0]=bad
            try:pack(n,cols,rank,changed)
            except ValueError:pass
            else:raise AssertionError('malformed integer accepted')
        for changed in [raw[:-1],raw[:-32]+bytes(32)]:
            try:unpack(changed)
            except ValueError:pass
            else:raise AssertionError('malformed frame accepted')
        (dest/f'{n}.bin').write_bytes(raw)
        cases.append(dict(tokens=n,original_frame_bytes=p.stat().st_size,compact_state_bytes=len(raw),
            bitwise_equal=True,source_request_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    result=dict(cases=cases,scope='Python state codec prototype only; lacks model/tensor binding and outer operation framing; not deployed or Wasm/instruction/full-graph validated')
    (dest/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
