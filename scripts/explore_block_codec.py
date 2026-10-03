#!/usr/bin/env python3
"""Lossless block-marker wire prototype; no canister or graph integration."""
import argparse
import hashlib
import json
import pathlib
import struct
import sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'client'))
from transport import decode


def encode_candidate(header,values,block=256):
    v=np.asarray(values,dtype='<f4').ravel();bits=v.view('<u4')
    if v.size>1000000 or not np.isfinite(v).all():raise ValueError('prototype bounds')
    blocks=(v.size+block-1)//block
    padded=np.zeros(blocks*block,dtype=bool);padded[:v.size]=(bits&65535)!=0
    flags=padded.reshape(blocks,block).any(axis=1)
    full=np.repeat(flags,block)[:v.size]
    pos=np.arange(v.size)+np.cumsum(full,dtype=np.int64)-full
    words=np.empty(v.size+int(full.sum()),dtype='<u2')
    words[pos]=(bits>>16).astype('<u2')
    words[pos[full]]=(bits[full]&65535).astype('<u2');words[pos[full]+1]=(bits[full]>>16).astype('<u2')
    h=json.dumps(dict(header,encoding='prototype-bf16-block-exact',block=block),separators=(',',':'),allow_nan=False).encode()
    b=struct.pack('<I',len(h))+h+struct.pack('<I',v.size)+np.packbits(flags,bitorder='little').tobytes()+words.tobytes()
    return b+hashlib.sha256(b).digest()


def decode_candidate(b):
    if len(b)<40 or hashlib.sha256(b[:-32]).digest()!=b[-32:]:raise ValueError('prototype checksum')
    length,=struct.unpack('<I',b[:4]);h=json.loads(b[4:4+length]);payload=b[4+length:-32]
    count,=struct.unpack('<I',payload[:4]);block=h['block'];blocks=(count+block-1)//block;size=(blocks+7)//8
    if count>1000000 or block not in (16,32,64,128,256):raise ValueError('prototype shape')
    raw=np.frombuffer(payload[4:4+size],dtype=np.uint8)
    if blocks%8 and int(raw[-1])>>(blocks%8):raise ValueError('prototype bitmap padding')
    flags=np.unpackbits(raw,bitorder='little')[:blocks].astype(bool);full=np.repeat(flags,block)[:count]
    if len(payload)!=4+size+2*(count+int(full.sum())):raise ValueError('prototype length')
    words=np.frombuffer(payload[4+size:],dtype='<u2');pos=np.arange(count)+np.cumsum(full,dtype=np.int64)-full
    bits=words[pos+full].astype('<u4')<<16;bits[full]|=words[pos[full]]
    padded=np.zeros(blocks*block,dtype=bool);padded[:count]=(bits&65535)!=0
    if not np.array_equal(flags,padded.reshape(blocks,block).any(axis=1)):raise ValueError('prototype noncanonical')
    out=bits.view('<f4')
    if not np.isfinite(out).all():raise ValueError('prototype nonfinite')
    return h,out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',default='artifacts/mlp-norm-v1-617');ap.add_argument('--output',default='artifacts/mlp-norm/block-codec-screen.json');a=ap.parse_args()
    source=ROOT/a.source;r=json.loads((source/'report.json').read_text());cases=[];seen=set();delta=None;hidden=None
    for q in r['queries']:
        p=source/'queries'/f"{q['index']:06d}";request=p.with_suffix('.request.bin').read_bytes();h,x=decode(request)
        if q['op']=='rms_bf16' and hidden is None:hidden=x
        if q['op']=='delta_input_integer' and delta is None:
            _,values=decode(p.with_suffix('.response.bin').read_bytes());delta=(h,values)
        if q['op'] in seen:continue
        seen.add(q['op'])
        for suffix in ['request','response']:
            original=p.with_suffix(f'.{suffix}.bin').read_bytes();header,values=decode(original)
            for block in [16,32,64,128,256]:
                candidate=encode_candidate(header,values,block);_,got=decode_candidate(candidate)
                assert np.array_equal(got.view(np.uint32),values.view(np.uint32))
                cases.append(dict(op=q['op'],direction=suffix,block=block,original_bytes=len(original),candidate_bytes=len(candidate),bitwise_equal=True,fits_2mb=len(candidate)<=2000000))
    h,values=delta;combined=np.concatenate([values,hidden])
    candidate=encode_candidate(h,combined);_,got=decode_candidate(candidate)
    assert np.array_equal(got.view(np.uint32),combined.view(np.uint32))
    header_size=len(json.dumps(h,separators=(',',':')).encode())
    old_bytes=4+header_size+4+(combined.size+7)//8+2*combined.size+2*int(((combined.view(np.uint32)&65535)!=0).sum())+32
    result=dict(scope='Prototype wire only; no deployed codec or fused QKV operation',source=a.source,cases=cases,combined_qkv_gates_hidden=dict(floats=combined.size,legacy_estimated_bytes=old_bytes,block_candidate_bytes=len(candidate),fits_2mb=len(candidate)<=2000000,bitwise_equal=True))
    dest=ROOT/a.output;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['combined_qkv_gates_hidden']))


if __name__=='__main__':main()
