#!/usr/bin/env python3
"""Host-only exact Delta-state codecs; no Wasm decoder or query-count claim."""
import argparse
import hashlib
import json
import lzma
import pathlib
import zlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',default='artifacts/activation-init-v1-prefix/queries')
    ap.add_argument('--output',default='artifacts/column16-explicit/state-codecs.json')
    a=ap.parse_args();source=ROOT/a.source;cases=[]
    for layer in [0,10,30]:
        path=source/f'states/layer-{layer:02d}.npz'
        with np.load(path,allow_pickle=False)as f:bits=f['delta'].view('<u4').copy()
        assert bits.shape==(32,128,128)
        raw=bits.tobytes();results=[]
        for transform in ['raw','byte-plane','xor-column','xor-row','bit-plane']:
            value=bits.copy()
            if transform=='xor-column':value[:,:,1:]=bits[:,:,1:]^bits[:,:,:-1]
            if transform=='xor-row':value[:,1:,:]=bits[:,1:,:]^bits[:,:-1,:]
            payload=value.tobytes()
            if transform in ['byte-plane','xor-column','xor-row']:
                payload=np.frombuffer(payload,np.uint8).reshape(-1,4).T.copy().tobytes()
            if transform=='bit-plane':
                lanes=np.unpackbits(np.frombuffer(payload,np.uint8).reshape(-1,4),axis=1)
                payload=np.packbits(lanes.T.copy().ravel()).tobytes()
            for codec in ['zlib1','zlib9','lzma3']:
                packed=lzma.compress(payload,preset=3) if codec=='lzma3' else zlib.compress(payload,int(codec[-1]))
                restored=lzma.decompress(packed) if codec=='lzma3' else zlib.decompress(packed)
                if transform=='bit-plane':
                    restored=np.packbits(np.unpackbits(np.frombuffer(restored,np.uint8)).reshape(32,-1).T.copy(),axis=1).tobytes()
                elif transform!='raw':restored=np.frombuffer(restored,np.uint8).reshape(4,-1).T.copy().tobytes()
                recovered=np.frombuffer(restored,'<u4').reshape(bits.shape).copy()
                if transform=='xor-column':recovered=np.bitwise_xor.accumulate(recovered,axis=2)
                if transform=='xor-row':recovered=np.bitwise_xor.accumulate(recovered,axis=1)
                assert recovered.tobytes()==raw
                # 87-token hidden + full32-head conv history in exact BF16.
                # This is only an optimistic sizing envelope, not an implemented frame.
                envelope=len(packed)+87*2560*2+3*8192*2+1024
                results.append(dict(transform=transform,codec=codec,bytes=len(packed),bitwise_roundtrip=True,
                                    optimistic_capture_bytes=envelope,under_2M=envelope<=2_000_000))
        row=dict(layer=layer,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_bytes=len(raw),results=results)
        cases.append(row);print(json.dumps(dict(layer=layer,best=min(results,key=lambda r:r['bytes']))),flush=True)
    report=dict(scope='Three real prefix Delta states, host lossless codec sizes only; no canister decoder, instruction/speed/accuracy/query-count improvement claim. Envelope excludes capture reply and prepared reuse state.',source=a.source,
                script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),cases=cases)
    out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
