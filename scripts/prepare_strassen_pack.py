#!/usr/bin/env python3
"""Prepare a partial benchmark pack: exact Strassen weights stay signed INT8.

Overflow is represented by sparse signed INT8 +/-256 corrections. This only
writes a new pack; it does not modify original weights or deploy a canister.
"""
import argparse, hashlib, json, pathlib, struct
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def transform_weights(w):
    if w.dtype!=np.int8 or w.ndim!=2 or not w.shape[0] or w.shape[0]%8 or not w.shape[1] or w.shape[1]%256:raise ValueError('Strassen INT8 matrix shape')
    rows,cols=w.shape;p=w.astype(np.int16).reshape(rows//2,2,cols//256,256)
    b11,b21,b12,b22=p[:,0,:,:128],p[:,0,:,128:],p[:,1,:,:128],p[:,1,:,128:]
    values=np.stack([b11+b22,b11,b12-b22,b21-b11,b22,b11+b12,b21+b22])
    packed=values.astype(np.int8);flags=((values-packed.astype(np.int16))//256).astype(np.int8)
    assert np.array_equal(values,packed.astype(np.int16)+flags.astype(np.int16)*256)
    flat=flags.reshape(-1,128);record,col=np.nonzero(flat);counts=np.bincount(record,minlength=len(flat));offsets=np.concatenate([np.zeros(1,np.uint32),np.cumsum(counts,dtype=np.uint32)])
    entries=np.empty((len(col),2),np.uint8);entries[:,0]=col;entries[:,1]=flat[record,col].view(np.uint8)
    csr=struct.pack('<I',len(flat))+offsets.astype('<u4').tobytes()+entries.tobytes()
    return packed,csr
def decode_corrections(blob,shape):
    if len(shape)!=4 or shape[0]!=7 or shape[-1]!=128 or len(blob)<4:raise ValueError('correction shape')
    records,=struct.unpack('<I',blob[:4]);expected=int(np.prod(shape[:-1]));header=4+4*(records+1)
    if records!=expected or len(blob)<header or (len(blob)-header)%2:raise ValueError('correction length')
    offsets=np.frombuffer(blob[4:header],dtype='<u4');entries=np.frombuffer(blob[header:],dtype=np.uint8).reshape(-1,2)
    if offsets[0]!=0 or offsets[-1]!=len(entries) or np.any(offsets[1:]<offsets[:-1]) or np.any(entries[:,0]>=128) or np.any(~np.isin(entries[:,1],[1,255])):raise ValueError('correction bounds')
    flags=np.zeros((records,128),np.int8)
    for r in np.flatnonzero(offsets[1:]!=offsets[:-1]):
        e=entries[offsets[r]:offsets[r+1]]
        if np.any(e[1:,0]<=e[:-1,0]):raise ValueError('correction column order')
        flags[r,e[:,0]]=e[:,1].view(np.int8)
    return flags.reshape(shape)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tensor',default='model.language_model.layers.3.self_attn.q_proj.weight');ap.add_argument('--output',default='artifacts/strassen-prepacked');a=ap.parse_args()
    original=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());t=next(v for v in original['tensors'] if v['name']==a.tensor)
    if t['dtype']!='int8' or t['rows']*t['cols']>30_000_000:raise ValueError('source tensor')
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'pack.bin').exists():raise ValueError('use a fresh output directory')
    w=np.memmap(ROOT/'checkpoints/full-int8.pack',mode='r',dtype=np.int8,offset=t['offset'],shape=(t['rows'],t['cols']));scales=np.memmap(ROOT/'checkpoints/full-int8.pack',mode='r',dtype='<f4',offset=t['offset']+w.size,shape=(t['rows'],))
    if not np.isfinite(scales).all() or not np.all(scales>0):raise ValueError('weight scales')
    derived,csr=transform_weights(w);decode_corrections(csr,derived.shape)
    data=bytearray();tensors=[]
    def append(name,rows,cols,dtype,b):
        while len(data)%256:data.append(0)
        tensors.append(dict(name=name,rows=rows,cols=cols,dtype=dtype,offset=len(data),bytes=len(b)));data.extend(b)
    append(t['name'],t['rows'],t['cols'],'int8',w.tobytes()+scales.tobytes())
    append(t['name']+'.strassen_i8',t['rows']//2,t['cols']//2,'strassen-i8-v1',derived.tobytes())
    append(t['name']+'.strassen_corrections',t['rows']//2,t['cols']//2,'strassen-csr-v1',csr)
    manifest=dict(version=1,model=original['model'],pack_hash=hashlib.sha256(data).hexdigest(),bytes=len(data),tensors=tensors,scope='Partial real-weight matrix-kernel benchmark; not a full model',source_pack_hash=original['pack_hash'],source_tensor=t['name'],preparation_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    (out/'pack.bin').write_bytes(data);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(bytes=len(data),derived_bytes=derived.nbytes,correction_bytes=len(csr),overflow_entries=(len(csr)-4-4*(7*(t['rows']//2)*(t['cols']//256)+1))//2,pack_hash=manifest['pack_hash'])))
if __name__=='__main__':main()
