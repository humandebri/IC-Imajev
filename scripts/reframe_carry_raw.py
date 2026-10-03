#!/usr/bin/env python3
"""Trade a small exact byte expansion for fewer Huffman decode loops."""
import argparse,hashlib,json,pathlib,struct,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import decode,frame_digest
from mlp_delta_carry import encode_plane,planar,HUFFMAN_NAME
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',default='artifacts/prefix_codec/carry-huffman-fusion-check');ap.add_argument('--directory',required=True);ap.add_argument('--raw-threshold',type=float,default=.1);a=ap.parse_args();assert 0<=a.raw_threshold<=1
d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);source=ROOT/a.source;cases=[]
for layer in [0,1,3]:
    _,state=decode((source/f'{layer:02d}-prepare-partial.response.bin').read_bytes());total=87;c=total*2560;q=total*9216
    prefix=ROOT/'artifacts/prefix_codec/full-state-layout-proof/prefix/queries/states'/f'layer-{layer+1:02d}.npz'
    with np.load(prefix,allow_pickle=False)as p:conv=p['conv'].ravel().copy();log=p['delta_log'].ravel().copy()
    for label,n in [('fused',87),('short-fused-profile',45)]:
        path=source/f'{layer:02d}-{label}.request.bin';original=path.read_bytes();hl=struct.unpack('<I',original[:4])[0];header=json.loads(original[4:4+hl]);assert header['dims'][:2]==[n,45] and header['encoding']==HUFFMAN_NAME
        values=np.concatenate([state[:n*2560],state[c:c+n*9216],state[c+q:c+q+total*36].reshape(total,36)[:n].ravel(),state[c+q+total*36:].reshape(total,64)[:n].ravel()])
        nc=n*2560;nq=n*9216;groups=[(planar((values[:nc].view('<u4')>>16).astype('<u2').tobytes(),2),2),(values[nc:nc+nq].astype(np.int8).tobytes(),1),(planar(values[nc+nq:].tobytes(),4),4),(planar((conv.view('<u4')>>16).astype('<u2').tobytes(),2),2),(planar((log[:45*2048].view('<u4')>>16).astype('<u2').tobytes(),2),2),(planar(log[45*2048:].astype('<f4').tobytes(),4),4)]
        parts=[];changed=[]
        for group,(raw,width)in enumerate(groups):
            count=len(raw)//width
            for plane in range(width):
                lane=raw[plane*count:(plane+1)*count];packed=encode_plane(lane)
                if packed[0]==1 and len(packed)>(1-a.raw_threshold)*len(lane):changed.append([group,plane]);packed=b'\0'+struct.pack('<I',len(lane))+lane
                parts.append(packed)
        h=json.dumps(header,separators=(',',':'),allow_nan=False).encode();body=struct.pack('<I',len(h))+h+b'\1'+b''.join(parts);frame=body+frame_digest(header,body);assert len(frame)<=2000000
        (d/f'{layer:02d}-{label}.request.bin').write_bytes(frame);cases.append(dict(layer=layer,tokens=n,changed_planes=changed,old_frame_bytes=len(original),frame_bytes=len(frame),frame_sha256=hashlib.sha256(frame).hexdigest(),source_frame_sha256=hashlib.sha256(original).hexdigest()))
report=dict(scope=__doc__,raw_threshold=a.raw_threshold,script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),cases=cases);(d/'reframe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
