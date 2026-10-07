#!/usr/bin/env python3
"""Unroll ascending-K score accumulation after checking complete operand bounds."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]


def main():
    helper=ROOT/'scripts/attention_key_lanes_bounded.rs'
    assert not helper.exists()
    old=ROOT/'scripts/attention_key_lanes.rs';text=old.read_text();edits=[]
    def change(before,after):
        nonlocal text
        assert text.count(before)==1
        text=text.replace(before,after);edits.append(dict(before=before,after=after))
    change('assert!(width>0 && q.len()==n*width && k.len()==(n+prefix)*width);',
           'assert!(n>0 && n<=132 && prefix<=132-n && width>0 && width<=256 && q.len()==n*width && k.len()==(n+prefix)*width);')
    change('transposed[(key/4*width+j)*4+key%4]=k[key*width+j];',
           '*transposed.get_unchecked_mut((key/4*width+j)*4+key%4)=*k.get_unchecked(key*width+j);')
    before='''            for j in 0..width {
                let a=f32x4_splat(q[t*width+j]);
                let b=v128_load(transposed.as_ptr().add((block*width+j)*4).cast());
                sums=f32x4_add(sums,f32x4_mul(a,b));
            }'''
    after='''            let qp=q.as_ptr().add(t*width);
            let kp=transposed.as_ptr().add(block*width*4);
            let mut j=0;
            while j+8<=width {
'''
    for i in range(8):
        after+=f'                sums=f32x4_add(sums,f32x4_mul(f32x4_splat(*qp.add(j+{i})),v128_load(kp.add((j+{i})*4).cast())));\n'
    after+='''                j+=8;
            }
            while j<width {
                sums=f32x4_add(sums,f32x4_mul(f32x4_splat(*qp.add(j)),v128_load(kp.add(j*4).cast())));
                j+=1;
            }'''
    change(before,after);helper.write_text(text)
    d=ROOT/'artifacts/attention-key-lanes-bounded-v1';d.mkdir(exist_ok=False)
    (d/'helper-edits.json').write_text(json.dumps(edits,indent=2)+'\n')
    p=ROOT/'scripts/build_attention_key_lanes_probe.py'
    source=p.read_text().replace('artifacts/attention-key-lanes-v1','artifacts/attention-key-lanes-bounded-v1').replace('scripts/attention_key_lanes.rs','scripts/attention_key_lanes_bounded.rs')
    (d/'frozen-builder.py').write_text(source)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    files=[Path(__file__),old,helper,p,d/'frozen-builder.py',d/'helper-edits.json']
    (d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
