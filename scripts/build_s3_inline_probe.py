#!/usr/bin/env python3
"""Inline single-use integer reconstruction DAG nodes into the Wasm operand stack."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s3-inline-v1';d.mkdir(exist_ok=False)
 gen=ROOT/'scripts/generate_s3_stream_probe.py';source=gen.read_text().replace("default='artifacts/s3-stream-v1/build'","default='artifacts/s3-inline-v1/generated'")
 new='''def reconstruct(c,result):
    roots=[v for row in result for v in row]
    uses=Counter(roots)
    for _,terms in c.nodes:uses.update(terms.keys())
    terms_by_name=dict(c.nodes)
    keep={n for n,_ in c.nodes if uses[n]>1 or n in set(roots)}
    mapping={f'p{i}':i for i in range(343)}
    mapping.update({n:343+i for i,(n,_) in enumerate((n,t) for n,t in c.nodes if n in keep)})
    def emit(name):
        if name in mapping:return [f'local.get $pc{mapping[name]}']
        out=[]
        for i,(parent,sign) in enumerate(terms_by_name[name].items()):
            if i==0 and sign<0:out.append('(v128.const i32x4 0 0 0 0)')
            out+=emit(parent)
            if i or sign<0:out.append('i32x4.'+('add' if sign>0 else 'sub'))
        return out
    code=[]
    for name,terms in c.nodes:
        if name not in keep:continue
        for i,(parent,sign) in enumerate(terms.items()):
            if i==0 and sign<0:code.append('(v128.const i32x4 0 0 0 0)')
            code+=emit(parent)
            if i or sign<0:code.append('i32x4.'+('add' if sign>0 else 'sub'))
        code.append(f'local.set $pc{mapping[name]}')
    assert all(r in mapping for r in roots)
    return code,[[mapping[r] for r in row] for row in result],len(mapping)

'''
 anchor='def main():';assert source.count(anchor)==1;source=source.replace(anchor,new+anchor)
 (d/'frozen-generator.py').write_text(source)
 exec(compile(source,str(gen),'exec'),dict(__file__=__file__,__name__='__main__'))
 p=ROOT/'scripts/build_s3_prepared_probe.py';source=p.read_text().replace("'artifacts/s3-prepared-v1/build'","'artifacts/s3-inline-v1/build'")
 before="wat = (original / 'kernel.wat').read_text()";assert source.count(before)==1;source=source.replace(before,"wat = (ROOT/'artifacts/s3-inline-v1/generated/kernel.wat').read_text()")
 (d/'frozen-builder.py').write_text(source)
 (d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),gen,p,d/'frozen-builder.py',d/'frozen-generator.py',d/'generated/kernel.wat',d/'generated/identity.json'] if p.exists()},indent=2)+'\n')
 exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
