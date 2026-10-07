#!/usr/bin/env python3
"""Independent symbolic stack proof for the fused first-use I16 weight cache."""
from pathlib import Path
import hashlib,json,re
from build_s1_winograd_fused_probe import weight
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d=ROOT/'artifacts/s1-winograd-fused-v1';old=ROOT/'artifacts/s1-winograd-v1'
    namespaces=dict(__name__='frozen_rank7',__file__=str(old/'plan.py'))
    exec(compile((old/'plan.py').read_text(),str(old/'plan.py'),'exec'),namespaces)
    a,b,c,leaves,roots,_=namespaces['plan']()
    cache={};expressions=[]
    for m in range(7):
        stack=[]
        for line in weight(m,0,0):
            hit=re.fullmatch(r'\(v128.load8x8_s offset=0\(local.get \$bp([0-3])\)\)',line)
            if hit:stack.append({[0,2,1,3][int(hit[1])]:1});continue
            hit=re.fullmatch(r'local\.(get|tee) \$w([0-6])_0_0',line)
            if hit:
                index=int(hit[2])
                if hit[1]=='get':assert index in cache;stack.append(dict(cache[index]))
                else:assert stack;cache[index]=dict(stack[-1])
                continue
            assert line=='i16x8.sub'
            right=stack.pop();left=stack.pop();result=dict(left)
            for k,v in right.items():result[k]=result.get(k,0)-v
            result={k:v for k,v in result.items()if v};stack.append(result)
            assert sum(abs(v)for v in result.values())*128<=512
        assert len(stack)==1 and stack[0]==b.symbols[leaves[m][1]]
        assert cache[m]==stack[0];expressions.append(dict(product=m,coefficients=stack[0],i16_absolute_bound=sum(abs(v)for v in stack[0].values())*128))
    assert sorted(cache)==list(range(7))
    for p in (old/'build/src').glob('*.rs'):
        assert p.read_bytes()==(d/'build/src'/p.name).read_bytes(),p.name
    new=(d/'build/kernel.wat').read_text();before=(old/'build/kernel.wat').read_text()
    assert new.count('i32x4.dot_i16x8_s')==before.count('i32x4.dot_i16x8_s')==14336
    for opcode in ['f32x4.mul','f32x4.add','f32x4.convert_i32x4_s','i8x16.shuffle','v128.store']:
        assert new.count(opcode)==before.count(opcode)
    assert new.count('v128.load8x8_s')==before.count('v128.load8x8_s')==4096
    build=json.loads((d/'build/report.json').read_text())
    assert build['locals']==7460 and all(p['wasmparser_validation']for p in build['patches'])
    for k in ['source_hashes','dependency_hashes']:
        assert all(sha(ROOT/p)==h for p,h in build[k].items())
    entry=json.loads((d/'entry-hashes.json').read_text());assert all(sha(ROOT/p)==h for p,h in entry.items())
    files=[Path(__file__),ROOT/'scripts/build_s1_winograd_fused_probe.py',old/'plan.py',d/'build/report.json',d/'entry-hashes.json',d/'build/kernel.wat']
    result=dict(complete=True,module=build['wasm_sha256'],leaf_weight_coefficients_equal=True,
                first_use_cache_initialized_before_reads=True,rust_sources_byte_equal=True,
                dot_float_store_counts_equal=True,integer_i16_safe=True,expressions=expressions,
                full_inference_verified=False,hashes={str(p.relative_to(ROOT)):sha(p)for p in files})
    (d/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,module=result['module'])))


if __name__=='__main__':main()
