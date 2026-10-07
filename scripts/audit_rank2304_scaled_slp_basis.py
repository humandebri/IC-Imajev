#!/usr/bin/env python3
"""Exact scaled author SLP basis; no per-leaf scales in reconstruction."""
import ast
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def program(path, width, input_scale):
    statements=[]
    symbolic={f'i{i}':[Fraction(input_scale*int(i==j)) for j in range(width)] for i in range(width)}
    def evaluate(node,env):
        if isinstance(node,ast.Name):
            return env[node.id]
        if isinstance(node,ast.Constant):
            assert type(node.value)is int
            return Fraction(node.value)
        if isinstance(node,ast.UnaryOp):
            assert isinstance(node.op,(ast.USub,ast.UAdd))
            x=evaluate(node.operand,env)
            return [-v for v in x] if isinstance(node.op,ast.USub) else x
        assert isinstance(node,ast.BinOp)
        x,y=evaluate(node.left,env),evaluate(node.right,env)
        if isinstance(node.op,(ast.Add,ast.Sub)):
            assert isinstance(x,list) and isinstance(y,list)
            sign=1 if isinstance(node.op,ast.Add) else -1
            return [a+sign*b for a,b in zip(x,y)]
        if isinstance(node.op,ast.Div):
            assert isinstance(y,Fraction) and y!=0 and isinstance(x,list)
            return [a/y for a in x]
        assert isinstance(node.op,ast.Mult)
        if isinstance(y,list):
            x,y=y,x
        assert isinstance(x,list) and isinstance(y,Fraction)
        return [a*y for a in x]
    for line in path.read_text().splitlines():
        if ':=' not in line:
            continue
        name,expression=line.split(';')[0].split(':=')
        name=name.strip()
        node=ast.parse(expression.strip(),mode='eval').body
        value=evaluate(node,symbolic)
        assert all(v.denominator==1 for v in value),(path,name)
        symbolic[name]=value
        statements.append((name,node))
    rows=[symbolic[f'o{i}'] for i in range(48 if width==16 else 16)]
    return statements,np.array(rows,dtype=np.int64)


def run(statements,x,scale,wrap):
    env={f'i{i}':x[i].astype(np.int64)*scale for i in range(len(x))}
    def evaluate(node):
        if isinstance(node,ast.Name):
            return env[node.id]
        if isinstance(node,ast.Constant):
            assert type(node.value)is int
            return node.value
        if isinstance(node,ast.UnaryOp):
            v=evaluate(node.operand)
            return -v if isinstance(node.op,ast.USub) else v
        a,b=evaluate(node.left),evaluate(node.right)
        if isinstance(node.op,ast.Add):
            out=a+b
        elif isinstance(node.op,ast.Sub):
            out=a-b
        elif isinstance(node.op,ast.Mult):
            out=a*b
        else:
            assert isinstance(node.op,ast.Div) and np.isscalar(b) and b!=0
            assert np.all(a%b==0)
            out=a//b
        if isinstance(out,(np.ndarray,np.integer)):
            if wrap:
                out=out.astype('<i4').astype(np.int64)
            else:
                assert out.min()>=-2**31 and out.max()<2**31
        return out
    for name,node in statements:
        env[name]=evaluate(node)
    count=48 if len(x)==16 else 16
    return np.array([env[f'o{i}'] for i in range(count)])


def transform(statements,x,scale):
    # Basis indices: outer row, outer column, inner row, inner column.
    v=x.reshape(4,4,4,4,16).transpose(0,2,1,3,4).reshape(16,16,16)
    inner=np.stack([run(statements,v[o],scale,False) for o in range(16)])
    outer=np.stack([run(statements,inner[:,i],scale,False) for i in range(48)],axis=1)
    return outer.reshape(2304,16)


def main():
    prior=ROOT/'artifacts/rank2304-latest-conditional-lift-v1'
    evidence=json.loads((prior/'report.json').read_text())
    for p,h in evidence['source_hashes'].items():
        assert sha(ROOT/p)==h,p
    primary=ROOT/'artifacts/rank48-latest-programs-v1'
    parser=ROOT/'scripts/audit_rank48_primary_coefficients.py'
    ns={'__name__':'parse','__file__':str(parser)}
    exec(compile(parser.read_text(),str(parser),'exec'),ns)
    programs={}
    matrices={}
    for kind in ['L','R','P']:
        scale=1 if kind=='P' else 2
        programs[kind],matrices[kind]=program(primary/f'4x4x4_48_204_{kind}.slp',48 if kind=='P' else 16,scale)
        expected=np.array([[int(v*scale) for v in row] for row in ns['parse'](primary/f'4x4x4_48_204_{kind}.sms')])
        assert np.array_equal(matrices[kind],expected)
    a,b,c=matrices['L'],matrices['R'],matrices['P']
    A=np.array([np.kron(x.reshape(4,4),y.reshape(4,4)).ravel() for x in a for y in a])
    B=np.array([np.kron(x.reshape(4,4),y.reshape(4,4)).ravel() for x in b for y in b])
    C=np.array([np.kron(c[:,i].reshape(4,4),c[:,j].reshape(4,4)).ravel() for i in range(48) for j in range(48)]).T
    tensor=np.einsum('om,ma,mb->oab',c,a,b,dtype=np.int64)
    expected=np.zeros_like(tensor)
    for t in range(4):
        for n in range(4):
            for k in range(4):
                expected[t*4+n,t*4+k,k*4+n]=4
    assert np.array_equal(tensor,expected)
    cases=[]
    for i in range(12):
        with np.load(prior/f'case-{i}.npz') as z:
            q=z['q'].astype(np.int64)
            w=z['w'].astype(np.int64)
            reference=z['expected'].astype(np.int64)
        qa=q.reshape(256,16)
        wb=w.reshape(16,16,16).transpose(0,2,1).reshape(256,16)
        av=transform(programs['L'],qa,2)
        bv=transform(programs['R'],wb,2)
        assert np.array_equal(av,A@qa) and np.array_equal(bv,B@wb)
        safe=bool(av.min()>=-32768 and av.max()<=32767 and bv.min()>=-32768 and bv.max()<=32767)
        products=np.sum(av*bv,axis=1).astype('<i4').astype(np.int64).reshape(48,48)
        inner=np.stack([run(programs['P'],products[o],1,True) for o in range(48)])
        outer=np.stack([run(programs['P'],inner[:,j],1,True) for j in range(16)],axis=1)
        numerator=outer.reshape(4,4,4,4).transpose(0,2,1,3).reshape(16,16)
        assert np.all(numerator%16==0)
        assert np.array_equal((C@products.ravel()).astype('<i4').reshape(16,16),numerator)
        assert np.array_equal(numerator>>4,reference)
        cases.append(dict(index=i,all256_integer_dots_exact=True,operands_i16_safe=safe,
                          exact_integer_divisions_in_input_slp=True))
    d=ROOT/'artifacts/rank2304-scaled-slp-basis-v1'
    d.mkdir(exist_ok=False)
    np.savez_compressed(d/'coefficients.npz',A=A.astype('<i2'),B=B.astype('<i2'),C=C.astype('<i2'))
    files=[Path(__file__),parser,prior/'report.json',d/'coefficients.npz']+[primary/f'4x4x4_48_204_{kind}.{ext}' for kind in ['L','R','P'] for ext in ['slp','sms']]+[prior/f'case-{i}.npz' for i in range(12)]
    report=dict(complete=True,all4096_base_tensor_identities_exact=True,
                all12_factorized_slp_and_matrix_and_scalar_dots_equal=True,
                reconstruction_leaf_rescalings_required=0,final_divisor=16,
                reconstruction_slp_base_add_sub=90,reconstruction_depth2_add_sub=90*(48+16),
                input_slp_intermediate_coefficients_all_integer_with_input_scale2=True,
                conditional_i16_gate_required=True,cases=cases,
                source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},
                wasm_execution_verified=False,ic_performance_verified=False,
                scope='Exact factorized integer-program proof only. Full inference/paid goal not achieved; a checked I16 gate and unchanged fallback must precede SIMD adoption.')
    (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['cases','source_hashes','scope']}))


if __name__=='__main__':
    main()
