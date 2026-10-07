#!/usr/bin/env python3
"""Exact integer basis identity, bounded intermediates and executable SIMD files."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    ns={'__name__':'audit','__file__':str(p)}
    exec(compile(p.read_text(),str(p),'exec'),ns)
    return ns
def main():
    source=ROOT/'scripts/plan_rank343_integer_basis.py';ns=load(source)
    a,b,c,leaves,roots,bound=ns['plan']()
    oa,ob,oc,ol,orr,_=ns['ns']['plan']()
    assert all(a.symbols[an]==oa.symbols[oan] and b.symbols[bn]==ob.symbols[obn]
               for (an,bn),(oan,obn) in zip(leaves,ol))
    for r in range(8):
        for n in range(8):
            expr={}
            for m,sign in c.symbols[roots[r][n]].items():
                for ai,av in a.symbols[leaves[m][0]].items():
                    for bi,bv in b.symbols[leaves[m][1]].items():
                        expr[ai,bi]=expr.get((ai,bi),0)+sign*av*bv
            assert {k:v for k,v in expr.items() if v}=={(r*8+k,k*8+n):1 for k in range(8)}
            assert c.symbols[roots[r][n]]==oc.symbols[orr[r][n]]
    ab=max(sum(abs(v) for v in e.values()) for e in a.symbols.values())*127
    bb=max(sum(abs(v) for v in e.values()) for e in b.symbols.values())*128
    assert ab<32768 and bb<32768
    rng=np.random.default_rng(343654);cases=[]
    def eval_dag(d,base,wrap):
        vals={f'{d.prefix}{i}':v for i,v in enumerate(base)};vals['zero']=np.zeros_like(base[0])
        for name,terms in d.nodes:
            v=sum(sign*vals[parent].astype(np.int64) for parent,sign in terms.items())
            vals[name]=v.astype('<i4') if wrap else v
        return vals
    for case in range(10):
        q=rng.integers(-127,128,(8,256),dtype=np.int64)
        w=rng.integers(-128,128,(256,8),dtype=np.int64)
        if case==1:q.fill(127);w.fill(-128)
        if case==2:q[:]=np.resize([-127,127],q.shape);w[:]=np.resize([-128,127],w.shape)
        av=eval_dag(a,q.reshape(64,32),False)
        bv=eval_dag(b,w.reshape(8,32,8).transpose(0,2,1).reshape(64,32),False)
        assert all(np.max(np.abs(v))<=ab for v in av.values())
        assert all(np.max(np.abs(v))<=bb for v in bv.values())
        products=[np.array((av[an]*bv[bn]).sum(),dtype=np.int64).astype('<i4') for an,bn in leaves]
        cv=eval_dag(c,products,True)
        actual=np.array([[cv[roots[r][n]] for n in range(8)] for r in range(8)])
        assert np.array_equal(actual,q@w)
        cases.append(dict(case=case,all64_modular_dots_equal=True))
    d=ROOT/'artifacts/rank343-integer-basis-v1';d.mkdir(exist_ok=False)
    frozen=d/'plan.py';frozen.write_bytes(source.read_bytes())
    emitter=ROOT/'scripts/generate_s3_k2_prepared_kernels.py';em=load(emitter)
    files=[Path(__file__),source,ns['BASE'],frozen,emitter];kernels=[]
    for seed in [False,True]:
        text,count,symbol=em['kernel'](ns,seed);p=d/('32_seed.wat' if seed else '32.wat');p.write_text(text);files.append(p)
        kernels.append(dict(path=str(p.relative_to(ROOT)),symbol=symbol,locals=count,seed=seed,tile=32))
    r=dict(complete=True,all262144_tensor_entries_exact=True,all343_leaf_coefficients_unchanged=True,
           all64_final_product_expressions_unchanged=True,input_i16_bound=ab,weight_i16_bound=bb,
           old_dag_nodes=[len(oa.nodes),len(ob.nodes),len(oc.nodes)],new_dag_nodes=[len(a.nodes),len(b.nodes),len(c.nodes)],
           reconstruction_nodes=len(c.nodes),unreduced_bound=bound,final_dot_bound=256*127*128,
           cases=cases,kernels=kernels,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},
           scope='Integer tensor basis and final inverse, including all preparation nodes. Identical rank343 leaves and original dots modulo I32. Host proof only; generated SIMD files not yet executed or metered. Prepacked weights retain 10.71875x raw capacity; no full-model adoption claim.')
    (d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    with zipfile.ZipFile(d/'evidence.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in files+[d/'report.json']:z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps({k:r[k] for k in ['old_dag_nodes','new_dag_nodes','input_i16_bound','weight_i16_bound','unreduced_bound']}))
if __name__=='__main__':main()
