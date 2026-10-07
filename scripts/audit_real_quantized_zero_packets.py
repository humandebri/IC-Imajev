#!/usr/bin/env python3
"""Independent index-based zero counts, runtime coefficients and frozen ZIP."""
import ast,hashlib,json,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    d=ROOT/'artifacts/real-quantized-zero-packets-v1';r=json.loads((d/'report.json').read_text())
    for p,h in r['source_hashes'].items():assert sha(ROOT/p)==h,p
    runtime=ROOT/'artifacts/update-rank7-prepare8-unrolled-v1/build/runtime/rank49_raw.rs'
    source=runtime.read_text();start=source.index('=[',source.index('const A:'))+1;end=source.index('];',start)+1
    terms=ast.literal_eval(source[start:end].replace('&',''))
    table=np.zeros((49,16),dtype=np.int64)
    for m,row in enumerate(terms):
        for i,v in row:table[m,i]=v
    with np.load(d/'coefficients.npz')as z:
        assert np.array_equal(table,z['rank49']);rank7=z['rank7'].astype(np.int64)
    extraction=json.loads((ROOT/'artifacts/terminal-mlp-reference-v1/report.json').read_text())
    audited=[]
    for row in r['rows']:
        carry=next(c for c in extraction['cases']if c['case']==row['case']and c['layer']==row['layer'])
        with np.load(ROOT/carry['carry'],allow_pickle=False)as z:q=z[row['input']][row['suffix_offset']:].astype(np.int64)
        assert q.shape==(row['rows'],row['cols'])
        width=4 if row['rank']==49 else 2;chunk=256//width;K=chunk//2
        if width==4:groups=len(q)//4;valid_n=groups*4;q=q[:valid_n]
        else:
            groups=(len(q)+1)//2
            if len(q)%2:q=np.concatenate([q,np.zeros((1,q.shape[1]),dtype=np.int64)])
        counts=[]
        for m,coeff in enumerate(table if width==4 else rank7):
            value=np.zeros((groups,q.shape[1]//256,chunk),dtype=np.int64)
            # Explicit original token/column indices, with no tensor reshape/einsum.
            for i,c in enumerate(coeff):
                if c:
                    tokens=np.arange(groups)*width+i//width
                    columns=np.arange(q.shape[1]//256)[:,None]*256+(i%width)*chunk+np.arange(chunk)[None,:]
                    value+=int(c)*q[tokens[:,None,None],columns[None,:,:]]
            active=value
            if width==2 and row['rows']%2 and m in [3,4,6]:active=value[:-1]
            zero=np.logical_and(active[:,:,0::2]==0,active[:,:,1::2]==0)
            count=int(zero.size);zeros=int(zero.sum());saved=row['leaf_fractions'][m]
            assert saved['leaf']==m and saved['packets']==count and saved['zero_packets']==zeros
            counts.append((zeros,count))
        zeros=sum(x for x,_ in counts);total=sum(y for _,y in counts)
        assert (zeros,total)==(row['zero_packets'],row['packets'])
        G=6 if width==4 else 16
        baseline=4*G*total;candidate=2*G*(total//K)+3*total+6*G*(total-zeros)
        assert baseline==row['modeled_original_instructions']and candidate==row['modeled_packet_gate_instructions']
        assert candidate>baseline
        audited.append(dict(case=row['case'],layer=row['layer'],input=row['input'],all_leaf_counts_exact=True))
    assert len(audited)==12
    files=[Path(__file__),d/'report.json',runtime]+[ROOT/p for p in r['source_hashes']]
    audit=dict(complete=True,all12_index_based_counts_exact=True,rank49_coefficients_equal_current_runtime=True,all_selected_packet_gate_models_regress=True,
        source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},ic_performance_verified=False,adopted=False,full_paid_goal_achieved=False,scope=r['scope'])
    out=d/'independent-audit.json';out.write_text(json.dumps(audit,indent=2)+'\n');files.append(out)
    archive=d/'frozen-evidence.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED)as z:
        for p in dict.fromkeys(files):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive)as z:
        assert z.testzip()is None
        for p in dict.fromkeys(files):assert hashlib.sha256(z.read(str(p.relative_to(ROOT)))).hexdigest()==sha(p)
    (d/'archive-identity.json').write_text(json.dumps(dict(complete=True,sha256=sha(archive),all_archived_bytes_verified=True),indent=2)+'\n')
    print('independent index/runtime coefficient/count/archive checks passed:',sha(archive))
if __name__=='__main__':main()
