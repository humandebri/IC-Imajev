#!/usr/bin/env python3
"""Historical exact quantized carries: zero-packet branch feasibility screen."""
import hashlib,json
from pathlib import Path
import numpy as np
from plan_rank49_integer_input_basis import plan
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    d=ROOT/'artifacts/real-quantized-zero-packets-v1';d.mkdir(exist_ok=False)
    parent=ROOT/'artifacts/terminal-mlp-reference-v1'
    extraction=json.loads((parent/'report.json').read_text());assert extraction['complete'] and extraction['extraction_only']
    paid=ROOT/'artifacts/paid-rank7-prepare8-unrolled-v1/proof/report.json'
    current=json.loads(paid.read_text());assert current['complete'] and current['baseline_restored']
    cases={r['case']:r for r in current['results']}
    ad,_,_,leaves,_,_=plan()
    A=np.array([[ad.symbols[a].get(i,0)for i in range(16)]for a,_ in leaves],dtype=np.int64)
    assert A.shape==(49,16)
    coefficients=np.array([[1,0,0,0],[0,1,0,0],[1,1,-1,-1],[0,0,0,1],[0,0,1,1],[-1,0,1,1],[1,0,-1,0]],dtype=np.int64)
    files=[Path(__file__),ROOT/'scripts/plan_rank49_integer_input_basis.py',parent/'report.json',paid]
    rows=[]
    for row in extraction['cases']:
        name=row['case'];c=cases[name];offset=c['quote']['prefix_tokens']-row['prefix'];assert offset>=0
        path=ROOT/row['carry'];assert sha(path)==row['carry_sha256'];files.append(path)
        with np.load(path,allow_pickle=False)as z:
            for key in ['q_input','product_q']:
                q=z[key][offset:].astype(np.int64);n,cols=q.shape
                expected_n=len(c['request']['request']['token_ids'])-c['quote']['prefix_tokens']
                assert n==expected_n and cols%256==0 and q.min()>=-127 and q.max()<=127
                if key=='product_q':assert cols==row['done']==1280
                # Existing current projection dispatcher selects rank49 for n56/57
                # at these full q/down shapes. Historical partial product columns
                # are only a subset, never a full-model sparsity claim.
                rank=49 if n in [56,57] else 7
                if rank==7:
                    padded=np.zeros(((n+1)//2*2,cols),dtype=np.int64);padded[:n]=q
                    x=padded.reshape(-1,2,cols//256,2,128).transpose(0,2,1,3,4).reshape(-1,4,128)
                    operands=np.einsum('mi,bik->bmk',coefficients,x)
                    active=np.ones(operands.shape[:2],dtype=bool)
                    if n%2:
                        start=((n+1)//2-1)*(cols//256);active[start:,[3,4,6]]=False
                    K=64;groups=16
                else:
                    # Complete quartets only. Actual odd final-token raw tail is
                    # excluded and explicitly reported rather than treated as49 leaves.
                    quartets=n//4
                    x=q[:quartets*4].reshape(quartets,4,cols//256,4,64).transpose(0,2,1,3,4).reshape(-1,16,64)
                    operands=np.einsum('mi,bik->bmk',A,x);active=np.ones(operands.shape[:2],dtype=bool)
                    K=32;groups=6
                assert operands.min()>=-32768 and operands.max()<=32767
                packets=operands.reshape(*operands.shape[:2],K,2)
                zero=np.all(packets==0,axis=-1)
                zeros=int(zero[active].sum());total=int(active.sum())*K
                fractions=[]
                for m in range(rank):
                    mask=active[:,m];count=int(mask.sum())*K
                    fractions.append(dict(leaf=m,zero_packets=int(zero[:,m,:][mask].sum()),packets=count,fraction=float(zero[:,m,:][mask].mean())))
                # Explicit schedule: reset G accumulators (2G), per K packet
                # local.get/any_true/if (3), and six instructions per active
                # accumulator update. Original long chains cost4GK.
                baseline=4*groups*total
                candidate=2*groups*int(active.sum())+3*total+6*groups*(total-zeros)
                threshold=1-(4*groups-3-2*groups/K)/(6*groups)
                result=dict(case=name,layer=row['layer'],input=key,rows=n,cols=cols,rank=rank,
                    current_prefix_tokens=c['quote']['prefix_tokens'],historical_prefix=row['prefix'],suffix_offset=offset,
                    raw_i16_zero_fraction=float((q==0).mean()),zero_packets=zeros,packets=total,zero_packet_fraction=zeros/total,
                    maximum_leaf_zero_fraction=max(v['fraction']for v in fractions),packet_gate_break_even_zero_fraction=threshold,
                    modeled_original_instructions=baseline,modeled_packet_gate_instructions=candidate,modeled_percent=(candidate/baseline-1)*100,
                    leaf_fractions=fractions,rank49_tail_tokens_excluded=n%4 if rank==49 else 0,
                    quantized_source_sha256=row['carry_sha256'])
                rows.append(result);print(json.dumps({k:v for k,v in result.items()if k not in ['leaf_fractions','quantized_source_sha256']}),flush=True)
    assert len(rows)==12
    np.savez_compressed(d/'coefficients.npz',rank7=coefficients,rank49=A);files.append(d/'coefficients.npz')
    report=dict(complete=True,rows=rows,all_selected_packet_gate_models_regress=all(r['modeled_packet_gate_instructions']>r['modeled_original_instructions']for r in rows),
                source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},ic_performance_verified=False,adopted=False,full_paid_goal_achieved=False,
                scope='Historical exact INT16 carry inputs from layers26/30; q_input full2560 and product_q first1280 only, sliced to current paid suffix. Counts/model only, not current additional Dense capture, actual Wasm timing or universal zero-skip rejection. Arithmetic model covers one specific packet-gated accumulator schedule; other schedules may differ.')
    (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
