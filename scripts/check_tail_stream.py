#!/usr/bin/env python3
"""Continue a proven six-query tail; separately verify prepared terminal/readout."""
import argparse,hashlib,io,json,pathlib,shutil,sys,zipfile
import numpy as np
if not __debug__:raise RuntimeError('Proof checker requires assertions')
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from proof_inputs import read_bytes,read_report,selections
from transport import Transport,decode,is_instruction_limit
from prefix_inference import verify_module
from mlp_attention_finish_codec import NAME as BRIDGE,STREAM_COMPLETE,TERMINAL,encode_request as bridge
from mlp_delta_stream_codec import NAME as PAIR,encode_request as pair,encode_continue_request as follow
from mlp_stream_codec import NAME as STREAM,length
from delta_mlp_start_codec import NAME as START,CONV,encode_request as start_packet
from mlp_codec import NAME as DOWN
from mlp_delta_carry import HUFFMAN_NAME,encode_request as finish

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['canister','wasm','directory','six-proof']:ap.add_argument('--'+name,required=True)
    ap.add_argument('--down29',type=int,default=0);ap.add_argument('--chain-terminal',action='store_true');ap.add_argument('--heads28',type=int,choices=range(2,32,2),default=6);ap.add_argument('--through8',action='store_true');ap.add_argument('--front28',default='6144');ap.add_argument('--heads29',default='26');ap.add_argument('--terminal-front',default='2560,3072,3584');a=ap.parse_args()
    if a.down29 and not (0<a.down29<2560 and a.down29%32==0):raise ValueError('down29 bounds')
    fronts=[]if a.through8 else list(map(int,selections(a.front28,tuple(map(str,range(128,9216,128))),'front28')))
    heads=list(map(int,selections(a.heads29,tuple(map(str,range(2,32,2))),'heads29')))
    tails=list(map(int,selections(a.terminal_front,tuple(map(str,range(256,9216,256))),'terminal front')))
    d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
    if any(d.iterdir()):raise ValueError('Use a fresh proof directory')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=list((ROOT/'client').glob('*.py'))+list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'canisters/inference/src').rglob('*.rs'))+[ROOT/p for p in ['Cargo.toml','Cargo.lock','crates/imajev-runtime/Cargo.toml','canisters/inference/Cargo.toml','MODEL_LOCK.json','checkpoints/full-int8.manifest.json','scripts/proof_inputs.py','scripts/build_full_prefix_candidate.py']]+[pathlib.Path(__file__)]
    hashes={str(p.relative_to(ROOT)):sha(p)for p in paths};refs={};rows=[];module=sha(ROOT/a.wasm);base=ROOT/'artifacts/prefix_codec/full-attention-q4-proof-v1';baseline=ROOT/'artifacts/output-pairs-v2-617';p=45;C,H,KV=2560,9216,2048
    m=json.loads(read_bytes(ROOT/'checkpoints/full-int8.manifest.json',ROOT,refs))
    t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec=STREAM,frame_version=3)
    def reference(layer,root=base/'617'):
        r=read_report(root/'report.json',ROOT,refs);q=next(q for q in r['queries']if q['op']=='mlp_full_integer'and f'.layers.{layer}.'in q['tensor'])
        h,x=decode(read_bytes(root/'queries'/f"{q['index']:06d}.request.bin",ROOT,refs));_,y=decode(read_bytes(root/'queries'/f"{q['index']:06d}.response.bin",ROOT,refs));return h,x,y
    def state(layer,prefix=False):
        root=base/'prefix'if prefix else base/'617'
        with np.load(io.BytesIO(read_bytes(root/'queries/states'/f'layer-{layer:02d}.npz',ROOT,refs)),allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
    def header(h,op,codec,dims):return dict(h,version=3,step=t.index,op=op,encoding=codec,dims=dims)
    def run(h,packet,scope):
        index=t.index
        try:y=t._run_encoded(h,packet)
        except RuntimeError as e:
            if not is_instruction_limit(e):raise
            failed=d/f'failed-{scope}.request.bin';shutil.copyfile(d/f'{index:06d}.request.bin',failed)
            row=dict(scope=scope,success=False,op=h['op'],error=str(e),request_file=failed.name,request_sha256=sha(failed));rows.append(row);print(json.dumps(row),flush=True);return None
        metric=t.measurements[-1];(d/f'{index:06d}.metric.json').write_text(json.dumps(metric,indent=2)+'\n');return y,metric
    def success(scope,calls,**extra):
        row=dict(scope=scope,success=True,full_bitwise_equal=True,calls=calls,**extra);rows.append(row);print(json.dumps(row),flush=True)
    def framed(h,build,scope,operands):
        try:packet=build()
        except ValueError as e:
            if 'frame bounds'not in str(e):raise
            saved=d/f'failed-{scope}.header.json';saved.write_text(json.dumps(h,indent=2)+'\n');files=[]
            for i,value in enumerate(operands):
                path=d/f'failed-{scope}.operand{i}.npy';np.save(path,value);files.append(dict(path=path.name,sha256=sha(path)))
            row=dict(scope=scope,success=False,stage='frame',op=h['op'],error=str(e),header_file=saved.name,operands=files);rows.append(row);print(json.dumps(row),flush=True);return None
        return run(h,packet,scope)
    def group(layer,k):
        z=state(layer,True);lv=z['delta_log'];cv=z['conv'];expected=state(layer)['conv']
        cols=lambda lo,hi:np.concatenate([np.arange(lo//2*128,hi//2*128),np.arange(2048+lo//2*128,2048+hi//2*128),np.arange(4096+lo*128,4096+hi*128)])
        logs=lambda lo,hi:np.concatenate([lv[:p*2048].reshape(p,2048)[:,lo//2*128:hi//2*128].ravel(),lv[p*2048:p*6144].reshape(p,4096)[:,lo*128:hi*128].ravel(),lv[p*6144:].reshape(p,32)[:,lo:hi].ravel()])
        first,rest=cols(0,k),cols(k,32);return cv[:,first],cv[:,rest],logs(0,k),logs(k,32),expected[:,first],expected[:,rest]
    try:
        verify_module(t,module)
        sixroot=ROOT/a.six_proof;six=read_report(sixroot/'report.json',ROOT,refs)
        if six['module_sha256']!=module:raise ValueError('six-query module mismatch')
        row=next(r for r in six['cases']if 'calls6'in r and r['full_bitwise_equal']);source=sixroot/f"{row['calls6'][-1]['index']:06d}.response.bin";_,v=decode(read_bytes(source,ROOT,refs))
        h26,x26,b26=reference(26);n=h26['dims'][0];begin=row['next_mlp_front'];v=v[:length(n,begin)]
        h27,x27,b27=reference(27);h28,x28,b28=reference(28);h29,x29,b29=reference(29);h30,x30,b30=reference(30,baseline)
        ref=read_report(base/'617/report.json',ROOT,refs);q=ref['queries'][-1];_,terminal=decode(read_bytes(base/'617/queries'/f"{q['index']:06d}.response.bin",ROOT,refs));expected_terminal=terminal[-(2*C+n*KV):]
        z=state(31,True);prefix31=np.concatenate([z['keys'].transpose(1,0,2).ravel(),z['values'].transpose(1,0,2).ravel()]);serving=json.loads(read_bytes(ROOT/'artifacts/reference-serving.json',ROOT,refs));t.decision_options=serving['records'][0]['options'];t.fuse_terminal_decision=True
        controls30={}
        if a.down29:
            t.wire_codec=DOWN;control29=t.run('mlp_prepare_partial_down',x29,[n,C,a.down29],[2.,1e-6],tensor=h29['tensor'],aux=h29['aux'],input_hash=h29['input_hash']);t.wire_codec=STREAM
        for front in tails:controls30[front]=t.run('mlp_stream_prepare',x30,[n,0,front],[2.,1e-6],tensor=h30['tensor'],aux=h30['aux'],input_hash=h30['input_hash'])
        z=state(27,True);prefix=np.concatenate([z['keys'].transpose(1,0,2).ravel(),z['values'].transpose(1,0,2).ravel()]);z=state(27);kv=np.concatenate([z['keys'][p:].ravel(),z['values'][p:].ravel()])
        h=header(h26,STREAM_COMPLETE,BRIDGE,[n,p,begin]);seven=run(h,bridge(h,v,prefix),'seven')
        if seven is not None:
            y7,q7=seven;expected=np.concatenate([b26[:n*C],x27[n*C:],kv]);assert y7.tobytes()==expected.tobytes();calls7=row['calls6']+[q7];success('seven',calls7)
            cv1,cv2,log1,log2,expected1,expected2=group(28,a.heads28)
            h=header(h27,'mlp_full_delta_partial',PAIR,[n,0,H,a.heads28,p]);eight=run(h,pair(h,y7[:2*n*C],cv1,log1),'eight')
            if eight is not None:
                y8,q8=eight;assert y8[:n*C].tobytes()==b27[:n*C].tobytes();assert y8[-3*a.heads28*256:].tobytes()==expected1.ravel().tobytes();calls8=calls7+[q8];success('eight',calls8)
                for front in fronts:
                    h=header(h27,'delta_partial_mlp_front',PAIR,[n,0,H,a.heads28,p,front]);nine=framed(h,lambda:follow(h,y8,cv2,log2,compress_base=True,compress_prefix=True,compress_hidden=True,compress_input=True),f'nine-front{front}',[y8,cv2,log2])
                    if nine is None:continue
                    y9,q9=nine;t.wire_codec=STREAM;control=t.run('mlp_stream_prepare',x28,[n,0,front],[2.,1e-6],tensor=h28['tensor'],aux=h28['aux'],input_hash=h28['input_hash'])
                    assert y9[:length(n,front)].tobytes()==control.tobytes();assert y9[length(n,front):].tobytes()==expected2.ravel().tobytes();calls9=calls8+[q9];success(f'nine-front{front}',calls9)
                    for k in heads:
                        c1,c2,l1,l2,e1,e2=group(29,k);h=header(h28,'mlp_complete_delta_partial',PAIR,[n,front,H-front,k,p]);ten=framed(h,lambda:pair(h,y9[:length(n,front)],c1,l1,compress_residual=True,residual_raw_threshold=.1,residual_dictionary=True),f'ten-front{front}-heads{k}',[y9[:length(n,front)],c1,l1])
                        if ten is None:continue
                        y10,q10=ten;assert y10[:n*C].tobytes()==b28[:n*C].tobytes();assert y10[-3*k*256:].tobytes()==e1.ravel().tobytes();calls10=calls9+[q10];success(f'ten-front{front}-heads{k}',calls10)
                        h=header(h28,'delta_partial_mlp_prepare_down'if a.down29 else'delta_partial_mlp_full',PAIR,[n,front,H-front,k,p]+([a.down29]if a.down29 else[]));eleven=run(h,follow(h,y10,c2,l2,compress_base=True),f'eleven-front{front}-heads{k}')
                        if eleven is None:continue
                        y11,q11=eleven;count29=control29.size if a.down29 else 2*n*C;expected29=control29 if a.down29 else b29
                        assert y11[:count29].tobytes()==expected29.tobytes();assert y11[count29:].tobytes()==e2.ravel().tobytes();calls11=calls10+[q11];success(f'eleven-front{front}-heads{k}',calls11)
                        if a.chain_terminal:
                            for terminal_front in tails:
                                z30=state(30,True);scope=f'twelve-front{front}-heads{k}-terminal{terminal_front}'
                                if a.down29:
                                    h=header(h29,'mlp_finish_delta_log_mlp_front',HUFFMAN_NAME,[n,p,a.down29,terminal_front]);packet=finish(h,y11[:count29],z30['conv'],z30['delta_log'])
                                else:
                                    h=header(h30,'delta_mlp_stream_prepare',START,[n,terminal_front,p]);packet=start_packet(h,y11[:n*C],y11[n*C:2*n*C],z30['conv'],z30['delta_log'])
                                twelve=run(h,packet,scope)
                                if twelve is None:continue
                                y12,q12=twelve;carry30=y12[n*C:-CONV]if a.down29 else y12[:-CONV]
                                if a.down29:assert y12[:n*C].tobytes()==b29[:n*C].tobytes()
                                assert carry30.tobytes()==controls30[terminal_front].tobytes();assert y12[-CONV:].tobytes()==state(30)['conv'].ravel().tobytes();calls12=calls11+[q12];success(scope,calls12)
                                h=header(h30,TERMINAL,BRIDGE,[n,p,terminal_front]);thirteen=run(h,bridge(h,carry30,prefix31),scope+'-decision')
                                if thirteen is None:continue
                                y13,q13=thirteen;assert y13.tobytes()==expected_terminal.tobytes()
                                for key in ['value','abstained','raw_logits','probabilities','unknown_probability']:assert t.terminal_decision[key]==ref['decision_query']['ok']['decision'][key],key
                                success(scope+'-decision',calls12+[q13],thirteen_query_chain=True)
        # An independent terminal input is a control, never a whole-inference proof.
        for front in tails:
            z30=state(30,True);h=header(h30,'delta_mlp_stream_prepare',START,[n,front,p]);control=run(h,start_packet(h,b29[:n*C],b29[n*C:],z30['conv'],z30['delta_log']),f'start30-front{front}')
            if control is not None:
                y,q=control;assert y[:-CONV].tobytes()==controls30[front].tobytes();assert y[-CONV:].tobytes()==state(30)['conv'].ravel().tobytes();success(f'start30-front{front}',[q],independent_control=True)
            h=header(h30,TERMINAL,BRIDGE,[n,p,front]);result=run(h,bridge(h,controls30[front],prefix31),f'terminal-front{front}')
            if result is None:continue
            y,q=result;assert y.tobytes()==expected_terminal.tobytes()
            for key in ['value','abstained','raw_logits','probabilities','unknown_probability']:assert t.terminal_decision[key]==ref['decision_query']['ok']['decision'][key],key
            success(f'terminal-front{front}',[q],independent_control=True)
        verify_module(t,module);assert sha(ROOT/a.wasm)==module and hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths} and all(sha(ROOT/p)==v for p,v in refs.items())
        (d/'report.json').write_text(json.dumps(dict(settings=vars(a),module_sha256=module,source_hashes=hashes,reference_hashes=refs,cases=rows,measurements=t.measurements,whole_inference_reduction_verified=False),indent=2)+'\n')
        with zipfile.ZipFile(d/'validated-source.zip','x',zipfile.ZIP_DEFLATED)as z:
            for path in paths:z.write(path,str(path.relative_to(ROOT)))
    finally:t.close()
if __name__=='__main__':main()
