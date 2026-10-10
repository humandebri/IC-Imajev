#!/usr/bin/env python3
"""LOCAL-only scalar/SIMD parity, stable receipt capacity and upgrade evidence.

The scalar Attention oracle shares the model's other numerical kernels and the
32-layer scheduler. This is not an independent implementation of the entire model.
"""
import argparse, hashlib, json, struct, subprocess, sys, uuid
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'client')]
from paid_update_transport import PaidTransport
from transport import Transport
from prefix_inference import verify_module
from check_canister_api_exports import check_exports


def bits(values):return struct.pack('<'+str(len(values))+'f',*values)
def measurement(call):
    v=call['result']['Ok'];w=v['workers']
    assert w and all(x['instructions']<40_000_000_000 for x in w)
    assert all(x['heap_pages']*65536<=4*1024**3 for x in w)
    return dict(seconds=call['seconds'],workers=len(w),instructions=sum(x['instructions'] for x in w),
                max_worker_instructions=max(x['instructions'] for x in w),max_heap_bytes=max(x['heap_pages'] for x in w)*65536)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['kernels','full','receipts','normal','restored'])
    for name in ['canister','relay','wasm','directory','helper','call-helper','source-proof']:p.add_argument('--'+name,required=True)
    a=p.parse_args();d=Path(a.directory);d.mkdir(parents=True,exist_ok=False)
    module=hashlib.sha256(Path(a.wasm).read_bytes()).hexdigest()
    check_exports(a.wasm,diagnostics=a.mode not in ('normal','restored'))
    status=json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--network','local','--identity','imajev-local','--json'],text=True,cwd=ROOT))
    assert status['status']=='Running' and status['module_hash'].removeprefix('0x')==module
    wire=PaidTransport(d/'calls',a.canister,helper=Path(a.helper),call_helper=Path(a.call_helper))
    source=json.loads(Path(a.source_proof).read_text());request=None if a.mode=='restored' else next(c for c in source['cases'] if c.get('tokens')==1024)['call']['request']['request']
    report=dict(complete=False,network='local',module=module,mode=a.mode,cases=[],scope=__doc__)
    def save(row):report['cases'].append(row);(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k not in ('call','debug','payload','receipt') }),flush=True)
    def infer(label,request):
        payload=dict(request=request,request_id=label+'-'+uuid.uuid4().hex[:12]);quote=wire.call('quote',request)['result']['Ok'];fee=quote['fee']
        call=wire.call('infer',payload,relay=a.relay,cycles=fee+12345);assert call['forward']['refunded']==12345
        assert call['result']['Ok']['paid_cycles']==fee
        duplicate=wire.call('infer',payload,relay=a.relay,cycles=fee);assert duplicate['result']==call['result'] and duplicate['forward']['refunded']==fee
        return dict(label=label,payload=payload,call=call,duplicate_no_charge=True,**measurement(call))
    try:
        if a.mode=='kernels':
            m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
            t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d/'queries',m['pack_hash'],wire_codec='bf16-block256-exact-v1',frame_version=3)
            try:
                verify_module(t,module)
                for seed,(n,total,width,heads) in enumerate([(1,1024,256,4),(2,1024,256,4),(57,1024,256,4),(57,823,256,4),(57,512,256,8),(7,1024,255,4),(7,1024,256,4)]):
                    rng=np.random.default_rng(seed);count=(heads*n+2*(heads//4)*total)*width
                    x=rng.normal(0,.5,count).astype('<f4');x=(x.view('<u4')&np.uint32(0xffff0000)).view('<f4');x[::127]=-0.
                    if seed==6:x[:heads*n*width]=np.float32(-0.);x[heads*n*width:]=np.float32(-0.)
                    np.save(d/f'kernel-{seed}.npy',x)
                    wire.call('paid_reference',False);got=t.run('gqa_suffix_bf16',x,[n,width,heads,total-n],[])
                    stride=n*width;kvstride=total*width;groups=heads//4;expected=[]
                    wire.call('paid_reference',True)
                    for head in range(heads):
                        group=head//4
                        part=np.concatenate([x[head*stride:(head+1)*stride],x[heads*stride+group*kvstride:heads*stride+(group+1)*kvstride],x[heads*stride+(groups+group)*kvstride:heads*stride+(groups+group+1)*kvstride]])
                        expected.append(t.run('gqa_suffix_bf16',part,[n,width,1,total-n],[]))
                    want=np.concatenate(expected);assert got.tobytes()==want.tobytes(),(n,total,width,heads)
                    save(dict(n=n,total=total,width=width,heads=heads,bitwise_equal=True,sha256=hashlib.sha256(got.tobytes()).hexdigest()))
                verify_module(t,module)
            finally:wire.call('paid_reference',False);t.close()
        elif a.mode=='full':
            rows=[]
            try:
                for scalar in [False,True]:
                    wire.call('paid_reference',scalar);row=infer('scalar' if scalar else 'simd',request)
                    row['debug']=wire.call('paid_debug')['result'];assert len(row['debug']['hidden_hashes'])==32 and len(row['debug']['state_hashes'])==32
                    save(row);rows.append(row)
                left,right=rows
                for field in ['hidden_hashes','state_hashes']:assert left['debug'][field]==right['debug'][field],field
                assert bits(left['debug']['final_hidden'])==bits(right['debug']['final_hidden'])
                x,y=[r['call']['result']['Ok']['decision'] for r in rows]
                for field in ['raw_logits','probabilities']:assert bits(x[field])==bits(y[field]),field
                assert bits([x['unknown_probability']])==bits([y['unknown_probability']])
                for field in ['value','abstained','calibration_version']:assert x[field]==y[field]
                save(dict(full32_hidden_hashes_equal=True,full32_state_hashes_equal=True,final_hidden_equal=True,logits_bitwise_equal=True,decision_equal=True))
            finally:wire.call('paid_reference',False)
        elif a.mode=='receipts':
            short=dict(request,token_ids=request['token_ids'][:94]);fee=wire.call('quote',short)['result']['Ok']['fee'];payloads=[]
            wire.call('paid_fault',dict(stage=0,trap=True,refund_fail=False))
            try:
                for i in range(130):
                    pending=i==0
                    wire.call('paid_fault',dict(stage=0,trap=True,refund_fail=pending))
                    payload=dict(request=short,request_id='archive-'+uuid.uuid4().hex[:12])
                    call=wire.call('infer',payload,relay=a.relay,cycles=fee)
                    assert call['result']['Err']['Failed']['refund']==('Pending' if pending else 'Done')
                    # Accepted fees return through deposit_cycles, not the original
                    # call's unaccepted-cycle refund counter.
                    forward=call['forward'];assert forward['refunded']==0
                    spent=forward['balance_before']-forward['balance_after']
                    assert (fee<=spent<fee+1_000_000_000) if pending else (0<=spent<1_000_000_000)
                    payloads.append(payload)
                    if i in (0,127,128,129):save(dict(admitted=i+1,pending=pending,payload=payload,call=call))
                for payload in [payloads[0],payloads[-1]]:
                    receipt=wire.call('inference_status',payload['request_id'],relay=a.relay)['result'];assert receipt and receipt['request_id']==payload['request_id'];save(dict(preserved_request_id=payload['request_id'],receipt=receipt))
                (d/'requests.json').write_text(json.dumps(payloads,indent=2)+'\n')
            finally:wire.call('paid_fault',dict(stage=None,trap=False,refund_fail=False))
        elif a.mode=='normal':
            save(infer('normal',request))
            denied=dict(request,token_ids=request['token_ids']+[198]);quote=wire.call('quote',denied)['result'];assert 'Invalid' in quote['Err']
            call=wire.call('infer',dict(request=denied,request_id='denied-'+uuid.uuid4().hex[:12]),relay=a.relay,cycles=1)
            assert 'Invalid' in call['result']['Err'] and call['forward']['refunded']==1;save(dict(tokens=1025,rejected_before_payment=True))
        else:
            # --source-proof in this mode selects the receipt stress report's directory.
            payloads=json.loads((Path(a.source_proof).parent/'requests.json').read_text())
            for payload in [payloads[0],payloads[-1]]:
                receipt=wire.call('inference_status',payload['request_id'],relay=a.relay)['result'];assert receipt and receipt['request_id']==payload['request_id'];save(dict(restored_request_id=payload['request_id'],receipt=receipt))
            payload=payloads[0];retry=wire.call('retry_inference_refund',payload['request_id'],relay=a.relay)
            assert retry['result']=={'Ok':'Done'}
            fee=receipt['quote']['fee'];received=retry['forward']['balance_after']-retry['forward']['balance_before']
            assert fee-1_000_000_000<received<=fee
            again=wire.call('retry_inference_refund',payload['request_id'],relay=a.relay);assert again['result']=={'Ok':'Done'} and again['forward']['balance_after']<=again['forward']['balance_before']
            replay=wire.call('infer',payload,relay=a.relay,cycles=fee)
            assert replay['result']['Err']['Failed']['refund']=='Done' and replay['forward']['refunded']==fee
            save(dict(upgrade_refund_retry_no_double_refund=True,replay_no_charge=True))
        report['complete']=True
    finally:(d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
