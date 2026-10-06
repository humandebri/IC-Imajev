#!/usr/bin/env python3
"""Measure exact W4 arithmetic separately from W8->W4 projection error.

Only installs/prepares a separately created diagnostic; inference is query.
Frozen current S1 measurements are reported as a distinct historical comparator.
"""
import argparse, hashlib, json, pathlib, subprocess, sys, time, zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import decode
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--directory',required=True);ap.add_argument('--build-directory',required=True);ap.add_argument('--labels',default='617,prefix,insufficient,maximum');ap.add_argument('--boundary-tokens',default='1,3,8');ap.add_argument('--full-lut',action='store_true');a=ap.parse_args()
    labels=a.labels.split(',');assert all(v in ['617','prefix','insufficient','maximum','normal'] for v in labels)
    boundaries=[int(v) for v in a.boundary_tokens.split(',')];assert all(1<=v<=109 for v in boundaries)
    d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
    if any(d.iterdir()):raise ValueError('Use fresh evidence directory')
    helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args'
    builddir=ROOT/a.build_directory;build=json.loads((builddir/'report.json').read_text());expected=sha((builddir/'diagnostic.wasm').read_bytes());assert expected==build['wasm_sha256']
    def status(cid):return json.loads(subprocess.check_output(['icp','canister','status',cid,'--network','local','--identity','imajev-local','--json'],text=True,cwd=ROOT))['module_hash']
    protected=['4caro-hl777-77775-aaaba-cai','6eydd-o3777-77775-aaama-cai'];assert a.canister not in protected
    before={cid:status(cid) for cid in protected};assert status(a.canister).removeprefix('0x')==expected
    calls=[]
    def call(method,arg=None,query=False):
        cmd=['icp','canister','call',a.canister,method,'--network','local','--identity','imajev-local','--output','hex']
        if arg:cmd+=['--args-file',str(arg),'--args-format','bin']
        else:cmd+=['()']
        if query:cmd+=['--query']
        start=time.monotonic();raw=subprocess.check_output(cmd,text=True,cwd=ROOT);elapsed=time.monotonic()-start
        path=d/f'{len(calls):04d}-{method}.hex';path.write_text(raw)
        v=json.loads(subprocess.check_output([str(helper),'decode',str(path),'measurement' if query else 'preparation'],text=True,cwd=ROOT));v.update(wall_seconds=elapsed,reply_sha256=sha(raw.encode()),request_bytes=arg.stat().st_size if arg else 6)
        calls.append(dict(method=method,query=query,result=v));return v
    manifest=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());tensor='model.language_model.layers.3.self_attn.q_proj.weight';t=next(v for v in manifest['tensors']if v['name']==tensor);rows,cols=t['rows'],t['cols'];assert(rows,cols)==(8192,2560)
    assert manifest['model']==sha((ROOT/'MODEL_LOCK.json').read_bytes())
    with (ROOT/'checkpoints/full-int8.pack').open('rb')as f:f.seek(t['offset']);raw=f.read(t['bytes'])
    w=np.frombuffer(raw[:rows*cols],dtype=np.int8).reshape(rows,cols).copy();sw=np.frombuffer(raw[rows*cols:],dtype='<f4').copy();assert sw.size==rows
    assert sha(raw)==json.loads((ROOT/'artifacts/column32/check/report.json').read_text())['tensor_bytes_sha256']
    for start in range(0,rows,128):
        path=d/f'weights-{start}.bin';path.write_bytes(w[start:start+128].tobytes()+sw[start:start+128].tobytes());arg=d/f'weights-{start}.args.bin';subprocess.run([str(helper),'chunk',str(start),str(path),str(arg)],check=True);call('prepare_chunk',arg)
    preparation=call('seal')
    blocks=cols//256;wb=w.reshape(rows,blocks,256);peak=np.max(np.abs(wb.astype(np.int16)),axis=2).astype(np.float32);factor=np.where(peak==0,np.float32(1),peak/np.float32(7)).astype(np.float32)
    w4=np.clip(np.rint(wb.astype(np.float32)/factor[:,:,None]),-7,7).astype(np.int8).reshape(rows,cols);s4=(factor*sw[:,None]).astype(np.float32)
    current_path=ROOT/'artifacts/s1_wide/check128/report.json';current=json.loads(current_path.read_text());comparators={v['label']:v for v in current['cases']}
    def native(x,weights,scales):
        n=x.shape[0];xb=x.reshape(n,blocks,256);peak=np.max(np.abs(xb),axis=2);sx=np.where(peak==0,np.float32(1),np.maximum(peak/np.float32(127),np.nextafter(np.float32(0),np.float32(1)))).astype(np.float32)
        q=np.clip(np.rint(xb/sx[:,:,None]),-127,127).astype(np.int32);out=np.zeros((n,rows),dtype=np.float32)
        for b in range(blocks):
            dots=q[:,b,:]@weights[:,b*256:(b+1)*256].astype(np.int32).T
            out+=((dots.astype(np.float32)*sx[:,b,None])*scales[None,:,b]).astype(np.float32)
        return out
    cases=[]
    for label in labels+[f'boundary-{n}' for n in boundaries]:
        source_info={}
        if label.startswith('boundary-'):
            n=int(label.split('-')[1]);x=np.resize(np.array([-127.,127.,0.,-0.,-1.,1.,.5,-.5,2**-126,-2**-126],dtype='<f4'),n*cols).reshape(n,cols)
        else:
            source=ROOT/f'artifacts/output-pairs-v2-{label}';report=json.loads((source/'report.json').read_text());query=next(v for v in report['queries']if v['op']in('attention_q_gqa_integer','attention_full_integer')and v['tensor']==tensor);request=source/'queries'/f'{query["index"]:06d}.request.bin';header,values=decode(request.read_bytes());n=header['dims'][0]
            assert n<=109,'Cold132 needs a distinct row count and separate comparison';x=values[:n*cols].astype('<f4').reshape(n,cols);source_info=dict(request=str(request.relative_to(ROOT)),request_sha256=sha(request.read_bytes()))
        ip=d/f'{label}.input.bin';ip.write_bytes(x.astype('<f4').tobytes());measured={};ref4=native(x,w4,s4)
        for method,name in [(0,'w4_dense_simd'),(1,'w4_lut_simd')]:
            tiles=[]
            for tile,start in enumerate(range(0,rows,1024)):
                arg=d/f'{label}-{method}-{tile}.args.bin';subprocess.run([str(helper),'query',str(method+tile*2),str(ip),str(arg)],check=True);v=call('project',arg,True)
                assert list(hashlib.sha256(ref4[:,start:start+1024].astype('<f4').tobytes()).digest())==v['digest']
                tiles.append(v)
            counts=['quantize_instructions','input_prepare_instructions','project_instructions','total_instructions','wall_seconds','request_bytes']
            measured[name]={k:sum(v[k]for v in tiles)for k in counts}
            measured[name]['tiles']=tiles
            measured[name]['single_preparation_estimate_instructions']=measured[name]['project_instructions']+tiles[0]['quantize_instructions']+tiles[0]['input_prepare_instructions']
        assert [v['digest']for v in measured['w4_dense_simd']['tiles']]==[v['digest']for v in measured['w4_lut_simd']['tiles']]
        if a.full_lut:
            arg=d/f'{label}-full.args.bin';subprocess.run([str(helper),'query','16',str(ip),str(arg)],check=True);v=call('project',arg,True)
            assert list(hashlib.sha256(ref4.astype('<f4').tobytes()).digest())==v['digest'];measured['w4_lut_full_query']=v
        ref8=native(x,w,np.broadcast_to(sw[:,None],(rows,blocks)));diff=ref4.astype(np.float64)-ref8;den=np.linalg.norm(ref8.astype(np.float64));error=dict(relative_l2=float(np.linalg.norm(diff)/den)if den else None,max_abs=float(np.max(np.abs(diff))),rmse=float(np.sqrt(np.mean(diff**2))),cosine=float(np.vdot(ref4.astype(np.float64),ref8.astype(np.float64))/(np.linalg.norm(ref4.astype(np.float64))*den))if den and np.linalg.norm(ref4)>0 else None)
        if label in comparators:assert list(hashlib.sha256(ref8.astype('<f4').tobytes()).digest())==comparators[label]['native']['digest']
        row=dict(label=label,tokens=n,input_sha256=sha(ip.read_bytes()),source=source_info,measurements=measured,projection_error_vs_int8=error,integer_lut_exact=True,native_w4_digest_exact=True)
        if label in comparators:
            old=comparators[label]['measurements']['wide']['total_instructions'];row.update(historical_current_s1_instructions=old,lut_change_vs_historical_percent=100*(measured['w4_lut_simd']['total_instructions']/old-1))
        cases.append(row);print(json.dumps(row),flush=True)
    assert status(a.canister).removeprefix('0x')==expected;assert before=={cid:status(cid)for cid in protected}
    assert build['source_hashes']=={p:sha((ROOT/p).read_bytes()) for p in build['source_hashes']}
    result=dict(scope='Single real Q projection split into eight output1024 queries/method; unchanged A8/block256. W4 symmetric/group256 requantized from the locked W8 pack, NOT from BF16. Exact I16 LUT; no activation/state/LUT quantization. No full-model or decision-accuracy validation. Historical S1 comparator is one query, so raw totals include repeated preparation for the eight tiles; single_preparation_estimate is a sum estimate, not an executed query.',canister=a.canister,module_sha256=expected,protected_module_hashes=before,build=build,model_lock_sha256=manifest['model'],tensor=tensor,tensor_sha256=sha(raw),rows=rows,cols=cols,packed_weight_bytes=rows*cols//2,group_scale_bytes=rows*blocks*4,original_weight_and_scale_bytes=len(raw),diagnostic_memory_note='Keeps original W8, dense W4 and packed W4 for controls; not a production memory measurement.',historical_comparator_sha256=sha(current_path.read_bytes()),historical_comparator_module=current['wasm_sha256'],preparation=preparation,ordinary_queries=sum(v['query']for v in calls),calls=calls,cases=cases)
    sources=[pathlib.Path(__file__),ROOT/'client/transport.py'];result['checker_source_hashes']={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in sources}
    with zipfile.ZipFile(d/'checker-source.zip','w',zipfile.ZIP_DEFLATED)as z:
        for p in sources:z.write(p,str(p.relative_to(ROOT)))
    (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
