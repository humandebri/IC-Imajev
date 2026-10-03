#!/usr/bin/env python3
"""Compare two ordinary LoRA tiles with capture/reuse, never install modules."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport, encode, decode
from prefix_inference import verify_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--directory', required=True)
    ap.add_argument('--canister')
    ap.add_argument('--wasm')
    ap.add_argument('--pack-only', action='store_true')
    ap.add_argument('--native', default='artifacts/projection-reuse/candidate-primitive')
    ap.add_argument('--verify-malformed', action='store_true')
    ap.add_argument('--compact-state', action='store_true')
    ap.add_argument('--mlp', action='store_true', help='Compare gate/up fusion with two cached A products')
    a = ap.parse_args()
    if a.mlp and not a.compact_state: ap.error('MLP reuse requires compact-state')
    ordinary_op='mlp_gate_up_integer' if a.mlp else 'lora_integer'
    capture_op='mlp_gate_up_capture' if a.mlp else 'lora_integer_capture'
    reuse_op='mlp_gate_up_reuse' if a.mlp else 'lora_integer_reuse'
    if bool(a.canister) != bool(a.wasm): ap.error('canister and wasm must be provided together')
    dest = ROOT / a.directory
    dest.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'artifacts/prepared-f32-v1-normal'
    graph = json.loads((source / 'report.json').read_text())
    query = next(q for q in graph['queries'] if q['op'] == ordinary_op)
    original = source / 'queries' / f"{query['index']:06d}.request.bin"
    header, values = decode(original.read_bytes())
    values = values.reshape(header['dims'][0], header['dims'][2])
    full = json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_text())
    names = [header['tensor']] + header['aux']
    if a.mlp:
        names=[]
        for base in [header['tensor'],header['aux'][0]]:
            prefix=base.removesuffix('.weight');names += [base,prefix+'.lora_A.weight',prefix+'.lora_B.weight']
    pack_dir = ROOT / ('artifacts/mlp-reuse/pack' if a.mlp else 'artifacts/projection-reuse/pack')
    pack_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, pack_path = pack_dir / 'manifest.json', pack_dir / 'pack.bin'
    if not manifest_path.exists():
        if pack_path.exists(): raise ValueError('Partial preparation exists; inspect it before retrying')
        tensors, pieces = [], []
        with (ROOT / 'checkpoints/full-int8.pack').open('rb') as source_pack, pack_path.open('xb') as output:
            for name in names:
                t = next(t for t in full['tensors'] if t['name'] == name)
                source_pack.seek(t['offset']); raw = source_pack.read(t['bytes'])
                assert len(raw) == t['bytes']
                pieces.append(dict(name=name, source_offset=t['offset'], sha256=hashlib.sha256(raw).hexdigest()))
                tensors.append(dict(t, offset=output.tell()))
                output.write(raw)
        manifest_path.write_text(json.dumps(dict(version=1, model=full['model'],
            pack_hash=hashlib.sha256(pack_path.read_bytes()).hexdigest(), bytes=pack_path.stat().st_size,
            tensors=tensors, source_pack_hash=full['pack_hash'], source_pieces=pieces,
            scope=('Two original INT8 gate/up projections and both original F32 LoRA A/B pairs' if a.mlp else 'One original INT8 projection and original F32 LoRA')+'; no new quantization'), indent=2)+'\n')
    m = json.loads(manifest_path.read_text())
    assert m['pack_hash'] == hashlib.sha256(pack_path.read_bytes()).hexdigest()
    assert m['model'] == full['model'] and m['source_pack_hash'] == full['pack_hash']
    assert [t['name'] for t in m['tensors']] == names
    if a.pack_only:
        print(json.dumps(dict(pack=str(pack_path), manifest=str(manifest_path), bytes=m['bytes']))); return
    candidate = ROOT / a.native
    oracle = ROOT / 'artifacts/strassen-wide/default-primitive'
    rows, cols = m['tensors'][0]['rows'], m['tensors'][0]['cols']
    rank = m['tensors'][1]['rows']
    sha = hashlib.sha256((ROOT / a.wasm).read_bytes()).hexdigest() if a.wasm else None
    t = Transport(m['model'], 'http://localhost:8001/', a.canister,
        str(ROOT / 'artifacts/imajev-local.pem'), dest, m['pack_hash']) if a.canister else None
    cases, updates, rejected = [], [], []
    def run(h, x, label, native):
        request, response = dest / f'{label}.request.bin', dest / f'{label}.response.bin'
        request.write_bytes(encode(h, x))
        if native:
            subprocess.run([str(native),str(request),str(response),str(manifest_path),str(pack_path)],check=True)
            result = None
        else:
            result = t.command(dict(op='step', input=str(request), output=str(response)))
            assert 'ok' in result, result
        raw=response.read_bytes();returned,output=decode(raw)
        if returned.get('encoding')=='projection-block256-exact-v1':
            assert encode(returned,output)==raw, 'Rust/Python compact frame canonical bytes differ'
        return output, result
    try:
        if t:
            verify_module(t,sha)
            status=t.command(dict(op='pack_status'))['ok']
            assert status['ready'] and status['pack_hash']==m['pack_hash'] and status['bytes']==status['received']==status['hashed']==m['bytes']
        for phase in (['cold','warm','cleared'] if t else ['native']):
            if t:
                if phase=='warm':
                    for name in names: updates.append(t.command(dict(op='warm_weights',name=name)))
                else: updates.append(t.command(dict(op='clear_weight_cache')))
                cache=t.command(dict(op='weight_cache_status'))['ok']['cache']
                assert set(cache['names'])==(set(names) if phase=='warm' else set())
                assert cache['bytes']==(m['bytes'] if phase=='warm' else 0)
            for n in [1,7,8,32,64,87,132]:
                # Both variants use the same two row boundaries; changed row
                # tiling does not alter per-row arithmetic or token padding.
                state_len=n*(cols+cols//256+rank*(2 if a.mlp else 1))
                first=min(rows//2//8*8,(900000-state_len)//n//8*8)
                assert 0<first<rows and n*(rows-first)<=900000
                outputs, measurements = {}, {}
                for row,width,label in [(0,first,'first'),(first,rows-first,'second')]:
                    h=dict(header,pack_hash=m['pack_hash'],dims=[n,width,cols,row],op=ordinary_op)
                    expected,_=run(h,values[:n],f'{phase}-{n}-{label}-oracle',oracle)
                    ordinary,result=run(h,values[:n],f'{phase}-{n}-{label}-ordinary',None if t else candidate)
                    assert np.array_equal(ordinary.view(np.uint32),expected.view(np.uint32))
                    outputs[label]=expected
                    measurements[label]=result
                h=dict(header,pack_hash=m['pack_hash'],dims=[n,first,cols,0],op=capture_op)
                if a.compact_state:h=dict(h,encoding='projection-block256-exact-v1',dims=h['dims']+[rank])
                captured,result=run(h,values[:n],f'{phase}-{n}-capture',None if t else candidate)
                assert len(captured)==n*first+state_len
                assert np.array_equal(captured[:n*first].view(np.uint32),outputs['first'].view(np.uint32))
                measurements['capture']=result
                h=dict(h,op=reuse_op,dims=[n,rows-first,cols,first]+([rank] if a.compact_state else []))
                reused,result=run(h,captured[n*first:],f'{phase}-{n}-reuse',None if t else candidate)
                assert np.array_equal(reused.view(np.uint32),outputs['second'].view(np.uint32))
                measurements['reuse']=result
                if t and a.verify_malformed and phase=='warm' and n==132:
                    if a.compact_state:
                        import struct
                        good=encode(h,captured[n*first:]);header_len,=struct.unpack('<I',good[:4]);payload_start=4+header_len
                        mutations=[]
                        for label,index,replacement in [('reserved',payload_start+1,b'\x80'),('scale',payload_start+1+n*cols,bytes(4)),('tag',payload_start,b'\x03')]:
                            raw=bytearray(good[:-32]);raw[index:index+len(replacement)]=replacement
                            mutations.append((label,bytes(raw)+hashlib.sha256(raw).digest()))
                        raw=good[:-33];mutations.append(('truncated',raw+hashlib.sha256(raw).digest()))
                        if a.mlp:
                            for which in [0,1]:
                                index=payload_start+1+n*cols+4*n*(cols//256)+which*4*n*rank
                                raw=bytearray(good[:-32]);raw[index:index+4]=struct.pack('<I',0x7fc00000)
                                mutations.append((f'ax-{which}',bytes(raw)+hashlib.sha256(raw).digest()))
                        for label,raw in mutations:
                            request=dest/f'bad-{label}.request.bin';response=dest/f'bad-{label}.response.bin';request.write_bytes(raw)
                            try:t.command(dict(op='step',input=str(request),output=str(response)))
                            except RuntimeError as error:
                                assert ('invalid activation' if label.startswith('ax-') else 'projection codec') in str(error), str(error)
                                rejected.append(dict(case=label,error=str(error)))
                            else:raise AssertionError('malformed compact frame accepted')
                if t and a.verify_malformed and not a.compact_state and phase=='warm' and n==132:
                    for label,index,value in [('low',0,-128.),('high',0,128.),('fraction',0,0.5),('scale',n*cols,0.)]:
                        changed=captured[n*first:].copy();changed[index]=value
                        request=dest/f'bad-{label}.request.bin';response=dest/f'bad-{label}.response.bin'
                        request.write_bytes(encode(h,changed))
                        try:
                            failed=t.command(dict(op='step',input=str(request),output=str(response)))
                        except RuntimeError as error:
                            assert 'prepared activation wire bounds' in str(error), str(error)
                            rejected.append(dict(case=label,error=str(error)))
                        else:
                            # The native client returns canister Result::Err as
                            # an error object rather than a replica rejection.
                            assert 'error' in failed and 'prepared activation wire bounds' in str(failed), failed
                            rejected.append(dict(case=label,error=failed))
                case=dict(phase=phase,tokens=n,rows=[first,rows-first],bitwise_equal=True,results=measurements)
                if t:
                    case['ordinary_instructions']=sum(measurements[k]['ok']['instructions'] for k in ['first','second'])
                    case['reuse_instructions']=sum(measurements[k]['ok']['instructions'] for k in ['capture','reuse'])
                    case['instruction_reduction']=1-case['reuse_instructions']/case['ordinary_instructions']
                    for variant,keys in [('ordinary',['first','second']),('reuse',['capture','reuse'])]:
                        case[variant+'_communication_bytes']=sum(measurements[k]['ok']['request_bytes']+measurements[k]['ok']['reply_bytes'] for k in keys)
                cases.append(case);print(json.dumps(case),flush=True)
        if t: verify_module(t,sha)
        (dest/'report.json').write_text(json.dumps(dict(cases=cases,updates=updates,rejected_states=rejected,wasm_sha256=sha,
            oracle_sha256=hashlib.sha256(oracle.read_bytes()).hexdigest(),
            candidate_native_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            source_request_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),pack_hash=m['pack_hash'],
            inference_queries=len(cases)*4 if t else 0,
            rejected_query_attempts=len(rejected),
            compact_state=a.compact_state, mlp=a.mlp,
            scope=('One real gate/up fusion with both original A products' if a.mlp else 'One real LoRA projection')+', seven input lengths, two row tiles; no full-model query or accuracy claim'),indent=2)+'\n')
    finally:
        if t:t.close()


if __name__=='__main__':main()
