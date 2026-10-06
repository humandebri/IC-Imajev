#!/usr/bin/env python3
"""Compare original/grouped F32 kernels in one dedicated local Wasm canister."""
import argparse,hashlib,json,pathlib,subprocess,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'client'))
from transport import decode

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--directory',default='artifacts/matrix-tail/check');a=ap.parse_args()
    d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
    helper=ROOT/'artifacts/matrix-tail/native-target/release/pair_args'
    wasm=ROOT/'artifacts/matrix-tail/target/wasm32-unknown-unknown/release/imajev_pair_bench.wasm'
    sha=lambda b:hashlib.sha256(b).hexdigest()
    files=[*sorted((ROOT/'crates/imajev-runtime/src').glob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml'],ROOT/'crates/imajev-runtime/Cargo.toml',ROOT/'crates/imajev-runtime/tests/matrix_tail.rs',*sorted((ROOT/'scripts/pair_bench/src').glob('*.rs')),ROOT/'scripts/pair_bench/src/bin/pair_args.rs',ROOT/'scripts/pair_bench/Cargo.toml',ROOT/'scripts/pair_bench/Cargo.lock',pathlib.Path(__file__)]
    hashes={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files}
    def native(*args):return json.loads(subprocess.check_output([str(helper),*map(str,args)],text=True,cwd=ROOT))
    def status():return json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--network','local','--identity','imajev-local','--json'],text=True,cwd=ROOT))['module_hash'].removeprefix('0x')
    expected=sha(wasm.read_bytes());assert status()==expected
    calls=[]
    def call(method,arg,query=False):
        cmd=['icp','canister','call',a.canister,method,'--network','local','--identity','imajev-local','--output','hex','--args-file',str(arg),'--args-format','bin']
        if query:cmd+=['--query']
        begin=time.monotonic();raw=subprocess.check_output(cmd,text=True,cwd=ROOT)
        reply=d/f'call-{len(calls):04d}.hex';reply.write_text(raw)
        result=native('decode',reply,'measurement' if query else 'preparation')
        result.update(wall_seconds=time.monotonic()-begin,request_bytes=arg.stat().st_size,reply_bytes=len(bytes.fromhex(raw.strip().removeprefix('0x'))))
        calls.append(dict(method=method,query=query,**result));return result
    manifest=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());tensors={t['name']:t for t in manifest['tensors']}
    root='model.language_model.layers.3.mlp.'
    cases=[];sources={};inputs={};axes={}
    for label in ['prefix','617','insufficient','maximum','normal']:
        src=ROOT/f'artifacts/column16-token48-v1-{label}';r=json.loads((src/'report.json').read_text())
        kv=next(q for q in r['queries'] if q['op']=='attention_kv_integer' and q['tensor']=='model.language_model.layers.3.self_attn.k_proj.weight')
        down=next(q for q in r['queries'] if q['op']=='lora_integer' and q['tensor']==root+'down_proj.weight')
        inputs[label]={}
        for kind,q in [('gate',kv),('down',down)]:
            path=src/'queries'/f'{q["index"]:06d}.request.bin';h,x=decode(path.read_bytes());n=h['dims'][0];cols=2560 if kind=='gate' else 9216
            assert len(x)==n*cols
            inputs[label][kind]=x.reshape(n,cols)
            sources[f'{label}-{kind}']=dict(path=str(path.relative_to(ROOT)),sha256=sha(path.read_bytes()),tokens=n)
    for kind in ['gate','down']:
        for side in ['A','B']:
            name=root+('gate_proj' if kind=='gate' else 'down_proj')+f'.lora_{side}.weight';t=tensors[name]
            assert t['dtype']=='f32' and t['bytes']==t['rows']*t['cols']*4
            with (ROOT/'checkpoints/full-int8.pack').open('rb') as f:f.seek(t['offset']);weights=f.read(t['bytes'])
            assert len(weights)==t['bytes'];wp=d/f'{kind}-{side}.weight.bin';wp.write_bytes(weights)
            arg=d/f'{kind}-{side}.begin.bin';subprocess.run([str(helper),'matrix_begin',str(t['rows']),str(t['cols']),str(arg)],check=True);call('matrix_begin',arg)
            for offset in range(0,len(weights),262144):
                chunk=d/f'{kind}-{side}-{offset}.weight.bin';chunk.write_bytes(weights[offset:offset+262144]);arg=d/f'{kind}-{side}-{offset}.args.bin'
                subprocess.run([str(helper),'chunk',str(offset//4),str(chunk),str(arg)],check=True);call('matrix_chunk',arg)
            for label in ['prefix','617','insufficient','maximum','normal']:
                x=inputs[label][kind] if side=='A' else axes[(label,kind)]
                n=len(x);rows=4096 if n>109 and t['rows']>4096 else t['rows']
                effective=wp
                if rows!=t['rows']:
                    effective=d/f'{kind}-{side}-{rows}.weight.bin';effective.write_bytes(weights[:rows*t['cols']*4])
                inp=d/f'{label}-{kind}-{side}.input.f32.bin';inp.write_bytes(x.astype('<f4').tobytes())
                ax=d/f'{label}-{kind}.ax.f32.bin'
                oracle=native('matrix_ax' if side=='A' else 'matrix_native',n,rows,t['cols'],effective,inp,*([ax] if side=='A' else []))
                if side=='A':axes[(label,kind)]=np.frombuffer(ax.read_bytes(),dtype='<f4').reshape(n,rows).copy()
                bf16=side=='A';payload=d/f'{label}-{kind}-{side}.input.bin'
                if bf16:
                    bits=x.astype('<f4').view('<u4');assert not np.any(bits&65535);payload.write_bytes((bits>>16).astype('<u2').tobytes())
                else:payload.write_bytes(inp.read_bytes())
                measured={}
                for candidate in [False,True]:
                    arg=d/f'{label}-{kind}-{side}-{candidate}.args.bin';subprocess.run([str(helper),'matrix_query',str(payload),str(n),str(bf16).lower(),str(candidate).lower(),str(arg)],check=True)
                    result=call('project_matrix',arg,True);assert result['digest']==oracle['digest'];measured['candidate' if candidate else 'baseline']=result
                row=dict(label=label,kind=kind,side=side,tokens=n,rows=rows,cols=t['cols'],tensor=name,weight_sha256=sha(weights),input_sha256=sha(inp.read_bytes()),oracle=oracle,measurements=measured)
                row['project_change_percent']=100*(measured['candidate']['project_instructions']/measured['baseline']['project_instructions']-1)
                cases.append(row);print(json.dumps(row),flush=True)
    assert status()==expected
    assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files}
    result=dict(scope='Pure F32 LoRA A/B kernels on saved representative inputs; no full graph or query-count improvement. gate A uses saved normalized Attention input. normal down uses the actual first96-token down chunk; gate B normal uses4096 output rows to preserve900K bound.',canister=a.canister,wasm_sha256=expected,model=manifest['model'],pack_hash=manifest['pack_hash'],sources=sources,source_hashes=hashes,cases=cases,ordinary_queries=sum(c['query'] for c in calls),preparation_updates=sum(not c['query'] for c in calls),module_status_update_reads=2,calls=calls)
    (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
