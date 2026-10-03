#!/usr/bin/env python3
"""Compare hash-only ordinary queries on explicit, dedicated local canisters."""
import argparse,hashlib,json,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--simd-canister',required=True);ap.add_argument('--portable-canister',required=True)
    ap.add_argument('--directory',required=True);ap.add_argument('--source',default='artifacts/mlp-reuse-v1-normal')
    args=ap.parse_args();dest=ROOT/args.directory;dest.mkdir(parents=True,exist_ok=True)
    helper=ROOT/'artifacts/hash-bench/target/release/hash_args'
    vectors_path=ROOT/'artifacts/hash-bench/official-vectors.json';vectors=json.loads(vectors_path.read_text())
    cases=[dict(name=f"vector-{v['input_len']}",data=bytes(i%251 for i in range(v['input_len'])),official_blake3=v['hash'][:64]) for v in vectors['cases']]
    source=ROOT/args.source;report_raw=(source/'report.json').read_bytes();graph=json.loads(report_raw)
    chosen=[]
    for op in ['lora_integer_capture','lora_integer_reuse','mlp_gate_up_capture','mlp_gate_up_reuse','delta_stage_bf16']:
        q=next(q for q in graph['queries'] if q['op']==op);chosen.append((f"{op}-{q['index']}",source/'queries'/f"{q['index']:06d}.request.bin"))
    largest=max(graph['queries'],key=lambda q:q['ok']['request_bytes'])
    chosen.append((f"largest-request-{largest['index']}",source/'queries'/f"{largest['index']:06d}.request.bin"))
    for name,path in chosen:
        raw=path.read_bytes();assert hashlib.sha256(raw[:-32]).digest()==raw[-32:]
        cases.append(dict(name=name,data=raw[:-32],source_file=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(raw).hexdigest()))
    cases.append(dict(name='maximum-hash-input',data=bytes(range(251))*(2_000_000//251)+bytes(range(2_000_000%251))))
    targets=dict(simd=args.simd_canister,portable=args.portable_canister);modules={}
    def status(cid):
        raw=subprocess.check_output(['icp','canister','status',cid,'--network','local','--identity','imajev-local','--json'],cwd=ROOT,text=True)
        return json.loads(raw)['module_hash'].removeprefix('0x')
    for mode,cid in targets.items():
        expected=hashlib.sha256((ROOT/f'artifacts/hash-bench/{mode}.wasm').read_bytes()).hexdigest()
        assert status(cid)==expected;modules[mode]=expected
    results=[]
    for case in cases:
        data=case.pop('data');input_path=dest/f"{case['name']}.input.bin";arg_path=dest/f"{case['name']}.args.bin";input_path.write_bytes(data)
        expected=json.loads(subprocess.check_output([str(helper),'args',str(input_path),str(arg_path)],cwd=ROOT,text=True))
        assert expected['sha256']==hashlib.sha256(data).hexdigest()
        if 'official_blake3' in case:assert expected['blake3']==case['official_blake3']
        measurements={}
        for mode,cid in targets.items():
            measurements[mode]={}
            for algorithm in ['sha256','blake3']:
                start=time.monotonic()
                raw=subprocess.check_output(['icp','canister','call',cid,'hash_'+algorithm,'--network','local','--identity','imajev-local','--query','--args-file',str(arg_path),'--args-format','bin','--output','hex'],cwd=ROOT,text=True)
                wall=time.monotonic()-start;reply_path=dest/f"{case['name']}.{mode}.{algorithm}.hex";reply_path.write_text(raw)
                got=json.loads(subprocess.check_output([str(helper),'decode',str(reply_path)],cwd=ROOT,text=True))
                assert got['digest']==expected[algorithm] and got['input_bytes']==len(data)
                got.update(wall_seconds=wall,request_bytes=arg_path.stat().st_size);measurements[mode][algorithm]=got
        result=dict(**case,input_bytes=len(data),input_sha256=hashlib.sha256(data).hexdigest(),measurements=measurements)
        results.append(result);print(json.dumps(dict(name=case['name'],input_bytes=len(data),instructions={m:{a:v['instructions'] for a,v in d.items()} for m,d in measurements.items()})),flush=True)
    for mode,cid in targets.items():assert status(cid)==modules[mode]
    result=dict(scope='Hash-only handler counter; not full inference, codec, query-count, speed or accuracy improvement',cases=results,canisters=targets,wasm_sha256=modules,
        official_vector_url='https://raw.githubusercontent.com/BLAKE3-team/BLAKE3/1.8.2/test_vectors/test_vectors.json',official_vectors_sha256=hashlib.sha256(vectors_path.read_bytes()).hexdigest(),
        source_report_sha256=hashlib.sha256(report_raw).hexdigest(),source_model=graph['model'],source_pack_hash=graph['pack_hash'],
        measured_queries=len(cases)*4,status_update_reads=4,compiler='rustc1.97.1',blake3_version='1.8.2',sha2_version='0.10.9',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pathlib.Path(__file__),ROOT/'scripts/hash_bench/Cargo.toml',ROOT/'scripts/hash_bench/Cargo.lock',ROOT/'scripts/hash_bench/src/lib.rs',ROOT/'scripts/hash_bench/src/bin/hash_args.rs']})
    (dest/'report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
