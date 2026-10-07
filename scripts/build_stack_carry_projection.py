#!/usr/bin/env python3
"""Directly replace12 bodies in both validated parents; preserve every source."""
import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    kernels=ROOT/'artifacts/stack-carry-projection-kernels-v1';audit=json.loads((kernels/'independent-audit.json').read_text());assert audit['all312_fresh_full_memory_cases_exact'] and audit['all3072_intervals_protect_held_stack_value']
    for p,h in audit['source_hashes'].items():assert sha(ROOT/p)==h,p
    kr=json.loads((kernels/'report.json').read_text());changes={k['symbol']:k for k in kr['kernels']}
    built={}
    for mode in ['update','paid']:
        old=ROOT/'artifacts'/f'{mode}-local-tee-projection-v1';parent=old/'build';base=json.loads((parent/'report.json').read_text());assert sha(parent/'full.wasm')==base['wasm_sha256']
        for group in ['source_hashes','dependency_hashes']:
            for p,h in base[group].items():assert sha(ROOT/p)==h,p
        d=ROOT/'artifacts'/f'{mode}-stack-carry-projection-v1';b=d/'build';b.mkdir(parents=True,exist_ok=False)
        for p in parent.glob('*.rs'):shutil.copyfile(p,b/p.name)
        if mode=='update':
            shutil.copytree(parent/'runtime',b/'runtime');shutil.copyfile(parent/'libimajev_runtime.rlib',b/'libimajev_runtime.rlib')
        else:shutil.copyfile(old/'optimized-paid-scheduler.rs',d/'optimized-paid-scheduler.rs')
        previous=parent/'full.wasm';patches=[];direct=[]
        for entry in base['patches']:
            if entry['export']in changes:
                k=changes[entry['export']];assert entry['source_sha256']==k['source_sha256']
                target=b/f'patched{len(direct)}.wasm'
                row=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(previous),str(ROOT/k['path']),str(target),entry['export']],text=True));assert row['wasmparser_validation']
                assert row['function_index']==entry['function_index'];patches.append(row);direct.append(row);previous=target
            else:patches.append(entry)
        assert len(direct)==12 and len(patches)==34;shutil.copyfile(previous,b/'full.wasm')
        report=dict(base);report.update(wasm_sha256=sha(b/'full.wasm'),patches=patches,parent=base['wasm_sha256'],changed_runtime_files=[],direct_patch_only=True,direct_patches=direct,scope='Direct twelve-WAT stack-carry patch of312-case validated parent; full paid parent measurement running; Rust sources unchanged. Compilation commands retained solely as parent provenance. Full paid/fidelity/IC proof pending; not adopted.')
        files=[Path(__file__),parent/'report.json',parent/'full.wasm',kernels/'report.json',kernels/'independent-audit.json']+list(b.glob('*.rs'))
        if mode=='update':files+=list((b/'runtime').glob('*.rs'))+[b/'libimajev_runtime.rlib']
        else:files+=[d/'optimized-paid-scheduler.rs']
        hashes=dict(base['source_hashes']);hashes.update({str(p.relative_to(ROOT)):sha(p)for p in files});hashes.update({k['path']:sha(ROOT/k['path'])for k in kr['kernels']});report['source_hashes']=hashes
        if mode=='paid':report['update_candidate']=built['update']['wasm_sha256']
        (b/'report.json').write_text(json.dumps(report,indent=2)+'\n');built[mode]=report
        (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),parent/'report.json',b/'report.json',kernels/'independent-audit.json']},indent=2)+'\n')
        if mode=='paid':
            for name in ['lib.rs','update_inference.rs','paid_types.rs','paid_inference.rs']:assert (b/name).read_bytes()==(parent/name).read_bytes()
            (d/'paid-source-equivalence.json').write_text(json.dumps(dict(complete=True,paid_contract_and_scheduler_and_lib_byteexact=True,module=report['wasm_sha256'],full_fidelity_and_instruction_proof_pending=True),indent=2)+'\n')
        print(json.dumps(dict(mode=mode,module=report['wasm_sha256'],direct_patches=12,total_projection_patches=34)),flush=True)
if __name__=='__main__':main()
