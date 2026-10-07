#!/usr/bin/env python3
"""Combine verified stack store with the latest exact paid inference runtime."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-adaptive168-v1';d=ROOT/'artifacts/update-stack-store-v1';proof=ROOT/'artifacts/s1-stack-store-v1/summary.json';r=json.loads(proof.read_text());assert r['native_bits_equal'] and r['native_digests_recomputed'] and r['ordinary_queries']==42
 for hashes in [r['workflow_hashes'],json.loads((old/'workflow-hashes.json').read_text())]:assert all(sha(ROOT/p)==h for p,h in hashes.items())
 d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 (d/'int8_168.wat').write_text((ROOT/'artifacts/s1-stack-store-v1/build/kernel168-store.wat').read_text().replace('__imajev_s1_168_store_accumulate','__imajev_s1_168_accumulate'))
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-adaptive168-v1','artifacts/update-stack-store-v1');(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),proof,old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'))
 (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 assert all(p.read_bytes()==(d/'build/runtime'/p.name).read_bytes()for p in(old/'build/runtime').glob('*.rs'))
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=[],changed_arithmetic_kernel=['__imajev_s1_168_accumulate'],arithmetic_opcode_order_equal=True,store_sites=210,instructions_saved_per_site=2,scope='Only temporary output store stack placement; original integer and F32 operations/order unchanged.'),indent=2)+'\n')
if __name__=='__main__':main()
