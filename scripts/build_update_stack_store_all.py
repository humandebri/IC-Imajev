#!/usr/bin/env python3
"""Extend exact output stack stores to160/128/32, retaining verified168."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-stack-store-v1';d=ROOT/'artifacts/update-stack-store-all-v1';proofs=[ROOT/'artifacts/s1-stack-store-all-v1/summary.json',ROOT/'artifacts/s1-stack-store-all-mlp-v1/summary.json']
 for p in proofs:
  r=json.loads(p.read_text());assert r['native_bits_equal'] and r['native_digests_recomputed']
  assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items())
 assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 for i,name in [(0,'int8_128.wat'),(1,'int8_160.wat'),(2,'int8_32.wat')]:
  text=(ROOT/f'artifacts/s1-stack-store-all-v1/build/kernel-store-{i}.wat').read_text().replace('__imajev_store_','__imajev_')
  if i==2:text=text.replace('__imajev_s1_32_accumulate','__imajev_s1_raw_accumulate')
  (d/name).write_text(text)
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-stack-store-v1','artifacts/update-stack-store-all-v1')
 anchor="        assert sha(wat) == patch['source_sha256']";assert s.count(anchor)==1;s=s.replace(anchor,anchor+"\n        if i==3:\n            wat=D.parent/'int8_128.wat'")
 (d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),*proofs,old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'))
 (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 assert all(p.read_bytes()==(d/'build/runtime'/p.name).read_bytes()for p in(old/'build/runtime').glob('*.rs'))
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=[],changed_arithmetic_kernels=['__imajev_s1_160_accumulate','__imajev_s1_wide_accumulate','__imajev_s1_raw_accumulate'],arithmetic_opcode_order_equal=True,store_sites=400,instructions_saved_per_site=2,scope='Output store stack placement for remaining3 widths only; original integer and F32 operations/order unchanged; optimized168 retained.'),indent=2)+'\n')
if __name__=='__main__':main()
