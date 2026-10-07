#!/usr/bin/env python3
"""Correct copy-count estimate to count valid output rows, not padded quartets."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'scripts/audit_s2_inline_roots_candidate.py';d=ROOT/'artifacts/s2-inline-roots-kernels-v1'
 source=original.read_text();before="(case['rows']//16)*4*8*((case['tokens']+3)//4)*(case['cols']//256)";assert source.count(before)==1
 source=source.replace(before,"(case['rows']//16)*8*case['tokens']*(case['cols']//256)").replace("d/'audit.json'","d/'audit-v2.json'").replace("d/'audit-evidence.zip'","d/'audit-v2-evidence.zip'")
 frozen=d/'frozen-auditor-v2.py';assert not frozen.exists();frozen.write_text(source)
 (d/'auditor-v2-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
