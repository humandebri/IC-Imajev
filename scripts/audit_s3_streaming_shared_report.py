#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'scripts/audit_s3_k2_prepared_chunked_report.py';d=ROOT/'artifacts/s3-streaming-shared-probe-v1'
 source=original.read_text().replace('s3-k2-prepared-chunked-probe-v1','s3-streaming-shared-probe-v1').replace('update-k2-odd-roots-v1','update-k2-pair-guard-fold-v1').replace('current_odd_roots_k2','current_guard_fold_k2').replace("len(b['patches'])==32","len(b['patches'])==31").replace("d/'frozen-evidence.zip','w'","d/'frozen-evidence.zip','x'")
 frozen=d/'frozen-auditor.py';assert not frozen.exists();frozen.write_text(source)
 (d/'auditor-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
