#!/usr/bin/env python3
"""Same paid accounting and owned scheduler with audited K2 runtime."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'artifacts/paid-gated-norm-finalize-v1';normal=ROOT/'artifacts/update-k2-odd-roots-v1';d=ROOT/'artifacts/paid-k2-odd-roots-v1';d.mkdir(exist_ok=False)
 audit=json.loads((normal/'source-audit.json').read_text());assert audit['all_original_22_wat_sources_identical']and audit['native_bits_equal']
 (d/'optimized-paid-scheduler.rs').write_bytes((old/'optimized-paid-scheduler.rs').read_bytes())
 s=(old/'frozen-builder.py').read_text().replace('artifacts/paid-gated-norm-finalize-v1','artifacts/paid-k2-odd-roots-v1').replace('artifacts/update-gated-norm-finalize-v1','artifacts/update-k2-odd-roots-v1')
 a=s.index('  wat=ROOT/[');b=s.index('\n  assert sha(wat)',a)
 s=s[:a]+"  paths=list(B.parent.glob('*.wat'))+list(B.glob('*.wat'))+[ROOT/'artifacts/single-quad/build-v2/kernel0.wat',ROOT/'artifacts/single-quad/build-v2/kernel2.wat'];mapping={sha(p):p for p in paths};wat=mapping[patch['source_sha256']]"+s[b:]
 s=s.replace("assert len({p['function_index'] for p in patches})==22","assert len({p['function_index'] for p in patches})==30")
 (d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),old/'frozen-builder.py',normal/'source-audit.json',d/'optimized-paid-scheduler.rs',d/'frozen-builder.py'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
