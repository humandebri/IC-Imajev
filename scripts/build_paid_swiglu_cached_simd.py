#!/usr/bin/env python3
"""Build paid inference with exact owned-gate SwiGLU and unchanged billing/bound prefixes."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 upstream=ROOT/'artifacts/paid-rms-ordered-simd-v1';runtime=ROOT/'artifacts/update-swiglu-cached-simd-v1'
 for v in [upstream,runtime]:assert all(sha(ROOT/p)==h for p,h in json.loads((v/'workflow-hashes.json').read_text()).items())
 d=ROOT/'artifacts/paid-swiglu-cached-simd-v1';d.mkdir(exist_ok=False);scheduler=upstream/'optimized-paid-scheduler.rs';(d/scheduler.name).write_bytes(scheduler.read_bytes())
 s=(upstream/'frozen-builder.py').read_text().replace('artifacts/paid-rms-ordered-simd-v1','artifacts/paid-swiglu-cached-simd-v1').replace('artifacts/update-rms-ordered-simd-v1','artifacts/update-swiglu-cached-simd-v1');(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),upstream/'workflow-hashes.json',upstream/'frozen-builder.py',runtime/'workflow-hashes.json',scheduler,d/scheduler.name,d/'frozen-builder.py'];(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(upstream/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 b=d/'build';r=json.loads((b/'report.json').read_text());assert all((b/n).read_bytes()==(ROOT/'canisters/inference/src'/n).read_bytes()for n in ['paid_types.rs','paid_inference.rs']);assert (b/'update_inference.rs').read_bytes()==scheduler.read_bytes()
 r.update(canonical_billing_sources_byte_equal=True,bound_prefix_and_bulk_digest=True,raw_packets_discarded_after_validation=True,paid_execution_verified=False);(b/'report.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
