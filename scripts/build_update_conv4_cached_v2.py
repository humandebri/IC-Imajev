#!/usr/bin/env python3
"""Save the table accessor as source, preserving its nested builder boundaries."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 proof=ROOT/'artifacts/conv4-cached-v1/summary.json';r=json.loads(proof.read_text());assert r['raw_replies_verified'] and r['all_bits_equal'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items())
 p=ROOT/'scripts/build_update_conv4.py';s=p.read_text().replace('artifacts/update-conv4-v1','artifacts/update-conv4-cached-v2').replace('artifacts/conv4-simd-v3','artifacts/conv4-cached-v1')
 api=ROOT/'artifacts/conv4-cached-v1/build/runtime/prepared_activation.rs';text=api.read_text();text=text[text.index('pub fn with_table<const K:usize,R>'):];assert text.strip().endswith('}')
 anchor="(d/'conv4.rs').write_bytes(helper.read_bytes())";assert s.count(anchor)==1
 s=s.replace(anchor,anchor+"\n (d/'activation-table-api.rs').write_text("+repr(text)+")")
 anchor="    p.write_text(text+(D.parent/'conv4.rs').read_text())";assert s.count(anchor)==1
 s=s.replace(anchor,anchor+"\n    p=D/'runtime/prepared_activation.rs'\n    p.write_text(p.read_text()+(D.parent/'activation-table-api.rs').read_text())")
 anchor="d/'frozen-builder.py',d/'conv4.rs'";assert s.count(anchor)==1;s=s.replace(anchor,anchor+",d/'activation-table-api.rs'")
 entry=ROOT/'artifacts/update-conv4-cached-v2-entry.py';entry.write_text(s)
 (ROOT/'artifacts/update-conv4-cached-v2-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),p,api,proof,entry]},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
