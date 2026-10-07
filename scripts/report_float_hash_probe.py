#!/usr/bin/env python3
"""Re-decode both hash replies and verify independent raw-byte SHA-256."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d=ROOT/'artifacts/float-hash-v1'
    r=json.loads((d/'check-v2/report.json').read_text())
    b=json.loads((d/'build/report.json').read_text())
    assert sha(d/'build/diagnostic.wasm')==b['module']==r['module']
    for group in ['source_hashes','dependency_hashes']:
        for p,h in b[group].items():assert sha(ROOT/p)==h,p
    assert r['source_sha256']==sha(ROOT/'scripts/check_float_hash_probe.py')
    helper=ROOT/'artifacts/f32-block-native/release/f32_args'
    assert sha(helper)==r['helper_sha256']
    for c in r['cases']:
        path=d/'check-v2'/(c['label']+'.input.bin')
        assert sha(path)==c['input_sha256']
        expected=list(hashlib.sha256(path.read_bytes()).digest())
        assert path.stat().st_size==4*c['values']
        for i,m in enumerate(c['measurements']):
            reply=d/'check-v2'/f'{c["label"]}-{i}.hex'
            assert sha(reply)==m['reply_sha256']
            decoded=json.loads(subprocess.check_output([str(helper),'decode',str(reply),'measurement'],text=True))
            assert decoded==m['measurement']
            assert decoded['digest']==expected and decoded['output_values']==c['values']
            assert decoded['total_instructions']==decoded['input_prepare_instructions']+decoded['project_instructions']
    files=[Path(__file__),ROOT/'scripts/build_float_hash_probe.py',ROOT/'scripts/check_float_hash_probe.py',
           d/'build/lib.rs',d/'build/report.json',d/'check-v2/report.json']
    result=dict(module=r['module'],raw_replies_verified=True,all_hashes_equal=True,cases=r['cases'],
                workflow_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},
                scope='Isolated SHA-256 cost; single-update byte view chosen only for slices of at least4096 floats. First check stopped on digest-list versus hex comparison; v2 preserves correct raw digests.')
    (d/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(d/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[d/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
    print('7 cases /14 queries; all byte hashes and saved replies verified')


if __name__=='__main__':main()
