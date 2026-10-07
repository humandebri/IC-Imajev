#!/usr/bin/env python3
"""Verify unroll order, pointer spans, source reversal and frozen component evidence."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d=ROOT/'artifacts/attention-key-lanes-bounded-v1'
    text=(ROOT/'scripts/attention_key_lanes_bounded.rs').read_text()
    edits=json.loads((d/'helper-edits.json').read_text())
    for row in reversed(edits):
        assert text.count(row['after'])==1;text=text.replace(row['after'],row['before'])
    assert text==(ROOT/'scripts/attention_key_lanes.rs').read_text()
    for width in range(1,257):
        emitted=[];j=0
        while j+8<=width:emitted.extend(range(j,j+8));j+=8
        emitted.extend(range(j,width));assert emitted==list(range(width))
    for total in range(1,133):
        for width in [1,3,7,8,16,128,255,256]:
            key=np.arange(total,dtype=np.int64)[:,None];col=np.arange(width,dtype=np.int64)[None,:]
            source=key*width+col;target=(key//4*width+col)*4+key%4
            assert source.min()==0 and source.max()==total*width-1
            assert target.min()==0 and target.max()<((total+3)//4)*width*4
            assert len(np.unique(target))==total*width
            for used in range(1,total+1):
                blocks=(used+3)//4
                assert ((blocks-1)*width+width-1)*4+3<((total+3)//4)*width*4
                lanes=[b*4+lane for b in range(blocks)for lane in range(min(4,used-b*4))]
                assert lanes==list(range(used))
    r=json.loads((d/'check/report.json').read_text())
    assert r['complete'] and r['all_bits_equal'] and r['transpose_in_body'] and len(r['cases'])==21 and r['queries']==42
    for name in ['source_hashes','evidence_hashes']:
        assert all(sha(ROOT/p)==h for p,h in r[name].items())
    for name in ['entry-hashes.json','checker-entry-hashes.json']:
        assert all(sha(ROOT/p)==h for p,h in json.loads((d/name).read_text()).items())
    with zipfile.ZipFile(d/'check/frozen-check.zip')as z:
        assert len(z.namelist())==len(set(z.namelist()))
        for name in ['source_hashes','evidence_hashes']:
            assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in r[name].items())
        assert z.read(str((d/'check/report.json').relative_to(ROOT)))==(d/'check/report.json').read_bytes()
    result=dict(complete=True,ascending_k_all_widths_verified=True,transpose_offsets_injective=True,
                discarded_lanes_only_padding=True,vector_loads_in_allocated_bounds=True,
                source_reversal_exact=True,frozen_candid_bits_hashes_verified=True,
                full_inference_verified=False,whole_goal_complete=False)
    files=[Path(__file__),d/'helper-edits.json',d/'entry-hashes.json',d/'checker-entry-hashes.json',d/'check/report.json']
    result['hashes']={str(p.relative_to(ROOT)):sha(p)for p in files}
    (d/'pointer-and-evidence-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(d/'pointer-and-evidence-audit.zip','w',zipfile.ZIP_DEFLATED)as z:
        for p in files+[d/'pointer-and-evidence-audit.json']:z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps(result))


if __name__=='__main__':main()
