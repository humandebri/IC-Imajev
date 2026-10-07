#!/usr/bin/env python3
"""Use measured ordered four-key scores only for the favorable full inference shapes."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old=ROOT/'artifacts/update-profile-off-v1'
    probe=ROOT/'artifacts/attention-key-lanes-bounded-v1'
    check=json.loads((probe/'check/report.json').read_text())
    assert check['complete'] and check['all_bits_equal'] and check['transpose_in_body']
    for hashes in [check['source_hashes'],check['evidence_hashes'],json.loads((old/'workflow-hashes.json').read_text())]:
        assert all(sha(ROOT/p)==h for p,h in hashes.items())
    d=ROOT/'artifacts/update-attention-key-lanes-v1';d.mkdir(exist_ok=False)
    for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
    helper=probe/'build/candidate.rs';(d/'attention_key_lanes.rs').write_bytes(helper.read_bytes())
    source=(old/'frozen-builder.py').read_text().replace('artifacts/update-profile-off-v1','artifacts/update-attention-key-lanes-v1')
    patch='''    (D/'runtime/attention_key_lanes.rs').write_bytes((D.parent/'attention_key_lanes.rs').read_bytes())
    p=D/'runtime/lib.rs';p.write_text(p.read_text()+'\\nmod attention_key_lanes;\\n')
    p=D/'runtime/attention_views.rs';text=p.read_text()
    before='    let divisor = (width as f32).sqrt();'
    assert text.count(before)==1
    text=text.replace(before,before+'\\n    let tiled=if n>=16 && width==256 && n+prefix<=132 {Some(crate::attention_key_lanes::scores(q,k,n,width,prefix))}else{None};\\n    let mut score_offset=0;')
    before='''+repr('''        for key in k[..(prefix + t + 1) * width].chunks_exact(width) {
            scores.push(bf(dot(query, key) / divisor));
        }''')+'''
    after='''+repr('''        if let Some(tiled)=tiled.as_ref() {
            let count=prefix+t+1;
            scores.extend_from_slice(&tiled[score_offset..score_offset+count]);
            score_offset+=count;
        }else{
            for key in k[..(prefix + t + 1) * width].chunks_exact(width) {
                scores.push(bf(dot(query, key) / divisor));
            }
        }''')+'''
    assert text.count(before)==1;text=text.replace(before,after);p.write_text(text)
'''
    anchor="    runtime = base['runtime_command'][:]";assert source.count(anchor)==1
    source=source.replace(anchor,patch+anchor);(d/'frozen-builder.py').write_text(source)
    files=[Path(__file__),old/'workflow-hashes.json',old/'frozen-builder.py',probe/'check/report.json',probe/'entry-hashes.json',helper,d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'))
    (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
    exec(compile(source,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
