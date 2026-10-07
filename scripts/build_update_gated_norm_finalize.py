#!/usr/bin/env python3
"""Reuse the RMS row buffer and borrow the raw-F32 SiLU table once per gate row."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
    old=ROOT/'artifacts/update-attention-key-lanes-v1';probe=ROOT/'artifacts/gated-norm-finalize-v1'
    check=json.loads((probe/'check/report.json').read_text());assert check['all_bits_equal']
    d=ROOT/'artifacts/update-gated-norm-finalize-v1';d.mkdir(exist_ok=False)
    for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
    helper=(ROOT/'scripts/gated_norm_finalize.rs').read_text().replace('gate:Vec<f32>','gate:&[f32]').replace('.zip(gate)', '.zip(gate.iter().copied())')
    (d/'gated_norm_finalize.rs').write_text(helper)
    s=(old/'frozen-builder.py').read_text().replace('artifacts/update-attention-key-lanes-v1','artifacts/update-gated-norm-finalize-v1')
    before='''                let values = rms(row, w, r.scalars[0]);
                for (j, v) in values.into_iter().enumerate() {'''
    after='''                let values = rms(row, w, r.scalars[0]);
                if r.op == "gated_norm" {
                    let start=d[0]*d[1]+i*d[1];
                    result.extend(gated_norm_owned(values,&x[start..start+d[1]]));
                    continue;
                }
                for (j, v) in values.into_iter().enumerate() {'''
    patch="    p=D/'runtime/lib.rs';text=p.read_text();before="+repr(before)+";after="+repr(after)+"\n    assert text.count(before)==1;text=text.replace(before,after)\n    p.write_text(text+(D.parent/'gated_norm_finalize.rs').read_text())\n"
    anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
    (d/'runtime-edits.json').write_text(json.dumps([dict(before=before,after=after)],indent=2)+'\n');(d/'frozen-builder.py').write_text(s)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    files=[Path(__file__),old/'frozen-builder.py',old/'source-audit.json',probe/'check/report.json',ROOT/'scripts/gated_norm_finalize.rs',d/'runtime-edits.json',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'))
    (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
    exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
