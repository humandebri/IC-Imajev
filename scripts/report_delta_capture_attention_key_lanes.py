#!/usr/bin/env python3
"""Audit latest runtime dense capture and independent saved-carry hidden30."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d=ROOT/'artifacts/delta-capture-attention-key-lanes-v1'
    source=(ROOT/'scripts/report_delta_capture.py').read_text().replace('artifacts/delta-capture-v2','artifacts/delta-capture-attention-key-lanes-v1').replace('artifacts/update-quantize-cached-v1/build','artifacts/update-attention-key-lanes-v1/build')
    source=source.replace("==10","==22").replace('ten_kernel_sources_equal=True','all22_kernel_sources_equal=True')
    (d/'frozen-reporter.py').write_text(source)
    exec(compile(source,str(d/'frozen-reporter.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
    summary=json.loads((d/'summary.json').read_text())
    proof=json.loads((d/'proof/report.json').read_text())
    audit=json.loads((d/'source-audit.json').read_text())
    assert audit['complete'] and audit['all22_kernel_sources_equal']
    host=ROOT/'artifacts/terminal-mlp-reference-v1'
    host_summary=json.loads((host/'summary.json').read_text())
    assert host_summary['complete'] and host_summary['all_32_hidden_verified']
    assert all(sha(ROOT/p)==h for p,h in host_summary['workflow_hashes'].items())
    rows=json.loads((host/'reconstruction/report.json').read_text())['results']
    hidden30=[]
    paths=[Path(__file__),ROOT/'scripts/report_delta_capture.py',d/'frozen-reporter.py',d/'source-audit.json',d/'proof-entry-hashes.json',host/'summary.json',host/'reconstruction/report.json']
    for case in proof['cases']:
        name=case['case'];bank='common' if name=='653' else 'voting'
        progress=json.loads((d/'proof'/bank/name/'stage-64.json').read_text())['ok']['progress']
        row=next(v for v in rows if v['case']==name and v['layer']==30)
        reference=ROOT/row['output'];assert sha(reference)==row['output_sha256']
        offset=0 if name=='653' else 11
        values=np.load(reference,allow_pickle=False)[offset:].astype('<f4').tobytes()
        assert hashlib.sha256(values).hexdigest()==progress['hidden_hashes'][30]
        hidden30.append(dict(case=name,layer=30,bit_equal=True,suffix_offset=offset))
        paths += [reference,d/'proof'/bank/name/'stage-64.json']
    summary.update(all32_hidden_verified=True,hidden30=hidden30,
                   source_audit=audit,paid_module=audit['paid_module'],
                   scope='Latest paid arithmetic counterpart with identical22 kernels and owned vector transitions. One stage per update enables direct dense capture. Payment/stage-grouping proof is separate. All72 dense states and pre-BF outputs are independently checked; hidden30 uses the independent saved-carry reference. No performance claim.')
    summary['workflow_hashes'].update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    (d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    archive=d/'frozen-latest-workflow.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in [ROOT/name for name in summary['workflow_hashes']]+[d/'summary.json']:
            z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        assert z.read(str((d/'summary.json').relative_to(ROOT)))==(d/'summary.json').read_bytes()
    print(json.dumps({'complete':True,'all32_hidden_verified':True,'dense_values':summary['dense_state_values_checked'],'baseline_restored':summary['baseline_restored']}))


if __name__=='__main__':
    main()
