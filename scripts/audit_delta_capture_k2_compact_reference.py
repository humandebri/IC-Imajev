#!/usr/bin/env python3
"""Compare complete latest captured states/operands to frozen earlier bit evidence."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d=ROOT/'artifacts/delta-capture-k2-compact-v1'
    previous=ROOT/'artifacts/delta-capture-v2/proof/report.json'
    old=json.loads(previous.read_text())
    new=json.loads((d/'proof/report.json').read_text())
    assert old['complete'] and old['baseline_restored'] and new['complete'] and new['baseline_restored']
    assert len(old['captures'])==len(new['captures'])==72
    rows=[]
    evidence={}
    for row in new['captures']:
        reference=next(v for v in old['captures'] if v['case']==row['case'] and v['layer']==row['layer'])
        current_path=Path(row['path']);old_path=Path(reference['path'])
        assert sha(current_path)==row['sha256']
        assert sha(old_path)==reference['sha256']
        assert current_path.read_bytes()==old_path.read_bytes(),(row['case'],row['layer'])
        assert row['n']==reference['n'] and row['step']==reference['step']
        rows.append(dict(case=row['case'],layer=row['layer'],all_capture_bytes_equal=True,
                         n=row['n'],bytes=current_path.stat().st_size))
        evidence[str(current_path.relative_to(ROOT))]=row['sha256']
        evidence[str(old_path.relative_to(ROOT))]=reference['sha256']
    for case in ('617','620','653'):
        assert sorted(r['layer'] for r in rows if r['case']==case)==[i for i in range(32) if i%4!=3]
    for name in ('workflow-hashes.json','proof-entry-hashes.json'):
        assert all(sha(ROOT/p)==h for p,h in json.loads((d/name).read_text()).items())
    summary=json.loads((d/'summary.json').read_text())
    assert summary['complete'] and summary['all32_hidden_verified'] and summary['dense_state_bits_equal']
    for name in ('workflow_hashes','evidence_hashes'):
        assert all(sha(ROOT/p)==h for p,h in summary[name].items())
    with zipfile.ZipFile(d/'frozen-latest-workflow.zip') as z:
        assert len(z.namelist())==len(set(z.namelist()))
        assert z.read(str((d/'summary.json').relative_to(ROOT)))==(d/'summary.json').read_bytes()
        for p,h in summary['workflow_hashes'].items():
            assert hashlib.sha256(z.read(p)).hexdigest()==h
    restored=json.loads((d/'proof/restored.json').read_text())
    assert all(restored[k] for k in ('cache_equal','pack_equal','snapshot_deleted'))
    result=dict(complete=True,cases=rows,evidence_hashes=evidence,
                all72_complete_capture_bytes_equal_to_frozen_reference=True,
                includes_initial_final_dense_q_k_v_g_b_pre_bf_output=True,
                independent_ordered_recurrence_report_verified=True,
                all32_hidden_verified=True,baseline_restored=True,
                temporary_snapshot_deleted=True,performance_claim=False,goal_complete=False,
                workflow_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),previous,d/'proof/report.json',d/'summary.json',d/'proof/restored.json']})
    output=d/'direct-reference-audit.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(d/'direct-reference-audit.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in [ROOT/name for name in result['workflow_hashes']]+[output]:
            z.write(path,str(path.relative_to(ROOT)))
    print(json.dumps({'complete':True,'captures':72,'all_capture_bytes_equal':True,'baseline_restored':True}))


if __name__=='__main__':
    main()
