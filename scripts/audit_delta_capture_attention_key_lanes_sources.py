#!/usr/bin/env python3
"""Reverse exactly the frozen capture edits and audit the latest paid counterpart."""
from pathlib import Path
import ast,hashlib,json,zipfile,textwrap
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d=ROOT/'artifacts/delta-capture-attention-key-lanes-v1'
    normal=ROOT/'artifacts/update-attention-key-lanes-v1'
    paid=ROOT/'artifacts/paid-attention-key-lanes-v1'
    builder=ROOT/'scripts/build_delta_capture_attention_key_lanes.py'
    tree=ast.parse(builder.read_text())
    injection=next(ast.literal_eval(node.value) for node in ast.walk(tree)
                   if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name)and t.id=='injection' for t in node.targets))
    tree=ast.parse(textwrap.dedent(injection))
    maps=[ast.literal_eval(node.value) for node in tree.body
          if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name)and t.id=='edits' for t in node.targets)]
    assert len(maps)==2 and [len(m)for m in maps]==[2,8]
    for p in (normal/'build/runtime').glob('*.rs'):
        text=(d/'build/runtime'/p.name).read_text()
        if p.name=='lib.rs':
            suffix='\npub mod delta_capture;\n';assert text.endswith(suffix);text=text[:-len(suffix)]
        elif p.name=='delta_full_log.rs':
            for before,after in maps[0].items():assert text.count(after)==1;text=text.replace(after,before)
        assert text==p.read_text(),p.name
    assert (d/'build/runtime/delta_capture.rs').read_bytes()==(ROOT/'scripts/delta_capture.rs').read_bytes()
    records=json.loads((d/'capture-scheduler-changes.json').read_text())
    text=(d/'build/update_inference.rs').read_text()
    suffix='\n#[ic_cdk::query]\nfn delta_capture_chunk(offset:u32,length:u32)->Result<Vec<u8>,String>{owner();imajev_runtime::delta_capture::chunk(offset,length)}\n'
    assert text.endswith(suffix);text=text[:-len(suffix)]
    for before,after in maps[1].items():
        assert text.count(after)==records[before]
        text=text.replace(after,before)
    assert text==(normal/'build/update_inference.rs').read_text()
    owned=json.loads((paid/'scheduler-comparison.json').read_text())['changes']
    assert {k:records[k]for k in list(maps[1])[:6]}==owned
    builds=[json.loads((p/'build/report.json').read_text())for p in [d,normal,paid]]
    assert all(len(b['patches'])==22 for b in builds)
    for i in range(22):
        assert len({b['patches'][i]['source_sha256']for b in builds})==1
        assert all(b['patches'][i]['wasmparser_validation']for b in builds)
    for b in builds:
        for name in ['source_hashes','dependency_hashes']:
            assert all(sha(ROOT/p)==h for p,h in b[name].items())
    assert json.loads((paid/'summary.json').read_text())['all_32_hidden_verified']
    result=dict(complete=True,all22_kernel_sources_equal=True,runtime_changes_only_capture_hooks=True,
                scheduler_reverse_to_normal_source_equal=True,owned_graph_changes_match_paid=True,
                one_stage_correctness_counterpart=True,paid_billing_stage_grouping_verified_separately=True,
                paid_module=builds[2]['wasm_sha256'],capture_module=builds[0]['wasm_sha256'],performance_claim=False)
    files=[Path(__file__),builder,d/'build/report.json',normal/'build/report.json',paid/'build/report.json',paid/'summary.json',paid/'scheduler-comparison.json',d/'capture-scheduler-changes.json']
    result['hashes']={str(p.relative_to(ROOT)):sha(p)for p in files}
    (d/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(d/'source-audit.zip','w',zipfile.ZIP_DEFLATED)as z:
        for p in files+[d/'source-audit.json']:z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps(dict(complete=True,capture_module=result['capture_module'])))


if __name__=='__main__':main()
