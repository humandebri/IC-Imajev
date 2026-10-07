#!/usr/bin/env python3
"""Rank343 input encoder327 with unchanged651-node reconstruction and latest control."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = ROOT / 'scripts/build_s3_k2_prepared_probe.py'
    basis = ROOT / 'artifacts/rank343-integer-basis-v1'
    report = json.loads((basis / 'report.json').read_text())
    execution = json.loads((basis / 'execution-report.json').read_text())
    assert report['complete'] and report['all343_leaf_coefficients_unchanged'] and report['all64_final_product_expressions_unchanged']
    assert report['new_dag_nodes'][:2] == [327, 327] and execution['complete'] and execution['conditions'] == 52
    for record in [report, execution]:
        for path, expected in record['source_hashes'].items():
            assert sha(ROOT / path) == expected, path
    code = original.read_text().replace('artifacts/s3-k2-prepared-probe-v1', 'artifacts/s3-integer-basis-probe-v1')
    code = code.replace('artifacts/update-k2-odd-roots-v1', 'artifacts/update-common-raw-hybrid-v1')
    old = "ns={'__name__':'plan','__file__':str(kernels/'plan.py')};exec(compile((kernels/'plan.py').read_text(),str(kernels/'plan.py'),'exec'),ns);ad,bd,cd,leaves,roots,_=ns['plan']()"
    new = "basis=ROOT/'artifacts/rank343-integer-basis-v1';ns={'__name__':'plan','__file__':str(ROOT/'scripts/plan_rank343_integer_basis.py')};exec(compile((basis/'plan.py').read_text(),str(basis/'plan.py'),'exec'),ns);ad,bd,cd,leaves,roots,_=ns['plan']();assert len(ad.nodes)==327 and all(v in (-1,1)for _,terms in ad.nodes for v in terms.values())"
    assert code.count(old) == 1
    code = code.replace(old, new)
    old = "mapping={sha(p):p for p in list(normal.glob('*.wat'))+list((normal/'build').glob('*.wat'))+[ROOT/'artifacts/single-quad/build-v2/kernel0.wat',ROOT/'artifacts/single-quad/build-v2/kernel2.wat']}"
    new = "mapping={sha(ROOT/p):ROOT/p for p in nb['source_hashes'] if p.endswith('.wat')}"
    assert code.count(old) == 1
    code = code.replace(old, new)
    code = code.replace("files=[Path(__file__),", "files=[ROOT/'scripts/build_s3_integer_basis_probe.py',basis/'report.json',basis/'execution-report.json',basis/'plan.py',Path(__file__),")
    code = code.replace('current_odd_roots_k2_control=True', 'current_common_raw_hybrid_control=True,input_dag_nodes=327,reconstruction_kernel_unchanged=True')
    compile(code, str(original), 'exec')
    directory = ROOT / 'artifacts/s3-integer-basis-probe-entry-v1'
    directory.mkdir(exist_ok=False)
    frozen = directory / 'frozen-builder.py'
    frozen.write_text(code)
    files = [Path(__file__), original, basis / 'report.json', basis / 'execution-report.json', frozen]
    (directory / 'entry-hashes.json').write_text(json.dumps({str(path.relative_to(ROOT)): sha(path) for path in files}, indent=2) + '\n')
    exec(compile(code, str(frozen), 'exec'), dict(__file__=str(original), __name__='__main__'))


if __name__ == '__main__':
    main()
