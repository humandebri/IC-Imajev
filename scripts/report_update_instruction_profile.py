#!/usr/bin/env python3
"""Aggregate nonoverlapping frozen update spans and verify saved Candid diagnostics."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/update-instruction-profile-v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    proof = D/'proof-v1'
    manifest = json.loads((proof/'report.json').read_text())
    assert manifest['complete'] and manifest['baseline_snapshot_restored'] and manifest['snapshot_deleted']
    assert all(sha(ROOT/p) == h for p,h in manifest['source_hashes'].items())
    results = []
    for bank in manifest['banks']:
        measurement = bank['measurement']
        assert measurement['complete']
        assert all(sha(ROOT/p) == h for p,h in measurement['reference_hashes'].items())
        for case in measurement['cases']:
            counts = collections.Counter()
            operations = collections.Counter()
            total = 0
            files = sorted((proof/bank['bank']/'measurement'/f"{case['case']}-r{case['repeat']}").glob('*.json'))
            assert len(files) == case['update_calls']
            for p in files:
                row = json.loads(p.read_text())
                total += row['ok']['progress']['instructions']
                for name,instructions in row['ok']['progress']['operations']:
                    operations[name] += instructions
                raw = p.with_suffix('.profile.candid')
                assert sha(raw) == row['profile_reply_sha256']
                text = raw.read_text()
                decoded,_ = json.JSONDecoder().raw_decode(text[text.index('"'):])
                spans = json.loads(decoded)
                assert spans == row['instruction_profile']
                for name,instructions,calls in spans:
                    assert instructions >= 0 and calls > 0
                    counts[name] += instructions
            assert total == case['total_handler_instructions']
            base = counts['base_project_inclusive']
            lora = sum(v for k,v in counts.items() if k.startswith('lora_matmul_'))
            gqa = counts['gqa_head_views']
            assert base+lora+gqa <= total
            groups = dict(int8_base=base,f32_lora=lora,gqa_head=gqa,other=total-base-lora-gqa)
            results.append(dict(case=case['case'],tokens=case['tokens'],suffix_tokens=case['suffix_tokens'],update_calls=case['update_calls'],profile_handler_instructions=total,groups=groups,percent={k:100*v/total for k,v in groups.items()},spans=dict(counts),operations=dict(operations),operation_percent={k:100*v/total for k,v in operations.items()},exported_bits_equal=case['all_exported_hidden_and_32_state_hashes_equal'] and case['decision_equal'] and case['final_norm_equal'],nominal_target=100_000_000_000,required_total_reduction_percent=100*(1-100_000_000_000/total),required_base_only_reduction_percent=100*(total-100_000_000_000)/base))
    assert {r['case'] for r in results} == {'617','620','653'}
    assert all(r['exported_bits_equal'] for r in results)
    output = dict(complete=True,proof_sha256=sha(proof/'report.json'),candidate=manifest['candidate'],cases=results,scope='Instrumented equivalent update graph, production kernels preserved. Inclusive base spans counted once. Other includes Delta, norm, activation, hashes, copies, quantization, loads and scheduler. No production improvement claim.')
    (D/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
