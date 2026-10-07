#!/usr/bin/env python3
"""Audit fresh query receipts, whole inputs and every final binary/tool route."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_fresh_proposal_assessment as fresh
from tools.proposal_assessment.improve_binary import approval_evidence_gate
from tools.proposal_assessment.binary_benchmark import binary_decision


def main():
    fresh.frozen()
    fixture = fresh.read(fresh.E / 'inputs.json')
    prepared = fresh.read(fresh.E / 'prepared.json')
    report = fresh.read(fresh.E / 'report.json')
    assert report['summary']['complete']
    assert report['summary']['new_distinct'] == len(fixture['records']) == 18
    assert report['summary']['reused_distinct'] == 0
    assert fixture == fresh.read(ROOT / 'artifacts/proposal-assessment-500-660-20261006/evaluation/inputs.json')
    assert prepared == fresh.read(ROOT / 'artifacts/proposal-assessment-500-660-20261006/evaluation/prepared.json')
    expanded = {p['proposal_id']: p for p in report['proposals']}
    assert set(expanded) == set(range(500, 661))
    routes = Counter(); inference_queries = 0; bit_equal = 0
    for entry in prepared['entries']:
        i = entry['record_index']; dest = fresh.E / 'runs' / f'{i:03d}'
        r = fresh.read(dest / 'report.json'); v = fresh.read(dest / 'verified.json')
        x = fixture['records'][i]
        assert not (dest / 'reuse.json').exists()
        assert v['report_sha256'] == fresh.sha(dest / 'report.json') and v['fresh_inference']
        assert r['tokens'] == len(x['token_ids']) <= 128
        assert r['input_hash'] == x['input_sha256'] and r['model'] == fixture['model_lock_sha256']
        assert r['wasm_sha256'] == r['deployed_wasm_sha256'] == fresh.MODULE
        assert r['query_count'] == r['executed_query_count'] == v['queries']
        assert not r['replayed_queries'] and not r.get('fallback')
        assert [l['layer'] for l in r['layers']] == list(range(32))
        assert r['comparison']['typed_output_valid']
        assert r['max_query_instructions'] < 5_000_000_000 and r['max_observed_heap_bytes'] < 2**32
        assert all(q['ok']['request_bytes'] < 2_000_000 and q['ok']['reply_bytes'] < 2_000_000 for q in r['queries'])
        logits = r['decision_query']['ok']['decision']['raw_logits']
        assert len(logits) == 3
        label, score = binary_decision(logits[:2], .6)
        final, reason = approval_evidence_gate(entry['task'], label)
        for pid in entry['proposal_ids']:
            p = expanded[pid]
            assert p['final_label'] == final and p['uncalibrated_score'] == score
            assert p['reason'] == reason and p['route'] == 'participation_model'
            assert not p['reused']
        inference_queries += r['executed_query_count']; routes[r['query_count']] += 1
        old = fresh.read(ROOT / 'artifacts/proposal-assessment-500-660-20261006/evaluation/runs' / f'{i:03d}' / 'report.json')
        same = logits == old['decision_query']['ok']['decision']['raw_logits']
        assert same == v['prior_logits_bit_equal']
        bit_equal += same
    preparation_queries = 0
    for n, bank in enumerate(fresh.read(fresh.D / 'plan.json')['banks']):
        prefix = fresh.D / 'prefixes' / f'{n:02d}'
        r = fresh.read(prefix / 'report.json'); c = fresh.read(prefix / 'queries/cache.json')
        assert r['wasm_sha256'] == r['deployed_wasm_sha256'] == fresh.MODULE
        assert not r['replayed_queries'] and r['query_count'] == r['executed_query_count'] == 66
        assert c['token_ids'] == bank['ids']
        assert r['max_query_instructions'] < 5_000_000_000 and r['max_observed_heap_bytes'] < 2**32
        codec = fresh.read(fresh.D / 'packets' / f'{n:02d}' / 'report.json')
        assert codec['preparation_queries'] == 24 and codec['codec_module'] == fresh.CODEC
        preparation_queries += r['executed_query_count'] + codec['preparation_queries']
    for gate in prepared['gates']:
        for pid in gate['proposal_ids']:
            p = expanded[pid]
            assert p['route'] == gate['route'] and p['final_label'] == gate['label']
    result = dict(verified=True, rerouted_proposals=161, modeled_proposals=50, new_distinct_inferences=18,
                  reused_predictions=0, full_input_tokens_preserved=True, inference_queries=inference_queries,
                  preparation_queries=preparation_queries, query_count_distribution=dict(routes),
                  logits_bit_equal_to_previous=bit_equal, human_vote_accuracy_measured=False)
    fresh.save(fresh.D / 'verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
