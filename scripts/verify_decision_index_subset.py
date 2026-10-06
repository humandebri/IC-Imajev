#!/usr/bin/env python3
"""Verify corpus mapping, provenance and actual query reports for the fixed sample."""
import hashlib,json,pathlib
import evaluate_prompt_accuracy as e
R=e.ROOT;D=R/'artifacts/decision-index-v1'

def main():
    fixture=json.loads((D/'inputs.json').read_text())
    sources=[json.loads(line) for line in (D/'selected-source-rows.jsonl').open()]
    assert len(sources)==len(fixture['records'])==100
    assert all(e.sha(R/p)==h for p,h in fixture['source_hashes'].items())
    for path in D.glob('*evaluation-session.json'):
        session=json.loads(path.read_text())
        assert session['input_hash']==e.sha(D/'inputs.json')
        assert all(e.sha(R/p)==h for p,h in session['source_hashes'].items()),path
    proof=json.loads((D/'prefix-equivalence.json').read_text())
    assert proof['all_hidden_bit_equal']
    reports=[]
    for i,(record,source) in enumerate(zip(fixture['records'],sources)):
        question=next(iter(source['questions'].values()))
        assert record['source_id']==source['id']
        assert record['options']==list(question['criteria'])
        assert record['gold']==next(iter(source['expected'].values()))
        assert question['instructions'] in record['prompt']
        assert all(f'{key} — {description}' in record['prompt'] for key,description in question['criteria'].items())
        assert record['input_sha256']==hashlib.sha256(json.dumps(record['token_ids']).encode()).hexdigest()
        path=D/'runs'/f'{i:03d}'/'report.json'
        report=json.loads(path.read_text())
        assert report['model']==fixture['model_lock_sha256']
        assert report['input_hash']==record['input_sha256'] and report['wasm_sha256']==e.MODULE
        assert report['comparison']['typed_output_valid'] and report['comparison']['gold']==record['gold']
        decision=report['decision_query']['ok']['decision']
        assert decision['value'] in record['options'] or decision['abstained']
        assert len(decision['probabilities'])==len(record['options'])
        assert abs(sum(decision['probabilities'])+decision['unknown_probability']-1)<1e-5
        assert 0<report['max_query_instructions']<5_000_000_000
        assert all(q['ok']['instructions']<5_000_000_000 for q in report['queries'])
        assert report['query_count']>=len(report['queries'])
        reports.append(dict(index=i,report_sha256=e.sha(path),replayed_queries=report['replayed_queries'],query_count=report['query_count']))
    result=dict(verified=True,completed_inferences=100,source_mapping_verified=True,all_input_and_module_hashes_verified=True,
                all_probabilities_valid=True,all_successful_query_handlers_under_5B=True,prefix_hidden_bit_equal=True,
                replayed_queries=sum(r['replayed_queries'] for r in reports),reports=reports)
    e.atomic_json(D/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='reports'}))

if __name__=='__main__':main()
