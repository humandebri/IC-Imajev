#!/usr/bin/env python3
"""Verify a completed short-input benchmark and its source mapping."""
import hashlib,json,collections,statistics,struct
import evaluate_prompt_accuracy as e
import benchmark_decision_index_short as b
R=e.ROOT;D=b.D

def f1(rows,label):
    tp=sum(r['gold']==label and r['prediction']==label for r in rows)
    fp=sum(r['gold']!=label and r['prediction']==label for r in rows)
    fn=sum(r['gold']==label and r['prediction']!=label for r in rows)
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.

def main():
    fixture=json.loads((D/'inputs.json').read_text());source=[json.loads(line) for line in (D/'selected-source-rows.jsonl').open()]
    assert len(source)==len(fixture['records'])==100
    assert all(e.sha(R/p)==h for p,h in fixture['source_hashes'].items())
    session=json.loads((D/'evaluation-session.json').read_text())
    assert session['input_hash']==e.sha(D/'inputs.json')
    assert all(e.sha(R/p)==h for p,h in session['source_hashes'].items())
    result=b.summarize();assert result['complete'] and not result['errors']
    checks=[]
    for i,(record,row) in enumerate(zip(fixture['records'],source)):
        question=next(iter(row['questions'].values()))
        assert record['source_id']==row['id'] and record['gold']==next(iter(row['expected'].values()))
        assert record['options']==list(question['criteria'])
        assert question['instructions'] in record['prompt']
        assert all(f'{key} — {description}' in record['prompt'] for key,description in question['criteria'].items())
        assert 1<=len(record['token_ids'])-record['prefix_tokens']<=87
        assert record['input_sha256']==hashlib.sha256(json.dumps(record['token_ids']).encode()).hexdigest()
        cache,_=b.caches(record['prefix_tokens']);tokens=json.loads((cache/'cache.json').read_text())['token_ids']
        assert record['token_ids'][:len(tokens)]==tokens
        path=D/'runs'/f'{i:03d}'/'report.json';report=json.loads(path.read_text());decision=report['decision_query']['ok']['decision']
        assert report['wasm_sha256']==e.MODULE and report['input_hash']==record['input_sha256']
        assert report['comparison']['typed_output_valid'] and len(decision['probabilities'])==len(record['options'])
        assert decision['value'] in record['options'] or decision['abstained']
        assert abs(sum(decision['probabilities'])+decision['unknown_probability']-1)<1e-5
        assert all(q['ok']['instructions']<5_000_000_000 for q in report['queries'])
        assert report['max_query_instructions']<5_000_000_000
        assert not report.get('fallback') and report['replayed_queries']==0
        checks.append(dict(index=i,report_sha256=e.sha(path),query_count=report['query_count']))
    for family,m in result['by_dataset'].items():
        rows=[r for r in result['cases'] if r['dataset']==family];labels=rows[0]['options']
        m.update(macro_f1=sum(f1(rows,label) for label in labels)/len(labels),
                 majority_accuracy=max(m['gold_counts'].values())/len(rows),median_queries=statistics.median(r['query_count'] for r in rows),
                 median_candid_bytes=statistics.median(r['candid_bytes'] for r in rows))
        if family=='iSarcasmEval-A-En':m['sarcastic_class_f1']=f1(rows,'yes')
    previous=json.loads((R/'artifacts/decision-index-v1/inputs.json').read_text())
    repeated=[]
    for row in result['cases']:
        record=fixture['records'][row['index']]
        matches=[(i,r) for i,r in enumerate(previous['records']) if r['input_sha256']==record['input_sha256']]
        for i,old in matches:
            path=R/'artifacts/decision-index-v1/runs'/f'{i:03d}'/'report.json'
            if not path.exists():continue
            decision=json.loads(path.read_text())['decision_query']['ok']['decision']
            current_probs=[*row['probabilities'],row['unknown_probability']];previous_probs=[*decision['probabilities'],decision['unknown_probability']]
            assert struct.pack('<'+'f'*len(current_probs),*current_probs)==struct.pack('<'+'f'*len(previous_probs),*previous_probs)
            repeated.append(dict(short_index=row['index'],previous_index=i,probabilities_bit_equal=True))
    result['verification']=dict(verified=True,actual_canister_inferences=100,source_mapping_verified=True,all_inputs_and_model_hashes_verified=True,
                                all_successful_query_handlers_under_5B=True,replayed_queries=0,fallbacks=0,repeated_input_checks=repeated,reports=checks)
    result['limitations']=['Input-length filtering changes the task and difficulty distribution; no full-dataset or long-context claim.','HellaSwag has only 60 of 10,042 source rows within the short-input condition (about 0.6%).',
                          'Four underlying benchmarks, five tracks; 40% of the sample is iSarcasmEval, with human taste/language tasks favored.',
                          'iSarcasmEval-A-En has 17 no and 3 yes labels; 85% accuracy is possible by always answering no. Its positive-class F1 is also reported.',
                          'Current INT8 canister with short prompt and one presentation order; no Clef evaluation on these exact 100 inputs.',
                          'Query counts and Candid bytes exclude prefix preparation, module-hash checks, HTTP/CBOR/signatures; handler instructions exclude Candid decode/encode.',
                          'Local timing is affected by memory pressure and parallel execution and is not a production latency comparison.']
    e.atomic_json(D/'report.json',result);e.atomic_json(D/'verification.json',result['verification'])
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','verification')},ensure_ascii=False))

if __name__=='__main__':main()
