#!/usr/bin/env python3
"""Freeze 100 eligible source-suite questions before observing any prediction."""
import collections, hashlib, json, pathlib, subprocess, sys
from prepare_text_legacy import TextPreparer
ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/decision-index-v1'
KIT = D / 'source/decision-index'
sys.path.insert(0, str(KIT))
from decision_index.suite.build.freeze import gid

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    target = D / 'inputs.json'
    if target.exists():
        raise RuntimeError('Refusing to replace the frozen sample')
    source = D / 'work/artifacts/benchmark-suite/partial-source/selected-rows.jsonl'
    with source.open() as stream:
        rows = [json.loads(s) for s in stream]
    excluded = json.loads((KIT / 'hub/excluded-questions.json').read_text())
    exclusions = set(excluded['rows'])
    preparer = TextPreparer()
    records, selected, inventory = [], [], {}
    seed = 20261005
    for family in ['ANLI', 'WinoGrande', 'HellaSwag', 'CLadder', 'VAST']:
        pool = [r for r in rows if r['family'] == family]
        pool.sort(key=lambda r: hashlib.sha256(f'{seed}:{family}:{gid(r)}'.encode()).hexdigest())
        rejected = collections.Counter()
        eligible = []
        scanned = 0
        for row in pool:
            scanned += 1
            if row['_evaluation']['run_id'] in exclusions:
                rejected['release_exclusion'] += 1
                continue
            if len(row['questions']) != 1:
                rejected['multiple_questions'] += 1
                continue
            key, question = next(iter(row['questions'].items()))
            criteria = question['criteria']
            if question['type'] != 'choice' or not 2 <= len(criteria) <= 7:
                rejected['unsupported_options_or_type'] += 1
                continue
            gold = row['expected'][key]
            assert gold in criteria
            case = dict(id=f'di_{len(eligible)}', state=row['state'], question=question['instructions'],
                        options=[dict(value=k, description=v) for k, v in criteria.items()], gold=gold)
            try:
                record = preparer.prepare(case)
            except ValueError as e:
                if 'text prefill requires' not in str(e):
                    raise
                rejected['over_512_tokens'] += 1
                continue
            assert record['prefix_tokens'] == 26
            eligible.append((row, record))
            if len(eligible) == 20:
                break
        assert len(eligible) >= 20
        inventory[family] = dict(source_rows=len(pool), scanned_in_hash_order=scanned, eligible_in_scanned=len(eligible), excluded_in_scanned=dict(rejected), selected=20)
        for row, record in eligible[:20]:
            record['id'] = f'di_{len(records):03d}'
            record['dataset'] = family
            record['source_id'] = row['id']
            records.append(record)
            selected.append(row)
    selected_path = D / 'selected-source-rows.jsonl'
    selected_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    manifest = dict(scope='Fixed 100-question text subset; not the complete Decision Index or its index score',
                    seed=seed, selection='SHA256(seed:family:group_id), first 20 eligible per dataset; no outcome selection',
                    adaptation='One choice field per request; preserve state, instructions, option keys and descriptions; current short prompt and reserved unknown',
                    official_kit_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=KIT,text=True).strip(),
                    model_lock_sha256=sha(ROOT/'MODEL_LOCK.json'), records=records, inventory=inventory,
                    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [source,selected_path,KIT/'hub/excluded-questions.json',ROOT/'scripts/prepare_text.py',ROOT/'scripts/prepare_text_legacy.py',pathlib.Path(__file__)]})
    target.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(inventory=inventory, total=len(records), min_tokens=min(len(r['token_ids']) for r in records), max_tokens=max(len(r['token_ids']) for r in records))))

if __name__ == '__main__':
    main()
