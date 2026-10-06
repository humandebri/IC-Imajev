#!/usr/bin/env python3
"""Compare pinned native Bosun inference with the frozen Imajev evaluations."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import importlib.metadata
import json
import math
import pathlib
import statistics
import sys
import time
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'artifacts/bosun-short-v1'
UNKNOWN = '__unknown__'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def native_module():
    package = types.ModuleType('bosun_pinned')
    package.__path__ = [str(DIRECTORY / 'model')]
    sys.modules['bosun_pinned'] = package
    return importlib.import_module('bosun_pinned.modeling_bosun')


def prepare():
    import transformers

    path = DIRECTORY / 'inputs.json'
    if path.exists():
        raise RuntimeError('Refusing to replace frozen inputs')
    module = native_module()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        DIRECTORY / 'model/tokenizer', local_files_only=True)
    baseline_dir = ROOT / 'artifacts/decision-index-short-v1'
    baseline = read(baseline_dir / 'report.json')
    assert baseline['complete'] and baseline['verification']['verified']
    frozen = read(baseline_dir / 'inputs.json')
    source = [json.loads(line) for line in (baseline_dir / 'selected-source-rows.jsonl').open()]
    records = []
    for i, row in enumerate(source):
        question = next(iter(row['questions'].values()))
        old = baseline['cases'][i]
        assert old['index'] == i and old['source_id'] == row['id']
        criteria = question['criteria']
        assert list(criteria) == old['options'] == frozen['records'][i]['options']
        records.append(dict(
            index=len(records), suite='short100', dataset=old['dataset'], id=row['id'], offset=0,
            state=row['state'], instructions=question['instructions'],
            candidates=[dict(id=key, label=key, description=description)
                        for key, description in criteria.items()],
            gold=old['gold'], imajev_prediction=old['prediction'], imajev_tokens=old['tokens'],
            imajev_probabilities=old['probabilities'] + [old['unknown_probability']],
            imajev_report=old['report'], imajev_report_sha256=old['report_sha256']))
    extra_dir = ROOT / 'artifacts/prompt-accuracy-v2'
    extra_inputs = read(extra_dir / 'inputs.json')
    extra_report = read(extra_dir / 'report.json')
    assert extra_report['complete'] and extra_report['completed_inferences'] == 64
    controls = [c for c in read(ROOT / 'benchmarks/cases.json') if c['gold'] is not None]
    controls += read(ROOT / 'benchmarks/prompt_accuracy_extra.json')
    lookup = {c['id']: c for c in controls}
    for pair in extra_inputs['pairs']:
        case = lookup[pair['id']]
        ordered = case['options'][pair['offset']:] + case['options'][:pair['offset']]
        assert ordered == pair['options'] and case['gold'] == pair['gold']
        old = next(r for r in extra_report['cases'] if
                   r['variant'] == 'short' and r['id'] == case['id'] and r['offset'] == pair['offset'])
        records.append(dict(
            index=len(records), suite='curated32', dataset=pair['category'], id=case['id'], offset=pair['offset'],
            state=case['state'], instructions=case['question'],
            candidates=[dict(id=option, label=option, description='') for option in ordered],
            gold=case['gold'], imajev_prediction=old['prediction'], imajev_tokens=old['tokens'],
            imajev_probabilities=old['probabilities'] + [old['unknown_probability']],
            imajev_report=old['report'], imajev_report_sha256=old['report_sha256']))
    assert len(records) == 132
    for record in records:
        record['candidates'].append(dict(id=UNKNOWN, label='unknown', description=''))
        record['row_id'] = f"{record['suite']}:{record['id']}:o{record['offset']}"
        # The native renderer shuffles candidates. Select the first seed that
        # preserves the baseline's presentation order, without consulting gold.
        identity = list(range(len(record['candidates'])))
        for seed in range(100_000):
            content, order, mapping = module.render_decision_prompt(
                state=record['state'], instructions=record['instructions'],
                candidates=record['candidates'], decision_type='choice',
                decision_tokens=[f'<|decision_{j:03d}|>' for j in range(256)],
                seed=seed, row_id=record['row_id'])
            if order == identity:
                break
        else:
            raise RuntimeError('Could not preserve candidate order')
        prompt = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': module._SYSTEM_PROMPT}, {'role': 'user', 'content': content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        token_ids = tokenizer(prompt, truncation=False)['input_ids']
        record.update(seed=seed, content=content, prompt=prompt, token_ids=token_ids,
                      bosun_tokens=len(token_ids), candidate_to_slot=mapping)
        assert len(record['imajev_probabilities']) == len(record['candidates'])
    sources = [baseline_dir / name for name in ['inputs.json', 'report.json', 'selected-source-rows.jsonl', 'verification.json']]
    sources += [extra_dir / name for name in ['inputs.json', 'report.json']]
    sources += [ROOT / 'benchmarks/cases.json', ROOT / 'benchmarks/prompt_accuracy_extra.json']
    identities = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    atomic(path, dict(model_revision=read(DIRECTORY / 'source/hub-model.json')['sha'],
                     source_hashes=identities, records=records,
                     protocol='native model.predict, choice, one preserved option order, no truncation, explicit unknown candidate'))
    print(json.dumps(dict(prepared=len(records), min_tokens=min(r['bosun_tokens'] for r in records),
                          max_tokens=max(r['bosun_tokens'] for r in records))), flush=True)


def f1(rows, label):
    tp = sum(r['gold'] == label and r['bosun_prediction'] == label for r in rows)
    fp = sum(r['gold'] != label and r['bosun_prediction'] == label for r in rows)
    fn = sum(r['gold'] == label and r['bosun_prediction'] != label for r in rows)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.


def metrics(rows):
    fixes = sum(r['bosun_correct'] and not r['imajev_correct'] for r in rows)
    regressions = sum(r['imajev_correct'] and not r['bosun_correct'] for r in rows)
    discordant = fixes + regressions
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(fixes, regressions) + 1)) / 2**discordant) if discordant else 1.
    return dict(total=len(rows), bosun_correct=sum(r['bosun_correct'] for r in rows),
                imajev_correct=sum(r['imajev_correct'] for r in rows),
                bosun_accuracy=sum(r['bosun_correct'] for r in rows) / len(rows),
                imajev_accuracy=sum(r['imajev_correct'] for r in rows) / len(rows),
                imajev_errors_fixed=fixes, imajev_correct_regressed=regressions,
                both_correct=sum(r['bosun_correct'] and r['imajev_correct'] for r in rows),
                both_wrong=sum(not r['bosun_correct'] and not r['imajev_correct'] for r in rows),
                paired_mcnemar_exact_p=p,
                unknown_predictions=sum(r['bosun_prediction'] == UNKNOWN for r in rows),
                bosun_median_tokens=statistics.median(r['bosun_tokens'] for r in rows),
                imajev_median_tokens=statistics.median(r['imajev_tokens'] for r in rows),
                bosun_min_tokens=min(r['bosun_tokens'] for r in rows),
                bosun_max_tokens=max(r['bosun_tokens'] for r in rows))


def summarize():
    fixture = read(DIRECTORY / 'inputs.json')
    rows = []
    for record in fixture['records']:
        path = DIRECTORY / 'runs' / f"{record['index']:03d}.json"
        if not path.exists():
            continue
        actual = read(path)
        assert actual['inputs_sha256'] == digest(DIRECTORY / 'inputs.json')
        result = actual['output']
        assert result['prompt'] == record['prompt'] and result['content'] == record['content']
        assert result['candidate_to_slot'] == record['candidate_to_slot']
        probs = result['probabilities']
        assert len(probs) == len(record['candidates'])
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probs) and abs(sum(probs) - 1) < 1e-5
        choice = max(range(len(probs)), key=probs.__getitem__)
        prediction = record['candidates'][choice]['id']
        rows.append(dict(index=record['index'], suite=record['suite'], dataset=record['dataset'],
                         id=record['id'], offset=record['offset'], gold=record['gold'],
                         bosun_prediction=prediction, imajev_prediction=record['imajev_prediction'],
                         bosun_correct=prediction == record['gold'],
                         imajev_correct=record['imajev_prediction'] == record['gold'],
                         bosun_tokens=record['bosun_tokens'], imajev_tokens=record['imajev_tokens'],
                         probabilities=probs, seconds=actual['seconds'],
                         report_sha256=digest(path)))
    short = [r for r in rows if r['suite'] == 'short100']
    curated = [r for r in rows if r['suite'] == 'curated32']
    distinct = [r for r in curated if r['offset'] == 0]
    result = dict(complete=len(rows) == 132, completed=len(rows), planned=132, cases=rows)
    if short:
        result['short100'] = metrics(short)
        result['by_dataset'] = {}
        for dataset in dict.fromkeys(r['dataset'] for r in short):
            group = [r for r in short if r['dataset'] == dataset]
            m = metrics(group)
            m['gold_counts'] = dict(collections.Counter(r['gold'] for r in group))
            m['prediction_counts'] = dict(collections.Counter(r['bosun_prediction'] for r in group))
            if dataset == 'iSarcasmEval-A-En':
                m['sarcastic_class_f1'] = f1(group, 'yes')
            result['by_dataset'][dataset] = m
    if distinct:
        result['curated24'] = metrics(distinct)
        known = [r for r in distinct if r['gold'] != UNKNOWN]
        unknown = [r for r in distinct if r['gold'] == UNKNOWN]
        result['curated24'].update(known_total=len(known), known_correct=sum(r['bosun_correct'] for r in known),
                                   unknown_total=len(unknown), unknown_correct=sum(r['bosun_correct'] for r in unknown),
                                   false_abstentions=sum(r['bosun_prediction'] == UNKNOWN for r in known))
    if curated:
        result['curated32_orders'] = metrics(curated)
        changes = []
        for row in curated:
            if row['offset']:
                original = next((r for r in distinct if r['id'] == row['id']), None)
                if original and row['bosun_prediction'] != original['bosun_prediction']:
                    changes.append(dict(id=row['id'], original=original['bosun_prediction'], reordered=row['bosun_prediction']))
        result['order_changes'] = changes
    atomic(DIRECTORY / 'report.json', result)
    return result


def run(device):
    import torch

    fixture = read(DIRECTORY / 'inputs.json')
    assert all(digest(ROOT / p) == h for p, h in fixture['source_hashes'].items())
    module = native_module()
    # Only redirect the pinned base to its downloaded local snapshot.
    config = module.BosunConfig.from_pretrained(DIRECTORY / 'model', local_files_only=True)
    assert config.base_model_revision == '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
    config.base_model_name_or_path = str(DIRECTORY / 'base')
    assert device != 'mps' or torch.backends.mps.is_available()
    dtype = torch.bfloat16 if device == 'mps' else torch.float32
    paths = [DIRECTORY / 'inputs.json', pathlib.Path(__file__)]
    paths += [p for folder in ['model', 'base'] for p in sorted((DIRECTORY / folder).rglob('*'))
              if p.is_file() and '.cache' not in p.parts and p.suffix in ['.json', '.py', '.safetensors', '.txt', '.jinja']]
    print('Hashing pinned evaluation inputs and weights', flush=True)
    identities = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    session = dict(model_revision=fixture['model_revision'], base_revision=config.base_model_revision,
                   device=device, dtype=str(dtype), attention='sdpa', adapter_merged=False,
                   packages={x: importlib.metadata.version(x) for x in ['torch', 'transformers', 'peft', 'safetensors', 'accelerate']},
                   source_hashes=identities)
    path = DIRECTORY / 'evaluation-session.json'
    if path.exists():
        assert read(path) == session
    else:
        atomic(path, session)
    print('Loading native Bosun base, adapter and trained decision rows', flush=True)
    model = module.BosunForDecision.from_pretrained(
        DIRECTORY / 'model', config=config, local_files_only=True,
        torch_dtype=dtype, device_map={'': device}, attn_implementation='sdpa')
    model.eval()
    (DIRECTORY / 'runs').mkdir(exist_ok=True)
    for record in fixture['records']:
        path = DIRECTORY / 'runs' / f"{record['index']:03d}.json"
        if path.exists():
            continue
        start = time.perf_counter()
        output = model.predict(state=record['state'], instructions=record['instructions'],
                               candidates=record['candidates'], decision_type='choice',
                               seed=record['seed'], row_id=record['row_id'])
        if device == 'mps':
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        assert output['prompt'] == record['prompt']
        atomic(path, dict(index=record['index'], inputs_sha256=identities['artifacts/bosun-short-v1/inputs.json'],
                         seconds=elapsed, output=output))
        result = summarize()
        summary = result.get('short100', result.get('curated24', {}))
        print(json.dumps(dict(completed=result['completed'], suite=record['suite'],
                              bosun_correct=summary.get('bosun_correct'), tokens=record['bosun_tokens'], seconds=elapsed)), flush=True)
    assert all(digest(ROOT / p) == h for p, h in identities.items())
    assert all(digest(ROOT / p) == h for p, h in fixture['source_hashes'].items())
    result = summarize()
    assert result['complete']
    verification = dict(verified=True, native_inferences=132, source_hashes_verified=True,
                        source_mappings_verified=True, prompt_and_candidate_mappings_verified=True,
                        probabilities_valid=True, same_presented_candidate_order=True,
                        no_truncation=True, gold_never_passed_to_predict=True)
    result['verification'] = verification
    atomic(DIRECTORY / 'verification.json', verification)
    atomic(DIRECTORY / 'report.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare()
    elif args.operation == 'run':
        run(args.device)
    else:
        print(json.dumps({k: v for k, v in summarize().items() if k != 'cases'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
