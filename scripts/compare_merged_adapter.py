#!/usr/bin/env python3
"""Compare saved-pack host decisions and independent warm host timing runs."""
import argparse
import json
import math
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--directory', default='artifacts/merged-adapter-v2')
    args = ap.parse_args()
    directory = ROOT / args.directory
    before, after = [load(directory / name) for name in ('before.json', 'after.json')]
    provenance = load(directory / 'model.provenance.json')
    if before['model_lock_sha256'] != after['model_lock_sha256'] or not after.get('merged_adapter') or after['weight_pack_hash'] != provenance['pack_hash']:
        raise ValueError('Experiment identity')
    def indexed(report):
        result = {(r['id'], r['offset']): r for r in report['records']}
        if len(result) != len(report['records']):raise ValueError('Duplicate cases')
        return result
    a, b = indexed(before), indexed(after)
    if a.keys() != b.keys():raise ValueError('Case coverage mismatch')
    cases = []
    for key, x in a.items():
        y = b[key]
        for field in ('options', 'gold', 'token_ids', 'input_sha256', 'prompt', 'rotations'):
            if x[field] != y[field]:raise ValueError(f'Changed input: {key}, {field}')
        for r in (x, y):
            output = r['result']
            if output['calibration_version'] != 'p3-r2-s000291-authored':raise ValueError('Calibration changed')
            if set(output['scores']) != set(r['options']) | {'__unknown__'}:raise ValueError('Scores')
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in output['scores'].values()) or abs(sum(output['scores'].values()) - 1) > 1e-5:raise ValueError('Probabilities')
            if any(not math.isfinite(v) for v in output['raw_logits'].values()):raise ValueError('Logits')
            if output['value'] is not None and output['value'] not in r['options']:raise ValueError('Value')
        gold = x['gold']
        def correct(r):
            return None if gold is None else (r['result']['value'] is None if gold == '__unknown__' else r['result']['value'] == gold)
        cases.append(dict(id=key[0], offset=key[1], gold=gold, before=x['result']['value'], after=y['result']['value'],
                          label_identical=x['result']['value']==y['result']['value'] and x['result']['status']==y['result']['status'],
                          probability_max_abs_error=max(abs(v-y['result']['scores'][k]) for k,v in x['result']['scores'].items()),
                          before_correct=correct(x), after_correct=correct(y)))
    gold_cases = [c for c in cases if c['offset']==0 and c['gold'] is not None]
    report = dict(scope='23 host conditions, 10 source questions; changed rounding/quantization. No Wasm instruction or general accuracy claim.',
                  cases=cases, records=len(cases), label_identical=sum(c['label_identical'] for c in cases),
                  original_order_gold_records=len(gold_cases), original_order_before_correct=sum(c['before_correct'] for c in gold_cases),
                  original_order_after_correct=sum(c['after_correct'] for c in gold_cases),
                  probability_max_abs_error=max(c['probability_max_abs_error'] for c in cases),
                  source_bytes=provenance['source_bytes'], merged_bytes=provenance['merged_bytes'], saved_bytes=provenance['saved_bytes'],
                  dense_projection_mac_reduction=provenance['dense_projection_mac_reduction'],
                  removed_lora_macs_per_token=provenance['removed_lora_macs_per_token'], merged_pack_hash=provenance['pack_hash'])
    timing_paths = [directory / f'{name}-timing.json' for name in ('before', 'after')]
    if all(p.exists() for p in timing_paths):
        timing = [load(p) for p in timing_paths]
        if timing[0]['weight_pack_hash']!=before['weight_pack_hash'] or timing[1]['weight_pack_hash']!=after['weight_pack_hash']:raise ValueError('Timing pack mismatch')
        if len(timing[0]['records'])!=len(timing[1]['records']) or len(timing[0]['records'])<4:raise ValueError('Timing sample coverage')
        for x,y in zip(*(r['records'] for r in timing)):
            if (x['id'],x['offset'],x['repeat'],x['token_ids'])!=(y['id'],y['offset'],y['repeat'],y['token_ids']):raise ValueError('Timing input mismatch')
        samples = [[r['metadata']['forward_seconds'] for r in t['records'][1:]] for t in timing]
        spread = [max(s)/min(s) for s in samples]
        report['host_timing'] = dict(scope='Single 132-token question repeated in separate sequential GPU processes. First forward excluded; host block256 implementation, not IC speed.',
                                     samples_per_variant=len(samples[0]), before_seconds=samples[0], after_seconds=samples[1],
                                     before_median_seconds=statistics.median(samples[0]), after_median_seconds=statistics.median(samples[1]),
                                     observed_median_ratio=statistics.median(samples[0])/statistics.median(samples[1]),
                                     before_max_min_ratio=spread[0],after_max_min_ratio=spread[1],
                                     timing_inconclusive_due_to_spread=max(spread)>1.2)
    (directory / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__ == '__main__':
    main()
