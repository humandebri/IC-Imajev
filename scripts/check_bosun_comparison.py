#!/usr/bin/env python3
"""Check the native loader/readout, CPU precision, and unknown-input sensitivity."""
import json
import pathlib
import sys
import time

import torch

from evaluate_bosun_local import DIRECTORY, ROOT, UNKNOWN, atomic, digest, native_module, read


def main():
    fixture = read(DIRECTORY / 'inputs.json')
    report = read(DIRECTORY / 'report.json')
    assert report['complete'] and report['verification']['verified']
    module = native_module()
    config = module.BosunConfig.from_pretrained(DIRECTORY / 'model', local_files_only=True)
    config.base_model_name_or_path = str(DIRECTORY / 'base')
    device = sys.argv[1]
    torch.set_num_threads(8)
    dtype = torch.float32 if device == 'cpu' else torch.bfloat16
    model = module.BosunForDecision.from_pretrained(
        DIRECTORY / 'model', config=config, local_files_only=True,
        torch_dtype=dtype, device_map={'': device}, attn_implementation='sdpa')
    directory = DIRECTORY / ('cpu-check' if device == 'cpu' else 'closed-set-check')
    directory.mkdir(exist_ok=True)
    rows = []
    selected = [0, 20, 40, 60, 80] if device == 'cpu' else range(100)
    for index in selected:
        record = fixture['records'][index]
        candidates = record['candidates'] if device == 'cpu' else record['candidates'][:-1]
        seed = record['seed']
        if device != 'cpu':
            for seed in range(100_000):
                _, order, _ = module.render_decision_prompt(
                    state=record['state'], instructions=record['instructions'], candidates=candidates,
                    decision_type='choice', decision_tokens=model.decision_tokens,
                    seed=seed, row_id=record['row_id'])
                if order == list(range(len(candidates))):
                    break
            else:
                raise RuntimeError('Could not preserve order')
        started = time.perf_counter()
        output = model.predict(state=record['state'], instructions=record['instructions'],
                               candidates=candidates, decision_type='choice', seed=seed, row_id=record['row_id'])
        probabilities = output['probabilities']
        prediction = candidates[max(range(len(probabilities)), key=probabilities.__getitem__)]['id']
        baseline = report['cases'][index]
        actual = dict(index=index, dataset=record['dataset'], seed=seed, output=output,
                      prediction=prediction, correct=prediction == record['gold'],
                      baseline_prediction=baseline['bosun_prediction'],
                      same_prediction=prediction == baseline['bosun_prediction'],
                      input_tokens=len(model.tokenizer(output['prompt'])['input_ids']),
                      seconds=time.perf_counter() - started)
        if device == 'cpu':
            assert output['prompt'] == record['prompt']
            actual['max_probability_difference'] = max(abs(a-b) for a,b in zip(probabilities, baseline['probabilities']))
        atomic(directory / f'{index:03d}.json', actual)
        rows.append(actual)
        print(json.dumps(dict(completed=len(rows), index=index, prediction=prediction,
                              same_prediction=actual['same_prediction'])), flush=True)
    summary = dict(device=device, dtype=str(dtype), source_inputs_sha256=digest(DIRECTORY / 'inputs.json'),
                   native_inferences=len(rows), correct=sum(r['correct'] for r in rows),
                   same_predictions=sum(r['same_prediction'] for r in rows),
                   by_dataset={name: sum(r['correct'] for r in rows if r['dataset']==name)
                               for name in dict.fromkeys(r['dataset'] for r in rows)}, rows=rows)
    if device == 'cpu':
        summary['max_probability_difference'] = max(r['max_probability_difference'] for r in rows)
    atomic(directory / 'report.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)


if __name__ == '__main__':
    main()
