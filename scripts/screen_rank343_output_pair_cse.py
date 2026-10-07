#!/usr/bin/env python3
"""Deterministic signed-pair CSE of exact rank343 reconstruction expressions."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
import json
import random

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def search(original, references, seed):
    rng = random.Random(seed)
    values = [dict(x) for x in original]
    symbols = {f'p{i}': {i: 1} for i in range(343)}
    nodes = []
    while True:
        counts = Counter()
        for value in values:
            for (a, sa), (b, sb) in combinations(sorted(value.items()), 2):
                counts[(a, b, sa * sb)] += 1
        if not counts or max(counts.values()) <= 1:
            break
        best = max(counts.values())
        candidates = sorted(k for k, n in counts.items() if n == best)
        a, b, sign = candidates[rng.randrange(len(candidates))]
        name = f'c{len(nodes)}'
        expression = dict(symbols[a])
        for i, v in symbols[b].items():
            expression[i] = expression.get(i, 0) + sign * v
        symbols[name] = {i: v for i, v in expression.items() if v}
        nodes.append((name, {a: 1, b: sign}))
        for value in values:
            if a in value and b in value and value[a] * value[b] == sign:
                coefficient = value.pop(a)
                value.pop(b)
                value[name] = coefficient
    for value, expected in zip(values, references):
        actual = Counter()
        for name, sign in value.items():
            for i, v in symbols[name].items():
                actual[i] += sign * v
        assert {i: v for i, v in actual.items() if v} == expected
    # Each new pair takes one addition/subtraction. Root negative leading
    # terms require zero-minus as in the existing strict arithmetic emitter.
    root_operations = sum(len(v) - 1 + int(next(iter(v.values())) < 0)
                          for v in values)
    return dict(seed=seed, nodes=nodes, roots=values,
                add_sub_operations=len(nodes) + root_operations,
                all64_leaf_expressions_exact=True)


def main():
    p = ROOT / 'artifacts/s3-k2-prepared-kernels-v1/plan.py'
    ns = dict(__name__='plan', __file__=str(p))
    exec(compile(p.read_text(), str(p), 'exec'), ns)
    a, b, c, leaves, roots, _ = ns['plan']()
    original = [c.symbols[name] for row in roots for name in row]
    original = [{f'p{i}': v for i, v in e.items()} for e in original]
    # Keep integer-indexed reference vectors for independent expansion.
    references = [{int(name[1:]): v for name, v in e.items()} for e in original]
    trials = []
    for seed in range(16):
        result = search(original, references, seed)
        trials.append(result)
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ('nodes', 'roots')}), flush=True)
    best = min(trials, key=lambda x: x['add_sub_operations'])
    d = ROOT / 'artifacts/rank343-output-pair-cse-v1'
    d.mkdir(exist_ok=False)
    (d / 'best.json').write_text(json.dumps(best, indent=2) + '\n')
    report = dict(complete=True, baseline_add_sub_operations=len(c.nodes),
                  best_add_sub_operations=best['add_sub_operations'],
                  all64_leaf_expressions_exact=True, trials=16,
                  performance_verified=False, wasm_execution_verified=False,
                  source_hashes={str(x.relative_to(ROOT)): sha(x)
                                 for x in [Path(__file__), p, d / 'best.json']})
    (d / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
