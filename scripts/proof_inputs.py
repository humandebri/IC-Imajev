"""Freeze evidence before parsing and reject incomplete benchmark coverage."""
import hashlib
import json


def read_report(path, root, hashes):
    raw = path.read_bytes()
    key = str(path.relative_to(root))
    digest = hashlib.sha256(raw).hexdigest()
    if key in hashes and hashes[key] != digest:
        raise ValueError(f'reference changed during run: {key}')
    hashes[key] = digest
    return json.loads(raw)


def residual_cases(report):
    cases = [c for c in report['cases'] if c.get('success') is True]
    keys = [(c['label'], c['layer'], c['begin']) for c in cases]
    expected = {(label, layer, begin)
                for label in ('617', 'insufficient', 'maximum')
                for layer in (3, 7, 11, 15, 19, 23, 27)
                for begin in (5120, 5376)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('residual reference must contain all 42 unique successful cases')
    return cases


def selections(raw, allowed, name):
    values = raw.split(',')
    if not values or any(v not in allowed for v in values) or len(values) != len(set(values)):
        raise ValueError(f'invalid or duplicate {name}: {raw!r}')
    return values
