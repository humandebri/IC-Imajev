"""Freeze evidence before parsing and reject incomplete benchmark coverage."""
import hashlib
import json


def read_bytes(path, root, hashes):
    raw = path.read_bytes()
    key = str(path.relative_to(root))
    digest = hashlib.sha256(raw).hexdigest()
    if key in hashes and hashes[key] != digest:
        raise ValueError(f'reference changed during run: {key}')
    hashes[key] = digest
    return raw


def read_report(path, root, hashes):
    return json.loads(read_bytes(path, root, hashes))


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


def validate_join_settings(args):
    """Reject unsupported schedules before creating artifacts or calling queries."""
    for name in ('front', 'attention_front'):
        value = getattr(args, name)
        if not 0 < value < 9216 or value % (256 if name == 'front' else 128):
            raise ValueError(f'invalid {name}: {value}')
    if not 0 < args.down_rows < 2560 or args.down_rows % 32:
        raise ValueError('invalid down_rows')
    if not 0 <= args.residual_raw_threshold <= 1:
        raise ValueError('invalid residual_raw_threshold')
    if args.residual_dictionary and not args.compress_residual:
        raise ValueError('residual_dictionary requires compress_residual')
    if getattr(args, 'entry_start', False) and args.entry_heads is None:
        raise ValueError('entry_start requires entry_heads')
    if args.entry_heads is not None:
        if not 0 < args.entry_heads < 32 or args.entry_heads % 2:
            raise ValueError('invalid entry_heads')
        if not 0 < args.entry_front < 9216 or args.entry_front % (256 if getattr(args, 'entry_start', False) else 128):
            raise ValueError('invalid entry_front')
