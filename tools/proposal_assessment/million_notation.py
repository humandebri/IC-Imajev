"""Exact M (million) notation candidates; does not change the active benchmark.

The formatter uses integer/string arithmetic only. Token measurements use the
existing pinned tokenizer; they are not evidence of model decision quality.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import re
import shutil
import sys

from .token_sweep import ROOT, numeric_change, save, sha

NUMBER = re.compile(r'([0-9]+)(?:\.([0-9]+))?\Z')


def exact_value(text):
    """Decode a decimal or M value into a rational, without float rounding."""
    if not isinstance(text, str):
        raise ValueError('expected a decimal string')
    suffix = text.endswith('M')
    raw = text[:-1] if suffix else text
    match = NUMBER.fullmatch(raw)
    if not match:
        raise ValueError('expected unsigned decimal, optionally suffixed M')
    whole, fraction = match[1], match[2] or ''
    value = Fraction(int(whole + fraction), 10**len(fraction))
    return value * (1000000 if suffix else 1)


def format_million(text):
    """Use M at >=1 million, retaining every original decimal digit exactly."""
    if type(text) is int:
        text = str(text)
    if not isinstance(text, str) or not (match := NUMBER.fullmatch(text)):
        raise ValueError('expected unsigned integer or decimal string')
    whole, fraction = int(match[1]), match[2] or ''
    if whole < 1000000:
        return text
    millions, remainder = divmod(whole, 1000000)
    tail = (f'{remainder:06d}' + fraction).rstrip('0')
    result = str(millions) + ('.' + tail if tail else '') + 'M'
    if exact_value(text) != exact_value(result):
        raise AssertionError('numeric value changed')
    return result


def compact_numeric_change(row):
    original = numeric_change(row)
    name, values = original.split(': ', 1)
    old, new = values.split(' -> ', 1)
    # Missing/unsupported values remain untouched rather than being guessed.
    shorten = lambda v: format_million(v) if NUMBER.fullmatch(v) else v
    return f'{name}: {shorten(old)} -> {shorten(new)}'


def candidate_state(task, original_state):
    """Replace only checked numeric rows, retaining every effect and instruction."""
    state = json.loads(task['state'])
    if state['action'] != 'ManageNervousSystemParameters':
        return original_state
    rows = state['changes_or_requests']
    parts = original_state.split('; ')
    for i, row in enumerate(rows):
        expected = numeric_change(row).replace(': ', ' ').replace(' -> ', '→')
        if i >= len(parts) or parts[i] != expected:
            raise ValueError('source is not the verified numerical participation layout')
        parts[i] = compact_numeric_change(row).replace(': ', ' ').replace(' -> ', '→')
    return '; '.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--imajev', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh output directory')
    prepared_path, inputs_path = args.source/'prepared.json', args.source/'inputs.json'
    prepared, inputs = [json.loads(p.read_text()) for p in (prepared_path, inputs_path)]
    identity = json.loads((args.source/'identity.json').read_text())
    assert sha(prepared_path) == identity['prepared_sha256']
    assert sha(inputs_path) == identity['inputs_sha256']
    sys.path.insert(0, str(args.imajev/'scripts'))
    from prepare_text import TextPreparer
    preparer = TextPreparer()
    assert sha(args.imajev/'MODEL_LOCK.json') == inputs['model_lock_sha256']
    entries, candidates = [], []
    for entry in prepared['entries']:
        original = inputs['records'][entry['record_index']]
        case = entry['input_case']
        # Establish identity before making the isolated candidate.
        reproduced = preparer.prepare(case)
        assert reproduced['token_ids'] == original['token_ids']
        modified = {**case, 'state': candidate_state(entry['task'], case['state'])}
        candidate = preparer.prepare(modified)
        candidate['prefix_tokens'] = original['prefix_tokens']
        assert candidate['token_ids'][:26] == original['token_ids'][:26]
        assert candidate.get('gold') is None
        old, new = len(original['token_ids']), len(candidate['token_ids'])
        # Some non-round decimals become longer. Keep the original in that case.
        selected = candidate if new < old else original
        entries.append({'proposal_ids': entry['proposal_ids'], 'original_tokens': old,
                        'M_candidate_tokens': new, 'selected_tokens': len(selected['token_ids']),
                        'saved_tokens': old-len(selected['token_ids']),
                        'original_state': case['state'], 'M_candidate_state': modified['state'],
                        'M_selected': new < old, 'model_inference_executed': False})
        candidates.append(selected)
    args.output.mkdir(parents=True)
    save(args.output/'measurements.json', {'entries': entries, 'notation': {'M': 1000000},
         'source_prepared_sha256': sha(prepared_path), 'source_inputs_sha256': sha(inputs_path),
         'tool_sha256': sha(__file__), 'model_lock_sha256': inputs['model_lock_sha256'],
         'quality_verified': False, 'model_inference_executed': False,
         'selection': 'Exact M candidate only when full rendered input has fewer tokens.'})
    save(args.output/'candidate-inputs.json', {'model_lock_sha256': inputs['model_lock_sha256'],
                                              'records': candidates})
    shutil.copyfile(__file__, args.output/'formatter_source.py')
    lines = ['# Exact M notation token measurement', '',
             'M = 1,000,000. Integer/string arithmetic preserves exact values; no rounding or external LLM. Existing source inputs and model are unchanged. Candidate inputs have not been inferred; quality is unverified.', '',
             '|proposal / control|original tokens|M candidate tokens|selected tokens|saved|',
             '|---|---:|---:|---:|---:|']
    for e in entries:
        lines.append(f"|{','.join(map(str,e['proposal_ids']))}|{e['original_tokens']}|{e['M_candidate_tokens']}|{e['selected_tokens']}|{e['saved_tokens']}|")
    lines += ['', 'Candidates with more/equal tokens retain the original. Selection currently uses the pinned Python tokenizer; this is not a fully onchain integration. M interpretation by the model still needs decision-quality evaluation.', '']
    (args.output/'REPORT.md').write_text('\n'.join(lines))
    save(args.output/'files-sha256.json', {p.name: sha(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps(entries, ensure_ascii=False))


if __name__ == '__main__':
    main()
