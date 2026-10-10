#!/usr/bin/env python3
"""Calculate a current five-token POC tariff from completed local paid measurements."""
import argparse
import json
from pathlib import Path


def calculate(report):
    if not report['complete'] or report['prefix_tokens'] != 5 or report['network'] != 'local':
        raise ValueError('requires a completed current five-token proof')
    cases = [row for row in report['cases'] if row.get('success')]
    if not cases:
        raise ValueError('no successful paid measurements')
    base_fee, unit = 10_000_000_000, 100_000_000
    rows = []
    fee_per_token = unit
    for row in cases:
        value = row['call']['result']['Ok']
        suffix = row['suffix_tokens']
        if value['prefix_tokens'] != 5 or value['suffix_tokens'] != suffix or suffix != row['tokens'] - 5:
            raise ValueError('measurement prefix/suffix mismatch')
        instructions = sum(worker['instructions'] for worker in value['workers'])
        # 13-node execution coefficient, plus the existing message overhead floor.
        estimate = (instructions * 12 + 9) // 10 + 1_000_000_000
        per_token = max(unit, (max(0, estimate - base_fee) + suffix * unit - 1) // (suffix * unit) * unit)
        fee_per_token = max(fee_per_token, per_token)
        rows.append(dict(tokens=row['tokens'], suffix_tokens=suffix, instructions=instructions, estimated_cycles=estimate))
    return dict(module=report['module'], prefix_tokens=5, measurements=rows,
                config=dict(base_fee=base_fee, fee_per_token=fee_per_token, reserve_cycles=2_000_000_000_000),
                scope='POC execution estimate only; storage is subsidized and this does not measure actual subnet cycle burn.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = calculate(json.loads(args.proof.read_text()))
    args.output.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value['config']))


if __name__ == '__main__':
    main()
