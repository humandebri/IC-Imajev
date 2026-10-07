#!/usr/bin/env python3
"""Independent scalar interpreter: saved DAGs, padded block256, tail rows/cols."""
import hashlib
import json
from pathlib import Path
import random
import struct

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/rank23-symbolic-v1'


def evaluate(nodes, values, bound):
    for name, terms in nodes:
        acc = 0
        for parent, sign in terms.items():
            acc += values[parent] * sign
            assert -bound <= acc <= bound
        values[name] = acc
    return values


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def main():
    report = json.loads((D/'report.json').read_text())
    for p, expected in report['hashes'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == expected
    rng = random.Random(161256)
    cases = []
    for size, orders in ((6, ('mixed-3-2', 'mixed-2-3')),
                         (12, ('mixed-3-2-2', 'mixed-2-3-2', 'mixed-2-2-3'))):
        for mode in ('max', 'min', 'alternating', 'random'):
            for tails in ((size, size), (1, 1), (size//2, size-1), (size-1, size//2)):
                nr, nc = tails
                q = [[(127 if mode == 'max' else -127 if mode == 'min' else
                        (127 if (i+k)%2 else -127) if mode == 'alternating' else
                        rng.randrange(-127,128)) if i < nr else 0
                       for k in range(256)] for i in range(size)]
                w = [[(127 if mode == 'max' else -128 if mode == 'min' else
                        (127 if (j+k)%2 else -128) if mode == 'alternating' else
                        rng.randrange(-128,128)) if j < nc else 0
                       for k in range(256)] for j in range(size)]
                expected = [[sum(q[i][k]*w[j][k] for k in range(256))
                             for j in range(size)] for i in range(size)]
                for order in orders:
                    plan = json.loads((D/(order+'.json')).read_text())
                    products = [0]*plan['rank']
                    segment = plan['segment']
                    padded = ((segment+3)//4)*4
                    for k in range(padded):
                        av = {'zero': 0}
                        bv = {'zero': 0}
                        for i in range(size):
                            for s in range(size):
                                index = s*segment+k
                                valid = k < segment and index < 256
                                av[f'a{i*size+s}'] = q[i][index] if valid else 0
                                bv[f'b{s*size+i}'] = w[i][index] if valid else 0
                        evaluate(plan['nodes']['a'], av, 32767)
                        evaluate(plan['nodes']['b'], bv, 32767)
                        for m, (a, b) in enumerate(plan['leaves']):
                            products[m] += av[a]*bv[b]
                            assert abs(products[m]) < 2**31
                    cv = {'zero': 0, **{f'p{m}':v for m,v in enumerate(products)}}
                    evaluate(plan['nodes']['p'], cv, 2**31-1)
                    actual = [[cv[name] for name in row] for row in plan['outputs']]
                    assert actual == expected
                    # Same float conversion and multiplication boundaries as block256.
                    float_bits = []
                    for i in range(size):
                        for j in range(size):
                            sx, sw, prior = f32(.003125*(i+1)), f32(.0025*(j+1)), f32(.3*(i-j))
                            def scaled(dot):
                                return struct.pack('<f', f32(prior+f32(f32(f32(dot)*sx)*sw))).hex()
                            assert scaled(actual[i][j]) == scaled(expected[i][j])
                            float_bits.append(scaled(actual[i][j]))
                    cases.append({'mode': mode, 'tail': tails, 'order': order,
                                  'integer_and_f32_bits_equal': True,
                                  'output_hash': hashlib.sha256(bytes.fromhex(''.join(float_bits))).hexdigest()})
    result = {'complete': True, 'cases': cases, 'case_count': len(cases),
              'scalar_interpreter_independent_of_generator': True,
              'original_block_size': 256, 'segment_sizes': [43, 22], 'padded_segment_sizes': [44, 24],
              'padding_never_reads_next_block': True, 'measured_on_ic': False,
              'hashes': {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [Path(__file__), D/'report.json', *[D/(r['label']+'.json') for r in report['results']]]}}
    (D/'native-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'complete': True, 'case_count':len(cases)}))


if __name__ == '__main__':
    main()
