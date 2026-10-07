#!/usr/bin/env python3
"""Independent integer audit of Sun's MIT rank23 data and mixed rank161 DAGs.

Only ast.literal_eval of SIDES is read from pinned author data. No upstream
Python is executed. Attribution/license are preserved under upstream/.
"""
import ast
import hashlib
import json
from pathlib import Path
from generate_s3_stream_probe import DAG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/rank23-symbolic-v1'


def side(dag, inputs, spec):
    values = list(inputs)
    assert len(values) == spec['base_dim']
    for a, sign, b in spec['inter']:
        assert sign in (-1, 1) and max(a, b) < len(values)
        values.append(dag.add(values[a], values[b], sign))
    outputs = []
    for terms in spec['final']:
        value = 'zero'
        for index, sign in terms:
            assert sign in (-1, 1) and index < len(values)
            value = dag.add(value, values[index], sign)
        outputs.append(value)
    return outputs


def polynomial(c, name, a, b, leaves):
    poly = {}
    for m, sign in c.symbols[name].items():
        an, bn = leaves[m]
        for ai, av in a.symbols[an].items():
            for bi, bv in b.symbols[bn].items():
                key = (ai, bi)
                poly[key] = poly.get(key, 0) + sign * av * bv
    return {k: v for k, v in poly.items() if v}


def make(specs, label):
    n = 1
    rank = 1
    for size, spec in specs:
        n *= size
        rank *= len(spec['U']['final'])
    a, b, c = DAG('a', n*n), DAG('b', n*n), DAG('p', rank)
    leaves = []

    def mul(x, y, depth):
        if depth == len(specs):
            index = len(leaves)
            leaves.append((x[0][0], y[0][0]))
            return [[f'p{index}']]
        size, spec = specs[depth]
        h = len(x) // size
        def transform(z, dag, key):
            blocks = [[[z[i*h+u][j*h+v] for v in range(h)]
                       for u in range(h)] for i in range(size) for j in range(size)]
            result = [[[None]*h for _ in range(h)] for _ in spec[key]['final']]
            for u in range(h):
                for v in range(h):
                    values = side(dag, [block[u][v] for block in blocks], spec[key])
                    for m, value in enumerate(values):
                        result[m][u][v] = value
            return result
        ax, by = transform(x, a, 'U'), transform(y, b, 'V')
        products = [mul(xx, yy, depth+1) for xx, yy in zip(ax, by)]
        result = [[None]*len(x) for _ in x]
        for u in range(h):
            for v in range(h):
                values = side(c, [p[u][v] for p in products], spec['W'])
                for ij, value in enumerate(values):
                    result[(ij//size)*h+u][(ij%size)*h+v] = value
        return result

    result = mul([[f'a{i*n+j}' for j in range(n)] for i in range(n)],
                 [[f'b{i*n+j}' for j in range(n)] for i in range(n)], 0)
    assert len(leaves) == rank
    for i in range(n):
        for j in range(n):
            assert polynomial(c, result[i][j], a, b, leaves) == {
                (i*n+k, k*n+j): 1 for k in range(n)}
    segment = (256+n-1)//n
    input_a = max(sum(abs(v) for v in x.values()) for x in a.symbols.values()) * 127
    input_b = max(sum(abs(v) for v in x.values()) for x in b.symbols.values()) * 128
    assert max(input_a, input_b) < 32768
    leaf_bound = max(sum(abs(v) for v in a.symbols[an].values()) * 127 *
                     sum(abs(v) for v in b.symbols[bn].values()) * 128 * segment
                     for an, bn in leaves)
    recombine_bound = max(sum(abs(v) for v in polynomial(c, name, a, b, leaves).values())
                         * 127 * 128 * segment for name in c.symbols)
    assert max(leaf_bound, recombine_bound) < 2**31
    data = {'label': label, 'size': n, 'rank': rank, 'segment': segment,
            'input_i16_bounds': [input_a, input_b],
            'leaf_i32_bound': leaf_bound, 'reconstruction_i32_bound': recombine_bound,
            'integer_identity': True, 'brent_equations_checked': n**6,
            'nodes': {'a': a.nodes, 'b': b.nodes, 'p': c.nodes},
            'leaves': leaves, 'outputs': result,
            'add_sub_nodes': [len(a.nodes), len(b.nodes), len(c.nodes)],
            'measured_on_ic': False,
            'f32_preservation_requirement': 'Convert only the reconstructed original 256-element integer dot; keep original scale multiplication and block accumulation order.'}
    path = OUT / (label + '.json')
    path.write_text(json.dumps(data, indent=2) + '\n')
    return {k: v for k, v in data.items() if k not in ('nodes', 'leaves', 'outputs')}


def main():
    manifest = json.loads((OUT / 'upstream.json').read_text())
    for name, info in manifest['files'].items():
        raw = (OUT / 'upstream' / name).read_bytes()
        assert len(raw) == info['bytes']
        assert hashlib.sha256(raw).hexdigest() == info['sha256']
    tree = ast.parse((OUT / 'upstream/verify.py').read_text())
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'SIDES' for t in node.targets)]
    assert len(assignments) == 1
    r23 = ast.literal_eval(assignments[0].value)
    # Standard Strassen rank7, row-major A/B/C. Each inter refers to earlier bases.
    r7 = {
        'U': {'base_dim': 4, 'inter': [(0,1,3),(2,1,3),(0,1,1),(2,-1,0),(1,-1,3)],
              'final': [[(4,1)],[(5,1)],[(0,1)],[(3,1)],[(6,1)],[(7,1)],[(8,1)]]},
        'V': {'base_dim': 4, 'inter': [(0,1,3),(1,-1,3),(2,-1,0),(0,1,1),(2,1,3)],
              'final': [[(4,1)],[(0,1)],[(5,1)],[(6,1)],[(3,1)],[(7,1)],[(8,1)]]},
        'W': {'base_dim': 7, 'inter': [], 'final': [
            [(0,1),(3,1),(4,-1),(6,1)],[(2,1),(4,1)],
            [(1,1),(3,1)],[(0,1),(1,-1),(2,1),(5,1)]]}}
    results = [make([(3, r23)], 'rank23'), make([(2, r7)], 'rank7'),
               make([(3, r23),(2, r7)], 'mixed-3-2'),
               make([(2, r7),(3, r23)], 'mixed-2-3'),
               make([(3, r23),(2, r7),(2, r7)], 'mixed-3-2-2'),
               make([(2, r7),(3, r23),(2, r7)], 'mixed-2-3-2'),
               make([(2, r7),(2, r7),(3, r23)], 'mixed-2-2-3')]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [Path(__file__), ROOT/'scripts/generate_s3_stream_probe.py',
                           OUT/'upstream.json', *OUT.glob('upstream/*'),
                           *[OUT/(r['label']+'.json') for r in results]]}
    report = {'complete': True, 'author_program_executed': False,
              'source_commit': manifest['commit'], 'results': results, 'hashes': hashes}
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'complete': True, 'results': results}))


if __name__ == '__main__':
    main()
