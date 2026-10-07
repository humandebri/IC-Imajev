#!/usr/bin/env python3
"""Independent eight-K-lane audit for 12x12 rank1127 block256 SIMD layout."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import struct

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/rank23-symbolic-v1'


def f32(v):
    return struct.unpack('<f',struct.pack('<f',v))[0]


def evaluate(nodes,values,bound):
    for name,terms in nodes:
        result = [0]*8
        for parent,sign in terms.items():
            for lane in range(8):
                result[lane] += sign*values[parent][lane]
                assert abs(result[lane])<=bound
        values[name] = result


def main():
    prior = json.loads((D/'report.json').read_text())
    for p,h in prior['hashes'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    valid = [s*22+k*8+lane for s in range(12) for k in range(3) for lane in range(8)
             if k*8+lane<22 and s*22+k*8+lane<256]
    assert sorted(valid)==list(range(256)) and len(valid)==len(set(valid))
    rng = random.Random(1127256)
    cases = []
    counts = []
    for order in ('mixed-3-2-2','mixed-2-3-2','mixed-2-2-3'):
        plan = json.loads((D/(order+'.json')).read_text())
        roots = [n for row in plan['outputs'] for n in row]
        uses = Counter(roots)
        for _,terms in plan['nodes']['p']:
            uses.update(terms.keys())
        retained = sum(uses[n]>1 or n in roots for n,_ in plan['nodes']['p'])
        locals_bound = 2*1127*3+1127+retained+40
        assert locals_bound<10000
        counts.append({'order':order,'retained_reconstruction':retained,
                       'full_a_b_cache_locals_upper_bound':locals_bound})
        for mode in ('max','min','alternating','random'):
            for nr,nc in ((12,12),(1,1),(7,11),(11,7)):
                q = [[(127 if mode=='max' else -127 if mode=='min' else
                       (-127 if (i+k)%2 else 127) if mode=='alternating' else rng.randrange(-127,128))
                      if i<nr else 0 for k in range(256)] for i in range(12)]
                w = [[(127 if mode=='max' else -128 if mode=='min' else
                       (-128 if (j+k)%2 else 127) if mode=='alternating' else rng.randrange(-128,128))
                      if j<nc else 0 for k in range(256)] for j in range(12)]
                products = [[0]*4 for _ in range(1127)]
                for k in range(3):
                    av,bv = {},{}
                    for i in range(12):
                        for s in range(12):
                            positions = [s*22+k*8+lane for lane in range(8)]
                            av[f'a{i*12+s}'] = [q[i][pos] if k*8+lane<22 and pos<256 else 0
                                               for lane,pos in enumerate(positions)]
                            bv[f'b{s*12+i}'] = [w[i][pos] if k*8+lane<22 and pos<256 else 0
                                               for lane,pos in enumerate(positions)]
                    evaluate(plan['nodes']['a'],av,32767)
                    evaluate(plan['nodes']['b'],bv,32767)
                    for m,(an,bn) in enumerate(plan['leaves']):
                        for lane in range(4):
                            products[m][lane] += av[an][lane*2]*bv[bn][lane*2]+av[an][lane*2+1]*bv[bn][lane*2+1]
                            assert abs(products[m][lane])<2**31
                # Use independent scalar-per-lane reconstruction, with original add/sub order.
                cv = {f'p{i}':v+[0]*4 for i,v in enumerate(products)}
                evaluate(plan['nodes']['p'],cv,2**31-1)
                output = []
                for i in range(12):
                    for j in range(12):
                        lanes = cv[plan['outputs'][i][j]][:4]
                        actual = (lanes[0]+lanes[1])+(lanes[2]+lanes[3])
                        expected = sum(q[i][kk]*w[j][kk] for kk in range(256))
                        assert actual==expected
                        sx,sw,previous = f32(.007*(i+1)),f32(.001*(j+1)),f32(.1*(i-j))
                        def finish(dot):
                            return struct.pack('<f',f32(previous+f32(f32(f32(dot)*sx)*sw)))
                        assert finish(actual)==finish(expected)
                        output.append(finish(actual))
                cases.append({'order':order,'mode':mode,'tail':[nr,nc],
                              'integer_and_float_bits_equal':True,
                              'output_sha256':hashlib.sha256(b''.join(output)).hexdigest()})
    result = {'complete':True,'cases':cases,'case_count':len(cases),
              'all_original_256_elements_covered_exactly_once':True,
              'dummy_K_positions_are_zero':True,'input_lanes':8,'partial_i32_lanes':4,
              'token_group':12,'output_tile':12,'rank':1127,'segment':22,'padded_segment':24,
              'integer_horizontal_reduction_before_float_conversion':True,
              'local_bounds':counts,'measured_on_ic':False,
              'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(__file__),D/'report.json',*[D/(c['order']+'.json') for c in counts]]}}
    (D/'rank1127-simd-layout-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete':True,'cases':len(cases),'local_bounds':counts}))


if __name__=='__main__':
    main()
