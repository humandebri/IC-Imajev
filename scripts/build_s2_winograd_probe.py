#!/usr/bin/env python3
"""Exact rank49 Winograd with shared input/weight/reconstruction DAG nodes."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s2-winograd-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(exist_ok=False)
 old=ROOT/'artifacts/s3-winograd-v1/frozen-generator.py';s=old.read_text();s=s[:s.index('def main():')]
 lo=s.index('def plan():');hi=s.index('\ndef expression',lo)
 plan=s[lo:hi].replace("DAG('a',64),DAG('b',64),DAG('p',343)","DAG('a',16),DAG('b',16),DAG('p',49)").replace('range(8)','range(4)').replace('*8+','*4+').replace('==343','==49').replace('*128*32','*128*64')
 s=s[:lo]+plan+s[hi:];s=s.replace('range(343)','range(49)').replace('capacity=343','capacity=49').replace('343+i','49+i')
 (D/'frozen-plan.py').write_text(s);ns=dict(__file__=str(D/'frozen-plan.py'),__name__='winograd49');exec(compile(s,str(D/'frozen-plan.py'),'exec'),ns)
 a,b,c,leaves,roots,_=ns['plan']();rec,regs,capacity=ns['reconstruct'](c,roots)
 bounds=[]
 for name,e in c.symbols.items():
  terms={}
  for m,sign in e.items():
   for ai,av in a.symbols[leaves[m][0]].items():
    for bi,bv in b.symbols[leaves[m][1]].items():terms[ai,bi]=terms.get((ai,bi),0)+sign*av*bv
  bound=sum(abs(v) for v in terms.values())*127*128*64;assert bound<2**31
  bounds.append(dict(name=name,bound=bound))
 proof=dict(rank=49,input_nodes=len(a.nodes),weight_nodes=len(b.nodes),reconstruction_nodes=len(c.nodes),reconstruction_registers=capacity,all_roots_are_original_dots=True,all_i32_intermediates_fit=True,bounds=bounds,max_bound=max(v['bound']for v in bounds),input_i16_bound=max(sum(abs(v)for v in e.values())for e in a.symbols.values())*127,weight_i16_bound=max(sum(abs(v)for v in e.values())for e in b.symbols.values())*128)
 (D/'integer-bounds.json').write_text(json.dumps(proof,indent=2)+'\n')
 upstream=ROOT/'artifacts/s2-pair88-v1/frozen-builder.py';s=upstream.read_text().replace('artifacts/s2-pair88-v1/build','artifacts/s2-winograd-v1/build')
 lo=s.index('    symbolic =');hi=s.index('    # The imported',lo)
 load="""    ns=dict(__file__=__file__,__name__='winograd49');exec(compile((ROOT/'artifacts/s2-winograd-v1/frozen-plan.py').read_text(),'<winograd49>','exec'),ns)
    a_dag,b_dag,c_dag,leaf_names,root_names,_=ns['plan']()
    leaves=[(a_dag.symbols[a],b_dag.symbols[b])for a,b in leaf_names]
    result=[[c_dag.symbols[v]for v in row]for row in root_names]
    reconstruction,root_registers,capacity=ns['reconstruct'](c_dag,root_names)
"""
 s=s[:lo]+load+s[hi:]
 lo=s.index('    positions =');hi=s.index("    prep = '''",lo);s=s[:lo]+s[hi:]
 lo=s.index('    for m, (a, _)');hi=s.index("    prep += '}}}}",lo)
 prep="""    for name,terms in a_dag.nodes:
        entries=list(terms.items());parent,sign=entries[0];value=('x'+parent[1:] if parent.startswith('a')and not parent.startswith('ad')else parent) if sign>0 else 'i16x8_sub(i16x8_splat(0),'+('x'+parent[1:] if parent.startswith('a')and not parent.startswith('ad')else parent)+')'
        for parent,sign in entries[1:]:
            arg='x'+parent[1:] if parent.startswith('a')and not parent.startswith('ad')else parent
            value=f'i16x8_{"add" if sign>0 else "sub"}({value},{arg})'
        prep+=f'let {name}={value};\\n'
    for m,(name,_)in enumerate(leaf_names):
        value='x'+name[1:] if name.startswith('a')and not name.startswith('ad')else name
        prep+=f'let value={value};let dst=out.add(({m}*groups+group)*(cols/2)+block*128+k*2);\\n'
        for half in range(2):
            idx=','.join(str(half*8+i)for _ in range(2)for i in range(8));prep+=f'v128_store(dst.add({half*8}).cast(),i8x16_shuffle::<{idx}>(value,value));\\n'
"""
 s=s[:lo]+prep+s[hi:]
 anchor="    shutil.copyfile(ROOT/'scripts/wat_s2_lane_bench/src/coeff.rs', src/'coeff.rs')"
 coeff="""    coeff='// Exact independently proved rank49 Winograd coefficients.\\n'
    for name,expressions,kind in [('A',[v[0]for v in leaves],'i16'),('B',[v[1]for v in leaves],'i16'),('C',[e for row in result for e in row],'i32')]:
        coeff+=f'pub(super)const {name}:[&[(usize,{kind})];{len(expressions)}]=[\\n'+''.join('&['+','.join(f'({k},{v})'for k,v in sorted(e.items()))+'],\\n'for e in expressions)+'];\\n'
    (src/'coeff.rs').write_text(coeff)
"""
 assert s.count(anchor)==1;s=s.replace(anchor,coeff)
 lo=s.index('    for m in [0,2,3,5,6]:');hi=s.index('    for m in range(49):',lo)
 s=s[:lo]+"    for name,_ in b_dag.nodes:lines.append(f'(local ${name} v128)')\n    for i in range(49,capacity):lines.append(f'(local $pc{i} v128)')\n"+s[hi:]
 lo=s.index('                    for m in [0,2,3,5,6]:');hi=s.index('            for m in range(49):',lo)
 bcode="""                    for name,terms in b_dag.nodes:
                        out+=ns['expression'](terms,lambda n:f'local.get ${n}','i16x8')+[f'local.set ${name}']
                    for m,(_,name)in enumerate(leaf_names):out+=[f'local.get ${name}',f'local.set $w{m}_{j}_{k}']
"""
 s=s[:lo]+bcode+s[hi:]
 lo=s.index('            for m in range(7):',s.index('    def row('));hi=s.index('            for ti in range(4):',lo)
 s=s[:lo]+"            import re\n            out += [re.sub(r'\\$pc(\\d+)',lambda m:'$p'+m[1]if int(m[1])<49 else m[0],v)for v in reconstruction]\n"+s[hi:]
 oldline="                    out += combine(single_c[(ti//2)*2+ri//2],lambda m:f'z{m}_{(ti%2)*2+ri%2}','i32x4')+[f'local.set $c{ri}']"
 assert s.count(oldline)==1;s=s.replace(oldline,"                    register=root_registers[ti][ri];out += [f'local.get ${\"p\" if register<49 else \"pc\"}{register}',f'local.set $c{ri}']")
 (D/'frozen-builder.py').write_text(s)
 (D/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),old,upstream,D/'frozen-plan.py',D/'frozen-builder.py',D/'integer-bounds.json']},indent=2)+'\n')
 exec(compile(s,str(D/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
