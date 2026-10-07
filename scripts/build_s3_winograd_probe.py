#!/usr/bin/env python3
"""Exact rank343 Winograd over I32 residues; F32 sees only bounded original dots."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s3-winograd-v1';d.mkdir(exist_ok=False)
 p=ROOT/'artifacts/s3-inline-v1/frozen-generator.py';s=p.read_text().replace('artifacts/s3-inline-v1/generated','artifacts/s3-winograd-v1/generated')
 lo=s.index('        p0=multiply(');hi=s.index('    result=multiply(',lo)
 formula='''        s1=a.matrix_add(a21,a22)
        s2=a.matrix_add(s1,a11,-1)
        s3=a.matrix_add(a11,a21,-1)
        s4=a.matrix_add(a12,s2,-1)
        t1=b.matrix_add(b12,b11,-1)
        t2=b.matrix_add(b22,t1,-1)
        t3=b.matrix_add(b22,b12,-1)
        t4=b.matrix_add(t2,b21,-1)
        p1=multiply(a11,b11)
        p2=multiply(a12,b21)
        p3=multiply(s4,b22)
        p4=multiply(a22,t4)
        p5=multiply(s1,t1)
        p6=multiply(s2,t2)
        p7=multiply(s3,t3)
        u1=c.matrix_add(p1,p2)
        u2=c.matrix_add(p1,p6)
        u3=c.matrix_add(u2,p7)
        u4=c.matrix_add(u2,p5)
        u5=c.matrix_add(u4,p3)
        u6=c.matrix_add(u3,p4,-1)
        u7=c.matrix_add(u3,p5)
        return [xx+yy for xx,yy in zip(u1,u5)]+[xx+yy for xx,yy in zip(u6,u7)]
'''
 s=s[:lo]+formula+s[hi:]
 before='max_bound=max(max_bound,bound); assert bound<2**31';assert s.count(before)==1
 s=s.replace(before,'max_bound=max(max_bound,bound) # Reconstruction is in Z/(2**32); final original dot is bounded.')
 before='    return a,b,c,leaves,result,max_bound';assert s.count(before)==1
 s=s.replace(before,'    assert 256*127*128 < 2**31 # Unique signed lift of every original dot.\n'+before)
 (d/'frozen-generator.py').write_text(s)
 ns=dict(__file__=__file__,__name__='winograd_plan');exec(compile(s,str(p),'exec'),ns)
 a,b,c,leaves,roots,bound=ns['plan']()
 proof=dict(rank=len(leaves),input_nodes=len(a.nodes),weight_nodes=len(b.nodes),reconstruction_nodes=len(c.nodes),input_i16_bound=max(sum(abs(v) for v in e.values()) for e in a.symbols.values())*127,weight_i16_bound=max(sum(abs(v) for v in e.values()) for e in b.symbols.values())*128,unreduced_reconstruction_bound=bound,final_dot_bound=256*127*128,symbolic_bilinear_identity=True,arithmetic='I16 transforms fit exactly; I32 dot/add/sub operate modulo2^32. Every final root is the original bounded integer dot residue, so signed conversion equals the original dot. F32 conversion/scaling occurs only afterward, in original block order.')
 (d/'ring-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
 p=ROOT/'scripts/build_s3_prepared_probe.py';s=p.read_text().replace('artifacts/s3-prepared-v1/build','artifacts/s3-winograd-v1/build')
 s=s.replace('from generate_s3_stream_probe import plan',"_ns=dict(__file__=__file__,__name__='winograd_plan');exec(compile((Path(__file__).resolve().parents[1]/'artifacts/s3-winograd-v1/frozen-generator.py').read_text(),'<winograd-plan>','exec'),_ns);plan=_ns['plan']")
 s=s.replace("wat = (original / 'kernel.wat').read_text()","wat = (ROOT/'artifacts/s3-winograd-v1/generated/kernel.wat').read_text()")
 anchor='    a, b, c, leaves, roots, bound = plan()';assert s.count(anchor)==1
 s=s.replace(anchor,"    (src/'prepare_s3.rs').write_bytes((ROOT/'artifacts/s3-winograd-v1/generated/prepare_s3.rs').read_bytes())\n"+anchor)
 (d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),p,ROOT/'artifacts/s3-inline-v1/frozen-generator.py',d/'frozen-generator.py',d/'frozen-builder.py',d/'ring-proof.json',d/'generated/generator.json',d/'generated/kernel.wat']
 (d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
