#!/usr/bin/env python3
"""Integer tensor basis for rank343; reuse the audited DAG/register emitter."""
from pathlib import Path
from itertools import product
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/s3-winograd-v1/frozen-generator.py'
ns={'__name__':'base','__file__':str(BASE)}
exec(compile(BASE.read_text(),str(BASE),'exec'),ns)
DAG=ns['DAG']
reconstruct=ns['reconstruct']

def axis(dag,values,shape,i,fn,size):
    result={}
    others=[j for j in range(len(shape)) if j!=i]
    for key in product(*(range(shape[j]) for j in others)):
        index=[0]*len(shape)
        for j,v in zip(others,key):index[j]=v
        args=[]
        for k in range(shape[i]):
            index[i]=k;args.append(values[tuple(index)])
        output=fn(dag,args)
        assert len(output)==size
        for k,v in enumerate(output):index[i]=k;result[tuple(index)]=v
    shape=list(shape);shape[i]=size
    return result,shape

def coords(row,col):
    return tuple(2*((row>>i)&1)+((col>>i)&1) for i in [2,1,0])

def ta(d,x):return [x[0],x[1],d.add(x[2],x[3]),x[3]]
def tb(d,x):return [x[0],x[2],x[3],d.add(x[1],x[0],-1)]
def la(d,x):
    s2=d.add(x[2],x[0],-1)
    return [x[0],x[1],d.add(x[1],s2,-1),x[3],x[2],s2,d.add(x[3],s2,-1)]
def lb(d,x):
    t2=d.add(x[2],x[3],-1)
    return [x[0],x[1],x[2],d.add(t2,x[1],-1),x[3],t2,d.add(t2,x[0],-1)]
def pc(d,p):
    u=d.add(d.add(p[0],p[5]),p[6])
    return [d.add(p[0],p[1]),d.add(u,p[3],-1),d.add(p[4],p[3]),d.add(p[2],p[6],-1)]
def tc_inv(d,z):
    c11=d.add(z[1],z[2])
    return [z[0],d.add(c11,z[3]),z[1],c11]

def plan():
    a,b,c=DAG('a',64),DAG('b',64),DAG('p',343)
    av={coords(r,k):f'a{r*8+k}' for r in range(8) for k in range(8)}
    bv={coords(k,n):f'b{k*8+n}' for k in range(8) for n in range(8)}
    for dag,v,transform,encode in [(a,av,ta,la),(b,bv,tb,lb)]:
        shape=[4]*3
        for i in range(3):v,shape=axis(dag,v,shape,i,transform,4)
        for i in range(3):v,shape=axis(dag,v,shape,i,encode,7)
        if dag is a:av=v
        else:bv=v
    keys=list(product(range(7),repeat=3))
    leaves=[(av[k],bv[k]) for k in keys]
    cv={k:f'p{m}' for m,k in enumerate(keys)};shape=[7]*3
    for i in range(3):cv,shape=axis(c,cv,shape,i,pc,4)
    for i in range(3):cv,shape=axis(c,cv,shape,i,tc_inv,4)
    roots=[[cv[coords(r,n)] for n in range(8)] for r in range(8)]
    bound=max(sum(abs(s)*sum(abs(v) for v in a.symbols[leaves[m][0]].values())*127*
                  sum(abs(v) for v in b.symbols[leaves[m][1]].values())*128*32
                  for m,s in expression.items()) for expression in c.symbols.values())
    return a,b,c,leaves,roots,bound

if __name__=='__main__':
    a,b,c,l,r,bound=plan()
    print(len(a.nodes),len(b.nodes),len(c.nodes),bound)
