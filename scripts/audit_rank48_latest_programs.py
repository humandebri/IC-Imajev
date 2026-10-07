#!/usr/bin/env python3
"""Parse author SLP as strict linear arithmetic data; compare every output toSMS."""
from pathlib import Path
from fractions import Fraction
import json,hashlib,ast,re,itertools,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/rank48-latest-programs-v1';r=json.loads((d/'report.json').read_text());assert all(sha(ROOT/i['path'])==i['sha256']for i in r['files']);parser=ROOT/'scripts/audit_rank48_primary_coefficients.py';ns=dict(__name__='parse',__file__=str(parser));exec(compile(parser.read_text(),str(parser),'exec'),ns);mats={p.stem:ns['parse'](p)for p in d.glob('*.sms')};reports=[]
 for path in d.glob('*.slp'):
  mat=mats.get(path.stem);cols=len(mat[0])if mat is not None else max(int(v)for v in re.findall(r'\bi(\d+)\b',path.read_text()))+1;assert cols<=64;outrows=len(mat)if mat is not None else max(int(v)for v in re.findall(r'^o(\d+)\s*:=',path.read_text(),re.M))+1;env={f'i{i}':tuple(Fraction(int(j==i))for j in range(cols))for i in range(cols)};counts=dict(add_sub=0,scalar_multiply_divide=0,assignments=0)
  def evaluate(node):
   if isinstance(node,ast.Name):assert node.id in env;return env[node.id]
   if isinstance(node,ast.Constant):assert type(node.value)is int;return Fraction(node.value)
   if isinstance(node,ast.UnaryOp)and isinstance(node.op,(ast.USub,ast.UAdd)):
    v=evaluate(node.operand);s=-1 if isinstance(node.op,ast.USub)else 1;return tuple(s*x for x in v)if isinstance(v,tuple)else s*v
   assert isinstance(node,ast.BinOp)and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div))
   x=evaluate(node.left);y=evaluate(node.right)
   if isinstance(node.op,(ast.Add,ast.Sub)):
    counts['add_sub']+=1;s=-1 if isinstance(node.op,ast.Sub)else 1;assert type(x)is type(y)
    return tuple(a+s*b for a,b in zip(x,y))if isinstance(x,tuple)else x+s*y
   counts['scalar_multiply_divide']+=1
   if isinstance(node.op,ast.Div):assert isinstance(y,Fraction)and y!=0;return tuple(v/y for v in x)if isinstance(x,tuple)else x/y
   assert not(isinstance(x,tuple)and isinstance(y,tuple))
   if isinstance(x,tuple):return tuple(v*y for v in x)
   if isinstance(y,tuple):return tuple(x*v for v in y)
   return x*y
  for line in path.read_text().splitlines():
   if ':='not in line:continue
   line=line.split(';')[0];name,expression=line.split(':=');name=name.strip();assert re.fullmatch(r'[a-z]\d{1,5}',name);env[name]=evaluate(ast.parse(expression.strip(),mode='eval').body);counts['assignments']+=1
  out=[env[f'o{i}']for i in range(outrows)]
  if mat is not None:assert out==[tuple(row)for row in mat],path
  else:mats[path.stem]=[list(v)for v in out]
  reports.append(dict(path=str(path.relative_to(ROOT)),all_symbolic_outputs_equal_SMS=mat is not None,paired32composition_checked=mat is None,shape=[outrows,cols],operation_counts=counts))
 exact=0
 L,R,P=[mats['4x4x4_48_204_'+x]for x in ['L','R','P']]
 for o,a,b in itertools.product(range(16),repeat=3):
  actual=sum((P[o][k]*L[k][a]*R[k][b]for k in range(48)),Fraction(0));assert actual==int(a//4==o//4 and a%4==b//4 and b%4==o%4);exact+=1
 compositions={}
 for size in [16,24]:
  for kind in ['L','R','P']:
   prefix=f'4x4x4_48_204-{size}';alt=mats[prefix+'ALT_'+kind];cob=mats[prefix+'CoB_'+kind];actual=ns['mm'](cob,alt)if kind=='P'else ns['mm'](alt,cob);assert actual==mats['4x4x4_48_204_'+kind];compositions[f'{size}/{kind}']=True
 assert ns['mm'](mats['4x4x4_48_204-32CoB_P'],mats['4x4x4_48_204-32ALT_P'])==P;compositions['32/P']=True
 files=[Path(__file__),parser,d/'report.json']+[ROOT/i['path']for i in r['files']]+[ROOT/'artifacts/rank48-primary-coefficients-v1/Licence_CeCILL-B_V1-en.txt',ROOT/'artifacts/rank48-primary-coefficients-v1/AUTHORS.md'];result=dict(complete=True,all_4096_base_tensor_identities_exact=True,all_SLPSymbolic_outputs_validated_against_SMS_or32composition=True,programs=reports,all16_and24_basis_compositions_exact=True,compositions=compositions,source_commit=r['source_commit'],source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Pinned primary-data symbolic linear-program validation only. SLP parsed with Fraction/strictAST, no Python/C++ source executed. Operation counts syntactic, not IC prices. NoWasm/fullpaidperformanceclaim.')
 (d/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(d/'evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'audit.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(programs=len(reports),tensor_identities=exact,all_compose=True)))
if __name__=='__main__':main()
