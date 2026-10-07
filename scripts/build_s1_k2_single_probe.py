#!/usr/bin/env python3
"""Add a layout-compatible exact single-token path without generic rank7 preparation."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'artifacts/s1-k2-compact-v1';d=ROOT/'artifacts/s1-k2-single-v1';d.mkdir(exist_ok=False)
 s=(old/'frozen-builder.py').read_text().replace('s1-k2-compact-v1','s1-k2-single-v1')
 patch=" p=src/'winograd.rs';p.write_text(p.read_text()+(ROOT/'scripts/s1_k2_single.rs').read_text())\n p=src/'lib.rs';text=p.read_text()\n before='let wa=(method==4).then(||winograd::operands(&q));';assert text.count(before)==1;text=text.replace(before,'let wa=(method==4&&n>1).then(||winograd::operands(&q));')\n before='let out=if method==4{f.win.as_ref().unwrap().project_wide(&q,wa.as_ref().unwrap(),&f.scales[..rows],rows).unwrap()}';after='let out=if method==4&&n==1{f.win.as_ref().unwrap().project_single(&q,&f.scales[..rows],rows).unwrap()}else if method==4{f.win.as_ref().unwrap().project_wide(&q,wa.as_ref().unwrap(),&f.scales[..rows],rows).unwrap()}';assert text.count(before)==1;text=text.replace(before,after);p.write_text(text)\n"
 anchor=" p=src/'lib.rs'\n cmd=nb['command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 (d/'frozen-builder.py').write_text(s);files=[Path(__file__),ROOT/'scripts/s1_k2_single.rs',old/'frozen-builder.py',old/'build/report.json',d/'frozen-builder.py'];(d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
