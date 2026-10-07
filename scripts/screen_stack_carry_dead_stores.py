#!/usr/bin/env python3
"""Read-only, bounded dead-store screening of the current actual WAT bodies."""
import collections,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parent=ROOT/'artifacts/update-stack-carry-projection-v1/build';r=json.loads((parent/'report.json').read_text())
    assert sha(parent/'full.wasm')==r['wasm_sha256']
    paths={h:ROOT/p for p,h in r['source_hashes'].items()if p.endswith('.wat')}
    allowed={'local.get','local.set','local.tee','v128.const','v128.load','v128.store','f32x4.convert_i32x4_s','i32x4.dot_i16x8_s','i32x4.add','i32x4.sub','i16x8.add','i16x8.sub','f32x4.mul','f32x4.add','f32x4.sub','i8x16.shuffle'}
    rows=[];files=[Path(__file__),parent/'report.json',parent/'full.wasm']
    for e in r['patches']:
        p=paths[e['source_sha256']];assert sha(p)==e['source_sha256'];files.append(p);text=p.read_text()
        reads=collections.Counter(re.findall(r'local\.get\s+(\$[^\s()]+)',text))
        tees=collections.Counter(re.findall(r'local\.tee\s+(\$[^\s()]+)',text))
        unread={name:n for name,n in tees.items()if reads[name]==0}
        bounded=[];selected=e['export'].startswith(('__imajev_win7_','__imajev_s2_k2_'))
        if selected:
            lines=text.splitlines()
            for i,line in enumerate(lines):
                m=re.fullmatch(r'local\.tee (\$\w+)',line.strip())
                if not m:continue
                for j in range(i+1,min(i+501,len(lines))):
                    fields=lines[j].strip().split()
                    if not fields:continue
                    if fields[0]not in allowed:break
                    if fields[0].startswith('local.')and fields[1]==m[1]:
                        if fields[0]in {'local.set','local.tee'}:bounded.append(dict(tee_line=i,overwrite_line=j,local=m[1]))
                        break
        rows.append(dict(symbol=e['export'],globally_unread_tee_writes=unread,bounded_scan_selected=selected,bounded_overwritten_before_read=bounded))
    assert len(rows)==34 and sum(x['bounded_scan_selected']for x in rows)==12
    result=dict(complete=True,parent=r['wasm_sha256'],kernels=rows,global_unread_tee_sites=sum(sum(x['globally_unread_tee_writes'].values())for x in rows),bounded_dead_tee_sites=sum(len(x['bounded_overwritten_before_read'])for x in rows),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},scope='All34 source WAT local names read counts; twelve main kernels flat whitelist scan at most500 lines, stops at first access/control/folded/unknown. Zero does not exclude CFG-aware or reordered optimizations. No module/canister change, IC timing or goal/adoption claim.')
    d=ROOT/'artifacts/stack-carry-dead-store-screen-v1';d.mkdir(exist_ok=False);(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]for k in ['global_unread_tee_sites','bounded_dead_tee_sites']}))
if __name__=='__main__':main()
