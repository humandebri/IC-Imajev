#!/usr/bin/env python3
"""Fresh scalar-basis audit of every operand byte and measured Candid reply."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    d=ROOT/'artifacts/rank7-prepare8-probe-v1';b=json.loads((d/'build.json').read_text());r=json.loads((d/'report.json').read_text())
    files={Path(__file__),d/'build.json',d/'report.json'}
    for v in [b,r]:
        for p,h in v['source_hashes'].items():assert sha(ROOT/p)==h,p;files.add(ROOT/p)
    coefficients=np.array([[1,0,0,0],[0,1,0,0],[1,1,-1,-1],[0,0,0,1],[0,0,1,1],[-1,0,1,1],[1,0,-1,0]],dtype=np.int64)
    fresh={}
    for c in b['cases']:
        q=np.frombuffer((ROOT/c['input']).read_bytes(),dtype='<i2').reshape(c['pairs']*2,c['cols'])
        expected=np.full((7,c['pairs'],c['cols']//2),0x5a5a,dtype='<i2')
        for pair in range(c['pairs']):
            for block in range(c['first'],c['end']):
                x=np.stack([q[pair*2+t,block*256+k*128:block*256+(k+1)*128] for t,k in [(0,0),(0,1),(1,0),(1,1)]]).astype(np.int64)
                values=coefficients@x;assert values.min()>=-32768 and values.max()<=32767
                expected[:,pair,block*128:(block+1)*128]=values
        fresh[c['index']]=expected.tobytes()
        assert fresh[c['index']]==(ROOT/c['expected']).read_bytes()
        for width in [4,8]:assert fresh[c['index']]==(d/f'check/node-{width}-{c["index"]}.bin').read_bytes()
    for v in r['replies']:
        p=ROOT/v['reply'];assert sha(p)==v['reply_sha256']
        raw=bytes.fromhex(p.read_text().strip().removeprefix('0x'))
        assert raw[:10]==bytes.fromhex('4449444c016d7b027800')
        count=int.from_bytes(raw[10:18],'little');assert count==v['instructions']
        at=18;size=0;shift=0
        while True:
            byte=raw[at];at+=1;size|=(byte&127)<<shift;shift+=7
            if byte<128:break
        assert size==len(fresh[v['index']]) and raw[at:]==fresh[v['index']]
    assert len(r['replies'])==14
    for v in r['comparisons']:
        assert v['prepare4']-v['prepare8']==v['saved'] and v['saved']>=0
        assert v['saved']>0 or v['first']==v['end']
    s=json.loads((d/'check/post-status.json').read_text());assert s['status']=='Stopped' and s['module_hash'].removeprefix('0x')==r['module']
    audit=dict(complete=True,all14_replies_independently_redecoded=True,all7_fresh_coefficient_matrix_operand_buffers_exact=True,all14_node_buffers_exact=True,
               partial_and_empty_ranges_preserve_unwritten_sentinels=True,all_nonempty_measured_cases_improve=True,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},adopted=False,full_paid_goal_achieved=False,scope=b['scope'])
    p=d/'independent-audit.json';p.write_text(json.dumps(audit,indent=2)+'\n');files.add(p)
    archive=d/'frozen-evidence.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED)as z:
        for p in sorted(files):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive)as z:
        assert z.testzip()is None
        for p in files:assert hashlib.sha256(z.read(str(p.relative_to(ROOT)))).hexdigest()==sha(p)
    (d/'archive-identity.json').write_text(json.dumps(dict(complete=True,sha256=sha(archive),all_archived_bytes_verified=True),indent=2)+'\n')
    print('independent matrix/reply/archive audit passed:',sha(archive))
if __name__=='__main__':main()
