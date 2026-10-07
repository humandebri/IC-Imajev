#!/usr/bin/env python3
"""Independent line/token/type audit and fresh SIMD execution."""
import hashlib,json,re,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    d=ROOT/'artifacts/stack-carry-projection-kernels-v1';r=json.loads((d/'report.json').read_text())
    for p,h in r['source_hashes'].items():assert sha(ROOT/p)==h,p
    count=0
    for k in r['kernels']:
        old=(ROOT/k['source']).read_text();new=(ROOT/k['path']).read_text()
        types=dict(re.findall(r'\(local (\$[^\s()]+) (i32|i64|f32|f64|v128)\)',old))
        lines=old.splitlines(keepends=True);expected=list(lines);changes=0;last=-1
        arities={'local.get':(0,1),'v128.const':(0,1),'local.set':(1,0),'local.tee':(1,1),'v128.load':(1,1),'f32x4.convert_i32x4_s':(1,1),'v128.store':(2,0)}
        for op in ['i32x4.dot_i16x8_s','i32x4.add','i32x4.sub','i16x8.add','i16x8.sub','f32x4.mul','f32x4.add','f32x4.sub','i8x16.shuffle']:arities[op]=(2,1)
        for site in k['sites']:
            a,b,x=site['set_line'],site['get_line'],site['local']
            assert last<a<b and types[x]=='v128';last=b
            assert lines[a].strip()=='local.set '+x and lines[b].strip()=='local.get '+x
            stack=['HELD']
            for index in range(a+1,b):
                fields=lines[index].strip().split()
                if not fields:continue
                op=fields[0];assert op in arities,(index,op)
                assert not(op.startswith('local.')and fields[1]==x)
                pop,push=arities[op];assert len(stack)>=pop
                popped=[stack.pop()for _ in range(pop)];assert 'HELD'not in popped
                stack.extend([f'value@{index}']*push)
            assert stack==['HELD']
            expected[a]=lines[a].replace('local.set','local.tee',1);expected[b]='';changes+=1
        assert ''.join(expected)==new and changes==k['replacements'];count+=changes
        # All opcode sequences outside local access are literally unchanged.
        def other_ops(s):return [line for line in s.splitlines()if not line.startswith(('local.set ','local.get ','local.tee '))]
        assert other_ops(old)==other_ops(new)
    assert count==r['replacements']==3072
    config=json.loads((d/'config.json').read_text());config['output']=str(d/'independent-node-results.json')
    p=d/'independent-config.json';p.write_text(json.dumps(config,indent=2)+'\n')
    run=subprocess.run(['node',str(d/'runner.cjs'),str(p)],capture_output=True,text=True)
    (d/'independent-stdout.txt').write_text(run.stdout);(d/'independent-stderr.txt').write_text(run.stderr);run.check_returncode()
    results=json.loads((d/'independent-node-results.json').read_text());assert len(results)==312 and all(c['all_memory_bytes_exact']for c in results)
    files=[Path(__file__),d/'report.json',p,d/'independent-node-results.json',d/'independent-stdout.txt',d/'independent-stderr.txt']+[ROOT/p for p in r['source_hashes']]
    audit=dict(complete=True,all12_independent_line_and_v128_type_checks_exact=True, all3072_intervals_protect_held_stack_value=True,all_nonlocal_opcodes_byte_equal=True,all312_fresh_full_memory_cases_exact=True,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},ic_performance_verified=False,adopted=False,full_paid_goal_achieved=False,scope=r['scope'])
    out=d/'independent-audit.json';out.write_text(json.dumps(audit,indent=2)+'\n');files.append(out)
    archive=d/'frozen-evidence.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED)as z:
        for p in dict.fromkeys(files):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive)as z:
        assert z.testzip()is None
        for p in dict.fromkeys(files):assert hashlib.sha256(z.read(str(p.relative_to(ROOT)))).hexdigest()==sha(p)
    (d/'archive-identity.json').write_text(json.dumps(dict(complete=True,sha256=sha(archive),all_archived_bytes_verified=True),indent=2)+'\n');print(run.stdout,end='')
if __name__=='__main__':main()
