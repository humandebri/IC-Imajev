#!/usr/bin/env python3
"""Hold saved SIMD value across a balanced straight-line interval; execute all312 memory fixtures."""
import hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
BINARY={'i32x4.dot_i16x8_s','i32x4.add','i32x4.sub','i16x8.add','i16x8.sub','f32x4.mul','f32x4.add','f32x4.sub','i8x16.shuffle'}
def effect(line):
    op=line.strip().split(' ')[0]
    if op in {'local.get','v128.const'}:return 0,1
    if op=='local.set':return 1,0
    if op in {'local.tee','v128.load','f32x4.convert_i32x4_s'}:return 1,1
    if op in BINARY:return 2,1
    if op=='v128.store':return 2,0
    if not op:return 0,0
    return None
def transform(text):
    lines=text.splitlines(keepends=True);positions=[];i=0
    while i<len(lines):
        match=re.fullmatch(r'local\.set (\$\w+)',lines[i].strip())
        if match:
            h=0;name=match[1]
            for j in range(i+1,min(i+501,len(lines))):
                line=lines[j].strip();ef=effect(line)
                if ef is None:break
                if re.fullmatch(r'local\.(?:get|set|tee) '+re.escape(name),line):
                    if line=='local.get '+name and h==0:
                        positions.append(dict(set_line=i,get_line=j,local=name))
                        lines[i]=lines[i].replace('local.set','local.tee',1);lines[j]='';i=j
                    break
                pop,push=ef
                if h<pop:break
                h+=push-pop
        i+=1
    return ''.join(lines),positions
def main():
    parent=ROOT/'artifacts/update-local-tee-projection-v1/build'
    base=json.loads((parent/'report.json').read_text());assert sha(parent/'full.wasm')==base['wasm_sha256']
    paths={h:ROOT/p for p,h in base['source_hashes'].items()if p.endswith('.wat')}
    d=ROOT/'artifacts/stack-carry-projection-kernels-v1';d.mkdir(exist_ok=False)
    files=[Path(__file__),parent/'report.json'];kernels=[]
    for entry in base['patches']:
        path=paths[entry['source_sha256']];assert sha(path)==entry['source_sha256'];text=path.read_text()
        if not(entry['export'].startswith('__imajev_win7_')or entry['export'].startswith('__imajev_s2_k2_')):continue
        modified,positions=transform(text);count=len(positions)
        if not count:continue
        # Each interval has no control boundary or access to the held local.
        # Its complete pop/push trace never consumes the underlying saved value.
        assert [s for s in text.splitlines()if s.startswith('f32x4.')]==[s for s in modified.splitlines()if s.startswith('f32x4.')]
        target=d/(entry['export']+'.wat');target.write_text(modified)
        imported=d/(entry['export']+'-imported.wat');imported.write_text(modified.replace('(module(func','(module(import "env" "memory" (memory 1))(func',1))
        wasm=d/(entry['export']+'.wasm')
        subprocess.run([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-audit'),'wat',str(imported),str(wasm)],check=True)
        files.extend([path,target,imported,wasm]);kernels.append(dict(symbol=entry['export'],source=str(path.relative_to(ROOT)),source_sha256=sha(path),path=str(target.relative_to(ROOT)),wasm=str(wasm),replacements=count,sites=positions))
    assert len(kernels)==12 and sum(k['replacements']for k in kernels)==3072
    runner=d/'runner.cjs';runner.write_text('''const fs=require('fs');
const config=JSON.parse(fs.readFileSync(process.argv[2]));
let results=[];
for(const group of config.groups){
 const memory=new WebAssembly.Memory({initial:1});const instances={};
 for(const[key,item]of Object.entries(group.modules))instances[key]=new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(item.path)),{env:{memory}});
 for(const c of group.cases){
  const input=fs.readFileSync(c.input),expected=fs.readFileSync(c.expected);
  if(input.length>memory.buffer.byteLength)memory.grow(Math.ceil((input.length-memory.buffer.byteLength)/65536));
  new Uint8Array(memory.buffer).set(input);
  for(let start=0;start<c.cols;start+=256){
   const key=c.rows+'/'+Number(c.seed&&start===0),item=group.modules[key];
   instances[key].exports[item.symbol](c.q,c.w,c.cols,start,c.sx+start/256*4,c.stride,c.sw,c.out,c.n);
  }
  if(!Buffer.from(memory.buffer,0,input.length).equals(expected))throw Error(JSON.stringify({family:group.family,rows:c.rows,cols:c.cols,n:c.n,seed:c.seed}));
  results.push({family:group.family,rows:c.rows,cols:c.cols,n:c.n,seed:c.seed,all_memory_bytes_exact:true});
 }
}
fs.writeFileSync(config.output,JSON.stringify(results,null,2)+'\\n');console.log(`PASS ${results.length} complete memory comparisons`);
''')
    groups=[]
    for family,folder in [('rank7','k2-contiguous-roots-kernels-v1'),('rank49','s2-cached-tail-dot-reuse-kernels-v1')]:
        prior=ROOT/'artifacts'/folder;r=json.loads((prior/'report.json').read_text());execution=json.loads((prior/'execution-report.json').read_text());assert execution['complete']
        # Validate all old fixture/oracle/runner identity before reusing them.
        for p,h in execution['source_hashes'].items():assert sha(ROOT/p)==h,p;files.append(ROOT/p)
        files.extend([prior/'report.json',prior/'execution-report.json',prior/'execution/config.json'])
        config=json.loads((prior/'execution/config.json').read_text());modules={}
        for old in r['kernels']:
            symbol=old['symbol'];new=next(k for k in kernels if k['symbol']==symbol)
            assert old['symbol']==new['symbol'] # Fixtures remain pinned scalar reference data.
            modules[f'{old["tile"]}/{int(old["seed"])}']=dict(path=new['wasm'],symbol=symbol)
        groups.append(dict(family=family,modules=modules,cases=config['cases']))
    config=d/'config.json';config.write_text(json.dumps(dict(groups=groups,output=str(d/'node-results.json')),indent=2)+'\n')
    run=subprocess.run(['node',str(runner),str(config)],capture_output=True,text=True)
    (d/'node-stdout.txt').write_text(run.stdout);(d/'node-stderr.txt').write_text(run.stderr);run.check_returncode()
    results=json.loads((d/'node-results.json').read_text());assert len(results)==312 and all(c['all_memory_bytes_exact']for c in results)
    files.extend([runner,config,d/'node-results.json',d/'node-stdout.txt',d/'node-stderr.txt'])
    report=dict(complete=True,kernels=kernels,replacements=3072,conditions=312,all_memory_bytes_exact=True,parent_module=base['wasm_sha256'],source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},ic_performance_verified=False,full_paid_goal_achieved=False,adopted=False,scope='Only straight-line balanced stack intervals hold same-local tee value until later get. No interval control or local mutation; pop/push trace protects held value. Identical arithmetic/float/memory order. Actual SIMD full-memory312 fixtures. No IC or full inference savings claim.')
    (d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(run.stdout,end='')
if __name__=='__main__':main()
