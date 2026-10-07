#!/usr/bin/env python3
"""Replace only adjacent same-local set/get, then execute all saved SIMD cases."""
import hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
PATTERN=re.compile(r'(?m)^local\.set (\$[A-Za-z0-9_]+)\nlocal\.get \1(?=\n)')
def main():
    parent=ROOT/'artifacts/update-rank7-prepare8-unrolled-v1/build'
    base=json.loads((parent/'report.json').read_text());assert sha(parent/'full.wasm')==base['wasm_sha256']
    paths={h:ROOT/p for p,h in base['source_hashes'].items()if p.endswith('.wat')}
    d=ROOT/'artifacts/local-tee-projection-kernels-v1';d.mkdir(exist_ok=False)
    files=[Path(__file__),parent/'report.json'];kernels=[]
    for entry in base['patches']:
        path=paths[entry['source_sha256']];assert sha(path)==entry['source_sha256'];text=path.read_text()
        modified,count=PATTERN.subn(lambda m:'local.tee '+m[1],text)
        if not count:continue
        # Inverse expands exactly to the original bytes. Every changed operation
        # keeps the identical local value and operand-stack value, for any type.
        positions=[];cursor=0;rebuilt=''
        for match in PATTERN.finditer(text):
            rebuilt+=text[cursor:match.start()]+'local.tee '+match[1];cursor=match.end()
            positions.append(dict(offset=match.start(),local=match[1]))
        rebuilt+=text[cursor:];assert rebuilt==modified
        assert [s for s in text.splitlines()if s.startswith('f32x4.')]==[s for s in modified.splitlines()if s.startswith('f32x4.')]
        target=d/(entry['export']+'.wat');target.write_text(modified)
        imported=d/(entry['export']+'-imported.wat');imported.write_text(modified.replace('(module(func','(module(import "env" "memory" (memory 1))(func',1))
        wasm=d/(entry['export']+'.wasm')
        subprocess.run([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-audit'),'wat',str(imported),str(wasm)],check=True)
        files.extend([path,target,imported,wasm]);kernels.append(dict(symbol=entry['export'],source=str(path.relative_to(ROOT)),source_sha256=sha(path),path=str(target.relative_to(ROOT)),wasm=str(wasm),replacements=count,sites=positions))
    assert len(kernels)==12 and sum(k['replacements']for k in kernels)==524
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
            assert new['source_sha256']==sha(ROOT/old['path'])
            modules[f'{old["tile"]}/{int(old["seed"])}']=dict(path=new['wasm'],symbol=symbol)
        groups.append(dict(family=family,modules=modules,cases=config['cases']))
    config=d/'config.json';config.write_text(json.dumps(dict(groups=groups,output=str(d/'node-results.json')),indent=2)+'\n')
    run=subprocess.run(['node',str(runner),str(config)],capture_output=True,text=True)
    (d/'node-stdout.txt').write_text(run.stdout);(d/'node-stderr.txt').write_text(run.stderr);run.check_returncode()
    results=json.loads((d/'node-results.json').read_text());assert len(results)==312 and all(c['all_memory_bytes_exact']for c in results)
    files.extend([runner,config,d/'node-results.json',d/'node-stdout.txt',d/'node-stderr.txt'])
    report=dict(complete=True,kernels=kernels,replacements=524,conditions=312,all_memory_bytes_exact=True,parent_module=base['wasm_sha256'],source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},ic_performance_verified=False,full_paid_goal_achieved=False,adopted=False,scope='Only adjacent same-local set/get to tee. Identical stack/local value and arithmetic/float/memory order. Actual SIMD full-memory312 fixtures. No IC or full inference savings claim.')
    (d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(run.stdout,end='')
if __name__=='__main__':main()
