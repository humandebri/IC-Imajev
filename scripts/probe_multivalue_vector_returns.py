#!/usr/bin/env python3
"""Build a local-only ABI probe; not an inference kernel."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/multivalue-vector-returns-v1';d.mkdir(exist_ok=False)
 text='''(module
 (import "ic0" "performance_counter" (func $counter(param i32)(result i64)))
 (import "ic0" "msg_reply_data_append" (func $append(param i32 i32)))
 (import "ic0" "msg_reply" (func $reply))
 (memory(export "memory")1)
 (data(i32.const 0) "DIDL\\00\\02\\78\\78")
'''
 methods={}
 for n in [16,64,128,256]:
  constants='\n'.join(f'v128.const i32x4 {i+1} {i+1} {i+1} {i+1}'for i in range(n))
  text+=f'(func $return{n} (param '+ ' '.join(['i32']*9)+') (result '+' '.join(['v128']*n)+')\n'+constants+')\n'
  for mode in ['inline','helper']:
   name=f'{mode}{n}';methods[name]=dict(results=n,expected_checksum=n*(n+1)//2,iterations=1000)
   locals_=''.join(f'(local $r{i} v128)'for i in range(n))
   body=constants if mode=='inline' else '\n'.join(['i32.const 0']*9)+f'\ncall $return{n}'
   body+='\n'+'\n'.join(f'local.set $r{i}'for i in reversed(range(n)))
   checksum='i64.const 0\n'+'\n'.join(f'local.get $r{i} i32x4.extract_lane 0 i64.extend_i32_u i64.add'for i in range(n))
   text+=f'''(func(export "canister_query {name}")(export "canister_update {name}_update")
    {locals_}(local $i i32)(local $begin i64)
    i32.const 1000 local.set $i
    i32.const 0 call $counter local.set $begin
    (loop $again {body}
     local.get $i i32.const 1 i32.sub local.tee $i br_if $again)
    i32.const 8 i32.const 0 call $counter local.get $begin i64.sub i64.store
    i32.const 16 {checksum} i64.store
    i32.const 0 i32.const 24 call $append call $reply)
'''
 text+=')\n';(d/'probe.wat').write_text(text)
 subprocess.run([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-audit'),'wat',str(d/'probe.wat'),str(d/'probe.wasm')],check=True)
 for mode in ['query','update']:
  (d/f'{mode}.did').write_text('service:{'+''.join(f'{name if mode=="query" else name+"_update"}:()->(nat64,nat64){"query"if mode=="query"else""};'for name in methods)+'}')
 files=[Path(__file__),d/'probe.wat',d/'probe.wasm',d/'query.did',d/'update.did']
 r=dict(module=hashlib.sha256((d/'probe.wasm').read_bytes()).hexdigest(),methods=methods,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},scope='Actual multi-value V128 ABI and compound call cost probe. No inference claim.')
 (d/'build.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
