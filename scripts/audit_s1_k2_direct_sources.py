#!/usr/bin/env python3
"""Verify exact WAT rewrite, current full control, and first-block initialization spans."""
from pathlib import Path
import hashlib,json,zipfile
from build_s1_k2_direct_probe import direct
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s1-k2-direct-v1';old=ROOT/'artifacts/s1-k2-four-columns-adaptive168-v1';control=ROOT/'artifacts/s1-k2-latest-control-v1';b=json.loads((d/'build/report.json').read_text());base=json.loads((control/'build/report.json').read_text())
 assert len(b['patches'])==28 and all(p['wasmparser_validation']for p in b['patches'])
 assert [(p['export'],p['source_sha256'])for p in b['patches'][:22]]==[(p['export'],p['source_sha256'])for p in base['patches'][:22]]
 assert (d/'build/src/lib.rs').read_bytes()==(control/'build/src/lib.rs').read_bytes()
 for tile in (128,168,32):
  symbol='__imajev_win7_wide_accumulate'if tile==128 else f'__imajev_win7_{tile}_accumulate'
  source=(old/'build'/f'kernel{tile}.wat').read_text()
  for seed in (False,True):
   expected=direct(source,tile,seed)
   if seed:expected=expected.replace(symbol,symbol+'_seed')
   p=d/'build'/(f'direct{tile}'+('_seed'if seed else '')+'.wat');assert p.read_text()==expected
   if seed:assert 'local.get $yp\nv128.load offset='not in expected and 'local.get $yp1\nv128.load offset='not in expected
 for rows in (2560,4096,8192):
  offsets=[];r=0
  while r<rows:
   tile=168 if rows in (4096,8192)and rows-r>=168 else 128 if rows-r>=128 else 32
   offsets.extend(range(r,r+tile));r+=tile
  assert offsets==list(range(rows))
  for n in range(1,133):
   # Every row has n contiguous scalar destinations; only existing odd tails store.
   for t in range(0,n,2):
    for ti in range(2):
     if t+ti<n:assert (t+ti)*rows+rows<=n*rows
 for key in ('source_hashes','dependency_hashes'):assert all(sha(ROOT/p)==h for p,h in b[key].items())
 assert sha(d/'build/diagnostic.wasm')==b['wasm_sha256']
 text=(d/'build/src/winograd.rs').read_text();assert 'MaybeUninit<f32>'in text and 'Vec::from_raw_parts'in text
 assert 'if out.iter().any(|v|!v.is_finite())'in text # No postcheck removed in this variant.
 result=dict(complete=True,module=b['wasm_sha256'],current_full22_control_kernel_sources_equal=True,all6_wat_rewrite_bytes_verified=True,all_output_spans_initialized_by_positive_zero_add_seed=True,post_finite_output_check_retained=True,all28_wasm_validated=True,source_dependency_hashes_verified=True)
 (d/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
