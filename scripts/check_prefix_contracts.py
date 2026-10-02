#!/usr/bin/env python3
"""Reject wrong identity, altered files, and causal positions without modifying the source cache."""
import argparse,copy,json,pathlib,sys,tempfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from prefix_inference import load_cache,file_hash
p=argparse.ArgumentParser();p.add_argument('--cache',required=True);a=p.parse_args();source=ROOT/a.cache;m=json.loads((source/'cache.json').read_text());manifest=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());wasm=file_hash(ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm');results=[]
with tempfile.TemporaryDirectory(prefix='prefix-contract-',dir=ROOT/'artifacts') as temporary:
 d=pathlib.Path(temporary);(d/'states').mkdir()
 for f in m['files']:(d/f).symlink_to((source/f).resolve())
 def reject(name,metadata,expected):
  (d/'cache.json').write_text(json.dumps(metadata))
  try:load_cache(d,manifest,wasm)
  except ValueError as e:
   assert expected in str(e),(name,str(e));results.append(dict(case=name,rejected=True,error=str(e)));return
  raise AssertionError(name)
 for field in ['model','pack_hash','wasm_sha256','graph_sha256']:
  bad=copy.deepcopy(m);bad[field]='0'*64;reject('wrong-'+field,bad,'identity mismatch')
 bad=copy.deepcopy(m);bad['files']['../outside.npy']='0'*64;reject('outside-file',bad,'file manifest')
 bad=copy.deepcopy(m);bad['files']['layer-00.npy']='0'*64;reject('altered-file',bad,'file hash')
 f='states/layer-03.npz';(d/f).unlink()
 with np.load(source/f,allow_pickle=False) as raw:v={k:raw[k] for k in raw.files}
 v['positions']=v['positions']+1;np.savez(d/f,**v);bad=copy.deepcopy(m);bad['files'][f]=file_hash(d/f);reject('wrong-causal-position',bad,'prefix positions')
(ROOT/'docs/prefix-contracts.json').write_text(json.dumps(dict(cache_sha256=file_hash(source/'cache.json'),cases=results),indent=2)+'\n');print('Rejected',len(results),'invalid cache cases')
