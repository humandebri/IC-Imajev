#!/usr/bin/env python3
"""Prepare the fixed voting template before the new-value digits/question."""
import json,hashlib,sys,subprocess,zipfile
from pathlib import Path
from tokenizers import Tokenizer
from evaluate_prompt_accuracy import base_flags
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/voting-template-prefix-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(exist_ok=False)
 source=ROOT/'artifacts/text-short-v2/inputs.json';fixture=json.loads(source.read_text());tokenizer=Tokenizer.from_file(str(ROOT/'checkpoints/base/tokenizer.json'))
 expected='<|im_start|>user\nInspect the available evidence and answer the question using the stated criteria. Return only the single option code.\nState: "Minimum voting dissolve delay changes from 1 day to '
 ids=fixture['records'][0]['token_ids'][:38];actual=tokenizer.decode(ids,skip_special_tokens=False);assert actual==expected,(actual,expected)
 assert ids==fixture['records'][1]['token_ids'][:38]
 assert 'Question:' not in actual and actual.endswith(' to ') and '20000' not in actual
 paths=[Path(__file__),source,ROOT/'checkpoints/base/tokenizer.json',ROOT/'scripts/run_prefix_canister.py',ROOT/'scripts/prepare_prefix_reuse.py',ROOT/'artifacts/query-packing-v3/build/full.wasm',ROOT/'artifacts/query-packing-v3/build/imajev-client']
 hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
 metadata=dict(scope='Reusable fixed voting stem; ends before the entire variable new value and question. Other proposal templates retain the common27-token prefix.',tokens=38,stem=actual,token_ids=ids,source_hashes=hashes)
 (D/'template.json').write_text(json.dumps(metadata,indent=2)+'\n')
 with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
 cmd=base_flags()
 for flag,value in [('--reference',str(source)),('--cache',str(D/'prefix/queries'))]:cmd[cmd.index(flag)+1]=value
 cmd+=['--record','0','--prefix-tokens','38','--prepare-prefix','--directory',str(D/'prefix')]
 with (D/'prefix-preparation.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,check=True,stdout=log,stderr=log)
 with (D/'packet-preparation.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'scripts/prepare_prefix_reuse.py'),'--prefix-directory',str(D/'prefix'),'--directory',str(D/'packets'),'--canister','2vn5i-k3777-77775-aaaua-cai','--codec-module','e9644266f9bea8efdff5c1d5017b7c82beb67c3030279e6f27248b90fa51ef6e','--run-report',str(D/'packet-preparation.json')],cwd=ROOT,check=True,stdout=log,stderr=log)
 assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in paths}
 print(json.dumps(dict(stage='fixed-voting-template-ready',tokens=38)),flush=True)
if __name__=='__main__':main()
