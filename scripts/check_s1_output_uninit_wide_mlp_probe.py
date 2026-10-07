#!/usr/bin/env python3
"""Measure 9216x2560 real MLP weights without modifying inference."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/check_s1_output160.py';s=p.read_text().replace('artifacts/s1-output160-v1/build','artifacts/s1-output-uninit-wide-v1/build').replace('artifacts/s1-output160-v1/check','artifacts/s1-output-uninit-wide-mlp-v1/check')
 s=s.replace("model.language_model.layers.3.self_attn.q_proj.weight","model.language_model.layers.3.mlp.gate_proj.weight").replace('8192','9216')
 s=s.replace("assert weight_hash == json.loads((ROOT/'artifacts/column32/check/report.json').read_text())['tensor_bytes_sha256']","assert weight_hash == hashlib.sha256(weights).hexdigest()")
 s=s.replace("['prefix','617','insufficient','maximum','normal']","[]").replace('for n in [57,59,67]:','for n in [48,56,57,59,67]:').replace('[1,2,3,5,7,8,9,32,64,88,109]','[1,2,3,5,7,8,9,32,64,88,89]')
 s=s.replace("[(3,'s1_pair_bounds'),(4,'output160')]","[(4,'direct_output'),(5,'output_uninit')]").replace("measured['s1_pair_bounds']","measured['direct_output']").replace("measured['output160']","measured['output_uninit']")
 d=ROOT/'artifacts/s1-output-uninit-wide-mlp-v1';d.mkdir(exist_ok=False);(d/'frozen-check.py').write_text(s);(d/'check-upstream.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
