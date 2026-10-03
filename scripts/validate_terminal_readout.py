#!/usr/bin/env python3
"""Validate terminal pruning and projection fusion on the selected local canister."""
import argparse,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--run-name',default='terminal-v3');ap.add_argument('--baseline',default='fused-stage');ap.add_argument('--fuse-mlp-norm',action='store_true');ap.add_argument('--fuse-norm-rope',action='store_true');ap.add_argument('--wire-codec',choices=['bf16-exact','bf16-block256-exact-v1'],default='bf16-exact');ap.add_argument('--reuse-projection-inputs',action='store_true');ap.add_argument('--frame-checksum',choices=['sha256','blake3'],default='sha256');ap.add_argument('--fuse-attention',action='store_true');ap.add_argument('--fuse-delta-projected',action='store_true');ap.add_argument('--fuse-delta-finish',action='store_true');ap.add_argument('--fuse-mlp-pipeline',action='store_true');ap.add_argument('--fuse-attention-full',action='store_true');ap.add_argument('--fuse-mlp-full',action='store_true');ap.add_argument('--fuse-delta-full-log',action='store_true');ap.add_argument('--mlp-full-token-cap',type=int,choices=[87,89],default=87);a=ap.parse_args()
 if not a.run_name.replace('-','').isalnum() or not a.baseline.replace('-','').isalnum():ap.error('run names must be alphanumeric with hyphens')
 common=['--frame-checksum',a.frame_checksum,'--canister',a.canister,'--arithmetic','int8','--wire-codec',a.wire_codec,'--compact-lossless','--fuse-add-norm','--fuse-mlp','--delta-head-cap','16','--attention-head-cap','8','--row-cap','16384','--work-cap','2500000000','--token-cap','132','--compact-heads','--fuse-delta','--wide-mlp','--terminal-readout','--fuse-delta-input']
 if a.fuse_delta_projected:common.append('--fuse-delta-projected')
 if a.fuse_delta_full_log:common.append('--fuse-delta-full-log')
 if a.fuse_delta_finish:common.append('--fuse-delta-finish')
 if a.fuse_mlp_pipeline:common.append('--fuse-mlp-pipeline')
 if a.fuse_attention_full:common.append('--fuse-attention-full')
 if a.fuse_mlp_full:common.extend(['--fuse-mlp-full','--mlp-full-token-cap',str(a.mlp_full_token_cap)])
 if a.fuse_attention:common.append('--fuse-attention')
 if a.reuse_projection_inputs:common.append('--reuse-projection-inputs')
 if a.fuse_mlp_norm:common.append('--fuse-mlp-norm')
 if a.fuse_norm_rope:common.append('--fuse-norm-rope')
 def run(script,*args):subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True)
 output_name='terminal-readout' if a.run_name=='terminal-v3' else a.run_name
 cache=f'artifacts/{a.run_name}-prefix/queries'
 run('run_prefix_canister.py',*common,'--prepare-prefix','--cache',cache,'--directory',f'artifacts/{a.run_name}-prefix')
 run('compare_compact_heads.py','--before',f'artifacts/{a.baseline}-prefix','--after',f'artifacts/{a.run_name}-prefix','--output',f'docs/{output_name}-prefix-results.json')
 for label,record in [('617',0),('insufficient',19),('maximum',11)]:
  options=[] if label=='617' else ['--reference','artifacts/reference-serving.json','--record',str(record)]
  directory=f'artifacts/{a.run_name}-{label}'
  run('run_prefix_canister.py',*common,'--cache',cache,'--directory',directory,*options)
  run('compare_compact_heads.py','--before',f'artifacts/{a.baseline}-{label}','--after',directory,'--output',f'docs/{output_name}-{label}-results.json')
 run('run_full_canister.py',*common,'--directory',f'artifacts/{a.run_name}-normal')
 run('compare_compact_heads.py','--before',f'artifacts/{a.baseline}-normal','--after',f'artifacts/{a.run_name}-normal','--output',f'docs/{output_name}-normal-results.json')
if __name__=='__main__':main()
