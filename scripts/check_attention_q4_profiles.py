#!/usr/bin/env python3
"""Check frozen Q4 bridge coverage and that shared operands execute only once."""
import argparse,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def check(directory,all_layers=False):
 directory=pathlib.Path(directory);report=json.loads((directory/'report.json').read_text())
 if not report.get('q4'):raise ValueError('not a Q4 bridge proof')
 successes=[c for c in report['cases']if c['success']]
 if all_layers:
  expected={(label,layer,begin)for label in ['617','insufficient','maximum']for layer in [3,7,11,15,19,23,27]for begin in [5120,5376]}
  actual={(c['label'],c['layer'],c['begin'])for c in successes}
  if len(successes)!=42 or actual!=expected or len(report['cases'])!=42:raise ValueError('incomplete all-layer Q4 coverage')
 for c in successes:
  for name in ['kv_bitwise_equal','prepared_mlp_bitwise_equal','finished_hidden_norm_bitwise_equal']:
   if c.get(name)is not True:raise ValueError('missing bitwise comparison: '+name)
  first={name:count for name,_,count in c['first_profile']['ok']['spans']};second={name:count for name,_,count in c['profile']['ok']['spans']}
  if any(first.get(name)!=1 for name in ['bridge_q_A_once','bridge_q_first4','bridge_attention_quantize_once']):raise ValueError('first query shared operand count')
  if 'bridge_q_A_once'in second or second.get('lora_matmul_A')!=1:raise ValueError('second query repeated Q A; only out A should remain')
  if max(c['first_call']['ok']['instructions'],c['call']['ok']['instructions'])>=5_000_000_000:raise ValueError('query handler instruction bound')
 for c in report['cases']:
  if not c['success']:
   name=f"failed-{c['label']}-layer{c['layer']}-first.request.bin"if c['stage']=='mlp_complete_kv'else f"failed-{c['label']}-layer{c['layer']}-begin{c['begin']}.request.bin"
   if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=c['request_sha256']:raise ValueError('failed request evidence mismatch')
 return dict(successes=len(successes),failures=len(report['cases'])-len(successes),coverage=[[c['label'],c['layer'],c['begin']]for c in successes],profile_operand_counts_valid=True)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory');ap.add_argument('--all-layers',action='store_true');a=ap.parse_args();print(json.dumps(check(ROOT/a.directory,a.all_layers)))
if __name__=='__main__':main()
