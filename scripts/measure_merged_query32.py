#!/usr/bin/env python3
"""Paired local measurements of the frozen 32-query schedule with merged INT8 weights."""
import argparse,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/merged-query32-v1'
sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
from prefix_inference import verify_module
from evaluate_prompt_accuracy import base_flags
VARIANTS={'before':('2hlkr-gl777-77775-aaaxa-cai','checkpoints/full-int8.manifest.json'),'after':('2akmf-lt777-77775-aaaxq-cai','artifacts/merged-adapter-v2/model.manifest.json')}
WASM='artifacts/merged-query32-v1/build/full.wasm';FIXTURE='artifacts/text-short-v2/inputs.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def module():return json.loads((D/'build/report.json').read_text())['wasm_sha256']
def command(variant):
 cid,manifest=VARIANTS[variant];cmd=base_flags();cmd[1]=str(ROOT/'scripts/run_merged_query32.py')
 for flag,value in [('--canister',cid),('--wasm',WASM),('--reference',FIXTURE),('--bridge-binary','artifacts/query32-v1/client-build/imajev-client')]:cmd[cmd.index(flag)+1]=value
 return cmd+['--manifest',manifest]
def run(cmd,path):
 with path.open('w') as log:subprocess.run(cmd,cwd=ROOT,check=True,stdout=log,stderr=log)
def caches(variant,record):
 if variant=='before':
  bank=ROOT/('artifacts/voting-template-prefix-v1' if record<2 else 'artifacts/query-packing-v3')
  return (bank/'prefix/queries',bank/'packets') if record<2 else (bank/'prefix-v2/queries',bank/'packets-v2')
 return D/f'after-prefix-{38 if record<2 else 27}'/'queries',D/f'after-packets-{38 if record<2 else 27}'
def prepare_prefix():
 for count in (27,38):
  dest=D/f'after-prefix-{count}';packets=D/f'after-packets-{count}'
  if not (dest/'report.json').exists():
   cmd=command('after');cmd[cmd.index('--cache')+1]=str(dest/'queries')
   print(json.dumps(dict(stage='prefix-start',tokens=count)),flush=True)
   run(cmd+['--record','0','--prefix-tokens',str(count),'--prepare-prefix','--directory',str(dest)],D/f'after-prefix-{count}.log')
  if not (packets/'cache.json').exists():
   run([sys.executable,'scripts/prepare_prefix_reuse.py','--prefix-directory',str(dest),'--directory',str(packets),'--canister','2vn5i-k3777-77775-aaaua-cai','--codec-module','e9644266f9bea8efdff5c1d5017b7c82beb67c3030279e6f27248b90fa51ef6e','--run-report',str(D/f'after-packets-{count}-report.json')],D/f'after-packets-{count}.log')
  print(json.dumps(dict(stage='prefix-ready',tokens=count)),flush=True)
def prepare_fixed(variant):
 dest=D/f'{variant}-fixed-prefix';assert not dest.exists();dest.mkdir()
 source,packets=caches(variant,2);source=source.parent
 cid,manifest=VARIANTS[variant];m=json.loads((ROOT/manifest).read_text());r=json.loads((source/'report.json').read_text());assert r['tokens']==27 and r['pack_hash']==m['pack_hash']
 cache=json.loads((packets/'cache.json').read_text());assert cache['identity']['source_report_sha256']==sha(source/'report.json')
 helper=ROOT/'artifacts/prefix-state-cache-v1/prefix-state-args';did=dest/'prepare.did';did.write_text('service:{prepare_fixed_prefix_state:(vec nat8,vec nat8)->(variant {Ok:record {nat32;nat64;nat64};Err:text})}')
 rows=[]
 for number,q in enumerate([q for q in r['queries'] if q.get('op')=='delta_full_log_integer'],1):
  layer=int(q['tensor'].split('.')[3]);frame=source/'queries'/f'{q["index"]:06d}.response.bin';header,values=decode(frame.read_bytes());assert header['dims']==[27,32,0,1]
  log=values[27*2560+24576:].astype('<f4').tobytes();assert len(log)==27*6176*4
  lp=dest/f'layer-{layer:02d}.log.f32';lp.write_bytes(log);packet=packets/f'layer-{layer:02d}.npf1';arg=dest/f'{layer:02d}.args.bin';reply=dest/f'{layer:02d}.reply.hex'
  subprocess.run([str(helper),'args',str(lp),str(packet),str(arg)],check=True);assert arg.stat().st_size<1_990_000
  raw=subprocess.check_output(['icp','canister','call',cid,'prepare_fixed_prefix_state','--network','local','--identity','imajev-local','--candid',str(did),'--args-file',str(arg),'--args-format','bin','--output','hex'],text=True);reply.write_text(raw)
  result=json.loads(subprocess.check_output([str(helper),'decode',str(reply)],text=True));assert result['count']==number and result['bytes']==number*2_097_152
  rows.append(dict(layer=layer,source_sha256=sha(frame),packet_sha256=sha(packet),result=result))
 assert len(rows)==24
 save(dest/'report.json',dict(update_calls=24,state_bytes=50_331_648,rows=rows,pack_hash=m['pack_hash']))
 print(json.dumps(dict(stage='fixed-prefix-ready',variant=variant)),flush=True)
def status(variant):
 cid,manifest=VARIANTS[variant];m=json.loads((ROOT/manifest).read_text());t=Transport(m['model'],'http://localhost:8001/',cid,str(ROOT/'artifacts/imajev-local.pem'),D/f'{variant}-status',m['pack_hash'])
 try:
  verify_module(t,module());pack=t.command(dict(op='pack_status'))['ok'];cache=t.command(dict(op='weight_cache_status'))['ok']['cache']
  assert pack['ready'] and pack['pack_hash']==m['pack_hash'];weights=[w for w in m['tensors'] if w['dtype'] in ('int8','f32') and 'embed_tokens' not in w['name']]
  assert set(cache['names'])=={w['name'] for w in weights} and cache['bytes']==sum(w['bytes'] for w in weights)
  return dict(module=module(),cache=cache,pack=pack)
 finally:t.close()
def measure(repeats):
 snapshot={v:status(v) for v in VARIANTS};files=[ROOT/'scripts/run_merged_query32.py',ROOT/'scripts/measure_merged_query32.py',ROOT/'client/query32_templates.py',ROOT/FIXTURE,ROOT/WASM]+list((ROOT/'client').glob('*.py'));sources={str(p.relative_to(ROOT)):sha(p) for p in files}
 save(D/'experiment.json',dict(scope=__doc__,cache_before=snapshot,sources=sources,repeats=repeats))
 for repeat in range(repeats):
  for record in range(3):
   for variant in (['before','after'] if (repeat+record)%2==0 else ['after','before']):
    dest=D/f'{variant}-r{record}-n{repeat}';assert not dest.exists(),dest
    cache,packets=caches(variant,record);cmd=command(variant);cmd[cmd.index('--cache')+1]=str(cache)
    cmd+=['--hybrid-cache',str(packets),'--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision','--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start','--roll-start','--query32-start','--record',str(record),'--directory',str(dest),'--step-offset',str((repeat+1)*100000+record*1000)]
    print(json.dumps(dict(stage='measure-start',variant=variant,record=record,repeat=repeat)),flush=True)
    load_before=os.getloadavg()
    run(cmd,D/f'{dest.name}.log')
    with (D/'host-load.jsonl').open('a') as f:f.write(json.dumps(dict(variant=variant,record=record,repeat=repeat,load_before=load_before,load_after=os.getloadavg()))+'\n')
    r=json.loads((dest/'report.json').read_text());assert r['query32_effective'] and r['query_count']==r['executed_query_count']==32 and r['replayed_queries']==0 and not r.get('fallback') and r['max_query_instructions']<5_000_000_000
    print(json.dumps(dict(stage='measure-ready',variant=variant,record=record,repeat=repeat,instructions=r['total_instructions'],seconds=r['wall_seconds_this_run'])),flush=True)
 assert sources=={p:sha(ROOT/p) for p in sources}
 save(D/'experiment.json',dict(scope=__doc__,cache_before=snapshot,cache_after={v:status(v) for v in VARIANTS},sources=sources,repeats=repeats))
 summarize(repeats)
def summarize(repeats):
 results=[]
 for record in range(3):
  runs={v:[json.loads((D/f'{v}-r{record}-n{n}/report.json').read_text()) for n in range(repeats)] for v in VARIANTS}
  for variant,rows in runs.items():
   assert len({json.dumps(r['decision_query']['ok']['decision']['raw_logits']) for r in rows})==1
   assert len({r['total_instructions'] for r in rows})==1
   assert all(r['wasm_sha256']==r['deployed_wasm_sha256']==module() and r['query_count']==r['executed_query_count']==32 and not r['replayed_queries'] for r in rows)
  a,b=runs['after'][0],runs['before'][0];assert a['input_hash']==b['input_hash'] and a['processed_tokens']==b['processed_tokens'] and a['tokens']-a['processed_tokens']==b['tokens']-b['processed_tokens']
  row=dict(record=record,proposal=['617','620','653'][record],prefix_tokens=b['tokens']-b['processed_tokens'],suffix_tokens=b['processed_tokens'],instruction_reduction=1-a['total_instructions']/b['total_instructions'])
  for v,rows in runs.items():
   times=[r['wall_seconds_this_run'] for r in rows];query_times=[sum(q['wall_seconds'] for q in r['queries']) for r in rows];r=rows[0]
   row[v]=dict(instructions=r['total_instructions'],seconds=times,median_seconds=statistics.median(times),query_seconds=query_times,median_query_seconds=statistics.median(query_times),query_count=32,max_query_instructions=r['max_query_instructions'],candid_bytes=r['total_candid_bytes'],heap_bytes=r['max_observed_heap_bytes'],decision=r['decision_query']['ok']['decision'])
  row['time_reduction']=1-row['after']['median_seconds']/row['before']['median_seconds'];row['decision_equal']=(row['before']['decision']['value'],row['before']['decision']['abstained'])==(row['after']['decision']['value'],row['after']['decision']['abstained'])
  row['max_probability_difference']=max(abs(x-y) for x,y in zip(row['before']['decision']['probabilities']+[row['before']['decision']['unknown_probability']],row['after']['decision']['probabilities']+[row['after']['decision']['unknown_probability']]))
  results.append(row)
 save(D/'comparison.json',dict(scope=__doc__,repeats=repeats,module=module(),results=results,instruction_scope='Handler performance counters; CDK Candid decoding/encoding excluded',timing_scope='Runner wall clock including bridge/HTTP/serialization/journal, excludes prefix/weights preparation',cache_control='Unique step offsets across repeats and fresh journals; replica query cache configuration unchanged',protocol_scope='Original rank64 wire operand slots retained as zero padding; merged runtime skips all absent A/B matrix products; internal zero descriptors never uploaded'))
 print(json.dumps(results,indent=2),flush=True)
def verify_full():
 import numpy as np
 rows=[]
 before_checks=[]
 for record in range(3):
  prior=ROOT/'artifacts/voting-template-prefix-v1/proof-v1'/['617','620','653'][record]
  new=D/f'before-r{record}-n0'
  old_report=json.loads((prior/'report.json').read_text());new_report=json.loads((new/'report.json').read_text())
  assert old_report['input_hash']==new_report['input_hash']
  assert np.load(prior/'final-hidden.npy').astype('<f4').tobytes()==np.load(new/'final-hidden.npy').astype('<f4').tobytes()
  assert old_report['decision_query']['ok']['decision']['raw_logits']==new_report['decision_query']['ok']['decision']['raw_logits']
  before_checks.append(dict(record=record,baseline_hidden_bitwise_equal=True,baseline_logits_equal=True,prior_report_sha256=sha(prior/'report.json')))
  dest=D/f'after-full-r{record}';assert not dest.exists()
  cid,manifest=VARIANTS['after']
  cmd=[sys.executable,'scripts/run_full_canister.py','--canister',cid,'--wasm',WASM,'--manifest',manifest,'--reference',FIXTURE,'--record',str(record),'--directory',str(dest),'--step-offset',str(900000+record*1000),'--arithmetic','int8','--wire-codec','bf16-exact','--compact-lossless','--fuse-add-norm','--fuse-norm-rope','--fuse-delta','--compact-heads','--delta-head-cap','8','--attention-head-cap','4','--row-cap','3072','--work-cap','3000000000']
  print(json.dumps(dict(stage='full-verification-start',record=record)),flush=True)
  run(cmd,D/f'after-full-r{record}.log')
  full=json.loads((dest/'report.json').read_text());fused=json.loads((D/f'after-r{record}-n0/report.json').read_text())
  a=np.load(dest/'final-hidden.npy')[-1];b=np.load(D/f'after-r{record}-n0/final-hidden.npy')[-1]
  assert a.astype('<f4').tobytes()==b.astype('<f4').tobytes(),'Merged fused hidden differs from independent full graph'
  da=full['decision_query']['ok']['decision'];db=fused['decision_query']['ok']['decision']
  assert da['raw_logits']==db['raw_logits'] and da['probabilities']==db['probabilities'] and da['value']==db['value']
  assert full['replayed_queries']==0
  rows.append(dict(record=record,hidden_bitwise_equal=True,logits_equal=True,probabilities_equal=True,full_report_sha256=sha(dest/'report.json'),query32_report_sha256=sha(D/f'after-r{record}-n0/report.json')))
  print(json.dumps(dict(stage='full-verification-ready',record=record)),flush=True)
 save(D/'verification.json',dict(scope='Independent unprefixed generic full graph compared against merged fused query32 graph for all three inputs',cases=rows,original_baseline_checks=before_checks))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prefix','fixed','measure','summary','verify']);ap.add_argument('--repeats',type=int,default=3);a=ap.parse_args();assert 1<=a.repeats<=10
 if a.stage=='verify':verify_full()
 elif a.stage=='prefix':prepare_prefix()
 elif a.stage=='fixed':
  for v in VARIANTS:prepare_fixed(v)
 elif a.stage=='measure':measure(a.repeats)
 else:summarize(a.repeats)
if __name__=='__main__':main()
