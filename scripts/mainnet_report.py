#!/usr/bin/env python3
"""Read-only final mainnet evidence collection after successful owner validation."""
import json,pathlib,time,sys
from mainnet_cli import MainnetCLI,ROOT,atomic_json,OWNER,MODULE,TARGET
sys.path.insert(0,str(ROOT/'client'))
from decision_validation import validate_decision

def integer(v):return int(str(v).replace('_',''))
def main():
    d=ROOT/'artifacts/mainnet-deploy-20261007';p=d/'preparation'
    state=json.loads((p/'preparation-state.json').read_text())
    # Require preparation runner's persisted terminal state before queries.
    if not state.get('bootstrap_done') or not state.get('examples_done'):raise RuntimeError('Owner validation not complete')
    results=json.loads((p/'owner-ui-results.json').read_text())['records']
    if len(results)!=3:raise RuntimeError('Three actual IC outputs required')
    cli=MainnetCLI(d/'final-checks');s=cli.status()
    if s['status']!='Running' or integer(s['settings']['wasm_memory_limit'])!=2**32:raise RuntimeError('Wrong live runtime settings')
    pack=cli.call('pack_status',query=True);cache=cli.call('weight_cache_status',query=True);config=cli.call('paid_config',query=True)
    if not pack['ready'] or pack['bytes']!=pack['received'] or pack['bytes']!=pack['hashed']:raise RuntimeError('Pack not sealed')
    if config['enabled']:raise RuntimeError('Paid service was unexpectedly activated')
    if cache['bytes']!=4_065_416_192 or len(cache['names'])!=721 or cache['rope_bytes']!=131072 or cache['activation_bytes']!=1048576 or cache['paired_weight_bytes']!=3_569_090_560:raise RuntimeError('Required cache preparation incomplete')
    m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
    if pack['model']!=m['model'] or pack['pack_hash']!=m['pack_hash'] or pack['bytes']!=m['bytes']:raise RuntimeError('Live pack identity mismatch')
    expected={t['name'] for t in m['tensors'] if t['dtype'] in ['int8','f32'] and 'embed_tokens' not in t['name']}
    if set(cache['names'])!=expected:raise RuntimeError('Wrong cache tensors')
    summaries=[]
    for r in results:
        progress=r['owner_progress'];decision=validate_decision(progress['decision'],r['input']['options'])
        if not progress['done'] or progress['stage']!=64 or len(progress['hidden_hashes'])!=32 or len(progress['state_hashes'])!=32:raise RuntimeError('Incomplete 32-layer graph')
        cost=r['measured_cost']
        if cost['already_completed_steps_at_measurement_start']!=0:raise RuntimeError('This run lacks full-example cost observation')
        summaries.append({'id':r['id'],'tokens':r['tokens'],'input':r['input'],'status':r['result']['status'],'value':decision['value'],'normalized_scores':r['result']['scores'],'raw_logits':r['result']['raw_logits'],'calibration_version':decision['calibration_version'],'owner_job_id':progress['id'],'graph_update_calls':cost['graph_update_calls'],'graph_instructions':cost['reported_graph_instructions'],'observed_canister_cycles_delta':cost['observed_canister_cycles_delta'],'illustrative_USD':cost['observed_canister_cycles_delta']/1e12*1.37,'wall_seconds':cost['wall_seconds'],'max_reported_heap_bytes':cost['max_reported_heap_bytes']})
    mint=json.loads((d/'mint-6.4999.intent.json').read_text());fund=json.loads((d/'funding-verified.json').read_text())
    before=mint['before_canister'];deposited=fund['topup']['amount_cycles']
    consumed=integer(before['cycles'])+integer(before['reserved_cycles'])+deposited-integer(s['cycles'])-integer(s['reserved_cycles'])
    daily=integer(s['idle_cycles_burned_per_day']);days=integer(s['settings']['freezing_threshold'])/86400
    report={'version':1,'observed_unix':time.time(),'canister_id':TARGET,'controller_and_runtime_owner':OWNER,'network':'ic','module_hash':MODULE,'model':pack['model'],'pack_hash':pack['pack_hash'],'pack_bytes':pack['bytes'],'pack_ready':True,'bootstrap_ready':True,'paid_enabled':False,'frontend_api_connected':False,'wasm_memory_limit_bytes':integer(s['settings']['wasm_memory_limit']),'freezing_threshold_days':days,'snapshots_created_by_this_deployment':False,'extra_canisters_created':False,'additional_model_phase_ICP_debit_including_fee':fund['observed_ICP_debit_e8s']/1e8,'after_conversion_ICP_balance':fund['after_ICP'],'target_cycles_deposited_model_phase':deposited,'observed_canister_cycles_consumed_model_phase':consumed,'remaining_available_canister_cycles':integer(s['cycles']),'reserved_cycles':integer(s['reserved_cycles']),'live_memory_size_bytes':integer(s['memory_size']),'idle_cycles_per_day':daily,'measured_idle_cycles_per_30days':daily*30,'illustrative_idle_USD_per_30days':daily*30/1e12*1.37,'estimated_freezing_reserve_cycles':daily*days,'estimated_idle_days_until_freezing_without_more_calls':max(0,(integer(s['cycles'])-daily*days)/daily),'immutable_cache_bytes':cache['bytes'],'immutable_cache_tensors':len(cache['names']),'prepared_rope_bytes':cache['rope_bytes'],'prepared_activation_bytes':cache['activation_bytes'],'paired_weight_bytes':cache['paired_weight_bytes'],'examples':summaries,'notes':['All outputs are actual IC replicated owner-update results, not native reference outputs or mock inference.','Observed example debits include management/ingress overhead and idle rent; these are not a configured public paid price.','USD uses illustrative 1.37 per trillion cycles, not a live FX quote.','The authorized 14-day freezing period lowers only the protected reserve, not monthly storage charges.','Actual idle operation until freezing shortens with further inference spending.','Runtime heap caches and prefix banks are lost on upgrades; preserve the pinned runtime and do not reinstall after upload.','Cloudflare UI remains API-disconnected. Public paid service was not activated.']}
    atomic_json(d/'mainnet-verified.json',report);atomic_json(d/'final-status.json',s)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
