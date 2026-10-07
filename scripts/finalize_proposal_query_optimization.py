#!/usr/bin/env python3
"""Independently verify full exported state parity and account preparation costs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/proposal-query-optimization-v1'
OLD=ROOT/'artifacts/proposal-assessment-500-660-20261006/evaluation'
sys.path.insert(0,str(ROOT/'tools'))
from proposal_assessment.binary_benchmark import binary_decision
from proposal_assessment.improve_binary import approval_evidence_gate

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 s=json.loads((D/'summary.json').read_text());assert s['complete']
 for path,h in json.loads((D/'source-hashes.json').read_text()).items():assert sha(Path(path))==h
 for path,h in json.loads((D/'input-identities.json').read_text()).items():assert sha(Path(path))==h
 for path,h in json.loads((D/'final-runner-identity.json').read_text()).items():assert sha(Path(path))==h
 for metadata in list(D.glob('schedule-refinement*.json'))+[D/'comparison-affinity.json']:
  data=json.loads(metadata.read_text())
  for path,h in data.get('source_hashes',data.get('sources',{})).items():assert sha(Path(path))==h
 fixture=json.loads((D/'inputs.json').read_text());original=json.loads((OLD/'report.json').read_text());plans=json.loads((D/'plan.json').read_text())
 assert fixture==json.loads((OLD/'inputs.json').read_text())
 assert len(s['rows'])==len(plans['records'])==len(fixture['records'])==18
 rows=[];counts=Counter();modeled_ids=[]
 for row,p,prior in zip(s['rows'],plans['records'],original['model_rows']):
  i=row['record'];assert i==p['record']==prior['record_index']
  dest=D/'runs'/f'{i:03d}';new=json.loads((dest/'report.json').read_text());old=json.loads((OLD/'runs'/f'{i:03d}'/'report.json').read_text())
  assert sha(dest/'report.json')==row['report_sha256']
  assert sha(OLD/'runs'/f'{i:03d}'/'report.json')==prior['report_sha256']
  assert fixture['records'][i]['token_ids'][:p['prefix']]==plans['banks'][p['bank']]['ids']
  assert new['input_hash']==old['input_hash']==fixture['records'][i]['input_sha256']
  assert new['model']==old['model'] and new['pack_hash']==old['pack_hash']
  assert new['wasm_sha256']==new['deployed_wasm_sha256']
  assert [r['layer'] for r in new['layers']]==list(range(32))
  assert new['query_count']==new['executed_query_count'] and not new['replayed_queries'] and not new.get('fallback')
  assert new['max_query_instructions']<5_000_000_000 and new['max_observed_heap_bytes']<2**32
  assert all(q['ok']['request_bytes']<2_000_000 and q['ok']['reply_bytes']<2_000_000 for q in new['queries'])
  h=json.loads((OLD/'runs'/f'{i:03d}'/'staging-intermediate-artifact-hashes.json').read_text())
  assert sha(dest/'final-hidden.npy')==h['final-hidden.npy']['sha256']
  matched={}
  for category in ('queries/layer-','queries/states/'):
   names=[]
   for file in dest.rglob('*'):
    name=str(file.relative_to(dest))
    if file.is_file() and name.startswith(category) and name in h:
     assert sha(file)==h[name]['sha256'],f'{i}: state mismatch {name}'
     names.append(name)
   matched[category]=len(names)
  assert matched['queries/layer-']==31 and matched['queries/states/']==32
  # layer30 hidden is omitted by the compact terminal graph; do not claim it.
  a=old['decision_query']['ok']['decision'];b=new['decision_query']['ok']['decision']
  assert all(a[k]==b[k] for k in ('raw_logits','value','abstained','probabilities','unknown_probability'))
  prediction,score=binary_decision(b['raw_logits'][:2],.6);label,_=approval_evidence_gate(prior['task'],prediction)
  assert label==prior['final_label']
  counts[label]+=len(prior['proposal_ids']);modeled_ids+=prior['proposal_ids']
  rows.append(dict(row,proposal_ids=prior['proposal_ids'],prefix=p['prefix'],suffix=p['suffix'],tokens=len(fixture['records'][i]['token_ids']),final_label=label,exported_hidden_equal=31,state_equal=32))
 baseline={k:sum(r['baseline_'+k] for r in rows) for k in ('queries','instructions','candid_bytes')}
 inference={k:sum(r[k] for r in rows) for k in ('queries','instructions','candid_bytes')}
 preparation={k:0 for k in ('queries','instructions','candid_bytes')}
 for b in range(len(plans['banks'])):
  dest=D/'prefixes'/f'{b:02d}';r=json.loads((dest/'report.json').read_text());c=json.loads((dest/'codec-report.json').read_text())
  assert not r['replayed_queries'] and r['query_count']==r['executed_query_count']
  assert r['tokens']==len(plans['banks'][b]['ids']) and c['preparation_queries']==24 and not c['cache_hit']
  for k in preparation:preparation[k]+=r['query_count' if k=='queries' else 'total_'+k]+c['preparation_'+k]
 failed_successes=[];rejections=[]
 for attempt in (D/'failed').iterdir():
  q=attempt/'queries'
  if not q.exists():continue
  failed_successes += [json.loads(p.read_text()) for p in q.glob('*.metric.json')]
  failures=q/'failures.jsonl'
  if failures.exists():rejections += [json.loads(line) for line in failures.read_text().splitlines()]
 exploration=dict(successful_queries=len(failed_successes),failed_queries=len(rejections),queries=len(failed_successes)+len(rejections),measured_instructions=sum(q['ok']['instructions'] for q in failed_successes),unmeasured_failed_instructions=None,measured_candid_bytes=sum(q['ok']['request_bytes']+q['ok']['reply_bytes'] for q in failed_successes),unmeasured_failed_candid_bytes=None)
 total={k:inference[k]+preparation[k] for k in inference}
 total['queries']+=exploration['queries']
 entry=json.loads((D/'entry-check-015/verified.json').read_text());entry_report=json.loads((D/'entry-check-015/report.json').read_text())
 assert entry['record']==15 and entry['queries']==32 and entry['raw_logits_bit_equal'] and entry['final_hidden_bit_equal']
 assert entry_report['replayed_queries']==0 and not entry_report.get('fallback')
 total_with_validation={k:total[k]+entry_report['query_count' if k=='queries' else 'total_'+k] for k in total}
 result=dict(verified=True,validation_query_count=entry['queries'],total_queries_with_entry_validation=total_with_validation['queries'],records=rows,baseline=baseline,inference=inference,preparation=preparation,first_pass_total=total,exploration=exploration,instruction_and_byte_totals_exclude_failed_attempts=True,
             online_query_reduction=1-inference['queries']/baseline['queries'],first_pass_query_reduction=1-total['queries']/baseline['queries'],
             modeled_proposal_counts=dict(counts),unchanged_all_161_counts=original['summary']['counts'],prefix_banks=len(plans['banks']),
             query32_inputs=sum(r['queries']==32 for r in rows),query50_inputs=sum(r['queries']==50 for r in rows),
             unknown_layer30_hidden_not_compared=True,no_model_or_canister_updates=True)
 (D/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 lines=['# proposal 500〜660：query回数削減','',
 '2026-10-06。前回の18種類の入力を一切変更せず、canisterが計算した前半状態を後半の融合経路へ渡した。モデル重み、質問、選択肢、閾値0.6を維持。canisterのupgrade・重み変更・投票は実行していない。','',
 f"18種類すべてで最終hidden、31層のexported hidden、32層のconv/KV/position、logits・確率・判断がbit一致。layer30 hiddenはcompact terminal graphがexportしないため直接比較していない。32 query達成{result['query32_inputs']}種類、50 query達成{result['query50_inputs']}種類。",'',
 '|項目|前回|改善後|','|---|---:|---:|',
 f"|各入力の推論query合計|{baseline['queries']:,}|{inference['queries']:,}|",
 f"|前半準備・圧縮query|0|{preparation['queries']:,}|",
 f"|初回総query|{baseline['queries']:,}|{total['queries']:,}|",
 f"|推論handler命令合計|{baseline['instructions']:,}|{inference['instructions']:,}|",
 f"|準備＋成功推論handler命令（失敗試行除外）|{baseline['instructions']:,}|{total['instructions']:,}|",
 f"|推論Candid合計bytes|{baseline['candid_bytes']:,}|{inference['candid_bytes']:,}|",
 f"|準備＋成功推論Candid bytes（失敗試行除外）|{baseline['candid_bytes']:,}|{total['candid_bytes']:,}|",'',
 f"推論query {result['online_query_reduction']:.2%}減、初回準備込み {result['first_pass_query_reduction']:.2%}減。11種類の前半bankは入力依存の値を含む。任意の新proposalで無料で使える共通prefixではない。新入力ではtoken一致を確認し、未知の先頭部分は再準備が必要。単発入力には準備でqueryが増える例もあり、上記は18種類全体の比較。",'',
 f"97-token入力の配分探索で成功{exploration['successful_queries']} query＋上限超過{exploration['failed_queries']} queryを記録。初回総queryにはこれらを含む。失敗handler命令・通信量は取得できないためnullとし、上の命令/bytes表は成功推論と準備のみ。さらに失敗試行で成功したqueryには{exploration['measured_instructions']:,} handler命令があり、探索費用もゼロではない。比較用canisterのmodule変更は準備の入口で拒否し、残り2種類は稼働中の別の検証済みmoduleへ移した。このworkflowによるupgradeや重み変更は行っていない。",'',
 '前回値は保存済み実測、改善後は18種類すべて新規queryで実測。両者を同時の負荷条件で時間比較したものではない。各queryの要求・返信2MB未満、handler50億命令未満、heap4GiB未満。モデルの正答率改善を主張しない。','',
 f"全161件の最終判定数は変更なし：{original['summary']['counts']}、除外88件。",'',
 '|proposal|tokens|前半/後半|前回query|今回query|最終判定|','|---|---:|---|---:|---:|---|']
 for r in rows:lines.append(f"|{','.join(map(str,r['proposal_ids']))}|{r['tokens']}|{r['prefix']}/{r['suffix']}|{r['baseline_queries']}|{r['queries']}|{r['final_label']}|")
 lines+=['',f"再実行入口 `scripts/run_assessed_proposal_queries.py --record 15 --directory <新しい出力先>`でも32 query・元とのbit一致を新規実測した。この追加確認32 queryは初回18種類の比較から別計上し、全試行・追加確認込みの今回総queryは{total_with_validation['queries']:,}。入口は18種類の固定入力と検証済み前半cacheを対象とする。新proposalでは新たな入力準備・prefix検証が必要。独立照合は `scripts/finalize_proposal_query_optimization.py`。",'']
 (D/'REPORT.md').write_text('\n'.join(lines))
 print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
