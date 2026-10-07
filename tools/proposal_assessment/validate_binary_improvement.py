"""Freeze a selected binary prompt and test remaining/new numerical cases."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from .binary_benchmark import IMAJEV, run, binary_decision
from .improve_binary import VARIANTS, encode_state, compact_exact_state, results,approval_evidence_gate
from .token_sweep import ROOT, save, sha

THRESHOLD = .6  # Chosen on the 3-case screen before validation outputs.
COMPACT_QUESTION = 'Approve lower barriers; reject prohibitive stake or lock. Consider all changes.'


def prepare(out):
    if out.exists():raise ValueError('fresh directory required')
    old=ROOT/'artifacts/proposal-assessment/heldout-128-20261006'
    base=json.loads((old/'prepared.json').read_text())['entries']
    selected=[]
    for i,e in enumerate(base):
        selected.append(dict(proposal_ids=e['proposal_ids'],task=e['task'],expected=e['reference_prediction'],
                             synthetic=e.get('synthetic',False),split='screen' if i in (0,3,4) else 'remaining_development'))
    corpus=ROOT/'artifacts/proposal-assessment/boomdao-validation-478-538-20261006'
    labels=json.loads((corpus/'preregistered-expectations.json').read_text())
    gates=[]
    for e in labels['entries']:
        state=json.loads(e['task']['state']);eligible=[r for r in state['changes_or_requests']
              if r['field'] in ('neuron_minimum_stake_e8s','neuron_minimum_dissolve_delay_to_vote_seconds')]
        row=dict(proposal_ids=[e['proposal_id']],task=e['task'],expected=e['expected'],synthetic=False,
                 split='new_real_validation',snapshot_sha256=e['snapshot_sha256'])
        if not eligible:
            gates.append({**row,'label':'hold','tokens':0,'route':'outside_direct_eligibility_scope',
                          'reason':'Only bonuses/voting time change; overall participation effect requires more evidence.'})
        else:selected.append(row)
    # New numerical controls, created before any validation inference.
    DAY=86400;E8=100000000
    controls=[('validation_lower_both','approve',[
        ('neuron_minimum_stake_e8s',1000*E8,2*E8),
        ('neuron_minimum_dissolve_delay_to_vote_seconds',7*DAY,DAY)]),
        ('validation_high_stake_mixed','reject',[
        ('neuron_minimum_stake_e8s',3*E8,12000000*E8),
        ('neuron_minimum_dissolve_delay_to_vote_seconds',5*DAY,2*DAY),
        ('max_dissolve_delay_bonus_percentage',10,20)]),
        ('validation_small_stake_unknown','hold',[
        ('neuron_minimum_stake_e8s',10*E8,11*E8)]),
        ('validation_small_lock_unknown','hold',[
        ('neuron_minimum_dissolve_delay_to_vote_seconds',5*DAY,6*DAY)])]
    for name,label,rows in controls:
        task={'state':json.dumps(dict(action='ManageNervousSystemParameters',changes_or_requests=[
            dict(field=f,previous=o,proposed=n) for f,o,n in rows],proposer_text=dict(summary='',proposal_title=''),unchanged_fields=[]))}
        selected.append(dict(proposal_ids=[name],task=task,expected=label,synthetic=True,split='new_synthetic_validation'))
    sys.path.insert(0,str(IMAJEV/'scripts'));from prepare_text import TextPreparer
    from vision_decision.scoring import verified_label_ids
    p=TextPreparer();records=[];entries=[]
    for e in selected:
        state=encode_state(e['task']);question=VARIANTS['ratio'];layout='ratio'
        prompt=f'State: {state}\nQuestion: {question}\nA: approve\nB: reject'
        ids=p.tokenizer.encode(p.render(prompt),add_special_tokens=False)
        if len(ids)>128:
            state=compact_exact_state(e['task']);question=COMPACT_QUESTION;layout='compact_ratio'
            prompt=f'State: {state}\nQuestion: {question}\nA: approve\nB: reject'
            ids=p.tokenizer.encode(p.render(prompt),add_special_tokens=False)
        if len(ids)>128:
            gates.append({**e,'label':'hold','tokens':0,'full_input_tokens':len(ids),'route':'budget_overflow'})
            continue
        assert verified_label_ids(p.tokenizer,p.render(prompt),['A','B'])==[p.binding['codes'][i]['token_id'] for i in range(2)]
        records.append(dict(id=f'validate_{len(records)}',options=['approve','reject'],gold=None,token_ids=ids,
            input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),prompt=prompt))
        entries.append({**e,'record_index':len(records)-1,'variant':layout,'tokens':len(ids),'state':state})
    out.mkdir(parents=True)
    save(out/'inputs.json',dict(model_lock_sha256=sha(IMAJEV/'MODEL_LOCK.json'),records=records))
    save(out/'prepared.json',dict(entries=entries,gates=gates,thresholds=[THRESHOLD],scope='fixed development and new validation, conditional numeric participation',
         selected_variant='ratio; compact_ratio only for overflow',threshold_selected_before_validation=True,
         selection_reason='3/3 screen at .6, labels preserved after order swap; scores remain order-sensitive',
         source_expectations_sha256=sha(corpus/'preregistered-expectations.json'),source_manifest_sha256=sha(corpus/'manifest.json'),
         gates_not_model_predictions=True,threshold_calibrated_as_correctness_probability=False))
    origin=ROOT/'artifacts/proposal-assessment/binary-improvement-screen-20261006'
    shutil.copyfile(origin/'runner.py',out/'runner.py')
    save(out/'identity.json',dict(inputs_sha256=sha(out/'inputs.json'),prepared_sha256=sha(out/'prepared.json'),
        runner_sha256=sha(out/'runner.py'),bridge_sha256=sha(IMAJEV/'target/release/imajev-client')))
    # Reuse only byte-identical completed screen inputs; never reference labels.
    oldinputs=json.loads((origin/'inputs.json').read_text())['records']
    for i,record in enumerate(records):
        j=next((j for j,r in enumerate(oldinputs) if r['token_ids']==record['token_ids'] and r['options']==record['options']),None)
        if j is not None:
            shutil.copytree(origin/'runs'/f'{j:03d}',out/'runs'/f'{i:03d}')
            save(out/'runs'/f'{i:03d}'/'reuse.json',dict(source=str((origin/'runs'/f'{j:03d}').resolve()),exact_token_identity=True))
    for name in ('validate_binary_improvement.py','improve_binary.py','binary_benchmark.py','million_notation.py'):
        shutil.copyfile(Path(__file__).parent/name,out/name.replace('.py','_source.py'))
    print(json.dumps([(e['proposal_ids'],e['split'],e['tokens']) for e in entries]));print('gates',len(gates))


def followup(out):
    if out.exists():raise ValueError('fresh directory required')
    origin=ROOT/'artifacts/proposal-assessment/binary-improved-validation-20261006'
    previous=json.loads((origin/'prepared.json').read_text())
    selected=[dict(e) for e in previous['entries']]
    for e in selected:
        if e['split']=='new_synthetic_validation':e['split']='development_feedback'
    DAY=86400;E8=100000000
    controls=[('confirmation_lower_both','approve',[
        ('neuron_minimum_stake_e8s',3000*E8,4*E8),
        ('neuron_minimum_dissolve_delay_to_vote_seconds',9*DAY,DAY)]),
        ('confirmation_mixed_restriction','reject',[
        ('neuron_minimum_stake_e8s',7*E8,2000000*E8),
        ('neuron_minimum_dissolve_delay_to_vote_seconds',6*DAY,3*DAY),
        ('max_dissolve_delay_bonus_percentage',25,10)]),
        ('confirmation_small_stake_unknown','hold',[
        ('neuron_minimum_stake_e8s',12*E8,13*E8)]),
        ('confirmation_small_lock_unknown','hold',[
        ('neuron_minimum_dissolve_delay_to_vote_seconds',8*DAY,10*DAY)])]
    for name,label,rows in controls:
        task={'state':json.dumps(dict(action='ManageNervousSystemParameters',changes_or_requests=[
            dict(field=f,previous=o,proposed=n) for f,o,n in rows],proposer_text=dict(summary='',proposal_title=''),unchanged_fields=[]))}
        selected.append(dict(proposal_ids=[name],task=task,expected=label,synthetic=True,split='fresh_synthetic_confirmation'))
    sys.path.insert(0,str(IMAJEV/'scripts'));from prepare_text import TextPreparer
    p=TextPreparer();records=[];entries=[];gates=previous['gates']
    for e in selected:
        state=encode_state(e['task']);question=VARIANTS['ratio'];layout='ratio'
        prompt=f'State: {state}\nQuestion: {question}\nA: approve\nB: reject'
        ids=p.tokenizer.encode(p.render(prompt),add_special_tokens=False)
        if len(ids)>128:
            state=compact_exact_state(e['task']);question=COMPACT_QUESTION;layout='compact_ratio'
            prompt=f'State: {state}\nQuestion: {question}\nA: approve\nB: reject'
            ids=p.tokenizer.encode(p.render(prompt),add_special_tokens=False)
        if len(ids)>128:
            gates.append({**e,'label':'hold','tokens':0,'full_input_tokens':len(ids),'route':'budget_overflow'});continue
        records.append(dict(id=f'confirmed_{len(records)}',options=['approve','reject'],gold=None,token_ids=ids,
            input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),prompt=prompt))
        entries.append({**e,'record_index':len(records)-1,'variant':layout,'tokens':len(ids),'state':state})
    out.mkdir(parents=True)
    save(out/'inputs.json',dict(model_lock_sha256=sha(IMAJEV/'MODEL_LOCK.json'),records=records))
    save(out/'prepared.json',{**previous,'entries':entries,'gates':gates,'approval_gate_enabled':True,
         'source_previous_prepared_sha256':sha(origin/'prepared.json'),
         'scope':'final numeric eligibility benchmark; post-feedback development and fresh confirmation',
         'canonical_field_order':True,'no_rejection_threshold_added':True})
    shutil.copyfile(origin/'runner.py',out/'runner.py')
    save(out/'identity.json',dict(inputs_sha256=sha(out/'inputs.json'),prepared_sha256=sha(out/'prepared.json'),
        runner_sha256=sha(out/'runner.py'),bridge_sha256=sha(IMAJEV/'target/release/imajev-client')))
    oldinputs=json.loads((origin/'inputs.json').read_text())['records']
    for i,record in enumerate(records):
        j=next((j for j,r in enumerate(oldinputs) if r['token_ids']==record['token_ids'] and r['options']==record['options']),None)
        if j is not None:
            shutil.copytree(origin/'runs'/f'{j:03d}',out/'runs'/f'{i:03d}')
            save(out/'runs'/f'{i:03d}'/'reuse.json',dict(source=str((origin/'runs'/f'{j:03d}').resolve()),exact_token_identity=True))
    for name in ('validate_binary_improvement.py','improve_binary.py','binary_benchmark.py','million_notation.py'):
        shutil.copyfile(Path(__file__).parent/name,out/name.replace('.py','_source.py'))
    print(json.dumps([(e['proposal_ids'],e['split'],e['tokens'],(out/'runs'/f"{e['record_index']:03d}"/'report.json').exists()) for e in entries]))


def report(out):
    rows=results(out);prepared=json.loads((out/'prepared.json').read_text())
    for row in rows:
        if row['status']=='evaluated':
            label=row['decisions'][str(THRESHOLD)]
            final,reason=approval_evidence_gate(row['task'],label) if prepared.get('approval_gate_enabled') else (label,None)
            row.update(final_label=final,approval_gate_reason=reason)
    gates=prepared['gates'];summary={}
    for split in ('screen','remaining_development','new_real_validation','new_synthetic_validation','development_feedback','fresh_synthetic_confirmation'):
        llm=[r for r in rows if r['split']==split];local=[r for r in gates if r['split']==split]
        evaluated=[r for r in llm if r['status']=='evaluated']
        summary[split]=dict(model_cases=len(llm),model_evaluated=len(evaluated),gated_cases=len(local),
             model_matches=sum(r['decisions'][str(THRESHOLD)]==r['expected'] for r in evaluated),
             pipeline_matches=sum(r['final_label']==r['expected'] for r in evaluated),
             gate_matches=sum(r['label']==r['expected'] for r in local),
             total=len(llm)+len(local))
    save(out/'quality.json',dict(complete=all(r['status']=='evaluated' for r in rows),threshold=THRESHOLD,
         summary=summary,rows=rows,gates=gates,accuracy_measured=False,human_golds=False,production_policy_approved=False))
    lines=['# 二択判定の改善と検証', '',
        '変更の倍率を整数・有理数で計算し、判定基準は「禁止的なstake/最低投票lockを拒否し、全変更を合わせて判断」に明確化。外部LLMによる前処理なし。hold/unknownはモデル入力に含めず、二択スコア0.6未満をToolでholdとする。0.6は3例の調整結果から選び、検証前に固定。確度の正解確率としての校正は未証明。', '',
        '|対象|区分|tokens／件|二択|スコア|Tool後|事前期待|','|---|---|---:|---|---:|---|---|']
    for r in rows:
        lines.append(f"|{','.join(map(str,r['proposal_ids']))}|{r['split']}|{r['tokens']}|{r.get('prediction',r['status'])}|{r.get('score',0):.3f}|{r.get('final_label','pending')}|{r['expected']}|")
    for r in gates:lines.append(f"|{','.join(map(str,r['proposal_ids']))}|{r['split']}|0|{r['route']}|—|{r['label']}|{r['expected']}|")
    lines+=['','|区分|二択＋閾値の期待一致|承認証拠検査後|対応範囲外hold|合計対象|','|---|---:|---:|---:|---:|']
    for split,s in summary.items():
        if s['total']:lines.append(f"|{split}|{s['model_matches']}/{s['model_cases']}|{s['pipeline_matches']}/{s['model_cases']}|{s['gate_matches']}/{s['gated_cases']}|{s['total']}|")
    if prepared.get('approval_gate_enabled'):
        lines+=['','最終版は変更項目をfield名順に統一。approveを採用するには、数値上の最低stake/最低投票lockの改善が確認でき、同時にどちらも厳しくならないことをToolで検査。確認できなければhold。これは安全性全般の保証・本番投票方針ではなく、この数値参加条件ベンチの保守的な承認条件。拒否の数値閾値は追加していない。', '',
                'この検査は最初の合成検証の失敗を受けて追加したため、その4例はdevelopment_feedbackへ移し、未見精度として扱わない。追加のfresh_synthetic_confirmationは最終版を固定してから実測。元の未使用実proposal2例は入力もreject結果も変更していない。']
    lines+=['','事前期待はassistantによる条件付き参加方針の判断で、人手goldではない。新実proposalの478〜538を別途取得して期待を推論前に固定。同じDAOの小規模検証であり、一般的なSNS採否精度・完全onchain実装は未証明。', '',
         'ボーナス/投票時間のみの変更は、最低stake/最低投票lockに限定したモデルの対応範囲外としてholdに回す。これらはLLMの正解として数えない。実行・投票認可はしない。多項目で128を超える場合は全旧新値を残して名称と冗長な倍率説明を圧縮し、それでも収まらなければhold。', '',
         '順序を入れ替えた調整用3例では採否は一致したが、スコアには差があった。順序を2回計算する構成は採用しておらず、表のtoken数は採用した1回分。調整実験の追加推論コストとは別。32層・同一モデル/Wasm・入力hash・query数・replay/fallbackなしを検査。','']
    (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(summary))
    shutil.copyfile(__file__,out/'report_source.py')
    save(out/'files-sha256.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='files-sha256.json'})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','followup','run','report']);p.add_argument('--output',type=Path,required=True);p.add_argument('--shards',type=int,default=1);p.add_argument('--shard',type=int,default=0);a=p.parse_args();out=a.output.resolve()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='followup':followup(out)
    elif a.mode=='run':run(out,a.shard,a.shards)
    else:report(out)


if __name__=='__main__':main()
