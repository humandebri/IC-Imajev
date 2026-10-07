"""Two prompt choices; uncertainty handled outside the model. No vote execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from .token_sweep import ROOT, save, sha

IMAJEV = ROOT
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
QUESTION = 'Approve easier participation; reject unreasonable voting barriers.'


def binary_decision(logits, threshold):
    if len(logits)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in logits):
        raise ValueError('two finite raw scores required')
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0.5<=threshold<=1:
        raise ValueError('threshold must be in [0.5,1]')
    a,b=logits
    score=1/(1+math.exp(-abs(a-b)))
    return ('hold' if a==b or score<threshold else 'approve' if a>b else 'reject'),score


def prepare(out):
    if out.exists():
        raise ValueError('fresh directory required')
    source = ROOT/'artifacts/proposal-assessment/heldout-128-20261006'
    old = json.loads((source/'prepared.json').read_text())
    sys.path.insert(0, str(IMAJEV/'scripts'))
    from prepare_text import TextPreparer
    from vision_decision.scoring import verified_label_ids
    p = TextPreparer()
    records, entries = [], []
    for e in old['entries']:
        # Only remove arithmetic direction commentary. Keep all numeric rows.
        state = '; '.join(x for x in e['input_case']['state'].split('; ')
                          if not x.startswith(('Voting eligibility ', 'Stake barrier ',
                                               'Voting time ', 'Affected participants/rationale ')))
        prompt = state+'\n'+QUESTION+'\nA: approve\nB: reject'
        rendered = p.render(prompt)
        ids = p.tokenizer.encode(rendered, add_special_tokens=False)
        assert verified_label_ids(p.tokenizer, rendered, ['A', 'B']) == [p.binding['codes'][i]['token_id'] for i in range(2)]
        records.append(dict(id=f'binary_{len(records)}', options=['approve', 'reject'], gold=None,
                            token_ids=ids, input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                            prompt=prompt, prompt_layout='binary-no-header-v1'))
        entries.append(dict(proposal_ids=e['proposal_ids'], synthetic=e.get('synthetic',False),
                            reference_prediction=e['reference_prediction'], record_index=len(records)-1,
                            tokens=len(ids), previous_tokens=e['tokens']))
    out.mkdir(parents=True)
    save(out/'inputs.json',dict(model_lock_sha256=sha(IMAJEV/'MODEL_LOCK.json'),records=records))
    save(out/'prepared.json',dict(entries=entries,thresholds=THRESHOLDS,question=QUESTION,
         reference_is_gold=False,scope='conditional participation; no full vote equivalence',
         confidence='sigmoid(abs(raw_logit_A-raw_logit_B)); uncalibrated binary score, not correctness probability',
         thresholds_preregistered=True,unknown_in_prompt=False,
         internal_unknown='Native readout still computes an unused third logit; only raw A/B logits are used.'))
    text=(IMAJEV/'scripts/run_full_canister.py').read_text().replace(
        'ROOT=pathlib.Path(__file__).resolve().parents[1]',f'ROOT=pathlib.Path({str(IMAJEV)!r})',1)
    oldrunner=(source/'run_existing_prefix.py').read_text()
    wrapper=oldrunner[oldrunner.index("if __name__=='__main__':\n import tempfile"):]
    text=text.replace("if __name__=='__main__':main()",wrapper)
    (out/'runner.py').write_text(text)
    shutil.copyfile(__file__,out/'prepare_source.py')
    save(out/'identity.json',dict(inputs_sha256=sha(out/'inputs.json'),prepared_sha256=sha(out/'prepared.json'),
         runner_sha256=sha(out/'runner.py'),source_sha256=sha(source/'prepared.json'),
         full_runner_sha256=sha(IMAJEV/'scripts/run_full_canister.py'),
         bridge_sha256=sha(IMAJEV/'target/release/imajev-client')))
    print(json.dumps(entries))


def run(out,shard,shards):
    ident=json.loads((out/'identity.json').read_text())
    for name in ('inputs','prepared'):
        assert sha(out/f'{name}.json')==ident[f'{name}_sha256']
    assert sha(out/'runner.py')==ident['runner_sha256']
    assert sha(IMAJEV/'target/release/imajev-client')==ident['bridge_sha256']
    sys.path.insert(0,str(IMAJEV/'scripts'))
    from evaluate_prompt_accuracy import base_flags
    entries=json.loads((out/'prepared.json').read_text())['entries']
    for e in entries:
        i=e['record_index']
        if i%shards!=shard:continue
        dest=out/'runs'/f'{i:03d}'
        if (dest/'report.json').exists():continue
        cmd=base_flags()
        for flag in ('--cache','--bridge-binary'):
            k=cmd.index(flag);del cmd[k:k+2]
        cmd[1]=str(out/'runner.py');cmd[cmd.index('--reference')+1]=str(out/'inputs.json')
        cmd.insert(1,'-B')
        cmd+=['--record',str(i),'--directory',str(dest),'--terminal-readout',
               '--fuse-terminal-attention','--fuse-terminal-decision']
        dest.mkdir(parents=True,exist_ok=True);save(dest/'command.json',cmd)
        print('starting',i,e['tokens'],flush=True)
        with (dest/'run.log').open('a') as log:
            result=subprocess.run(cmd,cwd=IMAJEV,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            save(dest/'error.json',dict(returncode=result.returncode,tail=(dest/'run.log').read_text()[-2500:]))
            print('failed',i,flush=True)
        else:print('completed',i,flush=True)


def report(out):
    prepared=json.loads((out/'prepared.json').read_text());inputs=json.loads((out/'inputs.json').read_text())
    ident=json.loads((out/'identity.json').read_text())
    assert sha(out/'inputs.json')==ident['inputs_sha256'] and sha(out/'prepared.json')==ident['prepared_sha256']
    rows=[]
    for e in prepared['entries']:
        path=out/'runs'/f"{e['record_index']:03d}"/'report.json';row=dict(e,status='pending')
        if path.exists():
            r=json.loads(path.read_text());record=inputs['records'][e['record_index']]
            assert r['wasm_sha256']==r['deployed_wasm_sha256']=='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
            assert r['input_hash']==record['input_sha256'] and r['model']==inputs['model_lock_sha256']
            assert r['tokens']==e['tokens'] and [x['layer'] for x in r['layers']]==list(range(32)) and not r['replayed_queries'] and not r.get('fallback')
            assert r['arithmetic']=='int8-block256-base-f32-lora-v1'
            assert not (path.parent/'active-staging.json').exists()
            assert r['executed_query_count']==r['query_count'] and record['gold'] is None
            logits=r['decision_query']['ok']['decision']['raw_logits'];assert len(logits)==3
            prediction,confidence=binary_decision(logits[:2],0.5)
            row.update(status='evaluated',binary_prediction=prediction,
                       confidence=confidence,raw_logits_AB=logits[:2],report_sha256=sha(path),
                       query_count=r['query_count'],instructions=r['total_instructions'])
            row['decisions_by_threshold']={str(t):binary_decision(logits[:2],t)[0] for t in prepared['thresholds']}
        elif (path.parent/'error.json').exists():row['status']='execution_error'
        rows.append(row)
    sweeps=[]
    for t in prepared['thresholds']:
        for group in ('real','synthetic'):
            subset=[r for r in rows if r['status']=='evaluated' and r['synthetic']==(group=='synthetic')]
            answered=[r for r in subset if binary_decision(r['raw_logits_AB'],t)[0]!='hold']
            matches=sum(binary_decision(r['raw_logits_AB'],t)[0]==r['reference_prediction'] for r in subset)
            correct_answered=sum(r['binary_prediction']==r['reference_prediction'] for r in answered)
            sweeps.append(dict(threshold=t,group=group,evaluated=len(subset),answered=len(answered),
                               held=len(subset)-len(answered),expectation_matches=matches,
                               correct_answered=correct_answered,wrong_answered=len(answered)-correct_answered))
    save(out/'results.json',dict(complete=all(r['status']=='evaluated' for r in rows),rows=rows,
         sweep=sweeps,accuracy_measured=False,confidence_calibrated=False,production_threshold_selected=False))
    lines=['# 二択＋Tool側の低確度hold', '',
           '共通指示、hold/unknown選択肢、数値の方向説明を入力から除去。全変更数値と参加方針は残す。前回と同じ未見4種類と合成3種類で比較。', '',
           '|対象|旧tokens|新tokens|二択判定|二択スコア|期待|','|---|---:|---:|---|---:|---|']
    for r in rows:lines.append(f"|{','.join(map(str,r['proposal_ids']))}|{r['previous_tokens']}|{r['tokens']}|{r.get('binary_prediction',r['status'])}|{r.get('confidence',0):.4f}|{r['reference_prediction']}|")
    lines+=['','|閾値|対象|回答|hold|期待一致|','|---|---|---:|---:|---:|']
    for s in sweeps:lines.append(f"|{s['threshold']}|{s['group']}|{s['answered']}|{s['held']}|{s['expectation_matches']}/{s['evaluated']}|")
    real=[r for r in rows if not r['synthetic']]
    old_total=sum(r['previous_tokens'] for r in real);new_total=sum(r['tokens'] for r in real)
    lines+=['',f'実proposal 4種類の入力合計は{old_total}→{new_total} tokens（{1-new_total/old_total:.1%}減）。', '',
            '二択にすると、期待したrejectをapproveへ振り分ける例が残った。hold閾値を上げればそれらの回答を抑制できる場合もあるが、期待どおりのapproveも抑制する。入力削減は達成、未見採否の品質達成・確度の校正は未達。']
    if all(r['status']=='evaluated' for r in real):
        previous=ROOT/'artifacts/proposal-assessment/heldout-128-20261006'
        reports=[json.loads((previous/'runs'/f"{r['record_index']:03d}"/'report.json').read_text()) for r in real]
        old_instructions=sum(r['total_instructions'] for r in reports)
        new_instructions=sum(r['instructions'] for r in real)
        old_queries=sum(r['query_count'] for r in reports);new_queries=sum(r['query_count'] for r in real)
        save(out/'cost-comparison.json',dict(scope='four unique real inputs; cache preparation excluded',
             old_instructions=old_instructions,new_instructions=new_instructions,
             old_queries=old_queries,new_queries=new_queries,actual_cycles_measured=False))
        lines+=['',f'同4種類のhandler命令数は{old_instructions:,}→{new_instructions:,}（{1-new_instructions/old_instructions:.1%}減）。query数は{old_queries}→{new_queries}。共通prefixを外したため経路も変わっている。実cycles料金は未測定。']
    lines+=['','スコアはA/B raw logitsをsoftmaxした未校正値。正解確率ではない。閾値は推論前に固定した比較候補で、本番閾値を選んでいない。期待は前回評価で公開済みのassistantによる条件付き参加方針で、人手goldではない。同じ評価集合の再利用であり、新たな独立検証ではない。', '',
            '入力にはunknownを含めないが、既存native readoutの契約上、内部では第三候補のlogitも計算される。その値とnative採否・棄権・校正確率は今回の判定に使用しない。Wasmやreadoutは変更していない。短縮により旧26-token共通prefixは使えず、全入力を32層で計算。tokens減が同率の実行費用減を意味するとは限らない。前処理・進行制御・閾値判定はPythonのベンチ実装で完全onchainではない。','']
    (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(sweeps))
    shutil.copyfile(__file__,out/'report_source.py')
    save(out/'files-sha256.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*')
                                if p.is_file() and p.name!='files-sha256.json'})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','report']);p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();out=a.output.resolve()
    assert 0<=a.shard<a.shards
    if a.mode=='prepare':prepare(out)
    elif a.mode=='run':run(out,a.shard,a.shards)
    else:report(out)


if __name__=='__main__':main()
