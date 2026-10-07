#!/usr/bin/env python3
"""Audit frozen evidence and compare existing scores with explicit reviewer judgments.

Reviewer judgments are this agent's conditional assessment, not human vote gold.
No new model calls, labels in prompts, or production policy changes.
"""
import hashlib
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/proposal-assessment-fresh-500-660-20261007'
OUT = ROOT / 'artifacts/proposal-evidence-review-20261007'
sys.path.insert(0, str(ROOT / 'tools'))
from proposal_assessment.binary_benchmark import binary_decision
from proposal_assessment.improve_binary import approval_evidence_gate

# Explicit review of each distinct input after reading its original proposal.
# These are development judgments; never presented as independent gold labels.
NOTES = {
    0: ('reject', '最低stake 5→100万tokens、最低投票lock 2→365.25日。参加条件を大幅に厳しくする。'),
    1: ('reject', '最低stake 5→100万tokens。bonus増加は新規参加のstake条件を緩めない。'),
    2: ('reject', '最低stake 5→500万tokens。新規参加の必要stakeが100万倍。'),
    3: ('reject', '最低stake 5→1,000万tokens。新規参加の必要stakeが200万倍。'),
    4: ('reject', '最低stake 5→1,500万tokens。新規参加の必要stakeが300万倍。'),
    5: ('reject', '最低投票lock 2→1日は緩和だが、最低stake 5→1,500万tokensの障壁が残る。'),
    6: ('reject', '最低stake 5→100万tokens、最低投票lock 2→1,461日。両方の参加条件が大幅に悪化。'),
    7: ('reject', '最低stake 5→1,500万tokens。投票lockの半減はstake障壁を解消しない。'),
    8: ('reject', '最低stake 5→100万tokens、最低投票lock 2→1,461日。reject costも100→1万tokens。'),
    9: ('reject', '最低stake 5→100万tokens、最低投票lock 2→1,461日。bonusの変更では参加条件を緩められない。'),
    10: ('reject', '最低stake 5→200万tokens、最低投票lock 2→1,461日。'),
    11: ('reject', '最低stakeは1,500万→600万tokensへ下がるが、最低投票lockは2→1,461日へ上がる。'),
    12: ('reject', '最低stake 5→600万tokens、最低投票lock 2→1,461日。'),
    13: ('reject', '最低stakeは1,000万→600万tokensへ下がるが、最低投票lockは2→1,461日へ上がる。'),
    14: ('hold', '最低投票lock 1→2日の強化だけ。2日を禁止的とする基準や実際の影響は示されていない。'),
    15: ('reject', '最低投票lock 1→2万日（約54.8年）。最大lock延長と最低投票条件を混同しない。'),
    16: ('approve', '最低stake 1,500万→5tokens、投票期間1→3日。数値上は参加条件を改善。ただしsummaryのtakeover言及は入力から落ちている。'),
    17: ('approve', '最低stake 1,500万→5tokens、最低投票lock 2→1日。lock bonus 1→100%の投票力配分への影響は別途検討が必要。'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def parsed_parameters(text, heading):
    """Separate parser of scalar Some values in the historical renderer."""
    assert text.count(heading) == 1
    section = text.split(heading)[1].split('\n## ')[0]
    pairs = re.findall(r'^\s*(\w+): Some\(\s*(\d+|true|false)\s*,?\s*\)', section, re.M)
    assert len(pairs) == len(dict(pairs))
    return {k: (v == 'true' if v in ('true', 'false') else int(v)) for k, v in pairs}


def main():
    OUT.mkdir(exist_ok=True)
    report_path = SOURCE / 'evaluation/report.json'
    report = json.loads(report_path.read_text())
    manifest_path = SOURCE / 'snapshots/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sources = {r['proposal_id']: r for r in manifest['records']}
    reviewed = []
    model_rows = report['model_rows']
    for row in model_rows:
        index = row['record_index']
        judgment, rationale = NOTES[index]
        task_state = json.loads(row['task']['state'])
        task_changes = {x['field']: (x['previous'], x['proposed']) for x in task_state['changes_or_requests']}
        for pid in row['proposal_ids']:
            source = sources[pid]
            snapshot_path = Path(source['snapshot'])
            assert sha(snapshot_path) == source['sha256']
            raw = json.loads(snapshot_path.read_text())
            assert int(raw['id']) == pid
            assert raw['proposal_action_type'] == 'ManageNervousSystemParameters'
            old = parsed_parameters(raw['payload_text_rendering'], '## Current nervous system parameters:')
            new = parsed_parameters(raw['payload_text_rendering'], '## New nervous system parameters:')
            changed = {}
            for field, value in raw['proposal_action_payload'].items():
                if value is None:
                    continue
                assert type(value) in (int, bool), (pid, field, 'non-scalar requires review')
                assert field in old and field in new, (pid, field, 'missing renderer value')
                assert type(new[field]) is type(value) and new[field] == value
                if old[field] != value:
                    changed[field] = (old[field], value)
            assert changed == task_changes, (pid, changed, task_changes)
            reviewed.append(dict(proposal_id=pid, record_index=index,
                source=str(snapshot_path), snapshot_sha256=source['sha256'],
                title=raw['proposal_title'], summary=raw['summary'],
                representative_proposer_text_matches=(
                    raw['proposal_title'] == task_state['proposer_text']['proposal_title'] and
                    raw['summary'] == task_state['proposer_text']['summary']),
                raw_proposed_matches_renderer=True, complete_changed_scalar_set=True,
                historical_old_values_chain_verified=False,
                changes=[dict(field=k, previous=v[0], proposed=v[1]) for k, v in changed.items()],
                current_label=row['final_label'], raw_binary=row['prediction'], score=row['score'],
                conditional_reviewer_label=judgment, rationale=rationale,
                human_gold=False, full_proposal_vote_verified=False,
                prompt=row['state'], input_tokens=row['tokens'], layout=row['variant']))

    gates = json.loads((SOURCE / 'evaluation/prepared.json').read_text())['gates']
    extra = []
    for gate in gates:
        if gate['route'] not in ('evidence_gate', 'budget_overflow') and gate['proposal_ids'] != [501]:
            continue
        for pid in gate['proposal_ids']:
            source = sources[pid]
            path = Path(source['snapshot'])
            assert sha(path) == source['sha256']
            raw = json.loads(path.read_text())
            extra.append(dict(proposal_id=pid, source=str(path), snapshot_sha256=source['sha256'],
                action=raw['proposal_action_type'], payload=raw['proposal_action_payload'],
                title=raw['proposal_title'], summary=raw['summary'],
                route=gate['route'], reason=gate['reason'], task=json.loads(gate['task']['state']),
                candidate_tokens=gate.get('full_input_tokens'), candidate_prompt=gate.get('candidate_prompt'),
                conditional_reviewer_label='hold', human_gold=False))

    threshold_rows = []
    for threshold in (.5, .52, .54, .55, .6, .7, .8, .9):
        counts = Counter()
        comparisons = Counter()
        for row in model_rows:
            prediction, _ = binary_decision(row['canonical_logits'], threshold)
            final, _ = approval_evidence_gate(row['task'], prediction)
            reviewed_label = NOTES[row['record_index']][0]
            n = len(row['proposal_ids'])
            counts[final] += n
            comparisons[reviewed_label + '→' + final] += n
        threshold_rows.append(dict(threshold=threshold, counts=dict(counts),
            conditional_review_confusion=dict(comparisons), human_gold=False,
            calibration=False, new_inference_count=0))

    summary = dict(reviewed_model_proposals=len(reviewed), distinct_inputs=len(model_rows),
        extra_tool_hold_proposals=len(extra), source_scalar_audit_passed=True,
        conditional_reviewer_counts=dict(Counter(r['conditional_reviewer_label'] for r in reviewed)),
        hold_with_substantial_participation_restrictions=[r['proposal_id'] for r in reviewed
            if r['current_label']=='hold' and r['conditional_reviewer_label']=='reject'],
        raw_approve_with_substantial_restrictions=[r['proposal_id'] for r in reviewed
            if r['raw_binary']=='approve' and r['conditional_reviewer_label']=='reject'],
        differing_proposer_text_in_numeric_groups=[r['proposal_id'] for r in reviewed
            if not r['representative_proposer_text_matches']],
        human_gold=False, accuracy_measured=False, new_inference_count=0,
        original_evaluation_changed=False, judgments='agent review under existing conditional participation policy',
        source_report_sha256=sha(report_path), source_manifest_sha256=sha(manifest_path),
        script_sha256=sha(Path(__file__)))
    assert len(reviewed)==50 and len(extra)==7
    assert len(summary['hold_with_substantial_participation_restrictions'])==22
    save(OUT/'review.json', dict(summary=summary, proposals=reviewed, tool_cases=extra,
        threshold_comparison=threshold_rows, primary_protocol_source='https://docs.internetcomputer.org/references/sns-settings/'))

    lines = [
        '# BOOM DAO proposal 500〜660：元proposalと数値参加判定の照合', '',
        '2026-10-07。approve 2件・reject 10件に加え、モデルhold 38件も全件確認した。モデル対象50件・18種類の入力、およびTool holdの代表・証拠不足ケース7件を確認した。参照したproposalは2026-10-06取得の固定snapshotで、最新chain状態の評価ではない。', '',
        '数値上の参加条件という限定された方針では、approve 2件とreject 10件の方向は支持できる。一方、hold 38件のうち22件は大幅なstake/投票lockの制限強化を含み、同じ方針での私のレビューはreject相当とする。残り16件は最低投票lock 1→2日であり、悪影響を断定する追加根拠がないためholdを支持する。', '',
        'これはこのエージェントによる条件付きレビューで、人間が確認した正解ラベルではない。正答率・実際の採否・本番投票推奨は確定していない。過去GPTラベルや採決結果を正解に使わず、現行推論結果も変更していない。', '',
        '## 元データと入力の照合', '',
        '全50件についてsnapshot hashを照合し、ハーネスとは別のparserでCurrent/Newの整数・bool値を抽出した。構造化payloadの指定値とNew renderingは一致し、旧値との差分はtaskの変更項目集合と完全一致した。数値が同じ指定値は変更から除外されている。旧値は履歴rendering由来であり、chain上の当時の状態との独立照合はしていない。', '',
        '圧縮は変更された旧新数値を保持している。元の推論検証に加え、今回全50件の元payloadとの差分を照合した。ただし数値保持は意味理解の保証ではない。長い入力ほどunit_groupedへ切り替えるため、項目数・内容・圧縮形式・質問の短縮が同時に変わる。現データだけで圧縮が失敗の原因だとは断定できない。', '',
        f"数値入力の重複排除では、title/summaryが代表taskと異なるproposalも同じ結果を共有する。今回の該当IDは{summary['differing_proposer_text_in_numeric_groups']}。review.jsonには各proposal自身の原文を残した。数値判定では同一入力だが、目的や意図まで同一とは扱えない。", '',
        '最低stakeはneuronの最低stake条件、最低投票lockは投票資格に必要なdissolve delay。最大lockは最大値とbonus計算に関わる値で、全員を自動的にその期間lockする意味ではない。bonusは投票力配分に関わる。[公式SNS設定仕様](https://docs.internetcomputer.org/references/sns-settings/)で用語を確認した。既存neuronへの実際の影響・保有分布・価格・DAO防衛との利益衡量は今回確認していない。', '',
        '## 18種類の入力ごとのレビュー', '',
        '|proposal ID|現在の判定|A/Bの方向・未校正スコア|数値参加方針でのレビュー|根拠|',
        '|---|---|---|---|---|',
    ]
    for row in model_rows:
        judgment, reason = NOTES[row['record_index']]
        lines.append(f"|{','.join(map(str,row['proposal_ids']))}|{row['final_label']}|{row['prediction']} / {row['score']:.4f}|{judgment}|{reason}|")
    lines += ['', '## approveの限界', '',
        '#656はstakeを1,500万→5tokensへ戻し、投票期間を1→3日へ延ばすため、数値参加条件のapproveには根拠がある。ただし元summaryには「The takeover can still happen」という言及がある。これは攻撃の立証ではないが、意図・支配権の評価を要する文脈で、数値promptには残っていない。全proposalの承認へ拡張するならholdとして追加検討すべきである。', '',
        '#659もstakeと最低lockを下げる。一方、lock bonusを1→100%へ変えるので、投票力配分の影響は参加資格の改善だけでは評価できない。両件とも「数値上の参加条件を改善」というラベルとして扱う。', '',
        '## holdの原因', '',
        '22件のモデルholdは入力に大幅な制限強化がある。#562/#566はstake 5→100万tokens、最低lock 2→1,461日なのに、A/Bの方向がapproveでスコア0.5354だった。#587/#588もapprove方向だった。低スコアholdと承認証拠gateが誤ったapproveの出力を止めているが、モデル自身が一貫して制限を識別したとは言えない。', '',
        '#602等16件は最低lock 1→2日だけで、A/Bはreject方向0.5385。要件の強化と「禁止的な障壁」は同義ではない。適切な許容基準がない状況でholdを外すのは妥当でない。', '',
        '|Tool holdのID|元proposalの内容|レビュー|', '|---|---|---|',
        '|584|SNS treasuryから2,000万tokens送金。タイトルはSNS Metadata Adjustment。|送金額は確認できるが、残高・受取人支配・用途の根拠不足。holdを支持。|',
        '|653|2億5,000万tokens mint。|供給量・受取人の保有と支配の根拠不足。holdを支持。|',
        '|654 / 655|8,900 ICP / 1,800万SNS tokensを同じprincipalへ送金。|関係と用途の根拠不足。holdを支持。|',
        '|658|frontend batch 44のcommit。構造化payloadは空配列、renderingは非空payload hashとbatch evidenceを記す。|データ源の不一致とコード未検証によりholdを支持。空配列が実際のon-chain payloadだったとは断定しない。|',
        '|585|stake 5→100万tokens、最低lock 2→1,461日を含む。全変更を保持した入力130tokens。|内容上は制限強化だが、実行予算超過のholdを維持。長さの問題を証拠不足と区別する。|',
        '|501|ledgerのtransfer_fee 100万e8s指定。|数値参加判定の対象外。内容を評価済みという意味ではない。|', '',
        '## 既存logitsによる閾値比較', '',
        '追加推論なしで、同じA/B logitsと現行承認証拠gateを使った。下表の比較対象は今回の条件付きレビューであり、人間goldに対する精度や校正結果ではない。', '',
        '|閾値|approve|reject|hold|レビューreject 32件のうちrejectを返す件数|レビューhold 16件をrejectにする件数|',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for row in threshold_rows:
        c, matrix = row['counts'], row['conditional_review_confusion']
        lines.append(f"|{row['threshold']}|{c.get('approve',0)}|{c.get('reject',0)}|{c.get('hold',0)}|{matrix.get('reject→reject',0)}|{matrix.get('hold→reject',0)}|")
    lines += ['',
        '閾値0.6ではレビューreject 32件中10件をrejectにできる。0.5では28件になるが、最低lock 1→2日の16件もrejectになる。さらに制限強化4件はA/Bがapprove方向なので、閾値だけではrejectへ直せない。全体の閾値を下げる解決は採用しない。', '',
        '## 次の改善に使う固定ケース', '',
        'review.jsonに全50件の旧新値・現行入力・入力形式・根拠・条件付きラベルとsource hashを保存した。内容と形式の影響を分離する比較では、#562（両要件悪化）、#590（stake緩和とlock悪化）、#570（長めのratio）、#602（小幅悪化）、#656/#659（緩和）を優先する。短い単一要件の#553も対照にする。', '',
        '表現比較は、質問・選択肢・モデル・閾値を固定し、全変更を残したまま資格条件を先に置く表現と現行表現を比較する。128tokensを超えるケースは除外理由を記録し、強制的な86token化や変更項目削除をしない。今回の7ケースで調整した後は別の未使用ケースで検証し、既知ケースへの合わせ込みと区別する。', '',
        '数値差分と単位はコードで厳密に処理し、参加条件の改善と全proposalの安全性を別項目として記録する設計が必要である。「禁止的」の許容基準を明示せず、増加ならすべてrejectという規則を追加してモデル改善と数えるべきではない。今回は推論・本番policy・元評価ファイルを変更していない。', '',
    ]
    (OUT/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'checks.json', dict(summary, artifacts={p.name: sha(p) for p in
        (OUT/'review.json', OUT/'REVIEW.md')}))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
