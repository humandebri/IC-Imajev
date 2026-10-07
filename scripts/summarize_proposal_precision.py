#!/usr/bin/env python3
"""Independently audit completed reports and summarize precision tradeoffs."""
import hashlib
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/proposal-full-161-20261007'
MODULE = '6ce099e1f128b835b153bb28cde9f27dee719711be3822a38976cb4688613de4'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder, rows):
    fixture = read(folder / 'inputs.json')
    manifest = read(ROOT / 'checkpoints/full-int8.manifest.json')
    for row in rows:
        dest = folder / 'runs' / f"{row['record']:03d}"
        r = read(dest / 'report.json')
        x = fixture['records'][row['record']]
        assert sha(dest / 'report.json') == row['report_sha256']
        assert r['input_hash'] == x['input_sha256']
        assert r['tokens'] == len(x['token_ids'])
        assert r['model'] == fixture['model_lock_sha256'] == manifest['model']
        assert r['pack_hash'] == manifest['pack_hash']
        assert r['wasm_sha256'] == r['deployed_wasm_sha256'] == MODULE
        assert [z['layer'] for z in r['layers']] == list(range(32))
        assert r['executed_query_count'] == r['query_count'] == row['queries']
        assert not r['replayed_queries'] and not r.get('fallback')
        assert r['comparison']['typed_output_valid']
        assert r['max_query_instructions'] < 5_000_000_000
        assert r['max_observed_heap_bytes'] < 2**32
        assert all(q['ok']['request_bytes'] < 2_000_000 and
                   q['ok']['reply_bytes'] < 2_000_000 for q in r['queries'])
        decision = r['decision_query']['ok']['decision']
        prediction = '__unknown__' if decision['abstained'] else decision['value']
        assert prediction == row['prediction']
        assert decision['raw_logits'] == row['raw_logits']
        assert decision['probabilities'] == row['probabilities']
        assert decision['unknown_probability'] == row['unknown_probability']
        if 'gold' in row:
            assert x['gold'] == row['gold']
            assert row['correct'] == (prediction == x['gold'])
            assert len(decision['raw_logits']) == len(x['options']) + 1
    return {'verified_runs': len(rows), 'queries': sum(x['queries'] for x in rows)}


def metrics(rows):
    answered = [r for r in rows if r['prediction'] != '__unknown__']
    unknown = [r for r in rows if r['gold'] == '__unknown__']
    known = [r for r in rows if r['gold'] != '__unknown__']
    return {
        'cases': len(rows), 'correct': sum(r['correct'] for r in rows),
        'accuracy': sum(r['correct'] for r in rows) / len(rows) if rows else None,
        'answered': len(answered),
        'answered_precision': sum(r['correct'] for r in answered) / len(answered) if answered else None,
        'coverage': len(answered) / len(rows) if rows else None,
        'unsupported_answers': sum(r['prediction'] != '__unknown__' for r in unknown),
        'unknown_cases': len(unknown),
        'correct_abstentions': sum(r['prediction'] == '__unknown__' for r in unknown),
        'known_cases': len(known),
        'false_abstentions': sum(r['prediction'] == '__unknown__' for r in known),
        'wrong_answers_on_known': sum(not r['correct'] and r['prediction'] != '__unknown__' for r in known),
    }


def main(allow_partial=False):
    paired = read(D / 'budget-comparison/report.json')
    all161 = read(D / 'report.json')
    assert paired['complete'], 'paired comparison still running'
    if not allow_partial:
        assert all161['summary']['complete'], 'all-proposal experiment still running'
        assert {r['record'] for r in all161['inputs']} == set(range(99))
        assert {p['proposal_id'] for p in all161['proposals']} == set(range(500, 661))
        assert all(p['model_complete'] for p in all161['proposals'])
        assert sum(p['completed_windows'] for p in all161['proposals']) == 721
    assert {r['record'] for r in paired['runs']} == set(range(64))
    checks = {
        'paired': audit(D / 'budget-comparison', paired['runs']),
        'all161': audit(D, all161['inputs']),
    }
    for folder, names in [
        (D, ['preparation-identities.json', 'execution-identities.json']),
        (D / 'budget-comparison', ['identities.json', 'execution-identities.json']),
    ]:
        for name in names:
            for path, digest in read(folder / name).items():
                assert sha(Path(path)) == digest, f'source changed: {path}'
    primary = [r for r in paired['runs'] if r['offset'] == 0]
    quality = {v: metrics([r for r in primary if r['variant'] == v])
               for v in ('full', 'compact')}
    assert all(q['cases'] == 24 for q in quality.values())
    categories = {p['id']: p['category'] for p in paired['paired']}
    by_category = {
        c: {v: metrics([r for r in primary if r['variant'] == v and categories[r['id']] == c])
            for v in ('full', 'compact')}
        for c in sorted(set(categories.values()))
    }
    route = read(D / 'budget-comparison/route-checks/verification.json')
    assert route['decision_logits_probabilities_bit_equal']
    assert route['final_hidden_bit_equal']
    route32 = read(D / 'budget-comparison/runs/001/report.json')
    route50 = read(D / 'budget-comparison/route-checks/001/report.json')
    assert route32['input_hash'] == route50['input_hash'] == route['input_hash']
    assert route32['query_count'] == 32 and route50['query_count'] == 50
    for key in ('value', 'abstained', 'raw_logits', 'probabilities', 'unknown_probability'):
        assert route32['decision_query']['ok']['decision'][key] == route50['decision_query']['ok']['decision'][key]
    assert sha(D / 'budget-comparison/runs/001/final-hidden.npy') == sha(D / 'budget-comparison/route-checks/001/final-hidden.npy')
    prep_queries = 0
    for folder in (D, D / 'budget-comparison'):
        prefix = read(folder / 'prefix/report.json')
        packets = read(folder / 'packets/report.json')
        prep_queries += prefix['executed_query_count']
        # Codec's preparation report has a different schema; count actual calls.
        prep_queries += packets['preparation_queries']
    summary = {
        'complete': all161['summary']['complete'], 'precision_comparison_complete': True,
        'primary_quality': quality,
        'ordered_quality': {v: metrics([r for r in paired['runs'] if r['variant'] == v])
                            for v in ('full', 'compact')},
        'category_quality': by_category,
        'single_window_predictions': dict(Counter(p['single_window_prediction']
            for p in all161['proposals'] if p['windows'] == 1 and p['model_complete'])),
        'actionable_distinct_inputs': sum(r['prediction'] in ('approve', 'reject') for r in all161['inputs']),
        'primary_regressions': [p for p in paired['regressions'] if p['offset'] == 0],
        'primary_improvements': [p for p in paired['improvements'] if p['offset'] == 0],
        'order_changes': paired['order_changes'], 'all161': all161['summary'],
        'route_equality': route, 'verification': checks,
        'preparation_queries': prep_queries, 'route_check_extra_queries': 50,
        'inference_queries': checks['paired']['queries'] + checks['all161']['queries'],
        'development_set': True, 'proposal_vote_accuracy_measured': False,
        'all161_methodology_valid_for_proposal_assessment': False,
        'all161_experiment_role': 'Input-path smoke test only; independent partial evidence is not a proposal assessment.',
        'recommendation': 'Preserve natural factual evidence; use 32 queries when unchanged input fits 86 tokens, otherwise allow 128 tokens and 50 queries. Do not mandate lossy shortening.',
    }
    (D / 'precision-assessment.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Imajev proposal検査の評価', '',
             '## 全件評価の設計上の問題（訂正）', '',
             '全161件の走査は入力経路の動作確認であり、161件のproposal評価としては成立していない。「161件を評価した」という説明は不適切だった。証拠の所在と実行完了は確認したが、判断に必要な文脈と判定単位を保持できていない。', '',
             '- 元のツールの数値・参加条件の問いから、広いapprove/reject/holdの問いへ変更した。これらは別の評価課題である。',
             '- 141件を独立した証拠断片へ分割した。全断片を処理しても、モデルがproposal全体を理解したことにはならない。',
             '- 全体判断へ統合する根拠ある仕組みと、人手で確認した賛否の正解がない。全件holdからモデルの能力や全件評価の価値を判断できない。', '',
             '以後、この全件走査を賛否評価の証拠から外す。正しい再評価では、元ツールの対象条件を固定し、各判定に必要な旧値・新値・単位・対象・否定・不足情報を一つの判定単位へ揃え、同じ問いと確認済みの正解で比較する。条件に必要な証拠が収まらなければ、賛否とは分けて予算超過／証拠不足として記録する。', '',
             '24種類の表現比較は限定した事実確認として有効だが、全proposalの賛否や128 tokenの一般的優越性を検証したものではない。以下の入力予算方針は、強制短縮を避ける暫定方針である。', '',
             '精度優先の運用では、86 tokenへの一律短縮を採用しない。自然文と数値・単位・否定・不足情報を保持し、同じ入力が86 tokenに収まれば32 query、長ければ128 token上限の50 queryを使う。27 + 59 = 86であり、87ではない。', '',
             '## 表現を短縮した場合の正答率', '',
             '同じ質問・選択肢・モデルによる24種類の開発用対照ケース。主要ケースを1回ずつ数え、選択肢順序違いは別途検査した。元の入力は71〜96 token、短縮後は67〜85 token。128 token全体の性能や未知proposalへの一般化を測った比較ではない。', '',
             '|入力|正解/24|正答率|回答した場合の正答率|情報不足への誤回答|答えられるのに保留|',
             '|---|---:|---:|---:|---:|---:|']
    for v, label in [('full', '自然文（上限128）'), ('compact', '短縮（上限86）')]:
        m = quality[v]
        lines.append(f"|{label}|{m['correct']}/24|{m['accuracy']:.1%}|{m['answered_precision']:.1%}|{m['unsupported_answers']}/{m['unknown_cases']}|{m['false_abstentions']}/{m['known_cases']}|")
    regressions = ', '.join(p['id'] for p in summary['primary_regressions']) or 'なし'
    improvements = ', '.join(p['id'] for p in summary['primary_improvements']) or 'なし'
    lines += ['', f'短縮による回帰: {regressions}。改善: {improvements}。選択肢順序で出力が変わった検査: {len(paired["order_changes"])}。', '',
              '今回の短縮方法での比較であり、24種類の小さな開発用標本から、128 tokenの一般的な優越性や他の短縮方法の限界を断定しない。', '',
              '同じ83 token入力を32 queryと50 queryで再実行した検査では、logits・確率・最終hiddenが完全一致した。query数そのものによる精度低下はこの例では見られず、短縮時の表現変更が問題となる。1入力の検査なので全入力の数値同一性を新たに証明したものではない。', '',
              '## proposal 500〜660の全件実験', '']
    a = all161['summary']
    lines += [f"161件に対するfacts投影の{a['completed_windows']}/{a['total_windows']}部分入力を実行した。重複を再利用し、異なる{a['completed_inputs']}/{a['distinct_inputs']}入力を推論・検証。全入力が完了したproposalは{a['completed_proposals']}/161件。最大{a['actual_max_tokens']} token、現時点の結果は{a['predictions_distinct_inputs']}。", '',
              f"単一入力でfacts投影全体を渡せたproposalは{a['single_window_proposals']}件。残りは部分入力なので、部分結果を全体の賛否へ合成していない。人手の賛否正解ラベルがなく、161件の投票精度は未測定。holdへの集中は精度の高さを証明しない。", '',
              'この実験で確認できたのは、従来除外したproposalも含め、保持した数値・識別子・説明文を予算内の入力へ通せることと、その部分入力の生出力である。画像・コード・リンク先の検証、および長いproposal全体の判断には別の証拠処理が必要となる。', '',
              f"広い賛否質問に対するapprove/rejectは{summary['actionable_distinct_inputs']}入力。単一入力の{summary['single_window_predictions']}も含め、今回の設計では追加の賛否判断を得られなかった。この結果から全件評価自体の価値やモデルの賛否能力は判断しない。改善評価は、正解を確認できる事実別の問いで行う。", '',
              f"全件実験の推論{checks['all161']['queries']:,} query、表現比較{checks['paired']['queries']:,} query、2実験のprefix/codec準備{prep_queries:,} query、経路同一性の追加検査50 query。モデルuploadとweight cache準備のupdateは別。共有ローカル環境の負荷が変動したため所要時間から速度優劣は判定しない。", '',
              '## 記録', '',
              '- [全161件の出力](REPORT.md)',
              '- [表現比較の各ケース](budget-comparison/REPORT.md)',
              '- [独立検証と集計](precision-assessment.json)',
              '- [入力のtoken数と証拠保持の検証](input-verification.json)',
              '- [実験条件と対象範囲](EXPERIMENT_PLAN.md)', '']
    (D / 'ASSESSMENT.md').write_text('\n'.join(lines))
    print(json.dumps({'quality': quality, 'all161': a, 'checks': checks}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-partial', action='store_true')
    main(parser.parse_args().allow_partial)
