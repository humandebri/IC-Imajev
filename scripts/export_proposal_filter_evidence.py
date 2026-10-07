#!/usr/bin/env python3
"""Export a small, portable evidence bundle without weights or private keys."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'docs/evidence/proposal-filter-20261007'
FRESH = 'artifacts/proposal-assessment-fresh-500-660-20261007'
REVIEW = 'artifacts/proposal-evidence-review-20261007'
OPT = 'artifacts/proposal-query-optimization-v1'
SOURCES = {
    f'{FRESH}/RESULT.md': 'RESULT.md',
    f'{FRESH}/evaluation/REPORT.md': 'DECISIONS.md',
    f'{FRESH}/evaluation/report.json': 'decisions.json',
    f'{FRESH}/evaluation/inputs.json': 'inputs.json',
    f'{FRESH}/verification.json': 'verification.json',
    f'{FRESH}/plan.json': 'plan.json',
    f'{FRESH}/execution-policy.json': 'execution-policy.json',
    f'{FRESH}/fresh-summary.json': 'fresh-summary.json',
    f'{FRESH}/evaluation/runs/000/report.json': 'run-000.json',
    f'{REVIEW}/REVIEW.md': 'REVIEW.md',
    f'{REVIEW}/review.json': 'review.json',
    f'{OPT}/REPORT.md': 'OPTIMIZATION.md',
    f'{OPT}/verification.json': 'optimization-verification.json',
    'artifacts/merged-query32-v1/build/report.json': 'build.json',
    'checkpoints/full-int8.manifest.json': 'pack-manifest.json',
    'checkpoints/adapter/RELEASE-SPEC.md': 'RELEASE-SPEC.md',
    'checkpoints/adapter/README.md': 'MODEL_CARD.md',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def portable(value):
    if isinstance(value, str):
        return value.replace(str(ROOT) + '/', '').replace(str(ROOT), '.')
    if isinstance(value, list):
        return [portable(x) for x in value]
    if isinstance(value, dict):
        return {portable(k): portable(v) for k, v in value.items()}
    return value


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def export(check=False):
    outputs, sources = {}, []
    for source, target in SOURCES.items():
        raw = (ROOT / source).read_bytes()
        if target.endswith('.json'):
            data = json.loads(raw)
            if target == 'decisions.json':
                data = {k: data[k] for k in ('summary', 'proposals')}
            content = encoded(portable(data))
        else:
            content = portable(raw.decode())
            if target == 'RESULT.md':
                content = content.replace('](evaluation/REPORT.md)', '](DECISIONS.md)')
            content = content.encode()
        outputs[target] = content
        sources.append(dict(source=source, source_sha256=sha(raw), export=target,
                            export_sha256=sha(content), bytes=len(content),
                            projection=target == 'decisions.json'))
    outputs['provenance.json'] = encoded(dict(
        date='2026-10-07', sources=sources,
        model_lock_sha256=sha((ROOT / 'MODEL_LOCK.json').read_bytes()),
        exporter_sha256=sha(Path(__file__).read_bytes()),
        transformations=['Repository absolute paths become repository-relative strings.',
                         'RESULT.md links target the exported decision report.',
                         'decisions.json contains summary and all 161 proposals; model_rows are omitted.'],
        new_inference_count=0, human_gold=False,
        scope='Export of saved evidence; hashes are provenance, not an independent execution attestation.'))
    outputs['README.md'] = ('''# proposalフィルター評価の共有用根拠資料

2026-10-07の保存済み結果から作成した公開用の写しです。新しい推論や判定変更は行っていません。全161件の処理結果、18種類のモデル入力、集計、条件付きレビュー、検証記録を含みます。人間の正解ラベルに対する精度評価ではありません。

- [全161件の判定・処理経路・理由](DECISIONS.md)、[機械可読の判定一覧](decisions.json)
- [新規実行の集計](fresh-summary.json)、[実行検証](verification.json)
- [入力token列とprompt](inputs.json)、[prefix計画](plan.json)、[実行方針](execution-policy.json)
- [元proposalとの根拠照合と閾値比較](REVIEW.md)、[各件の旧新値とレビュー](review.json)
- [128-token入力の実行記録例](run-000.json)
- [最適化の測定](OPTIMIZATION.md)、[照合結果](optimization-verification.json)、[build記録](build.json)
- [packの構成](pack-manifest.json)、[固定版のrelease specification](RELEASE-SPEC.md)、[モデルカード](MODEL_CARD.md)
- [原本と出力のSHA-256・変換範囲](provenance.json)

ファイル内の原本・ソース等のパス文字列はリポジトリ基準です。参照先の重み、Wasm、全snapshot、中間tensor、生ログはこの共有資料には含みません。別checkoutで元実験をそのまま再実行できるbundleではありません。検証記録は保存済み実験についての記録であり、この写しだけで独立した実行証明になるものではありません。

元のローカル証跡がそろった環境では、`python3 scripts/export_proposal_filter_evidence.py`で再出力、`--check`で原本からの変換一致を確認できます。原本の`artifacts/`と`checkpoints/`全体をGit対象へ変更する必要はありません。
''').encode()
    if not check:
        DEST.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        path = DEST / name
        if check:
            assert path.read_bytes() == content, f'export differs: {name}'
        else:
            path.write_bytes(content)
    print(json.dumps(dict(files=len(outputs), bytes=sum(map(len, outputs.values())),
                          source_match_checked=check, new_inferences=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    export(parser.parse_args().check)
