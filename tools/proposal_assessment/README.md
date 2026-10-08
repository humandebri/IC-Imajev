# Imajev proposal assessment

SNS snapshotの型検査、履歴renderingからの旧値抽出、重要項目の選別、証拠不足のhold、全変更を保持するexactな入力圧縮を提供する。実行時に外部のLaya checkout・checkpoint・classifierは使用しない。

2026-10-08に実装を置き換え、過去のprompt sweep・モデル比較・multilingual classifier CLIを削除した。現在のImajev driverが使う関数は維持する。質問文とフィールド説明は入力を変えないため`prompt_contract.json`にデータとして残し、出所とライセンスを`PROVENANCE.json`に記録する。MIT noticeは削除しない。

新規推論は`scripts/run_fresh_proposal_assessment.py`を使う。module・モデル・入力hash・全層・query実行数の検証を維持する。旧600〜660のprepare/reportは検証済みsnapshot archiveを読む互換入口であり、`run`は凍結された旧module用driverを実行する。現在のcanisterへそのまま使わない。

数値上の投票参加条件を検査する実験であり、proposal全体の安全性や採否の正解を保証しない。入力上限128を超える場合は切り捨てずhold。閾値0.6は未校正。除外は承認ではなく、投票は実行しない。

置き換えの検証:

```sh
PYTHONPATH=scripts:. .venv/bin/python -B scripts/check_proposal_replacement.py \
  --baseline artifacts/laya-removal-20261008/baseline \
  --archive artifacts/proposal-assessment-500-660-20261006/snapshots \
  --output artifacts/laya-removal-20261008/proposal-conformance.json
python3 -B -m tools.proposal_assessment.verify_local_copy --sources-only
```

過去runのsource hashと凍結manifestは変更しない。過去実装を要求する検証には、その実験の保存済みソースを使う。
