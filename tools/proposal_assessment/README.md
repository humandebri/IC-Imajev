# Proposal assessment: local binary harness

IC-Laya-Standaloneの`compact_units.py`と`benchmark_600_660_binary.py`を、必要な依存とともに取り込んだ。取り込み元リポジトリ名、相対パス、原本と現行コピーのSHA-256は`UPSTREAM.json`に記録している。この2ファイルの判定・圧縮ロジックは変更していない。600〜660の準備処理には、選択したarchive内のsnapshotをhash・proposal ID・SNS rootで検証する読み込みと`--snapshot-archive`を追加した。`binary_benchmark.IMAJEV`とtoken計測CLIの既定値は、このリポジトリのrootを使う。旧本文ベンチマーク用の`evaluation.py`も原本と同じ内容で取り込んだ。

数値上の投票参加条件の検査であり、proposal全体の安全性・賛否の正解を保証するものではない。

- 対象選別・証拠不足の判定を維持する。
- 旧値・新値・単位・全変更項目を保持し、必要な場合に正確な単位換算と倍率表記で圧縮する。断片を独立した賛否入力へ分割しない。
- モデルの選択肢はapprove/reject。holdは未校正の二択スコア閾値0.6と承認証拠検査などで決める。
- 上限128は質問・選択肢・chat特殊記号を含む。超過した場合は切り捨てずbudget_overflowとして記録する。
- 過去GPT出力は参照値であり、人手goldではない。参照ラベルやproposal IDをモデル入力へ入れない。

## ローカルでの使用

600〜660の元ベンチマークは、リポジトリrootから実行する。出力先は新しいディレクトリにする。

```sh
.venv/bin/python -B -m tools.proposal_assessment.benchmark_600_660_binary prepare \
  --output artifacts/proposal-assessment/my-600-660 --compact-ratio \
  --snapshot-archive ../retained-600-660/v1
.venv/bin/python -B -m tools.proposal_assessment.benchmark_600_660_binary report \
  --output artifacts/proposal-assessment/my-600-660
```

`--snapshot-archive`は`manifest.json`と`snapshots/`を含むディレクトリ。省略時はこのcheckoutの`artifacts/proposal-assessment/boomdao-600-660/v1`を使う。manifestに旧絶対パスが残っていても選択したarchive内だけを読み、移動先にsnapshotがなければ停止する。

500〜660の全161件には、同じローカル依存を使う`scripts/assess_proposal_range.py`を使用する。新しい出力ディレクトリの`snapshots/manifest.json`に、hash付きの取得済みsnapshot manifestを置く。

```sh
.venv/bin/python -B scripts/assess_proposal_range.py prepare \
  --directory artifacts/proposal-assessment/my-500-660
.venv/bin/python -B scripts/assess_proposal_range.py report \
  --directory artifacts/proposal-assessment/my-500-660
```

今回の移植確認は`artifacts/proposal-assessment-local-500-660-20261007/evaluation/REPORT.md`と`artifacts/proposal-assessment-local-600-660-20261007/REPORT.md`。入力token列・選択肢・モデルの同一性を照合した既存推論を再利用しており、新しいモデル推論ではない。`prepare`が既存結果を再利用できないケースはpendingのまま残す。snapshot・参照データ・旧推論reportはローカルの`artifacts/`に保管し、モデル重みやrequest/reply frameは移植しない。

元ベンチマークの`run`は旧6052 module用の実行ドライバで、現在の6ce moduleにそのまま使うものではない。新規推論では、入力を変えずに現在のImajev queryドライバへ渡し、module・モデル・入力hashを検証する。移植の一致確認では新規推論を実行していない。

## 新規の再判定（2026-10-07）

`scripts/run_fresh_proposal_assessment.py`は、同じ数値・二択・閾値・承認証拠検査を維持し、現在の検証済み6ce moduleに新規queryを送るドライバ。`assess_proposal_range.prepare(reuse=False)`で全161件の処理経路と入力を再生成し、既存予測と既存prefix状態を再利用しない。今回の実行内では、同一入力と同一前半状態をまとめて計算する。入力に必要な変更項目を独立した証拠断片へ分割することはない。

固定した出力先は`artifacts/proposal-assessment-fresh-500-660-20261007/`。新規18推論で50件のモデル対象を処理し、残り111件は元ルールで判定・対象外を記録した。モデル重みやcanisterの更新はしていない。再判定結果はapprove 2、reject 10、hold 61、対象外88。全18種類のlogitsが前回と一致。賛否の人手goldによる精度評価ではない。

```sh
# prepareは新しい出力先だけを受け付ける。runは同じ実行の未完了分を再開する。
.venv/bin/python -B scripts/run_fresh_proposal_assessment.py prepare
.venv/bin/python -B scripts/run_fresh_proposal_assessment.py run
.venv/bin/python -B scripts/verify_fresh_proposal_assessment.py
```

途中終了した推論・prefix準備・codec準備を再実行する際は、前の試行をrun内の`aborted/`へ番号付きで退避し、空のquery journalから開始する。検証済みの推論と完了した準備は維持する。prefixを作り直す場合は対応するcodec packetも作り直す。旧ソースhashを持つ過去runの凍結情報は更新せず、当時のソースで検証する。

移植の一致確認:

```sh
.venv/bin/python -B -m tools.proposal_assessment.verify_local_copy
```

## パスの移植と凍結ソース

`--imajev`の既定値はこのcheckoutのrootであり、別の配置先でも固定の外付けdiskパスを必要としない。`UPSTREAM.json`の`source`は参照元リポジトリ基準。`source_sha256`は取得時の原本、`local_sha256`は現在のコピーを表し、変更があるファイルは`adapted`で区別する。

```sh
python3 -B -m tools.proposal_assessment.verify_local_copy --sources-only
# 参照元checkoutもある場合は原本のhashを照合する
python3 -B -m tools.proposal_assessment.verify_local_copy --sources-only --source-root ../IC-Laya-Standalone
```

2026-10-07のパス整理前のソースはローカルの`artifacts/path-portability-20261007/`に保存した。過去実験のsource hashや凍結manifestは変更していないため、過去のsource hashを要求するdriverへ現行ソースを渡すと照合で停止する。これは判定結果の変更や実行失敗を示すものではない。過去実験の再検証には保存した当時のソースを使い、新規実験は現行ソースで新しい出力先・source hashを作る。[移植方針と検証](../../docs/PATH_PORTABILITY.md)。
