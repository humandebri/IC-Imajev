# ローカルパスの移植方針

[English](PATH_PORTABILITY.en.md)

2026-10-07に、共有対象の文書・来歴情報・実行スクリプトから、このMacの外付けdiskと個人ホームに依存するパスを取り除いた。自リポジトリの位置は`__file__`から求め、判定コードは`tools/proposal_assessment/`を使う。移植は実行環境の指定を変えるもので、数値圧縮・入力予算・判定gateを変更しない。

## 外部データを必要とするコマンド

引数で指定したデータは読み取り用である。必要な引数がなければ、実験の出力やcanister操作を始める前に停止する。

|スクリプト|指定する引数|入力|
|---|---|---|
|`analyze_laya_cost.py`|`--laya-root`|比較用Laya checkout|
|`summarize_optimizations.py`|`--laya-root`|参照ソースを持つLaya checkout|
|`benchmark_laya_kernels.py` / `benchmark_laya_peak.py`|`--laya-root`|Layaの固定ソース・重み。`--directory`で新しい出力先も指定|
|`make_benchmark.py`|`--source`|保存済み`boomdao_query_benchmark/summary.json`|
|`report_proposal_assessment.py`|`--source`|保存済み`boomdao-600-660/v1`ディレクトリ|
|`benchmark_proposal_assessment.py`|`--source-root`|旧benchmark archiveを持つroot。既定値はこのcheckout|
|`assess_proposal_range.py` / `run_fresh_proposal_assessment.py`|`--snapshot-archive`|`manifest.json`と`snapshots/`を持つ参照元archive|
|`prepare_all_proposal_windows.py` / `prepare_all_proposal_query32.py` / `prepare_all_proposal_compact86.py`|`--source-directory`|`snapshots/manifest.json`と`evaluation/report.json`を持つ保存済みrun|
|`checkpoint_goal_recovery_baseline.py`|`--backup`|snapshotを書き出す新しいディレクトリ。親ディレクトリは既存であること|

たとえば隣に読み取り用Laya checkoutがある場合、次のように指定する。

```sh
python3 scripts/analyze_laya_cost.py --laya-root ../IC-Laya-Standalone
python3 scripts/make_benchmark.py --source ../IC-Laya-Standalone/artifacts/boomdao_query_benchmark/summary.json
```

復旧用snapshotの保存先は自動選択しない。上表は指定方法の説明であり、snapshot操作や推論を今回実行したものではない。

`benchmark_proposal_assessment.py`は、manifestに旧絶対パスが残っていても、`--source-root`で選んだarchive内の`snapshots/proposal-ID.json`を読む。移動先にファイルがなければ停止し、元の絶対パスには戻らない。archive外への参照を拒否し、内容のSHA-256・proposal ID・SNS rootを検証する。凍結manifest自体は書き換えない。

同じ検証を`proposal_snapshots.py`へ共通化し、500–660用driverと3つの準備スクリプトにも適用した。fresh driverはmanifestだけを出力先へコピーし、snapshotは指定した参照元archiveから読む。評価用manifestと参照元manifestのhashが一致しなければ停止する。`--directory`には新しい出力先を指定する。

```sh
.venv/bin/python scripts/run_fresh_proposal_assessment.py prepare \
  --snapshot-archive ../retained-run/snapshots --directory artifacts/new-fresh-run
.venv/bin/python scripts/prepare_all_proposal_query32.py \
  --source-directory ../retained-run --directory artifacts/new-query32-inputs
```

参照元archiveは`manifest.json`と、その隣の`snapshots/proposal-ID.json`を含む。上記は指定方法の例で、移植後の入力準備では移動した161件のhash・ID・SNS rootと、既存18種類の入力との一致を確認した。推論は再実行していない。

## 原本と現在のソース

取り込み元は`tools/proposal_assessment/UPSTREAM.json`にリポジトリ名・相対パス・原本hashで記録する。`local_sha256`は現在のコピーを識別する。変更したCLI既定値と、原本のままの判定コードを`adapted`で区別する。

変更前の33ファイルはローカルの`artifacts/path-portability-20261007/frozen/`へ保存し、`original-sources.json`に原本hashと保存先を記録した。過去実験の凍結manifestやsource hashを書き換えて、現行ソースを当時のものと扱うことはしていない。過去driverのsource照合が現行ファイルで停止する場合は、当時のソースで検証する必要がある。新規実験には新しい出力先と現行ソースのhashを使う。

過去GPT参照結果の共有用JSONはコマンドの実行ファイルと出力先だけを一般化し、判定・使用token・計測値を維持した。正規化前の原本hashと変更範囲を`path_export`に記録し、原本は上記の凍結保存に含めた。

## 残した一時パス

個人・外付けdiskのパスとは分け、次の6か所の一時パスを残した。

- `test_adaptive_graph.py`、`test_packed_graph.py`、`test_limit_fallback.py`の計4か所は、fallbackコマンドを組み立てるテストのダミーパス。実際の保存先には使わない。
- `test_review_fixes.py`の1か所は、mock transportで演算のframe容量を検査するダミーパス。ファイルは書き込まない。
- `finalize_proposal_assessment.py`の1か所は、過去実験のSSD stagingを戻す際の削除範囲の検査。既存の履歴metadataに合わせた境界確認なので、汎用の一時ディレクトリへ置き換えて削除範囲を広げない。今回この復元処理は実行していない。

## 確認記録

別名かつ空白を含む一時checkoutへ必要なソースをコピーし、ROOT、ローカルimport、CLI既定値、外部引数の必須指定、元ベンチ入力の一致を確認した。Git worktreeは作成していない。判定の保存結果と凍結metadataも照合し、推論・canister操作は実行していない。

移植前後のソースhashと検証結果は[共有用記録](evidence/path-portability-20261007/verification.json)に保存する。重み・秘密鍵・当時のローカル絶対パスを含む原本はこの共有記録には含まない。
