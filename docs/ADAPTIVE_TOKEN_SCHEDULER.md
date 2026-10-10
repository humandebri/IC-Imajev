# 入力長に応じた有料update推論

受付上限は固定prefix 27 tokensを含む512 tokens。標準Cargo featureは512対応と演算ごとの分割を有効にし、有料updateに必要な全重みcacheの容量も有効にする。最適化済みWasmを作る `build_latest_common_prefix27.py` も512対応へ委譲する。公開canisterへの適用は別操作である。

116 tokens以下は従来の一括処理を維持する。117〜512 tokensは演算ごとのスケジュールを生成する。全512 tokens分を毎回計算することや、端数を埋めることはない。各層・各演算は実際のsuffixだけを処理する。

`token_plan.rs` はDelta、Attention、MLPの分割を独立して記述する。最終層のMLPは最後の1 tokenだけを処理する。進捗stageは実行する演算の数に対応し、継続・callback・完了判定も同じ上限を使う。117〜512の全入力長を列挙するnativeテストで、各演算の順序、全tokensの過不足ない処理、端数、stage上限を確認する。

検証済みの57-token固定版と比較し、Delta・MLPを最大89 tokens、Attentionを最大57 tokensとする `experimental-adaptive-token-tiles` を標準に採用した。`--fixed-token-tiles` またはCargoの `--no-default-features --features experimental-update-token-chunks` で以前の57-token分割を選べる。AttentionはKV履歴に応じて16 headsの入力・中間配列・演算量を判定し、必要な区間だけ8＋8 headsへ分割する。量子化・QのLoRA A積などをgroupごとに繰り返さない。

新規推論の受付上限とreceipt再取得の上限は独立している。互換性確認用の116-token版へ戻しても、保存済み512-token依頼の同じcaller・入力・IDは結果と返金状態を再取得できる。

## ローカル検証

`artifacts/adaptive-512-v1/build-candidate` は34個の投影kernel本体を従来の検証済み版と一致させた89-token候補。`proof-candidate` は70、84、100、116、117、132、160、192、256、359、512、513 tokensを測定する。513は受領前拒否、512以下は `--require-success` で推論完走を要求する。

70・100・132は新しい独立query参照を計算した。既存の参照がある入力長はモデル・version・全token IDs・選択肢の同一性を確認し、保存された独立queryの全32層のhidden・conv／suffix KV、最終hidden、判定・確率・logitsを新しい推論と比較する。保存参照の再利用は新しいquery計算とは区別してreportへ記録する。160・192・359は容量・命令数の測定で、今回の独立query参照との全層比較対象には含めない。

測定中は選択済み `http://localhost:8001/` のモデルcanisterをsnapshotで保護する。終了・失敗時とも元module・cache・packの復元を確認し、試験snapshotとcallerを後始末する。ネットワークを再起動しない。89-token分割は出力一致と命令数の改善を確認して標準へ採用した。

## ビルド

```sh
# 標準512-token版。診断endpointは含めない。
python3 scripts/build_latest_common_prefix27.py \
  --source-root "$PWD" \
  --directory artifacts/adaptive-512-v1/normal

# 互換性確認用の116-token版。
python3 scripts/build_latest_common_prefix27.py \
  --source-root "$PWD" --legacy-116 \
  --directory artifacts/adaptive-512-v1/legacy-116
```

通常Cargo buildは最適化済み凍結kernelを使うWasmと同じ性能とは扱わない。ローカルの命令数と時間は本番subnetでの性能保証ではない。

## 完了した測定

診断moduleは `a00c1e5ccf92f5027a42d8aa1719b9fa0fbbd512fd409c4b6aee086e8ad2e8a4`。測定した512以下の全11入力が完走し、513は受領前に拒否して添付cyclesを全額返却した。11入力は70／84／100／116／117／132／160／192／256／359／512 tokens。

| 全入力tokens | 内部worker | 時間（秒） | 命令合計（B） | 前版との命令数の差 |
| --- | --- | --- | --- | --- |
| 70 | 4 | 25.99 | 92.56 | 比較なし |
| 84 | 4 | 27.32 | 121.67 | +0.0023% |
| 100 | 5 | 38.76 | 154.20 | 比較なし |
| 116 | 7 | 53.80 | 187.08 | +0.0125% |
| 117 | 7 | 61.13 | 191.88 | −0.4888% |
| 132 | 8 | 56.36 | 223.36 | −0.1098% |
| 160 | 9 | 78.50 | 283.28 | 比較なし |
| 192 | 12 | 84.88 | 350.73 | 比較なし |
| 256 | 16 | 123.40 | 490.60 | −1.1207%（17→16 workers） |
| 359 | 24 | 177.49 | 717.20 | −0.9200% |
| 512 | 34 | 257.52 | 1,064.46 | −0.9447%（35→34 workers） |

84・116は従来の一括経路を維持しているが、命令数は完全に同一ではない。各1回の測定で、時間はホスト負荷を統制した反復benchmarkではないため、時間差の原因を断定しない。117・132の比較元は旧256版、その他の比較元は直前の512版。比較元の34投影kernel本体が今回と一致することも確認した。比較記録は `artifacts/adaptive-512-v1/comparison.json`。

70／84／100／116／117／132／256／512の全32層の中間値・状態、最終hidden、判定・確率・logitsが独立query参照と一致した。70・100・132の新規query数は66／66／909。残る5入力は同一入力の保存済み独立参照との新しい比較であり、今回はqueryを再計算していない。359は追加で直前の512版との全32層hash・状態・最終hiddenの一致を確認したが、独立query参照との比較とは扱わない。

512-token jobの途中trapを注入し、全額返金Doneとcaller実残高を照合した。失敗IDの再送は追加受領なし。fault解除後の新しい84-token jobも判定・確率・logitsが一致した。成功IDの再送時の追加受領なし、外部からのworker呼び出し拒否も通過した。

最大worker命令数は全ケースで33.53B、512では33.42B。いずれのupdateも実際に成功した。512の最大heapは4,203,151,360 bytesで、4 GiBまで87.5625 MiB残った。命令counterは返信末尾の小さい処理を含まないため、worker全費用の厳密な総計とは扱わない。

`proof-candidate/report.json` のcomplete、baseline_restored、snapshot_deleted、test_caller_stoppedがすべてtrue。module・cache・packの復元確認後に試験snapshotを削除し、試験callerを停止した。`audit.json` で77件の保存返信を再decodeし、全比較配列と判定を再検証した。nativeは標準canister 22 tests、116-token互換canister 19 tests、key-major状態を含むruntime 70 testsが通過した。

採用した診断APIなしの通常Wasmは `artifacts/adaptive-512-v1/normal/full.wasm`、SHA256 `d30407f84815a6d18f56e02d6d74e6e4a3577fb73aefcc5844cd9dabec19c9f4`。34投影kernelの本体一致とWasm validation、有料APIの存在とfault／probe／upgrade-probe endpointの不在を確認した。通常module自体は未インストールで、実測は同じ演算方針の診断moduleによる。公開canisterも未変更。
