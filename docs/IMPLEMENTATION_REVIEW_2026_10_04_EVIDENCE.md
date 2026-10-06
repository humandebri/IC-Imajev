# 実装レビューと測定証跡の修正

対象は `118b409` までの可逆残差dictionary・SIMD byte interleave・五連結の検証コード。Rustのunsafe SIMDは16 byte入出力境界、escapeを含むblockのscalar復号、無効indexのstore前拒否を確認した。現行全体54 queryの経路にはdictionaryの新形式をまだ接続していない。

## 修正した問題

- 五連結の参照queryを選択する `report.json` のhashが証跡から抜けていた。byte列を一度だけ読み、解析前にhashを固定し、同一実行中に参照が変わったら拒否する。
- 残差診断は参照成功ケースが0件や欠損でも完了reportを作れた。3入力×7層×2幅の42個すべてを要求し、重複も拒否する。
- 五連結のCLIケース・層・headの重複や不正値をquery開始前に拒否する。完了した連結数・失敗試行数・要求した連結数をreportに明示する。スクリプトのexit 0は診断完了であり、五連結の成立を意味しない。

共有処理は `scripts/proof_inputs.py`。参照変更・0件・欠損・重複・不正CLIと保存済み実42ケースを3テストで検証した。修正後のinterleave診断は `artifacts/prefix_update_hoist/interleave-proof-review-v5` に保存する。model演算・配備済みWasmは変更していない。

## 継続する制約

主87 tokenは全体54 query。第五queryとhead26の上限超過は未解決で、50/32 queryは未達。次は量子化block256とF32積和順を維持した128行単位のMLP分割を実装し、半端128行を元BF16で保持する。一般判断精度と採用INT8実装のbit一致は別に評価する。
