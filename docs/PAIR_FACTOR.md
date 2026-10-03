# 固定INT8重みのpair補正を準備する候補（不採用）

2026-10-02。毎queryで繰り返す重み側の演算を準備updateへ移す方向を検証した。専用診断canister `5tkpr-7d777-77775-aaaeq-cai` に実装し、全層推論とは分離して比較した。

隣接する2要素の整数積和を

`ae*be + ao*bo = (ae+bo)*(ao+be) - ae*ao - be*bo`

で計算する。重みはblock256内で偶数列・奇数列を並べ替え、`Σbe*bo`を準備updateで一度だけ計算・保存する。activation側の`Σae*ao`は量子化後に一度作り、全出力行で共有する。INT8値、元のblock256 scale、F32の掛け算・加算順序は維持する。I16加算とI32積和・補正は符号端値でもoverflowしない。

## 実測

固定packの第3層Q重み8192×2560と、採用済み全層実行から保存した5条件の投影入力を使った。132 tokenだけは既存900K出力値制限に従い4096行にする。準備64 upload update＋seal update、通常query10回、前後module status読み取り2回。nativeの従来INT8積和と候補、両Wasmの出力digestが全5条件で一致した。別途、nativeで1/7/8/32/64/87/132 token、符号端値のbit一致試験が通過した。

| 入力 | token数 | 出力行 | 従来命令 | 候補命令 | 増加率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix | 45 | 8192 | 671,802,633 | 682,678,310 | 1.6189% |
| 主問題suffix | 87 | 8192 | 1,225,437,294 | 1,239,863,755 | 1.1772% |
| 情報不足suffix | 80 | 8192 | 1,108,538,914 | 1,125,097,759 | 1.4937% |
| 最大変更suffix | 89 | 8192 | 1,326,713,758 | 1,347,071,355 | 1.5344% |
| prefixなし | 132 | 4096 | 948,028,026 | 966,869,623 | 1.9875% |

命令はF32入力復元・既存量子化・候補の入力準備・base投影を含むhandler区間。Candid、digest計算、LoRA、全層推論を含まない。nativeは全値のbit比較、Wasmは全F32出力のSHA256 digest比較。全モデルの判断精度やquery数の比較ではない。単回の時間が候補で短い例もあったが、query cache未制御で速度改善とは判定しない。

最初の試作ではscale処理にSIMD関数の呼び出しが残り、約29–30%悪化した。この測定は演算方式を公平に比較できないため、同じSIMD target featureを持つscale処理へ修正して上表を再測定した。修正版でも1.18–1.99%増えた。初回記録は `artifacts/pair-factor/check/report.json`、初回ソースは `artifacts/pair-factor/source.zip` に保存した。

重み側の準備は計1,507,878,425命令、各updateの壁時計時間の合計17.1614秒。診断では原重みと並べ替え重みを両方保持する。本番で置換する場合も補正項55,779,328 bytesが別に必要になる。SIMD dotの個数を半減しても、交差項のI16加算、補正、入力並べ替えが増え、今回の命令数は悪化した。候補は採用版に組み込まない。

記録 `artifacts/pair-factor/corrected-check/report.json`、検証時ソース `artifacts/pair-factor/corrected-source.zip`、Wasm SHA256 `9b0661b07fd49e61cc8ea982712e95d3ceb6d0bb3d5d52f0b752fe7be240b06c`。ソースは `scripts/pair_bench`、実行器は `scripts/check_pair_bench.py`。生成物はgitignore対象。診断canisterのseal後に再準備はできないため、再現時は新規診断canisterを作る。

```sh
CARGO_TARGET_DIR="$PWD/artifacts/pair-factor/target" cargo build --offline --release \
  --manifest-path scripts/pair_bench/Cargo.toml --target wasm32-unknown-unknown --lib
CARGO_TARGET_DIR="$PWD/artifacts/pair-factor/target" cargo build --offline --release \
  --manifest-path scripts/pair_bench/Cargo.toml --bin pair_args
# 新しいlocal診断canisterにowner,8192,2560をinit引数としてinstall後:
.venv/bin/python scripts/check_pair_bench.py --canister <diagnostic-id> --directory artifacts/new-pair-check
```
