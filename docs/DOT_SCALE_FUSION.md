2026-10-03：本書はdot/scale融合採用時の記録。現行は [全Attention融合](ATTENTION_FULL.md) を追加し、主122・初回245 queryへ削減した。

# 整数dotとscaleをつなぎ、一時配列の往復を除く

2026-10-03。現行のblock256 INT8 dotは、各blockの結果を整数の一時配列へ書き、次の関数が読み直してF32へ変換、activation scale・weight scaleを掛けて加算する。新候補は同じ整数reductionの直後に元の `(dot as f32 * sx) * sw` と `sum + value` を実行し、一時配列のゼロ初期化、store/load、関数境界を除く。weight scaleはoutput tileごとに一度ロードしtoken間で共有する。整数dot・block順・F32演算順・BF16境界・量子化・LoRA/readoutは変えない。

`crates/imajev-runtime/src/int8_dot_scale.rs` は現行のimmediate tileから生成したSIMD実装。Laya由来の整数tileについては既存の `docs/licenses/Laya-MIT.txt` を参照する。Wasmの9–64 token tileで使用し、8 token以下は従来のdelayed reduction、64超は従来経路を残す。ポインタの範囲は既存のopaque QuantizedRowsとprojectの形状検証・paddingで保証する。出力の有限値検証は残す。

runtimeの `experimental-dot-scale` は独立診断API、`experimental-adopt-dot-scale` は通常projectへの接続を有効にする。canisterでは `experimental-dot-scale` を追加する。既存のcolumn16-token48とMLP pipelineなどの全flagも維持する。

## nativeとfeatureの検証

全層構成58 runtime tests・2 INT8 integration・2 F32 matrix integration・3 compile-fail doctestが通過した。追加のscalar境界検証2 testsは1–132 token、256/512列、signed INT8極値・ゼロ・端数tokenで候補APIを含めてbit比較し、不正な寸法・scaleも拒否した。nativeでは同じscalar計算を使用するので、Wasm SIMDの証明は次の診断で行う。

前回のMLP型追加で `DecodedQueryInput` のfeature gateが隣のre-exportへ付いていたため、projection-reuseなしのbuildが失敗することを今回確認し修正した。featureなしnative cargo checkと、dot-scale/column16-token48のみのWasm cargo checkが通過。記録 `artifacts/dot-scale/{tests,scalar-tests,no-feature-check,check}.log`。

## 専用Wasm診断

新規ローカルcanister `5gn64-6l777-77775-aaaha-cai`、module `2092682e16884d25e7a77fb52ddf924e11597a82b0e24c0b5050619f0d3808c9`。同じ固定第3層Q投影のINT8重み、同じ入力で従来C16/token48と候補を測った。独立nativeの整数dotと元のblock256 F32 scaleを基準とし、全21条件で全出力digestが一致。５条件の実入力と16種類のtoken境界/signed/zero/極小値入力。native側も全出力を比較した。132 tokenのoutputは900K上限に従い4096行、その他は8192行、入力2560列。

65固定重み準備update、42通常query、前後module status read2 update。準備・digest・Candid encodingはkernel counterから除外。総counterは入力復元・量子化を含む。このcanisterはMLP/LoRA/readoutや全層精度・query数を測っていない。

| 実入力 | 総命令の変化 | project命令（従来→候補） |
| --- | ---: | ---: |
| prefix | -2.8904% | 624,745,533→606,543,930 |
| 617 | -2.5627% | 1,178,594,045→1,148,144,378 |
| insufficient | -2.8943% | 1,041,778,813→1,011,371,642 |
| maximum | -2.9070% | 1,242,388,221→1,205,985,018 |
| normal | -2.6648% | 898,288,509→873,962,874 |

記録 `artifacts/dot-scale/check/report.json`。11ソースhash/archiveは `source-hashes.json` / `source.zip`、native helperの実体hashも診断reportへ記録。通常SIMD関数属性を使用し、global target-featureは追加していない。全層への削減率は外挿しない。全層比較後、下記の採用版へ接続した。50/32 queryは未達。

## 全層比較と採用

| 条件 | query | handler命令 | 削減命令 | 削減率 | Candid bytes | 実測秒 | 最大query命令 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix | 139 | 158,249,118,328 | 3,097,685,192 | 1.9199% | 198,872,822 | 22.6927 | 1,987,899,352 |
| 617 | 138 | 294,078,969,609 | 5,031,326,024 | 1.6821% | 278,946,124 | 36.3652 | 3,751,031,346 |
| insufficient | 138 | 262,272,332,117 | 5,025,300,840 | 1.8800% | 260,895,070 | 33.2879 | 3,335,921,480 |
| maximum | 138 | 306,428,096,834 | 6,018,567,368 | 1.9263% | 284,124,623 | 38.0561 | 3,914,476,773 |
| normal | 294 | 447,482,600,283 | 9,862,947,056 | 2.1566% | 580,794,410 | 58.6386 | 2,882,723,911 |

全５実行で直前のMLP pipeline版と保持hidden/state・判断・確率がbit一致し、失敗/replay0。prefixは32層の全token hiddenと72状態配列、質問は31層の全token hidden＋最終層の最後のtokenと48保持状態配列を比較した。全32層を実行し、終端の不要状態を省く仕様は従来と同じ。公式BF16との差・最大変更gold=yesをnoとする既存誤判定は残る。型安全性の検証と判断精度は別で、一般的な精度改善は主張しない。

主問題で5,031,326,024命令（1.6821%）、prefixなしで9,862,947,056命令（2.1566%）減。queryは主138、prefix139、初回277、prefixなし294で変更なし。通信も同じ。共通prefix45＋主suffix87、情報不足45＋80、最大変更45＋89、prefixなし132 tokenを維持した。

最大query3,914,476,773命令、最大観測heap4,119,986,176 bytes。query5B/heap4GiB/frame2MB、通常float900K、MLP専用型segmentの制約を維持。counterはCDK Candid encode/decodeを除き、通信はHTTP/CBOR/signatureを含まない。heapは終端page数で瞬間ピークではない。時間はquery cache未制御の単回測定で、速度改善を保証しない。

固定721 tensor・4,065,416,192 bytesとRoPE131,072 bytesの準備は721 update、15,041,482,761命令、244.6841秒、request47,473/reply14,927,506 bytes。準備時pack status1/cache status2 query・認証read2、全層前後cache status2 query・認証read2は推論とは別計上。推論は通常query、中間状態はclient-held。Laya、mainnet、Git remoteを変更していない。生成物はgitignore対象。

採用module `b2ff85804b9d5e1cb919d9ea681dc4be463e994692a708e35f88c1292962dccf`、専用local canister `4caro-hl777-77775-aaaba-cai`、Wasm `artifacts/dot-scale/full.wasm`。認証module/cache bookendsと42実装hashは `artifacts/dot-scale-v1-cache-checks/report.json`。保存した42 hash・現在ソース・検証source archiveの一致を再確認済み。`artifacts/dot-scale/full-source-hashes.json` / `validated-source.zip`、全層記録 `artifacts/dot-scale-v1-*`、比較 `docs/dot-scale-v1-summary.json`。

## 再実行

```sh
cargo build --offline --release --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-projection-reuse,experimental-full-weight-cache,experimental-blake3,experimental-attention-fusion,experimental-delta-projected,experimental-prepared-rope,experimental-column16-token48,experimental-delta-finish,experimental-matrix-tail,experimental-mlp-pipeline,experimental-dot-scale
# 専用local canisterへupgrade後、固定モデルだけを一度準備:
.venv/bin/python scripts/prepare_weight_cache.py --canister <local-id> \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --directory artifacts/new-dot-scale-preparation --include-f32 --require-prepared-rope
.venv/bin/python scripts/validate_prepared_weights.py --canister <local-id> \
  --run-name new-dot-scale --baseline mlp-pipeline-v1 \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --preparation artifacts/new-dot-scale-preparation --reuse-projection-inputs \
  --frame-checksum blake3 --fuse-attention --fuse-delta-projected \
  --fuse-delta-finish --fuse-mlp-pipeline --require-prepared-rope
```

主handlerの合計だけでも5Bで割ると59 query、初回prefix込みは91 queryが必要。CDK・通信・依存関係を含む下限とは区別し、query結合だけで50回を達成したとは扱わない。残る主コストは整数dotと元F32 LoRAの積。Delta finishの主問題最大要求は1,941,119 bytesなので、residual hiddenを単に追加して次のMLPへ接続すると2MBを超える。次の候補も容量と実測命令の両方で検証する。
