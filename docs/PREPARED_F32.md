# 元F32重みの復元・検査を準備へ移す

2026-10-02。固定INT8重みの借用に続き、元のF32 adapter・norm・専用readoutも準備updateで一度復元して共有する実装を追加した。F32重みのstable read、byteからF32への復元、配列確保とコピー、同じ固定値の有限性の全走査をqueryごとに行わずに済む。

`PreparedF32`はconstructorでlittle-endian F32を復元し、全値の有限性を検証する。値はprivateな`Rc<[f32]>`で保持し、変更できないsliceとしてだけ公開する。非ゼロ行のsubrangeも元配列の参照を共有し、コピーしない。runtimeのprepared readerはこの型を受け取り、長さ・manifest・row範囲・work上限は引き続きqueryごとに確認する。既存のVec/AsRef reader APIはbyte-only wrapper経由で維持した。入力や新しい演算出力の有限性検査は省いていない。

重みの値、INT8量子化、整数dot、元F32 LoRA積和順、BF16境界、専用readout、calibration.jsonは変更していない。中間状態をserver cacheへ置かず、推論は通常queryで進める。質問を必要としない固定重みだけを準備する。

full cache budgetは4,080,000,000 bytesへ増やした。INT8 dense272個＋F32 tensor449個の実データ合計は4,065,416,192 bytes。embeddingと342,528 bytesのBF16重みは元のstable packから読む。default buildのcache budgetは256 MiBのまま。selected local canisterのheap limitは4 GiB、queryの5B上限は前版のまま。cacheはupgradeで空になるため、そのモデルの準備を再実行する。各質問の前に再準備する必要はない。

Rust43 tests、canister cache3 tests、compile-fail doctestが通過した。非ゼロ行のF32射影を通常readerと比較し全bit一致、prepared経路でbyte復元を呼ばないこと、同じ配列のpointerを共有すること、viewの寿命、negative zero・subnormal・有限極値・NaN/Infinity拒否、unaligned範囲・長さ不一致を確認した。Wasmの全層検証結果も以下に記録する。

## 再現

当プロジェクトの8001の専用local canister、seal済みfull packを使用する。初回pack uploadの費用は別。元pack・model lockは[BORROWED_WEIGHTS.md](BORROWED_WEIGHTS.md)と同じ。生成物はgitignore対象。Laya・mainnet・remote Gitは変更しない。

```sh
cargo test --offline --locked -p imajev-runtime -p imajev-inference --features experimental-full-weight-cache
cargo build --offline --release --locked --target wasm32-unknown-unknown -p imajev-inference --features experimental-full-weight-cache
cargo build --offline --release --locked -p imajev-client
icp canister settings update 4caro-hl777-77775-aaaba-cai --identity imajev-local --wasm-memory-limit 4gib --force
icp canister install 4caro-hl777-77775-aaaba-cai --identity imajev-local --mode upgrade --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm --args '()'
.venv/bin/python scripts/prepare_weight_cache.py --canister 4caro-hl777-77775-aaaba-cai --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm --include-f32 --directory artifacts/prepared-f32-fresh
.venv/bin/python scripts/validate_prepared_weights.py --canister 4caro-hl777-77775-aaaba-cai --run-name prepared-f32-fresh --baseline borrowed-weights-v1 --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm --preparation artifacts/prepared-f32-fresh
```

準備scriptは`--include-f32`で固定F32重みを追加し、宣言budgetを4.08 GBにする。tensor名、dtype別の実容量、元pack seal/model/hash、認証済みmodule hashを照合する。検証wrapperは全5条件の出力、前後のcache容量と名前、module hash、実装source hashを確認する。準備updateと診断queryを質問推論の回数に混ぜない。

## 全層検証・採用結果

新prefixから全5実行を通し、保持hidden/state、最終hidden、logits、確率、unknown、型付き判断が前版borrowed-weights-v1とbit一致した。失敗/replayは0。前後の認証済みmodule hashと全721個のcache名・実容量が同じで、実装source hashも不変だった。準備済みF32版を専用local canisterへ採用した。

| 実行 | query数 | handler命令数 | 前版からの削減 | Candid通信bytes | 単回時間s | 最大query命令数 | 終端heap最大観測bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix45 | 282 | 193,719,402,813 | 1.898% | 318,918,016 | 28.616 | 2,134,114,813 | 4,106,420,224 |
| 主問題132 / suffix87 | 305 | 354,704,377,597 | 1.038% | 506,366,104 | 46.327 | 4,021,004,507 | 4,114,612,224 |
| 情報不足125 / suffix80 | 305 | 310,181,654,931 | 1.226% | 469,941,494 | 66.125 | 3,489,465,358 | 4,114,022,400 |
| 最大変更134 / suffix89 | 305 | 365,457,218,434 | 1.005% | 516,788,871 | 53.500 | 4,157,302,093 | 4,115,005,440 |
| prefixなし132 | 485 | 523,750,449,327 | 0.917% | 814,232,972 | 77.853 | 3,612,832,522 | 4,114,612,224 |

主問題は3,720,301,153命令減（1.038%）。INT8重み借用前のquantized-invariants-v2からは10,746,383,720命令減（2.941%）。stepのphysical stable readは488,193,372→565,596 bytesになり、元F32 adapter等の毎回の読み出しを省いた。prefixなしは642,723,856→680,976 bytes。これはclient通信とは別で、通信量・query数は同じ。残る読み出しはembeddingの必要行と小さなBF16重み。decision_fastはread bytesを返さないため、この集計には含めない。

初回はprefix282＋主問題305＝587 query、prefixなし132は485 query。50/32 queryは未達。5Bを全てhandlerへ使う仮想的な下限でも主問題71、prefixなし105 query相当の命令数で、実際にはhandler外費用・分割・通信制約もある。

準備は721 update、15,027,997,708 handler命令、Candid要求47,473＋返信14,915,249 bytes、単回234.703秒。これを推論query数の削減には数えない。管理用pack status1 query・cache status2 query・認証module read2回を準備reportに記録する。全層検証wrapperの管理用cache status2 query・認証module read2回も別である。個々のgraph runnerのmodule readは各reportに含む。

handler命令はCandid decode/encodeを除き、通信はCandid要求＋返信でHTTP等を除く。時間は単回測定で、情報不足・最大変更・prefixなしでは前版より長かった。速度向上とは主張しない。heapはquery終端のlinear memory page数で、allocatorの使用量や途中のピークではない。最大観測4,115,005,440 bytesは4 GiB内だが、残り179,961,856 bytesのため、大きな一時重み変換は容量検証が必要。

bit一致は従来INT8版との比較であり、一般精度保証や未量子化公式版とのbit一致ではない。最大変更問題の既存誤判定（gold yes / 出力 no）は残る。型の妥当性は判断の正解とは別に検証した。

採用module `86d93cda02bb5bb6c6e4d17bea38b3315206aace76cdc94d3a4941a6edbec0df`。pack/modelは前版と同じ。生データ`artifacts/prepared-f32-v1-*`、集計`docs/prepared-f32-v1-summary.json`、準備`artifacts/prepared-f32/full-preparation/report.json`、cacheと実装照合`artifacts/prepared-f32-v1-cache-checks/report.json`。build時点の実装hashも最終照合した。

## 残るボトルネック

毎回の大きな重み読み出し・復元・検査は省けたが、整数dotと元F32 LoRAの積和は残る。prefixなしの同一入力の行分割62組・LoRA A再計算93回も残る。準備済み主問題では行分割重複は0組。今回の変更だけでは演算量を50 query相当にできない。

次の候補は、固定INT8係数の変換を準備時へ移すこと。既存Strassen wide版は毎queryでINT8＋carryをI16へ復元していたため、1-tokenでも736.9M命令が必要だった。準備済みI16係数を別の診断cacheで保持し、同じmoduleの通常積和と再比較する余地がある。過去のwide版は87-tokenで通常版より80.5%多く不採用であり、準備を移しただけで改善すると断定しない。フルモデルの係数拡張は4 GiBに収まるとは言えず、部分投影で実測してから判断する。[過去の実測](STRASSEN_WIDE.md)。
