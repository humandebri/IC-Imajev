# 固定重みの準備と借用による重複処理削減

2026-10-02。固定INT8 dense重みをモデル準備のupdateで一度読み込み、通常queryでは変更不能なsliceを借用する経路を実装した。これまで投影ごとに行っていたstable read、ゼロ初期化した読み出しバッファの確保、u8からi8への全重みコピーを省く。embedding、元のF32 adapter、readout、calibrationと演算順は変えていない。質問の中間状態はclient保持のまま。

runtimeの`evaluate_with_buffer`は借用bufferを扱う。既存のVec reader用`evaluate_with_reader`も残す。i8 viewは一度取得した同じsliceのpointerと長さから作り、同サイズ・alignment 1・全bit有効・bufferの寿命内という条件を満たす。複数回のAsRefで別sliceを返す安全な実装を用いた境界testを追加した。

## 準備とメモリ

専用local canister `4caro-hl777-77775-aaaba-cai`、URL `http://localhost:8001/`。272個のINT8 dense tensorと元のrow scaleをそのまま保持し、実容量は3,575,191,552 bytes。範囲、重複、tensor長、row scaleの有限性・正値、個別128 MiB・合計3.6 GBの予算を準備時に検証する。owner専用`warm_weights` updateは同じtensorに対して冪等。queryはcache miss時に元のstable packを読む。

full cacheは`experimental-full-weight-cache` featureで有効にする。通常buildのcache上限は256 MiBであり、全モデル準備には足りない。selected canisterのheap limitを3 GiBから4 GiBへ変更した。queryの5B上限は変更していない。[公式resource limits](https://docs.internetcomputer.org/references/resource-limits/)のWasm32 heap上限も4 GiBである。

準備は272 update、9,436,426,445 handler命令、Candid要求17,098 bytes＋返信2,038,854 bytes、単回85.521秒。pack uploadと質問推論は別で、このupdate数を推論query削減とは数えない。upgradeでcacheは空になるため再準備が必要。clearは保持データを解放するが、増えたWasm linear memoryのpage数は縮まらない。

## 再現

既にsealした固定full packを持つ専用local canisterへ、owner identityで実行する。初回installとpack uploadはREADMEの手順を使う。以下のupgradeは当プロジェクトのcanisterだけを指定する。

```sh
cargo test --offline --locked -p imajev-runtime -p imajev-inference --features experimental-full-weight-cache
cargo build --offline --release --locked --target wasm32-unknown-unknown -p imajev-inference --features experimental-full-weight-cache
cargo build --offline --release --locked -p imajev-client
icp canister settings update 4caro-hl777-77775-aaaba-cai --identity imajev-local --wasm-memory-limit 4gib --force
icp canister install 4caro-hl777-77775-aaaba-cai --identity imajev-local --mode upgrade --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm --args '()'
.venv/bin/python scripts/prepare_weight_cache.py --canister 4caro-hl777-77775-aaaba-cai --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm --directory artifacts/weights-preparation-fresh
.venv/bin/python scripts/validate_prepared_weights.py --canister 4caro-hl777-77775-aaaba-cai --run-name weights-validation-fresh --baseline quantized-invariants-v2 --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm --preparation artifacts/weights-preparation-fresh
```

準備scriptはmodel lock、sealed pack、認証済みmodule hash、全tensorの名前・容量を照合する。準備を毎質問行う必要はない。検証は新prefixから全5条件を実行し、前後のcacheとmodule、実装source hashが同じであることを確認する。中間結果の再利用はclient sessionの既存hash照合を使う。生成物はgitignore対象。

## 不採用のtoken tile候補

weight借用に加えて80〜136 tokenを一つのtileで処理する実験も行った。最初のWasm実験では64行を超えるunroll不足を実Wasm/native照合が検出し、主canisterへの採用前に修正した。全行unrollはcompilerがSIGKILLで終了した。共通loopへ変更した版は13実shape・cold/warm/clearの39 queryで整数出力が全bit一致したが、87-token Q投影は1,437,756,547→2,104,001,680命令（46.34%増）だった。45-tokenでは0.67%減だが、主条件では不利なので採用版のfeatureは無効。診断canisterは保存した旧moduleへの復帰を認証済みhashで確認した。

cache-onlyのRust41 tests、cache2 tests、改変禁止compile-fail doctestは通過。partial packの39条件では独立native基準とのbit一致を確認した。全層結果は次節に記録する。Layaのソース・Git・canister、mainnet、remote Gitは変更していない。

## 全層測定と採用結果

新prefixから全5実行を通し、保持hidden/state・最終hidden・logits・校正済み確率・unknown・型付き判断が前版quantized-invariants-v2とbit一致した。失敗/replayは0。準備後のcache全272個と容量が前後で不変で、認証済みmodule hash、source hashを照合した。cache-only版を専用local canisterへ採用した。

| 実行 | query数 | handler命令数 | 前版からの削減 | Candid通信bytes | 単回時間s | 最大query命令数 | 終端heap最大観測bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix45 | 282 | 197,468,189,614 | 3.458% | 318,918,016 | 29.729 | 2,183,511,819 | 3,591,241,728 |
| 主問題132 / suffix87 | 305 | 358,424,678,750 | 1.923% | 506,366,104 | 47.328 | 4,069,577,099 | 3,599,106,048 |
| 情報不足125 / suffix80 | 305 | 314,031,209,026 | 2.193% | 469,941,494 | 43.398 | 3,539,986,197 | 3,597,926,400 |
| 最大変更134 / suffix89 | 305 | 369,167,622,212 | 1.870% | 516,788,871 | 52.294 | 4,205,817,471 | 3,599,499,264 |
| prefixなし132 | 485 | 528,596,585,151 | 1.558% | 814,232,972 | 76.078 | 3,646,244,722 | 3,593,601,024 |

主問題は7,026,082,567命令減（1.923%）。stepのphysical stable readは4,063,384,924→488,193,372 bytesで、固定denseの3,575,191,552 bytesを全て省いた。これは通信量の削減ではなく、client通信量は同じ。prefixなしは8,368,121,581命令減（1.558%）。初回prefix45＋主問題は282＋305＝587 query、prefixなし132は485 query。重み準備updateはさらに別である。50/32 queryは未達。

handler命令数はCandid decode/encodeを除く。通信はCandid要求＋返信でHTTP等を除く。時間は単回測定で一般的な速度向上を保証しない。heap値はquery終端のlinear memory page数で、allocator使用量やquery途中のピークとは異なる。最大観測3,599,499,264 bytesで4 GiB設定内に収まった。今回のbit一致は従来INT8版との差を対象とし、未量子化公式版とのbit一致や一般精度保証ではない。最大変更問題の既存誤判定（gold yes / 出力 no）は残る。

モデルlock `7ef38ecb70f4afa4e92bad2fe9f0448699ee45041688b1110819143f41601330`、pack `364e3aa3f69a61a69e6df62e3e2a71d4f55457f1a8e1c16934aefc6130f04c84`、採用module `bc8e64b4c09ec78f9e6db87dddca1d27a355adb4321032faa8fd9efad1a2d8e8`。生データは`artifacts/borrowed-weights-v1-*`、検証付き集計は`docs/borrowed-weights-v1-summary.json`、準備は`artifacts/borrowed-weights/full-preparation/report.json`。全層validationのwrapperは準備・module・実装hashを`artifacts/borrowed-weights-v1-cache-checks/report.json`に保存する。管理用cache status query2回・認証module read2回は推論query数と分けて記録する。個々のgraph runnerの認証module readもそのreportに含む。

## 残る毎回処理

主問題の残るstable read約488 MBの大半は元F32 adapter。現実装ではこれを読み、F32配列へ復元して有限性を検査してからLoRA積和に渡す。準備時に型付きF32重みへ復元し、同じ不変データを借用することが次の固定処理削減候補である。追加adapter容量487,587,840 bytesと推論scratchを合わせ、4 GiB内で実測する必要がある。この時点では未実装だった。続いてF32重みの準備・共有を実装し、[PREPARED_F32.md](PREPARED_F32.md)に全層検証と採用結果を記録した。

prefixなし132では、同じ入力の行分割が62組あり、LoRA A再計算93回も残る。A結果を通常queryで一度取得してclientが保持し、後続行に返す方法が候補。追加通信・出力float上限・再開時の対応を伴うため、単に再利用できるだけで回数や総費用の改善とは断定しない。準備済み主問題ではこの行分割重複は0組。

主問題の命令の約83%はLoRA付きINT8投影、MLP、QKV/gates投影に残る。固定読み出しを省いても積和は残るため、今回だけで50 queryへ届くとは言えない。5Bのhandler予算だけで計算した理想下限でも主問題72回、prefixなし106回相当であり、実際には通信・分割・handler外費用が加わる。
