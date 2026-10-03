# LoRA入力共有とfull MLPの89-token対応

2026-10-03。両変更を実装し、通常release Wasmで全5条件を検証後、共有ソースへ最小差分を反映した。既存レビュー修正と他セッションのoutput-pairs変更を保持した。採用済み主canisterのupgradeはこの作業の範囲に含めていない。

## 全モデルの結果

固定baseline `output-pairs-v2` に対し、token IDから新たに推論した。全5条件で、保持hidden・状態・最終判断・logits・probabilitiesがbit一致。最終層はterminal readoutのためlast-tokenのみ保持する。失敗0、query replay0。推論は通常queryのみ、モデルpackと固定重み準備だけowner updateを使用する。

| 条件 | query数（前→後） | 全推論命令数（前→後） | Candid通信byte（前→後） |
|---|---:|---:|---:|
| prefix45 | 66→66 | 136,863,422,956→136,829,434,682（0.0248%減） | 71,105,691→71,105,691 |
| 主問題・suffix87 | 67→67 | 265,931,336,448→265,861,866,956（0.0261%減） | 113,709,005→113,709,005 |
| 情報不足・suffix80 | 67→67 | 244,255,114,705→244,164,792,824（0.0370%減） | 106,668,202→106,668,202 |
| 最大長・suffix89 | 98→67 | 275,371,260,925→272,246,195,657（1.1349%減） | 197,070,502→115,720,619 |
| prefixなし132 | 292→292 | 405,740,371,751→405,740,641,584（0.0000665%増） | 580,229,781→580,229,781 |

最大長は98→67 query、通信量41.2796%減、命令数1.1349%減。89-tokenのfull MLPは全31層で5B未満、最大4,699,809,975命令。最大heap 4,123,656,192 byteはbaselineと同じ。主問題は67 queryのまま、命令数0.0261%減。50/32-query目標は未達。prefixなし132-tokenは共有の対象外で、約0.0000665%の命令増が観測された。

最大長でbaselineがgold yesに対しnoを返す点も維持される。判断精度の改善は示していない。並行処理やコンパイルの影響があるため単回wall timeを速度改善の根拠にしない。

## 実装と使用フラグ

`experimental-lora-input-sharing` はMLP gate/upの異なるrank64 LoRA A行列に、一度だけ並べ替えたF32入力を渡す。各積は従来と同じ列順、F32乗算と加算の順序で実行する。n=4..89で共有し、範囲外・rank64以外・token-scale経路は既存処理へ戻す。質問依存の状態をcanisterへ保存しない。実装は `crates/imajev-runtime/src/lora_input_sharing.rs` と `lib.rs`。

`experimental-mlp-full89` はfull MLPの上限を87から89に拡張する。クライアント側は `--fuse-mlp-full --mlp-full-token-cap 89` で使用し、デフォルト87は維持する。旧moduleに89-tokenを誤送信しないため明示指定にした。0-token、90-token、不正epsilon、誤ったnext normは統合moduleの4通常queryでそれぞれ拒否を確認した。

検証に使用したrelease build：

```sh
cargo build --offline --release --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-full-weight-cache,experimental-lora-input-sharing,experimental-mlp-full89,experimental-attention-full,experimental-delta-full-log,experimental-delta-finish,experimental-matrix-tail,experimental-prepared-output-pairs,experimental-prepared-activation,experimental-prepared-rope,experimental-blake3
```

通常profile opt3/thinLTO/cgu1/overflow-checksありで13分10秒。初回のG16まで展開した共有kernelと、cgu8/LTOなしの再試行はコンパイル時メモリ圧迫でSIGKILLとなった。共有範囲を89まで、G8以下に限定し、共有関数の過剰なinlineを防いで通常profileのビルドを通した。実行時メモリとは区別する。

## 部分ごとの独立検証

LoRA A積の同一Wasm内比較：実入力45/80/87/89-token、層0/15/30の12組、24通常query。別々の二つのA行列を使用し、native oracleと両経路の出力digestおよび正規化入力が一致。A積kernel命令数2.63〜4.38%減。全モデルの削減率とは区別する。生結果 `artifacts/vision-followups/lora-pair-check/report.json`、抽出元hashと診断kernelは `diagnostic-extraction.json` と隔離ソース内 `scripts/lora_pair_bench/`。

MLP89の独立診断は、固定baseline moduleの二つの上限判定を計4byteだけ変更したもの。Wasm Validatorと0..512の境界真理値表を確認し、`mlp89-patch.json` に変更位置・hashを保存。LoRA共有は含まない。実MLP入力40通常queryで従来出力とbit一致し、別途全5条件でも一致した。この診断から得た通信削減を、最終的に上記のソースビルドで再確認した。

共有ソースに反映後、Rust runtime unit69件、LoRA共有integration1件、MLP上限integration1件が通過。default featureで87境界を維持するintegration1件も通過。Python MLP/prefix/scheduler/delta-full-log/legacy-fusedの計27件が通過。Wasm向けcanisterの型検査も通過。ログは `root-native-tests.log`、`root-default-bound-tests.log`、`root-python-tests.log`、`root-wasm-check.log`。

## 再現用の固定資料

統合module SHA256 `3459fc6d0910a40137bb7aa051a1114fc1522a4e1e5370ed723582a4447ce1d9`、Wasm `artifacts/vision-followups/integrated-v1.wasm`。271ソースファイルのビルド前・ビルド後・全検証後hashが一致し、`integrated-source-bookends.json` に保存した。固定ソースは `validated-source.zip`、hash一覧 `validated-source-hashes.json`、最小差分 `candidate.patch`。

721固定準備update、cache 4,065,416,192 byte、準備60,994,167,929命令。paired_weight 3,565,158,400 byte、rope 131,072 byte、activation 1,048,576 byte。準備結果 `integrated-v1-preparation/report.json`、全推論前後でcache不変を `artifacts/lora-mlp89-followup-v1-cache-checks/report.json` で確認した。各条件でmoduleを実行前後に確認している。

全5条件の生記録は `artifacts/lora-mlp89-followup-v1-*`、比較と集計は `artifacts/vision-followups/lora-mlp89-followup-v1-*-results.json` と `lora-mlp89-followup-v1-summary.json`。この固定moduleは隔離ソース `artifacts/vision-followups/source` からビルドした。rootへ反映時に保持した他セッションの変更は `root-integration-before.json` / `root-integration-after.json` に記録しているため、現在のroot全体と固定moduleのソースhashを混同しない。

専用全モデルcanister `7oxbz-ml777-77775-aaala-cai`、LoRA診断canister `7jwhn-bt777-77775-aaalq-cai`、network `http://localhost:8001/`。主canister `4caro-hl777-77775-aaaba-cai` は既存moduleのまま。他セッションには全結果・差分・統合完了を共有した。
