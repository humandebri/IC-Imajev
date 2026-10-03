2026-10-03：本書は全Attention採用時の記録。現行は [全MLP経路](MLP_FULL.md) を追加し、prefix90・初回212 queryへ削減した。主問題は122 queryのまま。

# Attention の途中返信をなくし、１通常queryで完了する

2026-10-03。整数dot/scale融合後の保存済み全層counterを調べると、K/V投影・Q/GQA・出力投影の合計は、主問題最大4,653,590,593命令、最大変更4,839,736,376命令になった。以前は上限を超えるため分割していたが、演算削減で１query化を試せる範囲に入った。これは別handlerの合計で、融合後の上限証明は実queryで行う。

`attention_full_integer` は正規化済み入力とprefixのK/Vを受け取り、K/V投影・K正規化/RoPE、Q投影・Q正規化/RoPE・GQA・gate、出力投影を続けて実行する。途中K/Vを返信して再送する処理と、gated Attention出力を返信して出力投影へ再送する処理をなくす。通常層では同じINT8 block256入力量子化をK/V/Q間で一度だけ共有する。終端層は元と同じ最後の１tokenのQを別に量子化する。LoRA A/B・F32 scale2、積和順序、BF16境界、GQA causal範囲とhead順、専用readoutとcalibrationは変更しない。

要求dimsは `[tokens, prefix_tokens, terminal_last_only]`、tensorは当該層の `.self_attn.q_proj.weight`。payloadは入力token-major、prefix keys/valuesはそれぞれhead-major。返信は出力投影のtoken-major値、現在入力のkeys・values各token-major。クライアントがprefix K/Vと結合し、従来と同じ全履歴を保持する。canisterへ質問の履歴を永続化しない。

適用は1–89 tokenの全Attention層、およびlayer31で終端Qのみを計算する1–132 token。132 tokenの通常層は従来の分割を維持する。layerは3/7/11/15/19/23/27/31のみ、終端flagはlayer31のみ有効。固定14 weightのshape/dtype/bytes、要求の形状と有限値、元のmodel/pack/envelopeを検証する。通常codecの900K論理値とframe2MBの上限は変えない。

## 検証

native runtime59 tests、INT8 integration2、F32 matrix integration2、compile-fail doctest3が通過。featureなしnative cargo checkも通過。新クライアント2 testsと従来Attention fusion2 testsは、prefixのhead順・全K/V履歴・終端token位置・要求/返信形状を検証する。`artifacts/attention-full/{tests,no-feature-check,client-tests}.log`。

build時 `experimental-attention-full`、client実行時 `--fuse-attention-full` を追加する。既存の全flagも維持する。ローカル検証module `dd0f9ddefff81a8939c80b2e6e85cf62bf2948f00d62d24892f3b66af458f2c0`、専用canister `4caro-hl777-77775-aaaba-cai`。

実query境界比較は９か所で投影出力・現在K/Vの全bit一致。最大4,767,254,030命令、要求最大676,475/reply最大820,970 bytes。不正要求５通常queryを拒否した。９成功通常query、module認証read2。`artifacts/attention-full/partial/report.json`。この境界検証の値は全層最大の保証とは別に扱う。全５条件の完走比較も完了し、この経路を採用した。現時点で50/32 queryは未達。新しい量子化やhost inferenceによる処理代行は追加しない。Laya、mainnet、Git remoteは変更していない。生成物はgitignore対象。

## 全層比較と採用

| 条件 | query（旧→新） | handler命令 | 削減命令 | Candid bytes | 実測秒 | 最大query命令 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix | 139→123 | 157,952,392,894 | 296,725,434 | 189,636,054 | 21.1546 | 1,987,899,352 |
| 617 | 138→122 | 293,581,828,499 | 497,141,110 | 262,955,428 | 44.5318 | 4,582,737,601 |
| insufficient | 138→122 | 261,828,779,432 | 443,552,685 | 246,187,761 | 47.6343 | 4,067,143,457 |
| maximum | 138→122 | 305,919,347,928 | 508,748,906 | 267,767,242 | 37.5477 | 4,767,254,113 |
| normal | 294→292 | 447,475,675,371 | 6,924,912 | 580,229,781 | 58.3683 | 2,882,723,911 |

主問題は16 query（11.5942%）、15,990,696 Candid bytes（5.7325%）、497,141,110命令（0.1690%）減。prefix123＋主122で初回245 query（旧277）。prefixなし132 tokenは終端Attentionだけ融合するため292 query（旧294）。主問題は共通prefix45＋suffix87、情報不足45＋80、最大変更45＋89、prefixなし132 tokenを維持した。50/32 queryは未達。

全５実行で直前のdot/scale融合版と保持hidden/state・判断・確率がbit一致し、失敗/replay0。prefixは全32層の全token hiddenと72状態配列、質問は31層の全token hidden＋最終層の最後のtokenと48保持状態配列を比較した。全32層を実行し、終端の不要状態省略は従来仕様。公式BF16との差・最大変更gold=yesをnoとする既存誤判定は残る。型安全性と判断精度を区別し、一般的な精度改善は主張しない。

最大query4,767,254,113命令は5B上限内だが、最大変更のAttentionで余裕が約4.7%と小さい。最大観測heap4,119,986,176 bytesは旧版と同じ。query5B/heap4GiB/frame2MB、通常float900K、既存MLP専用型segmentの制約を維持した。counterはCDK Candid encode/decodeを除き、通信はHTTP/CBOR/signatureを含まない。heapはquery終端page数で瞬間ピークではない。時間はquery cache未制御の単回測定で、主問題・情報不足は前回より遅かったため速度改善を保証しない。

固定721 tensor・4,065,416,192 bytesとRoPE131,072 bytesの準備は721 update、15,041,482,761命令、232.6516秒、request47,473/reply14,927,506 bytes。準備時pack status1/cache status2 query・認証read2、全層前後cache status2 query・認証read2は推論とは別計上。質問依存の状態は通常query内で生成して返信し、クライアントが保持する。

採用Wasm `artifacts/attention-full/full.wasm`。認証module/cache bookendsと43実装hashは `artifacts/attention-full-v1-cache-checks/report.json`、保存hashと現在ソース・archiveの一致を再確認済み。`artifacts/attention-full/source-hashes.json` / `validated-source.zip`、全層記録 `artifacts/attention-full-v1-*`、比較 `docs/attention-full-v1-summary.json`。生成物はgitignore対象。

## 再実行

```sh
cargo build --offline --release --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-projection-reuse,experimental-full-weight-cache,experimental-blake3,experimental-attention-fusion,experimental-attention-full,experimental-delta-projected,experimental-prepared-rope,experimental-column16-token48,experimental-delta-finish,experimental-matrix-tail,experimental-mlp-pipeline,experimental-dot-scale
# 専用local canisterへupgrade後、固定モデルだけを一度準備:
.venv/bin/python scripts/prepare_weight_cache.py --canister <local-id> \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --directory artifacts/new-attention-full-preparation --include-f32 --require-prepared-rope
.venv/bin/python scripts/validate_prepared_weights.py --canister <local-id> \
  --run-name new-attention-full --baseline dot-scale-v1 \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --preparation artifacts/new-attention-full-preparation --reuse-projection-inputs \
  --frame-checksum blake3 --fuse-attention --fuse-attention-full \
  --fuse-delta-projected --fuse-delta-finish --fuse-mlp-pipeline --require-prepared-rope
```

主handler合計だけでも5Bで割ると59 query、初回prefix込みは91 queryが必要。CDK・通信・依存関係を含む下限とは区別し、query統合だけで50回に達するとは扱わない。次の対象はMLPとDeltaの境界、および整数dot/F32 LoRAの演算共有。主問題のDelta finish最大要求は約1.94MBで、residualを単に追加するとframe上限を超える。圧縮・演算分担の変更も実測と精度差の確認が必要になる。
