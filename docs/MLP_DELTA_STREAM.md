# 層境界をまたぐ再計算しないquery

2026-10-04。実験用canister `6eydd-o3777-77775-aaama-cai`、module `86b78d009465c448bd4443cc13ee30978ac3f98c363423921a4a6a86060d7f64`。

`mlp_complete_delta_partial` はclient保持のMLP carryを完了し、次層Deltaの入力量子化・元F32 LoRA A・gateを一度だけ作る。選択したhead群を元のprefix logからkey-major状態へ復元して処理し、out projectionの新しい列だけを計算する。hidden、norm整数/scale/A/gate、未丸めのbase/A部分和、処理済みheadのconv履歴を返す。

`delta_partial_mlp_prepare` はその整数/scale/A/gateとbase/A部分和を使って残りheadとout projectionを完了する。Delta側の再量子化・A・gate計算は行わず、次層MLPのproduct整数・scale・元F32 down Aまで作る。MLPのfinishはこのAPIに含まない。全状態はclient保持、推論は通常query、updateは固定重みの準備のみ。部分和をI8/BF16へ丸めず、元の列順序を維持する。

## 実測で成立した境界

主87・情報不足80・最大変更89 token、MLP layer0/1/3→次層Deltaを測定した。同じ一括MLPの保存出力、元のDelta capture/gated/conv履歴、独立した列継続query、一括MLP準備と比較した。39成立条件で、返却hidden・Delta準備・base/A部分和・conv履歴・次層MLP準備とfinish後のhidden/normがすべてビット一致。非返却のgated全体や最終Delta scratchを直接比較したとは扱わない。

|layer0の成立例|token|済みMLP行/次のhead|前半query命令|続くquery命令|2 queryのCandid合計bytes|
|---|---:|---:|---:|---:|---:|
|主|87|5120 / 22|4,840,105,366|4,096,025,051|6,804,190|
|情報不足|80|4352 / 24|4,837,219,106|3,562,670,452|6,277,048|
|最大変更|89|5376 / 22|4,866,548,202|4,185,038,990|6,960,114|

主は3条件、情報不足は24条件。最大変更は最初の24条件がすべて5B命令超過だったため、境界を5376/5632/5888/6144行へ移し、22 headの12条件で成立した。主の早い境界や24 headも21条件で超過した。合計45拒否は保存requestとエラーを残した。token比例の見積もりだけではquery境界を決められない。

89 token・22 headの続くrequest frameは1,995,287〜1,995,288 bytes。2MBに近く、head数や追加状態を無条件に増やせない。通信削減・全体query削減をこの診断だけで主張しない。

## ボトルネックと検証範囲

主87・layer0のprofileでは前半のMLP完了が2,388,587,375命令、Delta 22 headが1,885,062,417、out整数列継続が331,105,133、out Aが39,377,272。続くqueryは次層MLP準備が2,902,821,970、残り10 headが861,186,247、out整数継続が160,158,308、Bが38,890,557。wire decode/encodeは続くqueryで合計75,879,951。spanは包含関係があり、全部を足して総命令数にしない。Deltaの入力準備は続くqueryにない。次層MLPの量子化/Aは新しい計算として1回行う。

runtime109/canister9 unit・integration9・compile-fail doc9、通常featureのruntime48/canister5 unit・doc2が通過。新codecのPython5件、既存codec5件/保存payload315個、tail/replay7件も通過。Rust/Wasmビルドと4 kernel patchの検証済み。

新Wasmで既存全6条件が完走し、返却hidden・保持state・型付き判断・logits・確率は既存INT8版とビット一致した。非返却layer30 hiddenを直接比較したとは扱わない。失敗/replayは0。主は依然62 query、235,201,653,264命令、最大4,716,855,270命令、Candid 123,281,081 bytes、返却時heap最大4,130,144,256 bytes、単回34.851秒。前版比+79,532命令、query数/通信は同じ。時間差を高速化の証拠にしない。既存の最大変更の見逃しも残る。

この経路は全体graphへ未接続。さらに必要なのは最初のDelta/attentionとMLP前半を合わせるquery、続くqueryの空きへdown projectionの一部を移す処理、層/履歴/再開を記録するclient実行順の実装である。全体50 queryは未達。主canisterのmoduleは読取で `36c04a57…` のままと確認し、Layaは変更していない。

証拠は `artifacts/f32_k_continue/pair-follow-check-{617,insufficient,maximum,maximum-cuts}` のreport・source ZIP・保存frame・profile。通常成功診断282、命令上限拒否45、profile診断42を全体推論と別計上する。`scripts/archive_mlp_delta_stream_proof.py` は保存byteと参照の対応、source/build hash、全6条件のreportを検査する。全体は `artifacts/prefix_codec/full-mlp-delta-stream-proof-v3`、buildは `full-build-mlp-delta-stream-v3`。生成物はignore済み。

レビューで証拠検査のsource ZIP上書きを除き、保存ZIP自身のhashを検査するよう修正した。近接requestのRustテストも実ヘッダー長で2,000,000 bytes未満を確認する。変更は検証スクリプトとtest部分で、上記Wasm実測は保存sourceに対応する。修正後、境界Rust3件・Python5件と保存証拠検査が通過した。
