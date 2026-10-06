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

## down projectionの前倒し（追加候補）

実験module `ba3bb4423c378c5a393df6590c45ff799f4a40c877115598a7f68d59bcba8cd6`。`delta_partial_mlp_prepare_down` は6番目のdimsで処理済みdown行数を指定する（32の倍数、1〜2559）。product整数/scale/down Aを1回作り、指定行のresidualを完成hiddenへ更新する。続く `mlp_finish_partial_integer` はこの行を飛ばす。carryのサイズ・精度は従来と同じで、全状態はclientが保持する。通常の5 dims経路は維持した。

|layer0の追加経路|token|down済み行|MLP完了/Delta部分query|Delta完了/部分MLP query|残りMLP finish query|3 query Candid合計bytes|
|---|---:|---:|---:|---:|---:|---:|
|主|87|1600|4,840,104,959|4,827,972,602|681,618,594|8,710,851|
|最大変更|89|1280|4,866,547,795|4,758,181,780|818,726,980|8,910,750|

87 tokenのlayer0/1/3では1600行が成立し、89 tokenの1600行は3条件とも後半queryが5B命令超過。89 tokenは1280行へ減らして3層とも成立した。87 layer0のfinish計測を追加した計7成立条件で、carry・conv履歴・次MLPのfinish後hidden/normが参照とビット一致。通常成功診断61、命令上限拒否3、profile診断9を保存し、保存byte/source/build検査も通過した。続くqueryの2MB入力上限拒否を診断で記録するようにした。最大requestは1,995,298 bytes。境界を全tokenへ一律適用しない。

87 finishの追加profile1回（通常呼出し0）も出力一致。総681,654,203命令、演算 `carry_mlp_finish` 473,724,065、wire decode 192,480,395、その中の `carry_inflate` 165,077,908。可逆圧縮を展開する費用が残る。profileのwall timeは全体回帰との同時実行で44.398秒だったため、性能比較に用いない。次は2MB以内の未圧縮carryを直接受け取り、この復元を取り除く候補を検証する。

追加featureのruntime115 unit（保存fixture依存1件ignore）/integration15/doc9、canister8 unit、Python6件が通過した。buildは `full-build-mlp-delta-stream-down-v1`、固定721重み4,065,416,192 bytesの準備は721 update・78,739,929,194命令・266.356秒。推論のquery数・通信に準備を合算しない。診断は `pair-down{1600,1280}-check-*`、保存検査は `pair-down-archive-v2.json`、finish profileは `pair-down-finish-profile-v1`。

この候補も全体graphへ未接続。3 queryの表は境界の一部分で、最初のMLP準備queryも別に必要。全体query削減や総命令削減の達成値とは扱わない。

追加候補の全6条件も回帰比較が通過した（`full-mlp-delta-stream-down-proof-v1`）。返却hidden・保持state・型付き判断・logits・確率は既存INT8とビット一致、非返却layer30 hiddenは直接比較しない。主は62 query・235,201,656,645命令（前module比+3,381）、最大4,716,855,277命令・Candid 123,281,081 bytes・返却時heap最大4,130,144,256 bytes・単回34.979秒。query数/通信は変わらず、時間差を速度向上としない。既存INT8の最大変更の見逃しも残る。主canisterのmodule `36c04a57…` は読取で不変を確認した。

再現時は全体検証完了後に `snapshot_full_query_proof.py --directory <proof-dir>` を実行する。source hashを確認したbyteからZIPを新規作成し、既存ZIPへの上書きは拒否する。続いて `archive_mlp_delta_stream_proof.py --build <build-dir> --directories <診断directory名のcsv> --full-directory <proof-dir> --output <検査結果json>` で保存証拠を検査する。生成物はignore済み。
