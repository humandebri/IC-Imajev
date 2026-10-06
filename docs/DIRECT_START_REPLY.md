# Delta/MLP startのINT8直接返信

共通prefixとtoken IDsまたは直前hidden/normから開始する`delta_mlp_stream_start_ids`/`delta_mlp_stream_prepare`で、MLP streamのtyped payloadをそのまま外側返信へ接続する。従来のq/product INT8→F32展開、外側encoderでの有限値・整数全走査とINT8への書き戻しを省く。

`experimental-direct-mlp-reply`を有効にしたstep/profileだけが使用する。既存の数値APIとfeature無効時は従来経路。内側のMLP requestは検査済み外側requestから生成し、runtime内部のopaque payloadだけを接続する。historyはshape・有限値・BF16精度を確認し、外側のopaque返信も元request全field・step+1に拘束する。tag0、内側tag1、frame/checksum、量子化block256、F32積和順とBF16境界は維持する。

## Native・build

全featureで136 unit（1 ignored）、15 integration、10 doc通過。最小featureのDelta start関連4テストと、直接返信無効構成の関連テストも通過。1/7/80/87/89 token、front4096/4352/9216、frame1/2/3、通常/owner署名checksumで既存encoderと全frame byte一致。不正historyのshape/NaN/非BF16とrequest identity変更を拒否する。

`artifacts/direct_start_reply/full-build-v1`、source bookend一致、4 WAT patchはwasmparser検証済み。raw `d65d9fa33d0424d0a73c5a2c31dd41c5e1a42b0c570dd700047a10b5e6bc8867`、最終module `a3b3d91bcda5baf22ab402e70aee5cd9ca75aab1fccee53e9aa9e8ae48dc9ee2`。専用ローカル実験canisterだけをupgrade。

## 実測

主87 token・front4352のIDs startは**4,892,743,896命令**で成立し、前候補のIC0522を解消した。入口3query＋五連結5queryのcarry/hidden/norm/convが固定参照とbit一致。`entry-start-main-v1`に要求、metric、profile、source/reference bookendを保存。

`entry-start-all-v1`は要求21条件・成功14・失敗7。主87と情報不足80は全7箇所で8query連結が成立。89 tokenは7箇所ともstartでIC0522、既存62query fallbackを維持する。前候補で成立していた同じ主87のstart（layer4/8/12/16/20/24）では各63,781,458〜63,781,755命令減。80 tokenは約57.70〜58.72M命令減。幅の配分自体は前候補と同じで、量子化を変更していない。

## 通し回帰

`full-proof-v1`標準6条件、`rolled-proof-v1`連結3条件が、全保持hidden/state・最終判断/確率で既存INT8参照とbit一致。失敗/replayは0。source/module bookendとZIP保存も確認。さらに前候補の170通常queryと要求/返信340 frameが全byte一致し、`rolled-byte-comparison.json`に保存。

| 連結graph | query | 合計命令 | Candid bytes | 最大query命令 | 単回時間s |
|---|---:|---:|---:|---:|---:|

| 主87 token | 54 | 240,020,297,740 | 133,896,992 | 4,900,654,790 | 36.826 |
| 情報不足80 token | 54 | 219,566,755,785 | 125,114,808 | 4,492,999,701 | 33.641 |
| 最大変更89 token・fallback | 62 | 240,031,004,749 | 125,231,193 | 4,816,567,380 | 39.179 |

前候補比で主493,452,112命令減、情報不足454,459,620減、最大変更59,082減。最大変更はこの新経路を使わず、branch配置変更による微小差として記録する。通信/query数は同じ、主の観測heap終了最大4,144,037,888 bytesも同じ。単回時間から速度改善を断定しない。

主入力は共通prefix45＋suffix87＝132 token・rotations=1。新8query連結は全体へ未接続で、主は54query、50/32未達。次はMLP完了＋次Attentionを同じ通常queryへ接続し、連結graphで1往復ずつ減らす。元BF16モデルに対する判断精度とは別の回帰で、最大変更の見逃しは解消していない。
