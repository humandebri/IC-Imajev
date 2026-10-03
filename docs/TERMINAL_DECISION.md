# 最終層と専用判断readoutの往復を省く

2026-10-03。通常query `terminal_step_decision` を追加し、最終Attention・MLP・最終normと専用F32 readout/calibrationを同じ呼び出しで実行する。最後のnormalized hiddenを別decision queryへ再送しない。中間状態はクライアント保持、query内の状態は保存しない。model/base INT8・F32 adapter/readout・calibrationは既存のまま。

## 型とcheckpoint

戻り値はCandidの `TerminalDecisionMeasurement { measurement: Measurement, decision: ChoiceResult }`。従来の状態frameと型付き判断結果を一緒に返す。KVと最後のhidden/normを保持し、旧stepの返信とのbyte比較も可能にした。既存のstep/decision/decision_fastの契約は変更せず、新methodを追加した。

最終層31の正規tensor・shape・lossless encoding・step overflow・選択肢数/長さ/予約語/重複を検査してから推論する。readoutは既存decision_fastと同じcandidate数・temperature `1.3051569717552742`・calibration version `p3-r2-s000291-authored`。選択肢順序と結果をクライアントのmetric checkpointへ保存し、再開時に照合・復元する。checksumを認証とは扱わない。

`--fuse-terminal-decision` は `--fuse-terminal-attention` と32層を必要とする。prefixの準備では無効化する。判断の命令・通信は統合queryのmeasurementへ含め、decision_queryレポートには `included_in_terminal_query: true` を記録する。追加queryとして二重計上しない。

## 全モデルの実測

実験専用canister `6eydd-o3777-77775-aaama-cai`、module `8f7a2db44dca15917d4f1180e8430478558a76f4c4cacb4c1d907467071ea9d2`。直前の固定scale直接参照候補32a3e1d5…に対する比較。

|条件|query（前→後）|handler合計命令|削減命令|Candid bytes|削減bytes|秒（単回）|最大query命令|
|---|---:|---:|---:|---:|---:|---:|---:|
|prefix|66→66|130,501,685,597|0|71,105,691|0|16.884|2,287,186,824|
|baseline-617|66→65|250,004,003,524|495,771|113,687,128|10,619|23.145|4,309,015,826|
|617|64→63|248,171,056,450|495,771|124,624,018|10,619|25.809|4,309,015,826|
|insufficient|64→63|226,906,464,149|576,870|117,690,970|10,619|19.412|3,938,267,720|
|maximum|64→63|253,758,525,089|495,998|126,604,858|10,619|21.528|4,405,351,911|
|normal|291→290|386,042,514,337|455,502|580,207,902|10,620|46.468|2,389,018,388|

全5条件（旧log主問題対照を含む6実行）の保持hidden/stateと判断4条件のfinal hidden・value/abstain/logits/probabilities/unknownが既存INT8版とbit一致。型付き出力検査通過、graphの失敗/replay0。主は64→63 query、Candid10,619 bytes・495,771命令減。50/32 queryは未達。既存の重大変更の見逃しは残り、判断精度改善の主張ではない。

同じmoduleで旧step＋decision_fastと新queryを比較したcontrolもstate返信byteと判断が一致。旧は731,801,563＋920,577＝732,722,140命令、1,002,747＋10,782＝1,013,529 Candid bytes。新は732,226,369命令、1,002,910 bytes。実際の2呼び出しが1呼び出しになり、全体差と同じ495,771命令・10,619 bytes減を確認した。

controlは5通常query（3成功比較、重複選択肢/誤った層の2拒否）をgraph実行と分けて記録した。実canister checkpointをqueryなしで再開し、状態・判断を復元できること、選択肢順序変更を拒否することも確認した。hostはframing/比較だけを行い、推論しない。

runtime85 unit＋9 integration＋3 compile-fail doc、canister8 unit通過。クライアントcheckpoint3 test、既存scheduler/wire/terminal_readoutも通過。Rust bridge/Candid/sourceとbridge binaryのhashを保存・再照合した。

固定cache721 tensors / 4,065,416,192 bytesを維持。準備721 update / 78,739,929,194命令 / 293.407秒は推論と分離。観測heap終了値最大4,131,258,368 bytesで前候補と同じ（瞬間peakではない）。prefix packet二回目の準備query/命令/通信は0。handler counterはCDK Candid処理を含まず、通信はCandidのみ、時間は単回で速度改善率へ一般化しない。

証拠：`artifacts/prefix_codec/full-terminal-decision-proof/report.json` / `validated-source.zip` / `client-binary-check.json` / `codec-second.json`、`terminal-decision-control/report.json` / `validated-source.zip`、固定準備は `full-terminal-decision-preparation/report.json`。build source/extra-kernel ZIP・patch log・client build logも保存した。生成物はignore。主canister36c04a57…の不変は `main-unchanged-terminal-decision/report.json` のreadで確認。Layaを変更していない。

## 再現

[固定scale候補](FIXED_SCALE_REUSE.md)と同じ全モデルbuild flagsとpair/S1/F32の3 body patchを使う。Rust bridgeを `cargo build --offline --release -p imajev-client` でビルドする。新しいbuild directoryを指定し、実験用canisterだけへ導入・cache準備する。

```sh
.venv/bin/python scripts/check_full_prefix_hybrid.py \
  --canister 6eydd-o3777-77775-aaama-cai \
  --codec-canister 7hukf-2d777-77775-aaakq-cai \
  --wasm artifacts/prefix_codec/full-build-terminal-decision/full.wasm \
  --directory artifacts/prefix_codec/recheck-terminal-decision \
  --terminal-attention --terminal-decision --prefix-start
.venv/bin/python scripts/check_terminal_decision.py \
  --canister 6eydd-o3777-77775-aaama-cai \
  --wasm artifacts/prefix_codec/full-build-terminal-decision/full.wasm \
  --source artifacts/prefix_codec/recheck-terminal-decision/617 \
  --directory artifacts/prefix_codec/recheck-terminal-decision-control
```
