# 固定重みの安全範囲を一度だけ判定する候補

2026-10-03。通常queryの重み符号拡張と入力ロードを減らす診断実装。元INT8重み、block256 activation scale、元F32 adapter/readoutを変えず、追加量子化は行わない。全モデルへは未採用。

owner準備updateで重みをK4／2出力の配置に並べ、各行ペア・K256 blockに1 byteの証明modeを付ける。I8 SIMD拡張積をI16 laneにまとめるW32では、各lane・各32要素窓・両出力行の絶対weight和が258以下であることを検査する。activationは既存の不変型が保証する[-127,127]。したがって全partial sumは絶対値32766以下となる。該当しないblockはW16を使い、最悪の重み-128でも127×(128+128)=32512で安全。queryでは証明modeを読むだけで、重みの範囲判定を繰り返さない。

I16 partial sumはI32へ戻し、各block256の整数dotを正確に復元してから、従来と同じF32 scaleとblock加算順を使う。8-bit符号拡張を個別の準備命令として繰り返す代わりに拡張積の命令にまとめ、入力ベクトルのload数を半分にする。mode分岐や新しい入力配置の準備もqueryの計測対象に含める。事前処理へ移しただけで高速になるとは判断しない。

実重みの静的調査ではQ行ペアblockの82.759%、gate78.852%、down93.800%がW32以上の窓を使える。これは実測命令の削減率ではない。生結果`artifacts/bounded-address/pair-interval-profile.json`。mode表は対象3,565,158,400 weight B全体で6,963,200 Bの見積もりだが、全heap配置と全層推論は未検証。

実装`scripts/bounded_i16_bench`、生成器`scripts/generate_bounded_i16.py`、比較`scripts/check_bounded_i16.py`。nativeでは整数安全範囲、極値・ゼロ・1/7/45/87/132 token・複数blockの出力bit一致、悪いshapeとscaleの拒否を検査。通常queryの実測は別途記録する。中間状態はクライアントが保持し、推論は通常query、固定重み準備のみupdate。

## 専用canisterの実測（不採用）

module `5b3e3f1b2d05085f6b8b1ec6e22e6950f653c7c28241b7a7498c89fc34ba3c24`、22通常query、65固定準備update、準備2,148,004,155命令。native・既存配置・候補の全digestが一致し、moduleとソースhashを実行前後に確認した。主87-token Q投影は1,046,006,663→1,453,336,896命令（38.9415%増）。実5条件は38.6980〜38.9420%増、境界6条件も30.9984〜38.9588%増で、全ケースで悪化したため採用しない。追加の質問状態をcanisterへ保存していない。

生成Wasmは拡張積のソース演算を符号拡張・乗算へ分解しており、専用kernelの静的集計にはI16拡張と乗算が残る。さらにI16からI32へ戻すdotとmode分岐が必要になる。静的opcode数だけで動的差を説明し尽くしたとは扱わないが、事前の安全性判定だけではこの実行費用を補えなかった。単回wall timeの短縮は命令の削減と区別する。

生結果`artifacts/bounded_i16/check/report.json`、静的Wasm解析`opcodes.json`。ビルドは固定したcompiled runtimeを参照する直接rustc方式で、flags・依存hash・source bookendを`cached-build/report.json`に保存した。同一診断moduleの二経路を比較しており、全推論moduleの最適化結果や全モデルの判断精度を検証したものではない。採用済み主canisterは`36c04a57…`、主67 queryのまま。
