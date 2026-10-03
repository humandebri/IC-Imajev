# pair補正をWasmの共有operandで再測定（不採用）

2026-10-03。以前のRust pair-factorはdot数を半減しても命令が増えた。今回は固定重みの展開とscale/correctionを最初のtokenで一度だけ読み、後続tokenがWasm localを使う実装を作った。各tokenの入力も最初の出力行だけで読み、32出力行で共有する。既存の64-query推論候補と主canisterは変更しない。

## 演算と保存

隣接する整数値について `ae*be + ao*bo = (ae+bo)*(ao+be) - ae*ao - be*bo` を使う。整数の全dot値を復元してから、元の `(dot as F32 * activation_scale) * weight_scale + previous_sum` をK256順に適用する。追加の量子化・丸め・scale変更はない。

固定重みは256 byteを偶数/奇数列へ分け、4 byteの整数補正を付けて260 byteにする。入力は元のINT8量子化値をI16で保持し、K256ごとに516 byteへ並べ替えて入力補正を一度作る。この入力準備費用を各queryで計上する。

`scripts/generate_wat_factor.py` が標準SIMDの9×I32 ABI bodyを生成し、`scripts/wat_factor_bench` が固定補正を準備する。raw stubはfail-closed。`build_wat_factor.py` は明示的に固定した過去runtime rlibと新しいwrapperをリンクし、bodyだけを置換・parser validationする。現在のruntimeを再ビルドした診断とは扱わない。compiler/source/dependency/body/module hashを記録する。

## 実測

専用診断canister `5tkpr-7d777-77775-aaaeq-cai`、module `25b41b28c72425d621fe03c912eb69a0c3d99fd2f3db675d90359206f99816b5`。固定第3層Q重み8192×2560、65準備update、通常query22回。scalar native、旧prepared pair、新しいpair-factorの出力digestは全11条件で一致。符号端値・zero/tiny・1/7/8/32/64/88-tokenも含む。native unitは9 token形状と実際の−128重みを含む全bit比較で通過した。

|入力|tokens|出力行|旧pairの合計命令|候補の合計命令|変化|
|---|---:|---:|---:|---:|---:|
|prefix|45|8192|543,136,837|571,349,649|+5.1944%|
|主suffix|87|8192|1,044,316,378|1,090,457,569|+4.4183%|
|情報不足|80|8192|962,099,226|1,003,787,921|+4.3331%|
|最大変更|89|8192|1,067,862,058|1,115,913,921|+4.4998%|
|coldのQ operand|132|4096|798,455,526|838,578,061|+5.0250%|

主suffixの投影部分は1,034,754,331→1,070,922,996命令、候補の入力準備は9,972,740命令。投影自体も増えており、入力並べ替えだけを取り除けば改善するという結果ではない。7-token syntheticだけは約10.12%減るが、実質問5条件はすべて悪化した。

積の数は半減するが、I16交差項の加算、4つのdotの水平和、I32補正と結果の組み替えが必要。標準SIMDのdotがまとめて計算する費用に対し、乗算個数だけで改善を判断できない。候補を全モデルへ接続しない。判断精度や全モデルのquery数を改善した証拠として使わない。

診断の係数payloadは21,299,200 bytes（原重み20,971,520 bytes＋補正327,680 bytes）。本体で固定INT8重みを置き換えるなら、対象全tensorの補正がさらに必要。4 GiBでの全モデル配置は未実装・未証明。通信量は入力とdigest replyの診断だけで、全推論の通信ではない。

命令は入力復元・量子化・入力準備・投影のhandler区間。digest生成/Candid処理・LoRA・readoutを含まない。時間は単回・query cache未制御であり、命令が増えた候補の一部の秒数が短くても改善としない。

証拠：`artifacts/prefix_codec/factor-check/report.json` と `validated-source.zip`。native test `factor-native-tests.log`、build `factor-build/report.json`、置換検査 `factor-build/patch.json`。生成物はignore。

再現：`.venv/bin/python scripts/build_wat_factor.py` 後、独立の診断用canisterへowner/8192/2560をinit引数としてinstallする。`.venv/bin/python scripts/check_wat_factor.py --canister <診断用ID> --directory <未使用directory>` で測定する。既存の主・prefix codec・全モデル候補へinstallしない。

全モデルの検証済み候補は引き続き主suffix64 query、262,163,826,219命令。50/32 queryは未達。
