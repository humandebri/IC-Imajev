# LoRA Aも固定配置から直接計算する

2026-10-03。`experimental-f32-output-generic`で、以前のLoRA B再利用をLoRA Aへ拡張した。INT8 base、元のF32 adapter/readout/calibration、BF16境界を維持する。固定配置だけをowner updateで準備し、通常query内の入力転置・weight loadの繰り返しを減らす。質問状態はクライアントが保持する。

## 実装

`PreparedF32`は元容量のcolumn/output32配置とprivateな列数を保持する。既存64列constructorも維持する。LoRA Aの2560/9216列はK64ごとに計算し、F32部分和を引き継ぐ。mul/addとcolumn順は元のままで、追加量子化やFMAを使わない。元のF32/byte範囲の読み取りも保つ。

`matrix_loaded`で型付き配置へdispatchする。gate/upのA積は元inputから直接計算し、共有input転置bufferを作らない。従来readerでは必要時にのみ旧bufferを作る。down-A、Delta/AttentionのA積にも接続した。列数128/2560/9216、tileをまたぐ部分view、符号付きゼロ等をnativeで確認し、投影が元配置への逆変換を呼ばないことも検証した。

runtime unit84、integration9、compile-fail doc3、canister unit7件を通過。Wasmのpair/S1/F32の3 bodyをpatch・validateした。F32 bodyは単体gate-Aでbit一致と改善を確認したK64 kernelである。

## 全モデル実測

専用canister `6eydd-o3777-77775-aaama-cai`、module `8fd0c75cbcdc3504bd88ab478b8181939c64110d4739d69961c25e88087164c5`。直前のLoRA Bだけを直接計算する候補と比較。

|条件|query|handler合計命令|直前候補比|Candid bytes|秒（単回）|最大query命令|
|---|---:|---:|---:|---:|---:|---:|
|prefix45|66|130,717,136,249|−1.4465%|71,105,691|16.978|2,291,030,094|
|主87・旧log対照|66|250,346,978,422|−0.9688%|113,697,747|20.400|4,315,627,906|
|主87・packet再利用|64|248,528,396,096|−0.9834%|124,634,637|20.139|4,315,627,906|
|情報不足80|64|227,206,371,983|−0.8210%|117,701,589|19.010|3,944,585,240|
|最大変更89|64|254,139,632,144|−1.0322%|126,615,477|20.648|4,412,466,557|
|prefixなし132|291|386,470,358,818|−0.3907%|580,218,522|56.514|2,393,975,267|

主で2,468,264,695命令減。5条件の全32層の保持対象hidden/state、判断4条件の最終hidden、value/abstain/logit/確率/unknownが既存INT8版とbit一致。型付き出力検査も通過し、失敗/replay0。既存の最大変更の見逃しは残り、判断精度の改善を意味しない。

64 queryと通信は同じで50/32未達。248.53B命令を5Bで割った算術値から、実際に50 queryへ配置できたとは判断しない。Candid処理、境界の追加処理、2 MBの入出力上限、client-held carryの往復が残る。最大変更条件の合計命令も254.14Bあるため、全条件で50以下の証明ではない。

固定cache721 tensors / 4,065,416,192 bytes、INT8 pair3,569,090,560 bytes、RoPE131,072 bytes、activation1,048,576 bytesを維持。準備721 updates / 79,305,692,659命令 / 259.871秒は推論から分離した。観測heap終了値最大4,128,833,536 bytesはピークではない。prefix packetの2回目の準備はquery/命令/Candid bytesが0。主canister `36c04a57…`の変更がないことをreadのみで確認した。Layaは変更しない。

命令はhandler counter、通信はCandid送受信のみ。時間は単回で速度改善率に一般化しない。

証拠は `artifacts/prefix_codec/full-f32-generic-proof/report.json`、`validated-source.zip`、`codec-second.json`、`full-f32-generic-preparation/report.json`、`main-unchanged-f32-generic/report.json`。

## 削減後の内訳

実主87-tokenのMLP layer0/1/3を同moduleで3通常query profileし、保存済み返信byteと一致。layer0の全体4,313,342,761命令の内訳：

|span|命令|全体比|
|---|---:|---:|
|INT8 base投影3回（inclusive）|3,346,084,704|77.58%|
|gate/upのA積2回|71,233,909|1.65%|
|down-A積1回|127,573,770|2.96%|
|LoRA B積3回|320,188,137|7.42%|
|wire decode|28,246,038|0.65%|
|wire encode|27,403,037|0.64%|

inclusiveや内側のchecksum spanは重ねて加算しない。以前測れていなかったA積もここでは別に記録した。B積は直前319,437,514から約0.235%増え、K64の部分和読み取り等を含む新kernelの結果として残す。全モデル合計は改善したが、すべての部分演算が減ったとは扱わない。

証拠：`full-f32-generic-profile/report.json` / `validated-source.zip`。

## 再現

```sh
.venv/bin/python scripts/generate_f32_block.py
.venv/bin/python scripts/build_full_prefix_candidate.py \
  --directory artifacts/prefix_codec/full-build-f32-generic \
  --target-directory artifacts/prefix_codec/full-target \
  --direct-input --terminal-attention --prefix-start \
  --delta-state-layout --strassen-raw --f32-output-generic --instruction-profile
```

pairとS1 bodyを従来どおり順にpatchした後、F32 bodyに`artifacts/f32_block/build/kernel.wat`を使い、export `__imajev_f32_output64`をpatchする。build/extra-kernel source ZIPとpatch hashで一致を確認する。変更時は新しいディレクトリを使う。生成物はignore。
