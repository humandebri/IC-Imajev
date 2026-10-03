# 元容量のINT8重みとquery内の変換再利用

2026-10-03。固定重みの配置はowner updateで一度だけ準備する。入力依存の値は通常query内だけで共有し、中間状態はクライアントが持つ。INT8 base、F32 adapter/readout、calibration、block256 scaleとF32加算順を維持する。

## 繰り返す処理の除去

以前の整数Strassen S1は、固定7係数をI16で持つため元INT8の3.5倍容量だった。今回は元INT8を4象限に並べ替えるだけにし、重みの追加容量をなくした。WATは最初のtoken pairのdotで係数を作り、残りのtoken pairで同じSIMD localを再利用する。各積の水平和を先に取らず、出力を整数で再構成してから水平和を取る。初回loadをlocal.teeでそのままdotへ渡し、直後の読み直しを省く。

`QuantizedRows`はquery内の7つの入力変換を`OnceCell`で共有する。gate/up等が同じ量子化入力を使う場合、変換bufferを再度作らない。固定cache以外をcanisterへ保存しない。小さな32行gateは既存のpair kernelを維持する。

新featureは`experimental-strassen-raw`。`output_pairs.rs`のprepared layout、`strassen_raw.rs`、`int8_kernel.rs`に接続した。元の重みbyte範囲・scale・2行ずれの部分viewも復元できる。未patchのWasm stubはNaNを書き、有限値検査で失敗する。

## 単体実測

専用診断canisterの実layer3 Q、通常query22回、owner準備65 update。全5実入力とtoken境界1/7/8/32/64/88で独立native基準と出力digest一致。量子化・query入力準備込み、digest/Candidは命令counter外。

|入力|既存pair命令|元容量S1命令|差|
|---|---:|---:|---:|
|prefix45|543,136,819|535,111,151|−1.4777%|
|main87|1,044,316,360|1,004,100,928|−3.8509%|
|情報不足80|962,099,208|915,992,464|−4.7923%|
|最大変更89|1,067,862,040|1,026,483,004|−3.8749%|
|cold132/Q半分|798,455,508|757,085,108|−5.1813%|

固定係数20,971,520 bytesで元INT8と同容量。scaleは元F32のまま。診断は旧compiled runtimeと基準pair kernelを固定して比較したもので、全モデルの直接入力kernelとの差は別に検証する。判断精度の改善や50 query達成を示す数値ではない。1-token境界は43,733,786対26,283,462命令（66.39%増）で、短い入力へ一律に効果を一般化しない。全モデルの最終token処理も計測対象に含める。

証拠：`artifacts/wat_s1_raw/check/report.json`、`validated-source.zip`。生成物はignore。

## 全モデル候補の再現

```sh
.venv/bin/python scripts/generate_wat_s1_raw.py --directory artifacts/wat_s1_raw/build
cargo build --offline --release --manifest-path scripts/wasm_patch/Cargo.toml \
  --target-dir artifacts/wasm-audit-target

.venv/bin/python scripts/build_full_prefix_candidate.py \
  --directory artifacts/prefix_codec/full-build-strassen-raw \
  --target-directory artifacts/prefix_codec/full-target \
  --direct-input --terminal-attention --prefix-start \
  --delta-state-layout --strassen-raw --instruction-profile

artifacts/wasm-audit-target/release/imajev-wasm-patch \
  artifacts/prefix_codec/full-build-strassen-raw/raw.wasm \
  artifacts/prefix_codec/direct-input-build/direct-input.wat \
  artifacts/prefix_codec/full-build-strassen-raw/pair-patched.wasm

artifacts/wasm-audit-target/release/imajev-wasm-patch \
  artifacts/prefix_codec/full-build-strassen-raw/pair-patched.wasm \
  artifacts/wat_s1_raw/build/kernel.wat \
  artifacts/prefix_codec/full-build-strassen-raw/full.wasm \
  __imajev_s1_raw_accumulate
```

先に`generate_wat_s1_raw.py`でraw WATを生成し、patcherをビルドする。Wasmの2つのnamed bodyを別々にpatch・validateする。他のbody/sectionは保持する。buildの`source.zip`と追加`raw-kernel-source.zip`に一致するソースを使い、変更時は新しい出力ディレクトリにする。

## 全モデル実測

専用`6eydd-o3777-77775-aaama-cai`、module `690e418abac9ed84d0196b8f84ee583cbe1d485efa98d815e247df23cb23c6d9`。全5条件（旧log主問題の追加対照を含め6実行）の32層hidden/state、判断4条件の最終hidden・logit・確率・value・abstainが既存INT8版とbit一致。型付き出力も通過。失敗/replay0。最大変更を`no`と判断する既存の見逃しは残り、判断精度の改善を主張しない。

|条件|query|handler合計命令|直前state-layout候補比|Candid bytes|秒（単回）|最大query命令|
|---|---:|---:|---:|---:|---:|---:|
|prefix45|66|135,388,738,030|−0.8041%|71,105,691|15.110|2,376,678,216|
|主suffix87・旧log対照|66|257,470,117,642|−1.9274%|113,697,747|35.204|4,448,502,361|
|主suffix87・packet再利用|64|255,629,912,125|−1.9454%|124,634,637|22.752|4,448,502,361|
|情報不足suffix80|64|233,030,280,431|−2.5318%|117,701,589|19.020|4,051,774,791|
|最大変更suffix89|64|261,806,240,046|−1.9828%|126,615,477|29.877|4,555,779,191|
|prefixなし132|291|392,164,871,838|−2.5690%|580,218,522|45.352|2,389,765,422|

主suffixで5,071,710,217命令減。query数と通信は直前候補と同じで、50/32は未達。counterはhandler内、通信はCandid送受信でHTTP/TLSを含まない。時間は単回で速度改善率に一般化しない。

固定cacheは721 tensors / 4,065,416,192 bytes、pair配置部分3,569,090,560 bytes、RoPE131,072 bytesとactivation1,048,576 bytesも準備済み。追加係数3.5倍を保持せず全モデルが収まった。準備は721 owner updates、71,182,663,164命令、266.922秒で、推論合計から分けている。観測heap終了値は最大4,128,768,000 bytesで、ピークheapではない。

prefix packetの2回目の準備はcache hit、query/命令/Candid bytesとも0。主canisterのmodule hash `36c04a57…`が変わっていないことも読み取りで確認した。Layaは変更していない。

証拠は `artifacts/prefix_codec/full-strassen-raw-proof/report.json`、`validated-source.zip`、`codec-second.json`、`full-strassen-raw-preparation/report.json`、`main-unchanged-strassen-raw/report.json`。候補featureのruntime unit82件、integration9件、compile-fail doc3件、canister unit6件を通過。新配置単独のruntime unit48件も通過。

## 残る反復処理

同moduleで87-token実MLPのlayer0/1/3を通常queryでprofileし、返信byteが保存済み出力と一致することを確認した。layer0は4,446,175,068命令、そのうちINT8 base投影3回が3,346,252,756（75.26%）、F32 LoRA B積3回が407,553,860（9.17%）。wire decode/encodeは28,266,154 / 28,295,572。inclusive spanやその内部checksumを重ねて合算しない。残りを未計測のLoRA Aだけのコストとは断定しない。

次の候補はF32 inputの並べ替え・weight load・SIMD accumulatorの配列store/loadを繰り返す部分。固定F32 weight自体は既に一度decodeして借用しているため、再decode削減として扱わない。追加変更は元のF32 column加算順を保ち、単体と全モデルの一致を再検証する。profile証拠は `full-strassen-raw-profile-v2/report.json`、`validated-source.zip`。
