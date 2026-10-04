# codecの重複走査と復号bufferのゼロ埋めを除去

2026-10-03。主245,900,230,125→245,111,663,149 handler命令、788,566,976減（0.320686%）。62 query・Candid123,281,081 bytesは同じ。50/32 queryは未達。

## 実装

`bf16-block256-exact-v1` と通常block出力のhybridでは、encode前の全体有限値検査、全体BF16分類、blockごとの再分類をしていた。block256ごとの一回のSIMD走査で有限性とBF16/F32を分類し、bitmapをサイズ計算と出力へ再利用する。全BF16なら元の連続packを維持し、混在blockも同じbitmap/bytesを返す。

復号では、BF16/F32 bytesを出力へ書くロードに有限値検査を重ねる。F32 blockのcanonical判定も同じ走査で行う。Vecをゼロ埋めして直後に全要素を上書きする処理を省き、`MaybeUninit<f32>` のspare capacityへ書く。全spanの書込みと検査が成功してからlenを設定する。非有限値/非canonicalで途中終了したVecはlen0のままで、未初期化値・部分復号値を公開しない。サイズとbitmap/length/padding検査は書込み前に行う。

このcodecが既に有限性を検査した場合だけ、外側のencode/decodeの再走査を省く。他のcodecの外側検査は維持する。SHA256/BLAKE3 frame checksum、モデル/pack/step照合、量子化、元F32 adapter/readout/calibrationと数値演算順序は変更していない。

## 同一moduleの切り分け

専用 `scripts/codec_scan_bench` に旧block sourceを保存し、旧scalar finite＋codecと候補を同じmoduleで比較した。単体counterはcodec処理のみで、初期入力の準備・digest・Candid処理を除外する。入力は全推論保存frameの最大block要求を5条件から選び、元frame checksumもhostで確認した。独立旧nativeとPython digestで、符号化bytes・復号全F32 bitsを照合。

|実入力条件|要素|旧encode|候補encode|旧decode|候補decode|
|---|---:|---:|---:|---:|---:|
|prefix45|230,400|6,738,061|2,558,424|7,020,787|1,433,474|
|主87|537,600|15,711,955|5,959,518|16,386,837|3,349,924|
|情報不足80|501,760|14,666,001|5,563,724|15,298,442|3,130,649|
|最大変更89|455,680|13,326,915|5,060,558|13,890,557|2,840,204|
|cold132|884,736|25,869,060|9,819,359|26,961,051|5,506,090|

実入力でencode約62%、decode約79.5%減。全モデルの同率改善とは扱わない。追加n0/1/7/8/9/255/256/257/511/513、900K BF16、400K混在のfinite bit patternも一致。n1のencodeは1,148→1,261（113命令増）、n0のdecodeは410→418であり、全shapeが改善したとは報告しない。

17 finite入力のencode/decode旧新68通常queryと16不正入力のdecode旧新32通常query、計100回を検証。Inf/NaNの符号とsignaling/quiet、最後のblock、過大count・bitmap padding・非canonical・truncationは両者拒否。一部の複数条件に違反する入力では拒否理由の検出順が変わる。証拠 `artifacts/codec_scan/check/report.json` / `validated-source.zip` / raw Candid files。

実推論canisterのstepでも、checksumを正しく付けた不正payloadをversion1/2で計32通常query送り、全て拒否。新decoderが単体だけで検査し、実入口で検査を飛ばしている状態ではない。既存frame試験もversion1/2と3 codecの6通常queryが旧native replyとbyte一致し、checksum/header破損等8件を拒否。証拠 `codec_scan/frame-rejections` と `codec_scan/wire-check`。nativeにはBF16全65,536 bit patternの有限性分類と独立scalar形式oracle・奇数tail・遅い非有限値検出を追加し、runtime91＋canister9 unit、9 integration、4 compile-fail doc testが通過した。

## 全32層の実測

専用canister `6eydd-o3777-77775-aaama-cai`、module `3dab40a78d4e46d05cdfab7d3673436c8a940bc398a8a1fa52c0ce3c2659a4ba`。直前のquantize-scan版と比較。

|条件|query|handler命令|削減命令|Candid bytes|最大query命令|秒（単回）|
|---|---:|---:|---:|---:|---:|---:|
|prefix45|66|128,659,290,966|656,973,781|71,105,691|2,253,790,923|11.695|
|旧log主87|64|246,614,596,018|1,118,581,193|112,344,191|4,919,059,977|23.426|
|主87|62|245,111,663,149|788,566,976|123,281,081|4,919,059,977|20.245|
|情報不足80|62|224,101,326,932|727,382,944|116,455,579|4,519,088,177|18.606|
|最大変更89|63|250,668,239,072|819,901,798|126,604,858|4,339,526,866|20.573|
|cold132|290|377,486,451,311|4,924,493,174|580,207,902|2,382,068,181|42.361|

全条件の返却hidden・全保持state・最終hidden・typed判断・logits・probabilities/unknownが既存INT8版とbit一致、失敗/replay0。主/情報不足/旧logは前回同様layer30 hidden非返却なので、未返却hiddenを直接比べたとは扱わない。最大変更の既存見逃しも残る。cold最大queryは2,379,472,033→2,382,068,181（2,596,148増）だが合計は減少した。全query個別で削減したという結果ではない。

heap終了値の全条件最大4,131,258,368 bytesは同じ。固定cache721 tensors /4,065,416,192 bytes、準備721 owner updates /78,739,929,194命令 /264.490秒を別計上。prefix packet二度目の準備はcache hitで準備query・命令・通信0。主module `36c04a57…` とclient binary `23fb748c…` は不変。Layaを変更していない。時間は制御していない単回値で、主は前回19.898→20.245秒。命令削減を時間短縮と同一視しない。

証拠 `artifacts/prefix_codec/full-codec-scan-proof/{report.json,before-after.json,validated-source.zip,codec-second.json,main-unchanged.json}`、固定準備 `full-codec-scan-preparation/report.json`。build `full-build-codec-scan-v2` にsource ZIP、kernel hash/ZIP、3body patch logを保存。最初の追加unit testには整数型の曖昧さがあり、usizeを指定して再検証しv2 sourceを保存した。Wasm raw hashは同じ。生成物はignore。

## 改善後のprofileと次の調査

実MLP layer0/1/3の3通常profile queryが以前の返信frameとbyte一致。layer0ではhandler4,242,428,072、model evaluate inclusive4,205,472,937は同じで、旧quantize-scan profileからdecode10,802,062＋encode8,080,568＝18,882,630命令を減らした。base/A/Bの各spanも同じ。

wire decode17,624,053 /encode19,324,094のうち、BLAKE3は14,626,238 /14,626,241。checksum合計29,252,479はwire合計36,948,147の約79.2%で、次の重複処理候補。INT8 baseは3,319,242,044で全handlerの約78.2%を占め、全体の主要なボトルネックは引き続きbase射影。証拠 `full-codec-scan-profile/report.json` / `validated-source.zip`。

現時点ではchecksum計算を省いていない。次は署名付きIC request/query replyの検証を明示したhost bridgeと、client保存frameの破損検査を維持し、canister側のchecksum再計算を移せるかを調べる。単にfooter検査を消して保存stateの破損を見逃す変更は採用しない。50/32回への再配置は未達。

## 再現

```sh
cargo build --offline --release --manifest-path scripts/codec_scan_bench/Cargo.toml --target wasm32-unknown-unknown --lib --target-dir artifacts/codec_scan/target
cargo build --offline --release --manifest-path scripts/codec_scan_bench/Cargo.toml --bin args --target-dir artifacts/codec_scan/native-target
.venv/bin/python scripts/check_codec_scan.py --canister <専用local診断ID> --directory artifacts/<fresh>
```

診断initはowner principal。全モデルは [TERMINAL_TAIL.md](TERMINAL_TAIL.md) のbuild/3body patch/固定準備/全層比較と同じflagsを使う。追加の `check_codec_frame_rejections.py --canister <ID> --wasm <Wasm> --directory <fresh>` は同じ固定packの検証canisterと単体reportが必要。
