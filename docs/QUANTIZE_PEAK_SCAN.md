# 量子化の有限値検査と最大値探索を一走査へまとめる

2026-10-03。各量子化で入力全体を有限値検査した後、block256ごとに最大絶対値を探していた。一回の最大値探索で有限性も判定する実装に変更した。主247,133,233,383→245,900,230,125 handler命令、1,233,003,258減（0.498923%）。62 query・通信123,281,081 bytesは同じ。50/32 query未達。

## 数値と型の維持

F32の符号bitを除いた整数bit patternは、非負有限値に対して浮動小数点の値と同じ大小順である。Wasm SIMDのunsigned maximumで4 laneのpeakを同時に求め、最終peakが `0x7f800000` 以上ならInf/NaNとして拒否する。signaling/quiet NaNと両符号を検査する。有限値のpeakは従来のabs/maxと同じbitを返す。scaleの `max/127`・最小subnormal・全zeroのscale=1、RNE divide/nearest/clamp、整数laneの範囲を変更していない。

`quantize_simd::block` はResultを返し、非有限値ならそのblockの出力を書き込む前に拒否する。前のblockまで書き込んだ後に拒否してもVecのlenは0のままで、未初期化値を公開しない。constructorが全blockとpaddingの書込みに成功してからimmutable `QuantizedRows` を作る。nativeにも同じpeak判定を接続した。モデル、INT8重み、元F32 adapter/readout/calibration、BF16境界は変更していない。

## 同じmoduleの単体比較

`scripts/quantize_scan_bench` は元のSIMD sourceをcontrolに保存し、同じmoduleで旧二走査と新一走査を比較する。旧scalarを独立referenceにして、active/padded整数値と全scaleのdigestを比較した。5実canister入力、7有限bit pattern/shape境界、6 Inf/NaNを計36通常queryで確認。全finiteのdigest一致、全nonfinite拒否。queryは一時入力だけを使用し、model stateは保持しない。

|実入力|旧量子化命令|候補命令|削減率|
|---|---:|---:|---:|
|prefix45|4,321,716|1,709,458|60.445%|
|主87|8,392,882|3,342,524|60.174%|
|情報不足80|7,704,232|3,060,224|60.279%|
|最大変更89|8,633,382|3,466,924|59.843%|
|cold132|12,533,174|4,870,566|61.139%|

n1/7/8/32/45/88/512、cols256/512/2560/9216を含むsyntheticも一致。符号付きzero、最大有限値、subnormal、half付近を含む。単体counterは量子化だけで、Candid・入力byte復号・digestを含まず、全モデルの60%改善ではない。native bench2 testsとruntime独立scalar・late-failureの追加testも通過。

単体証拠 `artifacts/quantize_scan/check/report.json` / `validated-source.zip`、source-beforeとcontrol code、Wasm/helper build log、native test logを保存した。診断canister `5tkpr-7d777-77775-aaaeq-cai` の前後module照合とsource bookendsを確認。

## 全32層の比較

専用canister `6eydd-o3777-77775-aaama-cai`、module `0b26bfaf6de27e67a4c73225966ba81d67a0eb180b9d95029809aa31852cd96b`。直前のterminal-tail compact版と比較。

|条件|query|handler命令|削減命令|Candid bytes|最大query命令|秒（単回）|
|---|---:|---:|---:|---:|---:|---:|
|prefix45|66|129,316,264,747|655,155,968|71,105,691|2,263,557,873|11.450|
|旧log主87|64|247,733,177,211|1,233,003,258|112,344,191|4,935,422,031|20.073|
|主87|62|245,900,230,125|1,233,003,258|123,281,081|4,935,422,031|19.898|
|情報不足80|62|224,828,709,876|1,133,832,872|116,455,579|4,534,321,047|18.439|
|最大変更89|63|251,488,140,870|1,261,337,654|126,604,858|4,358,843,576|20.470|
|cold132|290|382,410,944,485|1,987,303,008|580,207,902|2,379,472,033|43.594|

全条件の返却hidden・全保持state・最終hidden・typed判断・logits・確率/unknownが既存INT8版とbit一致、失敗/replay0。主/情報不足/旧logは前回同様layer30 hiddenを返さないので、未返却hiddenの直接比較とは区別する。最大変更の見逃しも同じで、判断精度改善を主張しない。query数・通信量は全条件で変わらない。観測heap終了値の全条件最大4,131,258,368 bytesも同じ。

固定cache721 tensors /4,065,416,192 bytesを維持し、準備721 updates /78,739,929,194命令 /267.115秒を推論と分離した。prefix packet二度目の準備はcache hit、準備query・命令・通信0。既存mainのmodule `36c04a57…` は不変、Layaは変更しない。検証用canisterのupgradeがローカル模擬cycles不足で一度拒否されたため、20Tのローカル模擬cyclesを補充して成功を確認した。mainnet/cycles送金は実施していない。

証拠 `artifacts/prefix_codec/full-quantize-scan-proof/{report.json,before-after.json,validated-source.zip,codec-second.json,main-unchanged.json}`、`full-quantize-scan-preparation/report.json`、`full-build-quantize-scan` のbuild/patch logとkernel hash/ZIP。Rust89 runtime＋9 canister unit、9 integration、4 compile-fail doctest通過。生成物はignore。

実MLP layer0/1/3を3通常queryでprofileし、以前の実返信frame全体とbyte一致。layer0はhandler4,261,310,702命令、INT8 base inclusive3,319,242,044（77.89%）、A積合計198,929,924、B積320,383,609。wire decode28,426,115とencode27,404,662のうちBLAKE3は各14,626,238 /14,626,241。nested spanを重ねて合計しない。まだbase投影が最大のボトルネック。証拠 `full-quantize-scan-profile/report.json` / `validated-source.zip`。

## 再現と次の調査

```sh
cargo build --offline --manifest-path scripts/quantize_scan_bench/Cargo.toml --release --target wasm32-unknown-unknown --target-dir artifacts/quantize_scan/target
cargo build --offline --manifest-path scripts/quantize_scan_bench/Cargo.toml --release --bin args --target-dir artifacts/quantize_scan/native-target
.venv/bin/python scripts/check_quantize_scan.py --canister <専用local診断ID> --directory artifacts/<未使用directory>
```

診断canisterのinitはowner principalだけ。helperの `init` でCandidを作り、専用canisterへreinstallする。全モデルのbuild/patch/固定重み準備/比較は [TERMINAL_TAIL.md](TERMINAL_TAIL.md) と同じflagsで、directoryを新規にし、同じS1 address reuseとF32 K64 kernelを使う。

次の重複はcodecの全有限値検査とBF16分類の別走査、そしてframe全体のchecksum計算。現行clientの固定ic-agent0.49.2は署名付きidentityを使用し、query返信署名検証のdefaultはtrueで、無効化設定はないことをローカルsourceで確認した。ただし現時点ではframe checksumを省いていない。transport検証・client保存時の破損検査・checkpoint再開を保った上でcanister側の再計算を移せるかを、別候補として調べる。50回への配置は未実装で、合計命令/5Bの算術下限を達成回数とは扱わない。
