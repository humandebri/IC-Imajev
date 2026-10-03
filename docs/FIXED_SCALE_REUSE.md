# 固定INT8 scaleのquery内処理を省く

2026-10-03。固定scaleのbyte→F32復号を準備updateへ移し、元のscale tailを除去して同容量のF32配列を保持する。出力pair/S1四象限の重みとscaleを同じ不変型に束ね、通常queryが必要な行を直接参照する。中間状態はclient-held、推論は通常query。base INT8、LoRA/readout/calibrationと積和順序を維持する。

## 実装

`PreparedPairs`のprivate scaleは構築時に正の有限値を確認する。任意byte範囲の読み出しは従来通り元packと一致し、unaligned/cross-boundary viewも対応する。scale単独のaligned viewは既存`PreparedF32`で参照し、固定重みとscaleの論理payloadは増やさない。

第二候補では`PackedView`に束ねられたscaleを直接使う`project_prepared`へ接続する。各行列積のscale読み出し、F32配列確保・復号、配列全体の正値/有限値走査を省く。queryから型のfieldは変更できない。shape・work bounds・出力有限性検査を維持し、未準備readerとtoken-scale経路は従来の検証を使う。

## 第一候補：復号だけを移す

現在のpair配置で再試験した。以前の[不採用試験](PREPARED_SCALES.md)とは実行kernel・配置が異なるため再計測したが、これだけで性能向上を主張しない。

85 runtime unit・9 integration・3 compile-fail doctest、7 canister unitとdefault runtimeが通過。実入力のMLP3層の返信がbyte一致。ただしhandlerはそれぞれ150,941 / 150,653 / 204,457命令増加。有限性再走査とscale用viewの取得が残る。

全5条件（旧log主問題対照を含む6実行）で全保持hidden/stateと判断が既存INT8版とbit一致、失敗/replay0。次の増減は前候補8fd0c75c…に対するhandler計測で、CDK Candid処理を含まない。query数とCandid bytesは変わらない。

|条件|query|handler命令|前候補からの増減|
|---|---:|---:|---:|
|prefix|66|130,707,216,008|-9,920,241|
|baseline-617|66|250,358,787,185|+11,808,763|
|617|64|248,525,879,973|-2,516,123|
|insufficient|64|227,235,971,811|+29,599,828|
|maximum|64|254,119,117,287|-20,514,857|
|normal|291|386,563,096,529|+92,737,711|

第一候補module `e47a22ee9a29690a33160352ce8354fc832c69c3db5e84ff724aa440e1e78789`。cache payload4,065,416,192 bytes/721 tensors。準備721 update・78,739,929,194命令・323.026秒。生記録`artifacts/prefix_codec/full-fixed-scales-{proof,profile,preparation}`、sourceと差分`full-build-fixed-scales`。生成物はignore。

## 第二候補：scaleの読み出し・復号・再走査を省く

専用canister `6eydd-o3777-77775-aaama-cai`、module `32a3e1d5e7143847fa2771a183552ff8ab1fee20fbaf9a7a198921a5e231e7ad`。全保持hidden/state・判断4条件の最終hidden、value/abstain/logits/確率/unknownが既存INT8版とbit一致。型付き出力検査通過、失敗/replay0。以下はLoRA A採用候補8fd0c75c…との比較。再コンパイルによる他処理の変化も含み、scale処理だけの単独効果とは扱わない。

|条件|query|handler命令|削減命令|削減率|Candid bytes|秒（単回）|最大query命令|
|---|---:|---:|---:|---:|---:|---:|---:|
|prefix|66|130,501,685,597|215,450,652|0.1648%|71,105,691|18.490|2,287,186,824|
|baseline-617|66|250,004,499,295|342,479,127|0.1368%|113,697,747|20.294|4,309,015,826|
|617|64|248,171,552,221|356,843,875|0.1436%|124,634,637|20.089|4,309,015,826|
|insufficient|64|226,907,041,019|299,330,964|0.1317%|117,701,589|18.487|3,938,267,720|
|maximum|64|253,759,021,087|380,611,057|0.1498%|126,615,477|20.471|4,405,351,911|
|normal|291|386,042,969,839|427,388,979|0.1106%|580,218,522|44.813|2,389,018,388|

実入力MLP3層のprofileは1 queryあたり6,563,696 / 6,563,324 / 6,509,754命令減（0.1509〜0.1522%）、返信byte一致。layer0のbase inclusiveは3,346,084,704→3,339,421,744命令。残るbase演算は主MLPの約77.54%で、scaleを省くだけではquery数は減らない。

主問題は356,843,875命令（0.1436%）減、query64回・Candid124,634,637 bytesは同じ。50/32未達。全5条件でhandler合計が減った第二候補を実験用の次の基準とする。判断精度の向上を意味せず、既存の最大変更の見逃しも残る。

固定cacheは721 tensors / 4,065,416,192 bytes、pair payload3,569,090,560 bytes、RoPE131,072・activation1,048,576 bytes。scaleの二重保持はない。ただし観測heap終了値最大は前候補4,128,833,536→4,131,258,368 bytes（+2,424,832）。実allocation・scratchの観測値も報告し、payload不変をheap不変とは扱わない。瞬間peakは未測定。

準備721 update / 78,739,929,194命令 / 273.326秒を推論と分ける。prefix packetの2回目はquery/命令/Candid通信が0。命令はhandler内counterでCDK Candid処理を含まず、通信はCandidのみ、時間は単回。

証拠は`artifacts/prefix_codec/full-fixed-scales-direct-{proof,profile,preparation}`。proofの`validated-source.zip`とbookend hashes、buildのsource/extra-kernel ZIP・差分を保存した。主canister module36c04a57…が変わらないことは`main-unchanged-fixed-scales/report.json`のreadで確認。Layaを変更していない。全生成物はignore。

再現は`build_full_prefix_candidate.py --direct-input --terminal-attention --prefix-start --delta-state-layout --strassen-raw --f32-output-generic --instruction-profile`の新しいdirectoryを指定し、既存pair/S1/F32の3 bodyをpatch・validateする。実験用canisterだけをupgradeし、`prepare_weight_cache.py --include-f32 --require-prepared-rope --require-prepared-activation --require-all-output-pairs`、`check_full_prefix_hybrid.py --terminal-attention --prefix-start`を実行する。詳細コマンドは[F32 Aの手順](F32_A_REUSE.md)と同じ。
