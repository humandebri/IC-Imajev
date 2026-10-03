# 固定INT8配置と入力アドレス共有の測定

この文書は部分投影の診断時点の記録。その後[固定cacheへの統合](OUTPUT_PAIRS.md)を全5条件で検証し採用した。主問題の全モデルhandler削減率は2.3933%で、以下の部分投影2.5513%とは区別する。

2026-10-03。採用版[固定活性化表](PREPARED_ACTIVATION.md)の次の探索。新しい配置のbase Q投影で、元Column32 kernelより入力展開込み2.5513%の命令削減を確認した。全モデルへ未接続で、採用版は主67 query、50/32 queryには未達。部分投影の削減率を全モデルへそのまま当てはめない。

## 配置変更

元のINT8重みをK4ごとに隣接する2出力行を組にして並べる。各8-byteロードは`[w(row0,k..k+4), w(row1,k..k+4)]`になる。量子化済みI16入力は同じ4値を2組にする。重みの値・byte数・元row scaleを変えず、F32 scale適用とblock加算順も維持する。再量子化や中間精度の変更はない。

`i32x4.dot_i16x8_s`の各laneに2出力の部分和を持たせることで、4出力をまとめるshuffleを6→2個へ減らす。block256の整数和をすべてI32で完了し、それから元のF32 scaleを適用する。`256*127*128 < 2^31`なので、この整数加算順変更でoverflowは起きない。

固定重みの並べ替えは準備update、入力の複製は通常query内で一度行う。質問に依存する状態をcanisterへ保持しない。現在のprototypeは81〜88 token、32の倍数の出力行だけを扱う。

## 同じ実重み・実入力を使った3段階の通常query検証

layer3 Q投影8192×2560の実重みと、採用版の主問題から保存した87-token入力を使用した。重みのSHA256は以前のColumn32試験の同じtensorと照合した。各版で主問題1入力＋符号付き極値・0・微小値の81〜88 tokenの8入力を使い、元/新kernelの18通常queryを実行した。3版合計54 queryで全出力SHA256が元kernel・独立したnative scalarと一致した。

| 版 | 主問題の元kernel | 新配置（入力準備込み） | 元kernelからの変化 |
| --- | ---: | ---: | ---: |
| V1: array from_fn | 1,089,211,303 | 1,369,103,651 | 25.6968%増 |
| V2: 明示入力ロード | 1,089,211,303 | 1,133,727,011 | 4.0870%増 |
| V3: tokenごとのアドレス共有＋SIMD複製 | 1,089,211,303 | 1,061,422,355 | **2.5513%減** |

V1/V2は不採用。V3は部分投影の候補として保持する。V3の全8境界条件でも入力準備込み2.5504〜2.5567%減。主問題のkernel命令は1,079,649,270→1,050,791,575、入力準備は元232→候補1,068,979命令、入力byte復元・量子化は両方9,561,801命令。準備費を隠して評価しない。

各queryのCandid requestは890,894 byte、outputの全F32値数は87×8192=712,704。計測counterにはCDK Candid decode/encodeと出力SHA256を含めない。単回時間はCLI・他処理を含み、反復して負荷を統制した速度評価ではない。

V3の固定重み準備は64 chunk update＋sealの65 update、930,330,199 handler命令。元重みとscaleは21,004,288 byte、新配置の重みは20,971,520 byte。診断では元重みと新配置を**両方**保持するため、全モデルの4 GiB配置が可能だという証拠にはならない。終端heap観測1,059 pages（69,402,624 byte）も、この診断だけの値で一時peakではない。

## Wasmで特定した繰り返し処理

同じ44 token×32出力のkernelの静的コード解析。動的性能の採否は上の実query counterで決める。

| 項目 | 元Column32 | V1 | V2 | V3 |
| --- | ---: | ---: | ---: | ---: |
| integer dot | 45,056 | 45,056 | 45,056 | 45,056 |
| shuffle | 2,112 | 704 | 704 | 704 |
| loop | 2 | 44 | 0 | 0 |
| conditional branch | 10 | 257 | 2,895 | 144 |
| V128Load（8-byte sign-extension loadを除く） | 1,768 | 3,264 | 3,176 | 3,176 |
| local.get | 94,663 | 94,257 | 96,241 | 93,464 |

V1では64要素の`from_fn`が各tokenにloopを残した。V2で明示ロードに変えるとloopは消えたが、各ロードの整数アドレス計算のchecked branchが増えた。V3ではtokenの先頭pointerを一度だけ求め、固定offsetのpointerロードにした。入力複製は4つのI16を`v128.load64_splat`で複製し、一度のstoreで8要素を初期化する。global overflow-check設定や範囲検査は維持する。

privateな重み・入力型のconstructorが、block256、32出力の倍数、完全なpadded88入力、正の有限scaleを保証する。SIMDロードは8-byte重み、16-byte複製入力の完全な範囲内。spare capacityへの複製は全新要素を書き終えてからVec lengthを設定する。native testは81/87/88 token×256/512列で符号付き極値とscaleを検証した。Wasm経路の一致はnative testだけでなく上の18 queryで確認した。

## 全モデルへの次の実装条件

新配置で採用版のcacheを置き換える際、元配置の全コピーを追加して4 GiBを超えてはいけない。packedな重みの型をreaderから整数kernelへ渡し、通常のbyte readerには元配置を正確に復元できる経路を残す必要がある。prefix45・情報不足80・重大変更89・prefixなし132・終端1 tokenなども検証する。入力複製を量子化済み入力の寿命内で共有し、実際に使わないpadding行の積和を省く方向を次に試す。

現在はこのcache置換、全token形状のkernel、全5条件の精度/命令/query/通信の検証が未完了。50 queryの達成には、採用版の272.452B handler命令をさらに少なくとも約8.24%減らし、分割境界も再配置する必要がある。今回の部分投影だけで達成したとは扱わない。

## 証拠と再現

診断canister `7avmr-x3777-77775-aaaka-cai`。V1 module SHA256 `fafc0d385cf43c90e163f642125f4ab53f0adf8e8f64f9a4c56e315245e7ecbd`、V2 `8732da3c66cc75d03878218c157d8ca7e34436e90964f79e2c980371a000eca6`、V3 `e7582ae73d7c6a2062bcec6a190ec5c27c0bf1324bf0b2dd2194768b26257a88`。各実行の前後で認証module hashとsource hashを照合した。

raw proofは`artifacts/lane-pair/{check,explicit/check,shared-address/check}/report.json`。各版の正確なsourceは`v1-source.zip`、`v2-source.zip`、`v3-source.zip`、Wasmとopcode解析も同directoryへ保存した。V3 proofは43 source hashを保持する。生成物はignore対象。

実装は`scripts/lane_pair_bench`、kernel生成は`scripts/generate_lane_pair.py`、検証は`scripts/check_lane_pair.py --canister <id> --directory <new-directory>`。nativeとWasmを同crateでrelease buildし、専用の新しい診断canisterをowner付き8192×2560でinitする。正式な推論canisterへこのprototypeをinstallしない。

主canisterは`ba47cf87…`の固定活性化表版を維持し、認証statusと採用proofの50 core source hashを再照合した。Layaは変更していない。
