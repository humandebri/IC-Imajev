# IC上のINT8投影比較

2026-10-05。準備済みの合成入力を使ったINT8線形投影では、ImajevのIC命令数が比較先より38.5〜79.4%少なかった。モデル全体の性能比較ではない。

## 比較対象と結果

比較先はonicaiのllama.cpp fork `91c63a2284ff99c5761a27c3865b312fb8eca148` の [Wasm Q8_0 dot実装](https://github.com/onicai/llama_cpp_onicai_fork/blob/91c63a2284ff99c5761a27c3865b312fb8eca148/ggml/src/ggml-cpu/arch/wasm/quants.c)。関数本体とFP16変換を固定したソースから抽出し、各token・出力行に呼び出す。比較先のcanisterも [-msimd128でビルドする](https://github.com/onicai/llama_cpp_canister/blob/e987fca5ff91dbff3026a1ac5ff44cb372039212/icpp.toml)。SIMDの有無の比較ではない。

Imajevは現在の `output_pairs::project` と準備済み重みを使い、既存のpair、S1 raw、S1 output128のWAT置換を適用した。1 tokenではsingle-quad経路を使う。両者とも出力行数は512。ICの同じlocal networkで、通常query内の `performance_counter(0)` により投影の区間を測定した。

| token数 | 入力列数 | Imajev命令数 | GGML Q8_0命令数 | 命令数削減 |
|---:|---:|---:|---:|---:|
| 1 | 256 | 285,743 | 464,341 | 38.5% |
| 7 | 256 | 866,694 | 2,528,845 | 65.7% |
| 32 | 256 | 2,935,508 | 11,150,945 | 73.7% |
| 87 | 256 | 7,494,214 | 30,115,565 | 75.1% |
| 132 | 256 | 11,163,228 | 45,619,345 | 75.5% |
| 1 | 2560 | 1,615,726 | 3,736,597 | 56.8% |
| 7 | 2560 | 6,713,684 | 23,274,637 | 71.2% |
| 32 | 2560 | 22,596,914 | 104,723,137 | 78.4% |
| 87 | 2560 | 59,442,406 | 283,901,837 | 79.1% |
| 132 | 2560 | 88,473,037 | 430,477,137 | 79.4% |

代表形状の87 token × 512出力 × 2560列では59,442,406対283,901,837命令で、79.1%削減した。入力と出力の再利用、重み配置、ループ展開、scale処理の粒度が寄与する候補だが、各要因の寄与率は分離測定していない。

同じWasmのNode/V8実時間では、この形状はImajev 8.04 ms、GGML 5.66 msだった。複数tokenの全形状でImajevが遅く、1 tokenの2形状では速かった。IC命令費用とCPU上の実時間は別の指標である。Nodeは `--no-liftoff` で実行し、12回のwarmup後、実行順を交互にした9標本の中央値を採用した。専有CPUでの測定ではなく、この時間比を一般的な速度倍率とは扱わない。

## 同一出力の条件と測定範囲

GGML Q8_0はblock32・FP16 scale、Imajevはblock256入力・F32 scaleを使う。全scaleを1にし、各32要素の入力先頭を127、その他の入力と重みを決定的な整数にした。全形状の内積がF32の整数精度内に収まるため、このfixtureでは加算順の違いを含めても同じ数学的結果になる。通常のモデルデータで量子化精度が同じという証明ではない。

Nodeで全出力を独立したscalar内積と照合し、両実装のF32 bytesも一致した。ICでは全10形状を各3回、両実装で測定した計60 queryの出力digestをこの結果と照合した。命令数も各形状の3回で一致した。queryの引数を毎回変え、返信cacheの再利用を避けた。

入力量子化、重みの再配置、入力operand cacheの準備、FP16 lookup tableの初期化は測定区間外。準備用の20 updateは性能表に含めない。Imajevの区間にはAPIの検証、出力の確保・コピー・finite検査を含む。一方、GGML側は事前確保した出力へのdotループで、llama.cppのgraph全体は含まない。返信、digest計算、通信も区間外。heap情報は原記録に保存したが、固定サイズの診断用bufferを含むためモデルのメモリ効率を表さない。

計測はICで課金対象になる実行命令を対象とし、cycles残高の差やend-to-end時間は測っていない。Q4_K_M、Attention、KV cache、LoRA、token生成も未比較。従って「llama_cpp_canister全体より約5倍速い」とは結論しない。

## 証拠と再現

要約は [kernel-results.json](kernel-results.json)。原記録は `artifacts/runtime-comparison/kernel-ic-v2/{build.json,source.zip,report.json}` と `artifacts/runtime-comparison/ic-proof-v1/report.json`。要約に原記録のSHA256を保存した。`source.zip` は測定用moduleを作った時点のソースで、ビルド前後とNode測定後にhashの一致を検査した。後の整形等を含む現行ソースとは区別する。相手の取得ソースhashは [fork-sources.json](fork-sources.json) に保存した。生成したCソースとforkのMIT licenseもartifactに保存した。

IC測定には新規の診断専用canisterを2個だけ作り、測定後に両方stop/deleteした。既存canisterと外部canisterの変更は行っていない。

準備条件と実行コマンドは [benchmark harness README](../../../scripts/runtime_kernel_bench/README.md) を参照する。

## ライブラリ化への示唆

IC向けのINT8投影kernel、固定重み配置、準備済み入力の再利用は、他モデルへ持ち出して検証する価値がある。現在 `inference-core` に抽出したF32演算・codecに加え、このINT8経路を数値契約と一緒に分離するのが次の候補になる。モデルgraphや推論API全体が独立した汎用runtimeとして完成したことを、この測定は示さない。
