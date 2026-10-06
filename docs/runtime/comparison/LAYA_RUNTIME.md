# Layaランタイムとの比較と共通化

2026-10-05。実重みを使った準備済み整数投影の初回比較では、Laya側のIC命令数が全10条件で少なかった。Imajevのtoken-scale候補はLaya比2.2〜33.5%増。今回のtoken-scale候補へ置き換えて命令を減らす根拠は得られなかった。目的はLayaの命令削減に絞り、別の最適化を[探索・実測](LAYA_INSTRUCTION_SEARCH.md)している。

今回の候補は `imajev-runtime::int8_token_kernel::project`。実験用の列全体I32積算であり、Imajevの採用済みblock256/S1 kernelとは別の経路。全ランタイム・全モデルの優劣を示す測定ではない。

## 今回実施した比較

IC-Laya-Standaloneの実INT8 packから `encoder.0.qkv.weight`（3072×1024）と `encoder.0.wo.weight`（1024×2624）の先頭512出力行を選んだ。元tensor全体のSHA256をmanifestと照合し、選択した重みとscaleのhashも保存した。Layaの元ソース・Git・ネットワーク・canisterは変更していない。

入力は実モデルの中間activationではなく、決定的な合成INT8値とtokenごとに異なる正のF32 scale。同じ整数入力・重み・scaleを両経路へ渡す。Laya側は現在のWasm `dot_tile` と `q8_store_tile` をそのまま抽出し、64/32/16/8/4/2/1 token・16出力のdispatchを診断wrapperに再現した。biasはなし。出力groupのscratch組立と出力の再量子化は含めない。

両経路とも列全体のI32内積→F32変換→token scale→weight scaleの順序を保持する。候補には既存APIの入力検査・出力確保・finite検査を含み、両経路に共通の出力finite検査もある。従って検査量などのAPI overheadは同じではない。幅2624は候補側だけ2816へゼロ埋めし、追加演算は命令数に含めた。pad後の入力・重みの準備は区間外。

## Local IC命令数

出力行数はすべて512。同じ診断Wasm・同じlocal canisterで両経路を交互に通常queryとして実行し、投影区間の `performance_counter(0)` を測定した。

| token数 | 入力列数 | Laya命令数 | Imajev token候補命令数 | 候補の増加 |
|---:|---:|---:|---:|---:|
| 1 | 1024 | 799,766 | 992,596 | +24.1% |
| 7 | 1024 | 3,536,527 | 3,613,551 | +2.2% |
| 32 | 1024 | 11,972,859 | 12,289,575 | +2.6% |
| 65 | 1024 | 24,010,532 | 26,918,266 | +12.1% |
| 128 | 1024 | 46,582,513 | 47,739,642 | +2.5% |
| 1 | 2624 | 1,915,128 | 2,556,246 | +33.5% |
| 7 | 2624 | 8,430,934 | 9,072,566 | +7.6% |
| 32 | 2624 | 28,340,121 | 30,440,485 | +7.4% |
| 65 | 2624 | 56,793,931 | 67,240,385 | +18.4% |
| 128 | 2624 | 110,218,558 | 118,523,175 | +7.5% |

全10条件で両経路のF32出力bytesが一致し、独立したscalar整数計算とF32丸めによる参照にも一致した。Nodeでの照合後、IC上の全60queryの出力digestをこの結果と照合した。各条件の命令数も3回とも一致した。query引数には固有nonceを入れ、返信cacheを避けた。

Node実時間でも全条件でLaya側が速かった。128 token・2624列はLaya 5.13 ms、候補9.66 ms。ただし共有ホスト上の7標本の中央値であり、一般的な速度倍率には使わない。ICの同じ条件は110,218,558対118,523,175命令で候補が7.5%多い。時間と命令数を区別する。

候補は65 tokenを内部で72 tokenまで計算し、2624列は2816列まで計算する。一方、Layaの今回のdispatchは65 tokenを64＋1で処理でき、2624列もそのまま扱う。差の原因になり得るが、検査量・tile幅等の差もあるためpaddingだけの寄与率は未測定。

## 測っていない範囲

入力の量子化、出力の再量子化、丸め方式の変更、bias付き投影、Attention、norm、全モデル、logits、判定品質、通信、モデル準備費用は未比較。Layaのhalf-away-from-zeroとImajevのties-to-evenを置き換えていない。準備済み整数から始めることで、量子化による差を測定から除いた。

診断は両方の入力表現・重み表現を同じheapに置くので、記録したheapページを各runtimeのメモリ消費として比較できない。準備は10 update、推論は60通常query。digest計算と返信は命令の計測区間外。cycles残高とend-to-end時間は測っていない。

## 共通化の設計候補（現在は保留）

1. **数値契約を分けた共通INT8演算を設計する。** Layaのtoken scaleとImajevのblock256 scaleを明示し、整数積算・scale適用・出力再量子化の境界を定義する。Layaの既存kernelを基準実装として維持する。
2. **実activationと再量子化まで比較する。** 固定入力からLayaの層入力を取得し、bias、group別scale、丸め境界、幅2624と端数tokenを含める。今回の合成入力一致だけでモデルの数値一致とはしない。
3. **モデルadapterを接続する。** Layaのpack読込と演算順序を保持し、既存96入力のWasm参照logitsを共通runtimeで再現する。全モデルの命令・通信・query数を同じ入力と状態で測る。
4. **schedulerを改善する。** 全推論の一致確認後に分割や通信を変える。Imajevの50query配分はLayaへコピーせず、Laya側の予算から作る。

初回比較は完了。モデルadapter、実activation、再量子化、全推論の比較はまだ実装していない。

## 証拠と再現

数値と証拠hashは [laya-kernel-results.json](laya-kernel-results.json)。原記録は `artifacts/runtime-comparison/laya-v1/{build.json,source.zip,node-report.json,kernels.wasm}` と `artifacts/runtime-comparison/laya-ic-v1/report.json`。凍結ソースに元LayaのINT8ソース・manifestと抽出ソースを含め、ビルドからNode照合までのソース不変も確認した。LayaのMIT licenseを生成物に添付した。

測定用canister `37c7s-il777-77775-aaata-cai` は新規作成した診断専用の1個だけで、終了後stop/deleteが成功した。既存canisterの変更はない。

再現手順は [harness README](../../../scripts/laya_kernel_bench/README.md)。既存Laya全推論の基準は [INT8_QUERY_OPTIMIZATION_SEARCH](../../../../IC-Laya-Standalone/docs/INT8_QUERY_OPTIMIZATION_SEARCH.md) にあるが、この文書の投影数値とは集計範囲が異なる。
