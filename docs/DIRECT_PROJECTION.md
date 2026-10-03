# 分割再利用状態の直接INT8展開

2026-10-02。[可逆projection codec](PROJECTION_CODEC.md)で送る量子化済みINT8を、通常queryの入口で直接、計算用I16へ展開する実装。旧経路のINT8→F32→I16変換、整数用F32中間配列と再走査を省く。A結果は別配列へ直接復元する。新たな量子化・重み変更・F32演算順序変更は行わない。

## 省略する処理と残す検証

`decode_query`は既存frame/header/SHA256検査後、reuse入力をprivateな`PreparedProjection`へ復元する。INT8 byteからのWasm SIMD sign extensionはreserved -128検査と同時に行う。scaleは正の有限値、A結果は有限値、寸法と値数は既存の上限で検証する。型のprivate fieldによって復元後の改変を防ぎ、評価時にはrequestのmodel/pack/input/step/tensor/aux/dims/scalars/encodingを照合する。requestを変更した場合は重みを読む前に拒否する。

クライアントから戻る状態をquery間で検証済みとは扱わない。各queryで必要な不正状態検査は維持し、同じquery内の再変換と大きな整数配列の重複検査を省く。中間状態はクライアント保持、推論は通常query、固定重み準備だけupdateで行う。Candidと通信codecは前版と同一。

## 部分canister実測

元layer0 QKV8192×2560＋rank64、部分pack23,756,800 bytesを使用。4096+4096行の2queryを7入力長、cold/warm/clearedの3状態で実行し、84成功queryで独立旧native出力とbit一致。PythonとRustのresponse再エンコードもbyte一致。不正状態4件は全て拒否した。

| tokens | 通常2query命令 | capture/reuse命令 | 削減率 | reuse Candid bytes |
| --- | ---: | ---: | ---: | ---: |
| 1 | 66,728,965 | 60,219,072 | 9.7557% | 29,926 |
| 7 | 271,468,779 | 249,522,755 | 8.0842% | 193,245 |
| 8 | 177,456,555 | 174,007,286 | 1.9437% | 220,464 |
| 32 | 646,294,487 | 636,296,505 | 1.5470% | 873,730 |
| 64 | 1,273,271,313 | 1,253,779,677 | 1.5308% | 1,744,746 |
| 87 | 1,859,370,263 | 1,813,064,188 | 2.4904% | 2,370,789 |
| 132 | 2,672,028,200 | 2,633,596,413 | 1.4383% | 3,595,659 |

132-token warmの前compact版は2,651,654,611命令、本版は2,633,596,413命令（追加18,058,198命令、0.6810%減）。通信3,595,659 bytesは前compact版と同一で、通常2queryの3,517,884 bytesより大きい。部分projectionの改善率を全モデルへそのまま適用しない。時間は単回でquery cacheを制御していないため、速度向上の根拠としない。

feature有効時のruntime45、cache3、compile-fail doctest2、通常featureのruntime43、cache3、doctest1も通過。

部分module `95b16846b1bdaedf44967ca3b5001992d4229681fb823f150d9b7f198961e6c8`、selected canister `4qggx-l3777-77775-aaaca-cai`、生結果`artifacts/direct-projection/probe/report.json`。モデル、pack、独立oracle、native、入力のhashも同reportに保存し、生成物はgitignore対象。

## 残る繰り返し

前compact版の保存requestを再集計すると、prefixなし132-tokenには同一入力のMLP gate/up分割が31組残る。準備済みprefixからの主問題には同一入力のrow分割が0組である。この再利用だけで主問題の305 queryを減らせるとは扱わない。再集計は`analyze_preparation_work.py`、結果は`artifacts/direct-projection/repeated-work-{normal,hot}.json`。

固定INT8 dense272 tensorのscaleは計1,328,640個、5,314,560 bytes。現状は固定byte cacheから毎query復元し、projectionで正の有限値を再検査する。準備updateで検証済み型へ移す候補は未実装・未計上。capture返送側のI16→F32→INT8も残る。

より広いINT8 dotのrelaxed SIMD案については、[公式ICのWasm検証設定](https://github.com/dfinity/ic/blob/master/rs/embedders/src/wasm_utils/validation.rs)が`wasm_relaxed_simd(false)`を指定し、決定性のため無効化する理由も記している。ローカルの機能設定を変更して得た性能を通常ICの改善として扱わず、本案は採用しない。固定INT8→I16重み展開は過去のStrassen試験と同様にheap容量と準備費用を含めて判断する。

## 全層版の固定重み準備

全層module `e961dda8d6b2f2f71448ba1b21512b72f231f440ef3b7b41951ff3ae0209c423`を`experimental-projection-reuse,experimental-full-weight-cache`でbuildし、selected local主canister `4caro-hl777-77775-aaaba-cai`へupgradeした。sealed packを保持し、721 tensor・4,065,416,192-byte cacheを準備updateで復元。準備は15,027,997,708 handler命令、214.5919秒、Candid要求47,473／返信14,915,249 bytes。pack status1、cache status2、認証module read2は別計上。質問ごとの再準備は不要。生記録は`artifacts/direct-projection/full-preparation/report.json`。

## 全32層の比較と採用

全5実行を従来`projection-codec-v1`と比較した。prefixは32層hiddenと72 state arrays、それ以外は前31層の全hiddenと最終層の必要なlast-token hidden、48 state arraysを保持してbit一致を確認。final hidden、logits、確率、unknown、型付き判断も一致した。terminal省略後の保持要素を比較しており、最後の層の全token hiddenの比較とは扱わない。失敗とreplayは全て0。各実行前後の認証module、model/pack、cache721 tensorとbyte数、検証対象ソースhashの一致も確認した。

| 実行 | 通常query | handler命令数 | Candid bytes | 単回秒 | 最大query命令 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix | 282 | 193,719,819,209 | 318,918,016 | 40.5264 | 2,134,116,390 |
| 617 | 305 | 354,704,778,950 | 506,366,104 | 52.6768 | 4,021,006,084 |
| insufficient | 305 | 310,182,055,808 | 469,941,494 | 41.8105 | 3,489,466,935 |
| maximum | 305 | 365,457,619,923 | 516,788,871 | 50.6837 | 4,157,303,670 |
| normal | 485 | 522,650,865,435 | 816,643,811 | 73.0486 | 3,612,834,057 |

prefixなし132-tokenは523,244,923,486→522,650,865,435命令、追加594,058,051命令（0.1135%）減。31組のQKV capture/reuseで再変換を省いた。485 query、通信816,643,811 bytesは前compact版と同じ。原prepared-f32-v1の523,750,449,327命令からは計1,099,583,892命令減だが、通信はその版より2,410,839 bytes多い。

prefix準備済みの主問題にはcapture/reuse対象が0で、305 query、354,704,778,950命令。前版との差は+386,937命令（+0.0001091%）という分岐等の小さな増加で、主問題の命令削減とは扱わない。初回prefix込み587 query、prefixなし485 queryで50/32の目標は未達。現handler命令だけを5B/queryで使い切る理想値でも主問題71、初回prefix込み110、prefixなし105 query以上が必要。繰り返しの省略と並行して、積和自体とquery間の通信を減らす必要が残る。

終端heap観測最大4,115,005,440 bytesで前版と同じ。瞬間peakは未測定。handler命令はCDK Candid decode/encodeを含まず、Candid bytesはHTTP/CBOR/signatureを含まない。単回時間の揺れを速度改善と扱わない。既存の最大変更gold=yes/出力noは残り、bit一致は量子化済み従来実装からの数値劣化がないことの証拠であり、一般的な判断精度が向上したという意味ではない。

本moduleをselected local主canisterに保持した。明示オプション`--reuse-projection-inputs`と既存codecが必要。Layaのソース/Git/稼働canister、mainnet、remoteは変更していない。生記録は`artifacts/direct-projection-v1-*`、比較要約`docs/direct-projection-v1-summary.json`、cache/source照合`artifacts/direct-projection-v1-cache-checks/report.json`。いずれもgitignore対象。

再現は[前codecの全層手順](PROJECTION_CODEC.md)のbuild featureを維持し、prepareのdirectoryと検証run-nameを新しくする。今回の検証コマンド：

```sh
.venv/bin/python scripts/validate_prepared_weights.py \
  --canister 4caro-hl777-77775-aaaba-cai --run-name direct-projection-v1 \
  --baseline projection-codec-v1 --wasm artifacts/direct-projection/full.wasm \
  --preparation artifacts/direct-projection/full-preparation --reuse-projection-inputs
```

既存の固定pack、fixtureと比較元artifactが必要。prepareはモデル更新またはupgradeでcacheが消えた時だけ行う。

後続の[MLP分割再利用](MLP_REUSE.md)でgate/upの2つのLoRA Aも再利用し、全5実行を比較して追加31.57億命令減を確認した。現在の採用moduleと計測値は同文書を参照。上記の直接展開版の数値はその版の記録として残す。
