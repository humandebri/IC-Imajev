# Prefix復元の繰り返し読み込み削減とレビュー

2026-10-04。重みINT8、元F32 adapter/readout、BF16境界、加算順序は維持する。質問状態はクライアント保持、推論は通常query。50/32 queryは未達。

## 実装と修正

`delta_restore.rs` に既存復元処理を分離し、`experimental-prefix-update-hoist` で更新ベクトルのSIMDロードをkeyループの外へ移した。同じ128値を128回読み込む処理を一度にする。各状態要素は従来の `old * g + k * update` を同じ順序で計算する。FMA、再量子化、token切り詰めは使わない。入力長・有限値・decay範囲の検査を維持した。

レビューで、分離時に `delta_log` のfeature guardが脱落し、通常featureのビルドが壊れる問題を発見して修正した。修正後は通常nativeの48 unit/2 doctest、候補nativeの127 unit/15 integration/9 doctestが通過（fixture依存1件ignore）。Python join4件、codec16件、journal6件が通過し、保存MLP payload315件のbyte一致も確認した。

残差の可逆圧縮には `residual_raw_threshold` を追加した。圧縮効果が小さいbyte planeをraw送信へ切り替え、復号命令を減らす。既定値0は変更せず、診断では0.1を指定した。範囲外とNaNの拒否、raw/Huffmanで元byteが変わらないことを検証した。

## 復元単体

`artifacts/prefix_update_hoist/check-v1`。共通prefix45 token、全24 Delta層、二つのlayout、baseline/候補の計96通常query。返却digestは全F32ビットを含み、保存参照と一致した。counterは入力byteのF32展開と復元を含み、digest計算・CDK Candid decode/encodeを含まない。

|layout|変更前命令|変更後命令|削減|
|---|---:|---:|---:|
|value major|173,166,507|127,486,335|45,680,172（26.38%）|
|key major|165,565,274|119,436,184|46,129,090（27.86%）|

全層で同じcounter。heap136 pages、要求1,111,694/返信92 Candid bytesは同じ。これは復元だけの値で、全推論の改善率やquery削減として扱わない。ベンチは実モデルと同じ復元ソースを直接コンパイルする。baseline `6dzfx-dd777-77775-aaamq-cai` と候補 `6k2ol-vl777-77775-aaana-cai` は専用の診断canisterで、重み・質問状態は保持しない。

## 連結queryの境界

`compact-q4-raw10-main-v3` は旧moduleで残差threshold0.1だけを変更。87 token/MLP前半5888行/24 headのPairは4,862,445,223→4,841,047,326命令、Candid bytesは3,533,841→3,537,017。21,397,897命令減と引き換えに3,176 bytes増えた。返却状態は全ビット一致。

`prefix-hoist-chain-main-v1` は復元の最適化も組み込んだ実験。上と同じPairは4,806,737,778命令になり、さらに34,309,548命令減った。profileではprefix復元87,933,192、MLP完了2,157,858,841、Delta head計算2,021,664,733、out整数列継続359,638,410命令。包含spanは足し合わせない。

24 headの場合は第五query（残り8 headとfull MLP）がIC0522、26 headの場合は第四query自体がIC0522だった。成功したhidden/norm/carry/convは保存参照と全ビット一致。失敗要求、途中成功metric/profile、ソースZIPを保存した。部分query削減を全体query削減とは扱わない。

レビュー修正後の候補buildは `artifacts/prefix_codec/full-build-prefix-update-hoist-review-v2`、Wasm hash `3fbbd566964df38019ca46a54989cebead8996dffebf21a7a4af33384ae44da0`。専用実験canister `6eydd-o3777-77775-aaama-cai` にだけupgradeする。保護対象canisterとLayaは変更していない。

## 全体回帰

`full-prefix-update-hoist-review-proof-v2` の全6条件が完走し、保存対象のhidden/state/最終hidden/型付き判断/logits/確率が固定INT8参照と全ビット一致した。失敗/replayなし、source/moduleを前後確認し79ファイルのソースZIPを保存した。compact tailが返さないlayer30 hiddenは比較対象に含めない。

標準の主経路は62 query・234,721,319,375命令・123,281,081 Candid bytes。直前候補の同じ62-query経路235,201,653,167から480,333,792命令（0.2042%）減り、通信/query数は同じ。単回時間は34.298→47.970秒と悪化したので速度改善を主張しない。handler命令はCDKのCandid decode/encodeを含まず、Candid通信はHTTP/CBOR/署名を含まない。

共通prefixは66 query、hybridなし主問題64、情報不足62、最大変更62、prefixなし132 tokenは290で完走した。最大変更の見逃しを含め判断は以前と同じで、判断精度の改善はない。重み準備は721 update・4,065,416,192内部weight-read bytes・78,739,929,194命令・252.012秒で、推論queryの値とは分ける。

既存連結経路も `rolled-prefix-update-hoist-review-v2` の全3条件でbit一致、失敗/replayなし。69ファイルのソースZIPを保存した。主と情報不足は54 query、最大89 tokenは既存の62-query fallbackを維持する。

|条件|query|総命令数|Candid bytes|最大query命令|単回秒|
|---|---:|---:|---:|---:|---:|
|主87 token|54|240,546,092,310|133,896,992|4,900,654,472|41.140|
|情報不足80 token|54|220,045,106,719|125,114,808|4,492,999,333|49.071|
|最大変更89 token|62|240,030,890,467|125,231,193|4,816,563,899|37.064|

直前の54-query主経路241,643,997,846から1,097,905,536命令（0.4543%）減った。通信は同じ、終了時heap観測最大4,144,037,888 bytes。単回時間は36.864→41.140秒で悪化し、速度改善とは扱わない。全体のquery数削減は未達。

## 次の対象

Pairの大部分はMLP完了とDelta projection/recurrenceで、復元だけの削減では26 headを収められなかった。次は、直前のAttention/MLP前半queryへdownの整数部分和を前倒しし、元F32累積値をクライアントへ返す案を検証する。既存carryはdownのF32 A部分和だけを持ち、整数baseは完了queryで全部計算している。追加通信、5B命令上限、元のblock256加算順序を別々に測る。現時点では未実装・未成立の候補である。

別方向の探索として `residual-dictionary-capacity-v1.json` に実87-token carryのbyte分布を保存した。上位byteは22種類、最多15種類で99.916%を覆う。4-bit辞書+escapeの容量見積りは111,568 bytes（raw222,720、現Huffman約78,940）。低位byteは全256種類で同方式はrawより大きく不適切。Huffmanより通信は増えるが復号分岐が減る可能性がある。まだcodec未実装で命令/query削減は未測定。
