# 256列整数dotとgate/up/SwiGLUの融合

後続のINT8直接ロードと共通prefix cacheは[DIRECTIONS.md](DIRECTIONS.md)。以下は変更前の履歴。


2026-10-02。前回の整数方式・BF16境界・F32 LoRA/readout/calibrationを保存したまま、整数dotのaccumulator更新とMLPの送受信を減らした。中間状態はclient保持、推論は通常query、updateは専用local canisterの準備と重みuploadだけ。Layaのソース・Git・canisterは変更していない。

## 全32層の実測

BOOM DAO 617、132 tokens、rotations=1。

| 指標 | 前回 | 256列dotのみ | MLP融合4096行 | 広いMLP融合 |
| --- | ---: | ---: | ---: | ---: |
| 総handler命令数 | 846,799,463,736 | 776,924,555,896 | 746,859,932,863 | **739,286,168,442** |
| query数 | 868 | 868 | 740 | **708** |
| Candid通信bytes | 1,624,778,979 | 1,624,778,979 | 1,270,774,939 | **1,247,760,195** |
| 単回local時間 | 81.145秒 | 123.627秒 | 90.305秒 | **78.490秒** |

前回比で命令12.70%、query18.43%、通信23.20%減。当初3.682兆命令からは79.92%減。617の最大query4,580,151,047命令、handler終端heap最大観測82,968,576 bytes。失敗queryとcheckpoint replayはともに0。handler counterはCDK Candid decode/encodeを除外し、通信はHTTP/CBOR/署名を除外する。heapは瞬間peakではない。

**命令数の減少と時間の短縮は分ける。** 4096行版の単回時間は前回より長く、広い分割版は短かった。cache、host負荷、環境再作成を制御した比較ではなく、速度向上を主張しない。256列展開はWasmを約1.2→2.2 MBへ増やし、release buildも約6→65秒の候補測定となった。命令予算を優先した採用だが、code sizeやcompile負荷という代償はある。

両段階で全32層hidden・conv/DeltaNet/KV state・最終logit・確率・unknownが前回版と全bit一致。[256列dot](dot-unroll-full-results.json)、[4096行融合版](mlp-fused-results.json)、[広い融合版の全queryとbit比較](mlp-wide-results.json)、[最終費用集計](mlp-wide-costs.json)。これは以前の整数方式への追加誤差がゼロという検証であり、元F32方式や公式未量子化へ一致するとの意味ではない。

## 残る費用の内訳

最終617の命令数は、単独整数LoRA射影331.21B（44.80%）、融合MLP238.81B（32.30%）、DeltaNet heads72.78B（9.85%）、conv22.14B、add/norm20.75B、attention heads19.07B。射影関連が77.10%を占める。これはquery全体のop分類で、射影中のdot・quantization・LoRA・SHA等も含む。以前のquery内profileとは別の内訳であり、重複加算しない。全queryとop別費用を保存した。

広い融合の上限は観測4.58B handler命令で、保守的な4B目安を超える。今回の指定入力で通常queryのdecode/encodeを含む呼び出しが成功したことを確認した。任意入力の成功保証ではなく、`--fuse-mlp --row-cap 8192 --work-cap 2500000000`という実験設定で使用する。既定の小さいrow/work capは保つ。

## 別の入力長と判断

| 入力 | tokens | query | handler命令合計 | Candid bytes | 単回秒 | 最大query命令 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 情報不足 | 125 | 707 | 703,350,817,414 | 1,185,933,892 | 79.948 | 4,599,436,195 | unknown |
| 最大lock duration重大変更 | 134 | 708 | 754,751,549,085 | 1,265,424,603 | 104.378 | 4,615,277,865 | no（gold yesで誤り） |

両入力も全32層hidden/state/logit/確率が前回整数方式と全bit一致し、失敗query・checkpoint replayとも0。125-tokenでは幅6032＋3184行、134-tokenでは5624＋3592行を使い、最大の観測は46.15億handler命令だった。3入力の型付き出力は有効。重大変更の既存の見逃しは残り、型が有効であることは正解の証明にならない。[情報不足の全queryとbit比較](mlp-wide-insufficient-results.json)、[重大変更の全queryとbit比較](mlp-wide-maximum-results.json)、[最終検証とhash](mlp-wide-validation.json)。

今回の改善による追加誤差は測定入力でゼロだが、以前の整数化に伴う同じpackのF32演算との確率差（23入力で最大約8.48ポイント）は残る。公式未量子化モデルとの精度・校正の一般同等性を主張しない。[以前の判断診断](INTEGER_ARITHMETIC.md)。

## 同じ環境での候補比較

専用local networkは停止しており、再開後に旧canisterが存在しなかった。新しい`4caro-hl777-77775-aaaba-cai`を`http://localhost:8001/`へ作り、同じ固定4,702,451,200-byte packを復元した。upload/hashは3,201 update、Candid chunk request4,702,594,915 bytes、409.488秒。この準備費用は上の推論時間・通常query通信量へ含めていない。古い同名principal文字列のcanisterと同じinstanceではない。

初回upload状態とupgrade後ではheap/alloc counterに微差があったため、前回Wasmもupgradeしてから測り直した。64/128/256列候補を同じ復元環境・upgrade状態・実入力で比較。376整数射影のshape出現数で展開した命令削減は5.21%／7.82%／10.79%。これはshape標本の加重値で、全モデルの削減値は別に上表で実測した。各候補の全出力bit一致、source/Wasm hashを保存した。[候補の計測](dot-unroll-candidates.json)。

32列ごとのaccumulator更新を、256列のbalanced I32 add treeへまとめる。整数dotの絶対和は256×127×128でI32内に収まり、整数の結合順を変えても正確。activation scale・weight scaleの別乗算、block順のF32加算は変更しない。LoRAのF32積和もそのまま。

## MLP融合

`mlp_gate_up_integer`はgateとupのbase射影、元F32入力による各LoRA、各BF16丸め、従来のBF16 SwiGLUを一つのqueryで実行する。inputのblock256量子化は共有するが、adapterは混ぜず、出力丸めを省略しない。clientへgate/upの両中間配列を返さず、SwiGLUの積だけを返す。

まず132-tokenの9216出力を4096＋4096＋1024行へ分割し、各層の旧gate/up4 query＋SwiGLU3 queryを3 queryへ減らした。さらにwork予算から5704＋3512行へ広げて2 queryとし、32層で合計160 query削減した。5704行の実重み比較でbit一致と45.43億命令を確認してから、全層へ適用した。[広い分割の直接比較](fused-mlp-wide-check.json)。分割が増えてLoRA Aを再計算する費用も含めて全モデルを測定した。

dimsは`[tokens,rows,cols,row_start]`、tensorは同じ層の`.mlp.gate_proj.weight`、aux1本は`.mlp.up_proj.weight`、scalar1個はadapter scale。A/Bは固定manifestから名前とshapeを検証して読む。2射影の合計4B MACまで、token512、rows8の倍数、cols256の倍数、片射影weight tile30M bytes、出力900k floats、通常blob2,000,000 bytes以内。unsafe整数カーネルのbuffer/finite/padding/scale検証を保つ。instruction-limit時は行幅を8単位で半減し、失敗をjournalへ記録する。

`--fuse-mlp`は明示指定し、整数方式とlossless BF16 wireだけで使用できる。sessionに`gate-up-swiglu-v1`と`mlp_scheduler=work-budget-v2`を記録して旧journalとの混在を拒否する。既定のF32経路・単独projectionは残る。

## 検証と残る目標

Rust23件、Python全22件（scheduler7件、wire3件、INT8 wire4件、journal4件など）を確認。実重み27ケースで、非ゼロrow offset・token1/7/8/9/32/36/64/96/132の融合出力が旧3 queryと全bit一致。さらにscheduler幅3ケースで命令予算を確認し、metadata/row/workの不正5ケースを拒否。整数射影96ケースと量子化境界16ケースもnative scalarとbit一致。[融合契約と数値](fused-mlp-check.json)。

**50 queryは未達。** 0.739兆命令は計算だけを50億の枠へ詰めても148 query相当。50回の予算へはさらに約66.18%の総命令削減が必要。整数dotの残るload/一時配列、attentionやDeltaNetの再送、共有prefixの再計算が候補になる。

23参照入力の先頭45 tokenが厳密に共通であることも確認した。[prefix候補](prefix-cache-candidate.json)。clientに各層のconv/DeltaNet/KV stateを保持して再利用する候補だが、まだ推論cacheは未実装。初回prefix準備費用、絶対position、継続時の丸め順を含めて検証する必要がある。50 query到達や精度保存を計上していない。

## 再現

```sh
cargo test --workspace --offline
cargo build --release --target wasm32-unknown-unknown -p imajev-inference --offline
cargo build --release -p imajev-runtime --offline
# 自分の準備済み専用local canisterへupgrade後、新しいdirectoryで実行
.venv/bin/python scripts/check_fused_mlp.py --canister <id>
.venv/bin/python scripts/check_integer_parity.py --canister <id>
.venv/bin/python scripts/check_simd_quantization.py --canister <id>
.venv/bin/python scripts/run_full_canister.py --canister <id> --arithmetic int8 \
  --wire-codec bf16-exact --compact-lossless --fuse-add-norm --fuse-mlp \
  --delta-head-cap 8 --attention-head-cap 8 --row-cap 8192 \
  --work-cap 2500000000 --token-cap 132 --directory artifacts/new-mlp-fused-run
```

採用Wasm SHA256: `d019a32bb19b9672a2d68d6d7481e0be7fb9d7bfb4faf8afe1981dd5c543ab57`。固定モデル/packと公式prompt、tokenizer、adapter、専用readout、calibrationは変更していない。mainnet・push・PRは実施していない。
