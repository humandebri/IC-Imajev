# 整数ロード・タイル・共通prefixを別々に測る

2026-10-02。固定model/INT8 pack/未統合F32 LoRA/専用readout/calibrationを保ち、専用local canister `4caro-hl777-77775-aaaba-cai`（8001）で検証した。中間状態はクライアントが保持し、cacheの作成も継続推論も通常query。Layaは参照のみで変更していない。

## 全層の費用

BOOM DAO 617、132 token、rotations=1。

| 指標 | 前回 | 今回の通常実行 | prefix準備済みの実行 |
| --- | ---: | ---: | ---: |
| query | 708 | 708 | **611** |
| handler命令 | 739,286,168,442 | **711,017,272,458** | **492,550,510,942** |
| Candid bytes | 1,247,760,195 | 1,247,760,195 | **872,750,670** |
| 単回local秒 | 78.490 | 87.311 | 99.448 |
| 最大query命令 | 4,580,151,047 | 4,395,987,402 | 4,511,178,558 |
| handler終端heap最大観測bytes | 82,968,576 | **41,811,968** | 41,287,680 |

cacheを使わなくても命令3.82%減、heap最大観測は約半分。prefix準備済みでは今回の通常実行から命令30.73%、通信30.05%減、前回からの命令削減は33.37%。各段階で全32層hidden・conv/DeltaNet/KV state・logit・確率・unknownがbit一致。[通常実行](layout-full-results.json)、[prefix実行](prefix-hit-results.json)、[費用と全検証](prefix-costs.json)。

**速度向上は確認できていない。** 上表の617の時間は増えた。host負荷・query cache・cold/warm条件を制御した反復比較ではない。入力ごとの時間も下表に保存し、命令削減から待ち時間短縮を推定しない。handler counterはCDK Candid decode/encodeを除外、通信はHTTP/CBOR/署名を除外、heapは瞬間peakではない。固定4.702 GB packのstable配置容量は変わらない。

## タイルと重みロード

同じ保存実入力、整数射影の全10 shapeを同じlocal環境で順番に実測した。shapeの実出現数264で加重した比較を示す。全モデル合計は上表で別途実測した。

| 候補 | 射影命令の加重値（基準比） | 判断 |
| --- | ---: | --- |
| token tile最大32 | +0.91% | 不採用 |
| token tile最大16 | +4.23% | 不採用 |
| token tile最大8 | 大きなMLPで50億命令上限超過 | 不採用 |
| 出力tile8（以前16） | −1.87% | 改善 |
| INT8直接ロード | −3.05% | 改善 |
| 出力tile8＋直接ロード | **−4.93%** | 採用、全層で3.82%減 |

[候補ごとのWasm/source hashと比較](integer-layout-candidates.json)。32/16/8へのtoken分割は重みロードの共有を減らす。小さいtileが常に安いとは限らない。8-token版の上限拒否は実験候補の失敗であり、最終採用版の3入力は失敗query・checkpoint replayとも0。

採用版は重みをi16の大きな一時配列へ展開せず、8 byteのINT8を`v128_load64_zero`で読み、`i16x8_extend_low_i8x16`で符号拡張して共有する。最終行の末尾でも8 byteだけ読む。shape・padding・finite・scaleの既存チェックを保つ。I32 dotの正確さ、block256のF32 scaling/加算順、元入力によるF32 LoRA、BF16丸めは変更しない。

## prefixの初回費用

同じ公式prompt/tokenizerを用いた23入力に、厳密に共通の先頭45 tokenがある。canisterでその45 tokenを全32層実行して、client側へhidden・conv・DeltaNet state・KV/絶対positionを保存した。公式hiddenを推論入力へ流用していない。prefixの全32層hiddenは、通常132-token実行の先頭45行とbit一致。

| prefix準備（1回） | 実測 |
| --- | ---: |
| 通常query | **578** |
| handler命令 | **255,695,466,238** |
| Candid bytes | **463,530,574** |
| 単回local時間 | **30.197秒** |
| client cacheの64データファイル | **70,409,344 bytes** |

617の最初の1質問は準備も含めて**1,189 query・748,245,977,180命令・1,336,281,244 bytes**。今回の通常実行より高い。同じ617費用の質問を繰り返す計算では、命令・通信は2質問目から、query数は6質問目から累積費用が下回る。これは費用式の比較であり、異なる質問全体の待ち時間保証ではない。

cacheはmodel/pack/Wasm/source hash、厳密なtoken列、64ファイルのSHA256、dtype/shape/finite、連続した絶対positionを検証する。model/pack/Wasm/sourceの不一致・パス逸脱・ファイル改変・位置ずれの7ケースを拒否した。[契約検証](prefix-contracts.json)。cacheはこのclient内で再利用し、canisterへ推論状態を永続化しない。

suffixだけをembedding・線形射影・conv・DeltaNet・MLPで処理する。RoPEは絶対offset45。attentionはprefix KVを連結し、先頭45個のqueryをゼロとして既存のcausal kernelを使い、suffix出力だけ採用する。prefix queryの計算はまだ残るが、絶対query groupingと加算順を保って検証した。cacheは改変せず、各質問へstateのcopyを渡す。

## 異なる入力長の全層検証

| 入力 | 全token／処理suffix | query | handler命令 | Candid bytes | 単回秒 | 最大query命令 | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 617 | 132／87 | 611 | 492,550,510,942 | 872,750,670 | 99.448 | 4,511,178,558 | likely |
| 情報不足 | 125／80 | 579 | 432,645,254,849 | 798,181,466 | 55.779 | 4,197,249,944 | unknown |
| 重大変更 | 134／89 | 611 | 508,160,372,805 | 890,066,919 | 61.137 | 4,582,503,026 | no（誤判定） |

全3入力で全32層hidden/state/判断のbit一致、型付き出力有効、失敗query・checkpoint replay 0。[情報不足](prefix-insufficient-results.json)、[重大変更](prefix-maximum-results.json)。以前の整数化による同じpackのF32方式との確率差最大約8.48ポイント、重大変更の見逃し、選択肢順序依存は残る。今回の追加最適化の数値保存と、元モデルへの判断精度・校正同等性を分ける。

Rust23件、Python26件、実重みのnative scalarとの整数/LoRA比較96ケース、量子化境界16ケースを確認した。[整数比較](integer-parity.json)。

## 残る方向

**50 queryは未達。** cache hitの617でも、計算を50億命令へ理想的に詰めて99 query相当。50回の予算にはさらに49.24%の命令削減が必要で、初回準備は別に必要。

LoRA Aの同一入力再計算も集計した。cache hitで64組・64回の重複があり、避けられる積和は912,261,120 MAC、A重み読込41,943,040 bytes。F32のA出力を一度ずつ返すだけでも1,425,408 bytesになる。[再利用候補](lora-reuse-candidate.json)。まだ未実装であり、命令削減として計上しない。clientへA出力を返して次tileへ渡す方式は、追加通信・hash費用と検証して選ぶ必要がある。

prefix attentionの不要なquery計算、conv/QK norm/gated normの再送を減らす融合も候補。pruning・量子化の変更によって判断精度が保存されるとは仮定しない。

## 再現

```sh
cargo test --workspace --offline
cargo build --release --target wasm32-unknown-unknown -p imajev-inference --offline
cargo build --release -p imajev-runtime --offline
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
# 自分の専用local canisterへinstall/upgradeし、固定packを準備したあと：
.venv/bin/python scripts/run_full_canister.py --canister <id> --arithmetic int8 \
  --wire-codec bf16-exact --compact-lossless --fuse-add-norm --fuse-mlp \
  --delta-head-cap 8 --attention-head-cap 8 --row-cap 16384 \
  --work-cap 2500000000 --token-cap 132 --directory artifacts/new-standalone
# 共通prefixを実際に生成。cacheは指定directoryのqueries配下に作る。
.venv/bin/python scripts/run_prefix_canister.py --canister <id> --arithmetic int8 \
  --wire-codec bf16-exact --compact-lossless --fuse-add-norm --fuse-mlp \
  --delta-head-cap 8 --attention-head-cap 8 --row-cap 16384 \
  --work-cap 2500000000 --token-cap 132 --prepare-prefix \
  --cache artifacts/new-prefix/queries --directory artifacts/new-prefix
# --prepare-prefixを外し、同じcacheと新しい実行directoryで質問を処理。
.venv/bin/python scripts/run_prefix_canister.py --canister <id> --arithmetic int8 \
  --wire-codec bf16-exact --compact-lossless --fuse-add-norm --fuse-mlp \
  --delta-head-cap 8 --attention-head-cap 8 --row-cap 16384 \
  --work-cap 2500000000 --token-cap 132 \
  --cache artifacts/new-prefix/queries --directory artifacts/new-prefix-hit
```

採用Wasm SHA256: `806c1006b7effc726b0cb5ab77e7f4735fdbdb69308e6000d26856704e1f9664`。既定の通常実行にprefix利用を自動で足さない。異なるtoken prefixやWasm/sourceではcacheを再生成する。query journalのinput identity検証に加え、prefix用sessionはcache hashを束縛する。mainnet・push・PRは実施していない。
