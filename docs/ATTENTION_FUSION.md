# Attention投影からGQAまでのquery統合

2026-10-02。INT8/F32 LoRA投影、BF16境界、norm/RoPE、causal GQA、gateの既存演算順序を維持し、query境界の中間serialize・decode・hashを省く。K/Vの同じ入力量子化もquery内で1回へまとめる。

## 実装

`experimental-attention-fusion` featureと`--fuse-attention`を明示する。対象は固定Imajev/Qwen3.5 textモデルのhidden2560、Q16 head×512（Q256+gate256）、K/V4 head×256、rank64の元F32 A/B、scale2。元のnorm weightとRoPE theta10,000,000・rotary64を使う。query limit5B、heap4GiB、frame2MBは変更しない。推論は通常query、中間状態はclient-held、固定重み準備だけupdate。

- `attention_kv_integer` dims=`[tokens,position_offset]`、tensor=`<self_attn>.k_proj.weight`、aux/scalarsなし。K/V projectionを同じblock256量子化で実行し、K norm/RoPEも同queryで実行。返信はtoken-major Kと元のBF16 V。
- `attention_q_gqa_integer` dims=`[query_tokens,total_tokens,total_tokens-query_tokens,first_head,heads]`、tensor=`<self_attn>.q_proj.weight`、aux/scalarsなし。入力は正規化済みhidden＋該当GQA groupのhead-major full K/V history。Q/gate投影→Q norm/RoPE→causal GQA→既存BF16 attention gateを順に実行し、gated attentionをtoken-majorで返す。

すべてのheadと履歴を使う。headsは4の倍数、first_headも4で整列し、global GQA groupと同じ対応を保つ。89 token以下はQ16 headを一括、それより長い132 token経路は8 headずつ。終層readoutだけQ1 tokenを使う既存方針を維持し、K/Vは全tokenを計算する。queryが返すK/Vをそのままclient cacheへ保存する。新たな量子化、重み統合、token削減は行わない。

Runtimeはモデル/pack identity、lossless codec、metadata、finite input、token/position/head/payload/work、INT8 base形状、対応するF32 A/B/rank64を検査する。projection work2.5G、既存GQA work75Mの保守的上限を使う。これらのMAC上限自体を5B命令以内の証明とは扱わず、実queryでも確認する。

## 部分実測

直前の採用`blake3-v1`に保存された5条件の第3・第31層で21入力を構成。nativeの統合出力が元の分離kernel出力とbit一致し、Wasm返信がnativeとbyte一致した。native cacheのbinary SHA256・各request/reply SHA256を照合して再利用し、Wasm実行時にnative計算を繰り返していない。CLIには`--native-cache`の明示指定が必要。

| 実入力 | 統合op | 通常query命令 |
| --- | --- | ---: |
| 主問題87 token、第3層 | K/V＋K norm/RoPE | 562,050,408 |
| 主問題87 token、第3層 | Q16＋norm/RoPE/GQA/gate | 3,555,899,318 |
| prefixなし132 token、第3層 | Q先頭8 head＋後続処理 | 2,360,374,572 |
| 同132 token | Q後半8 head＋後続処理 | 2,360,435,164 |

21入力最大は3,675,135,031命令。別途、誤position・非整列head・未知tensor・lossy encodingの4通常queryを拒否。前後認証module read2回。部分probeだけで全モデルquery数や判断精度を証明したとはしない。

部分記録`artifacts/attention-fusion/{native-check-fixed,wasm-check}/report.json`。初回native probeは長いQのcapture codec名を試験headerへ引き継ぐ構成誤りで途中停止し、fixtureだけを修正して全21条件を新directoryで再検証した。その途中記録`native-check`を完了証拠として使わない。

Rust feature有効50 tests＋compile-fail doctest2、Python routing2・terminal2・compact heads2が通過。query数上限を満たす入力全体が同じhead/historyを使用すること、終層Qの絶対位置、K/V保存を確認した。

## 次のDelta統合に向けたpacket実測

`scripts/size_delta_projected_state.py`で、現実の5条件のhidden・conv history・F32 Delta stateを使い、16 head単位の「投影を内包したDelta query」のpacketをencodeした。入力最大でも2MBに収まり、4つのrank64 A結果・INT8入力/scaleのcapture byte予算を追加した返信も2MB未満だった。主問題87 tokenの入力1,519,315 bytes、capture付き返信上限697,813 bytes。prefix準備45 tokenのstate返信とcapture上限1,422,235 bytes。cold132 tokenのcapture付き返信上限1,045,259 bytes。

これは仮のop名を用いた通信容量の確認で、Delta投影統合のkernel・速度・精度・query数の証拠ではない。queryに同じ中間QKV/Zを送る処理を省ける見込みが具体的にあるため、Attention全層検証後の実装対象とする。記録`artifacts/attention-fusion/delta-projected-packet-sizes.json`。

## 再現

```sh
cargo build --release --offline --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-projection-reuse,experimental-full-weight-cache,experimental-blake3,experimental-attention-fusion
# 専用local canisterへupgradeし、sealed packの固定weight cacheを準備した後:
.venv/bin/python scripts/check_attention_fusion.py --canister <local-id> \
  --wasm artifacts/attention-fusion/full.wasm --native artifacts/attention-fusion/primitive \
  --directory artifacts/new-attention-probe
.venv/bin/python scripts/validate_prepared_weights.py --canister <local-id> \
  --wasm artifacts/attention-fusion/full.wasm --preparation artifacts/attention-fusion/full-preparation \
  --run-name attention-fusion-v1 --baseline blake3-v1 \
  --reuse-projection-inputs --frame-checksum blake3 --fuse-attention
```

## 全32層の実測と採用

固定prefix45 tokenを新規作成し、主問題・情報不足・最大変更・prefixなしまで5条件を全32層で完走。保持hidden/stateと4質問の終端hidden・raw logits・probabilities・unknown probability・型付き判断・abstentionは直前`blake3-v1`とbit一致。prefixは32層全token hidden＋72 state array、質問実行は31層全token hidden＋終層最後token hidden＋48 state arrayの比較である。失敗/replay0、固定cache721 tensor/4,065,416,192 bytesの名前・容量が前後不変、認証module hashも全実行の前後で一致。

| 条件 | token数（suffix） | 従来query | 新query | 新handler命令 | 命令削減率 | 新Candid bytes | 通信削減率 | 単回時間 秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix準備 | 45 | 282 | 242 | 174,064,328,178 | 0.4575% | 291,954,692 | 8.4546% | 25.6629 |
| 主問題 | 132（87） | 305 | 265 | 323,417,890,858 | 0.4147% | 459,917,414 | 9.1729% | 42.8754 |
| 情報不足 | 125（80） | 305 | 265 | 281,168,309,548 | 0.4331% | 427,221,092 | 9.0906% | 40.1023 |
| 最大変更 | 134（89） | 305 | 265 | 333,521,767,242 | 0.4117% | 469,274,957 | 9.1941% | 68.4913 |
| prefixなし | 132 | 485 | 438 | 469,252,607,675 | 0.3575% | 750,180,202 | 8.6426% | 68.3129 |

主問題はquery13.1148%減、通信46,448,690 bytes減、命令1,346,873,394減。初回prefix込み587→507 query、prefixなし485→438。50/32 queryには未達。現在のhandler総命令をquery limit5Bで単純に割り切り上げても主問題65・初回100・prefixなし94である。これはCDK・通信・依存を無視した下限で、実現可能query数の予測ではない。

命令はhandler counterで、CDK Candid encode/decodeは除外。通信はCandid request/replyで、HTTP/CBOR/signatureを除外。時間はquery cache未制御の単回計測で速度改善を主張しない。最大変更の時間は直前48.6687秒から68.4913秒へ増えている。原因を確認していないので、query数・命令・通信削減から実時間の改善を推定しない。元公式参照との量子化由来の誤差と、最大変更gold=yesに対してnoを返す従来の見逃しも残る。今回証明したのは直前INT8実装から追加劣化がないこと。

主問題最大query3,873,386,783命令、最大変更4,006,293,787命令。最大観測heap4,115,333,120 bytes。selected localのquery limit5B/heap4GiBは変更していない。

採用module `58cf64265a5ae8ec45eef1fd5fdabaae281a0cd66790d7d344b51dc78171524f`、canister `4caro-hl777-77775-aaaba-cai`。Wasm/native/build/probe記録は`artifacts/attention-fusion`、全層生記録は`artifacts/attention-fusion-v1-*`、比較は`docs/attention-fusion-v1-*-results.json`と`docs/attention-fusion-v1-summary.json`。`artifacts/attention-fusion-v1-cache-checks/report.json`で36 implementation hashの前後一致と固定cache/module bookendsを確認。対応sourceとglobal release Wasm hashも照合した。

固定重み準備は別途721 update・15,027,997,708命令・246.2345秒、Candid request47,473/reply14,915,249 bytes。準備のpack status1/cache status2 query・認証module read2は推論集計の外。全層前後のcache status2 query・認証module read2も別。モデル変更/upgrade後に固定重みを準備し、質問ごとの推論ではupdateしない。

native/Wasm・pack・測定JSON・全実行状態はgitignore対象。Layaのソース・Git・canisterには触れていない。
