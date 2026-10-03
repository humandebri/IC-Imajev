この文書の測定はDelta投影統合時点の記録。現在の採用版は[入力バッファの二重初期化削減](ACTIVATION_BUFFERS.md)で、query数と数値出力を維持し命令を追加削減した。

# Delta投影・gate・再帰をclient-held準備状態で統合

2026-10-02。QKV/Zを別queryで返してからconv/Deltaへ再送する境界を減らす候補。最初のhead群でblock256入力量子化、QKV/Zそれぞれの元F32 LoRA A、全32 headのgateを一度だけ計算し、続くhead群で再利用する。INT8 base・LoRA A/B・BF16境界・F32再帰・readout/calibrationは維持する。

## プロトコルと処理

`experimental-delta-projected` feature、`--fuse-delta-projected`を明示する。既存のlossless fused DeltaとINT8 projectionが前提である。

`delta_project_capture` / `delta_project_reuse`のdimsは`[tokens,heads,first_head,keep_recurrent_state]`、tensorは`<linear_attn>.in_proj_qkv.weight`、aux/scalarsなし。tokens1〜132、headsは2〜16の偶数、first_headも偶数で32 head内に収める。

capture入力は正規化済みhidden・該当Q/K/V channelの3-token conv履歴・元F32 Delta状態。量子化と2種類のA積・gateは同query内で共有する。QKVのQ/K/V別の出力行では同じA結果を使い、Zは元の別A結果を使う。gate用A/B projectionには固定adapterにLoRAがないことを既存helperで検査し、同じ量子化を共有する。

返信はgated Delta出力、更新後のraw conv履歴、必要ならF32再帰状態、capture時だけ準備状態を付ける。準備状態の論理値数はtokens×2762で、INT8整数値2560、raw F32 scale10、QKV A64、Z A64、raw F32 g32/beta32。reuse入力ではこの状態と別head群の履歴・再帰状態を受け取る。canisterに質問依存の状態を保存しない。

準備状態は既存`bf16-block256-exact-v1`または`bf16-exact`で可逆に送る。INT8値は整数のままで、今回のwireでは正確なBF16値として2 bytesを使う。scale/A/gate/再帰状態を新たに量子化しない。受信した整数範囲・整数性・scale正の有限値を`QuantizedRows`で検査し、A/gateを含む全F32入力の有限性を検査する。独自のINT8 wire codecはこの候補に追加していない。

各frame2MB/header16KB/900K valuesを維持し、capture返信の保守的なbyte上限を重み読み出し前に検査する。clientは同じ上限でhead数を選ぶ。通常の45-token prefixと87/80/89-token質問、prefixなし132 tokenの状態省略経路は16 headずつ。132 tokenでF32状態も返信する一般経路では10 head単位にする。全headと全tokenを処理する。

query内で組み立てるconv/Z/gate/stateのscratchだけ最大1,089,664 F32値まで認めるprivate helperを使う。これはserialized入力ではなく、従来の通常`delta_stage_bf16`は900K上限のまま。n<=132・h<=16でこの内部上限を固定し、公開wire上限とquery limit5B/heap4GiBを広げていない。

## nativeとWasmの切り分け

最初のnative対保存Wasm比較ではF32再帰状態に差が出た。第0層prefix入力のgate122値に最大5.96e-8の差があり、従来の独立native binaryと統合nativeのgateは一致した。保存Wasmのgateを従来native Deltaへ渡すと再帰状態は保存Wasmと一致した。統合で生じた差ではなく、既存native/Wasmのgate計算差と判断した。初回停止記録`artifacts/delta-projected/native-check`は完了証拠に使わない。

nativeは独立した従来binary `artifacts/strassen-wide/default-primitive`（SHA256 `4c51d29e213c6d1777d284a188e5f4b1056c4a8802c7a7b10f79c4c980e1574f`）でQKV/Z projection・gate・8 headずつのconv/Deltaを分離実行して比較した。5条件の第0・第30層、capture/reuse合計20条件がbit一致。Wasm由来の同じ入力hidden/初期履歴/初期再帰状態を使い、gateを含む演算は両nativeでそれぞれ計算した。各request/responseと独立oracle実行を保存する。

native対保存Wasmには20条件合計365,813値の差があり、その大半はprefixのF32再帰状態である。nativeとWasmがbyte一致したとは主張しない。Wasm候補は保存された従来Wasm出力へ直接bit比較し、captureで返したWasm準備状態を次のreuse入力へ使う。この準備状態をnative由来gateへ置き換えて検証しない。

native完了記録`artifacts/delta-projected/native-paired-final/report.corrected.json`。Rust feature有効52 tests＋compile-fail doctest2、default44＋doctest1、canister cache3が通過。新client試験1つで5構成を確認し、fused Delta1・Delta input1・Attention fusion2・journal4も通過。capture/reuseで全headを覆うこと、conv/F32 state保存、長い状態保持経路の事前reply容量チェックを含む。

## 再現

```sh
cargo build --release --offline --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-projection-reuse,experimental-full-weight-cache,experimental-blake3,experimental-attention-fusion,experimental-delta-projected
# 専用local canisterへupgradeし、固定pack/cacheを準備した後:
.venv/bin/python scripts/check_delta_projected.py --canister <local-id> \
  --wasm artifacts/delta-projected/full.wasm --native artifacts/delta-projected/primitive \
  --directory artifacts/new-delta-probe
.venv/bin/python scripts/validate_prepared_weights.py --canister <local-id> \
  --wasm artifacts/delta-projected/full.wasm --preparation artifacts/delta-projected/full-preparation \
  --run-name delta-projected-v1 --baseline attention-fusion-v1 \
  --reuse-projection-inputs --frame-checksum blake3 --fuse-attention --fuse-delta-projected
```

## Wasm部分実測

保存された5条件の第0・第30層、capture/reuse計20通常queryが従来Wasmのgated出力・conv履歴・保持F32再帰状態とbit一致。最初のcaptureから返されたWasm準備状態をreuseへ渡している。入力の非整数・予約値-128・scale0・scale負値・奇数first headの5通常queryを拒否。前後認証module read2回。記録`artifacts/delta-projected/wasm-check/report.corrected.json`。

| 入力 | query | handler命令 |
| --- | --- | ---: |
| 主問題87 token、第0層 | capture先頭16 head | 1,921,961,599 |
| 同 | reuse後半16 head | 1,747,188,678 |
| prefixなし132 token、第30層 | capture先頭16 head | 2,743,127,149 |
| 同 | reuse後半16 head | 2,548,359,710 |

部分最大2,743,127,149命令で5B以内。native対Wasmの既存差をこの一致確認から隠さず、native同士・Wasm同士をそれぞれ検証した。部分probeだけで全モデル精度やquery数を主張しない。

## 全32層の実測と採用

prefix45 tokenから新規に準備し、主問題・情報不足・最大変更・prefixなしの5実行を全32層で完走。直前`attention-fusion-v1`と保持hidden/state、4質問の終端hidden・raw logits・probabilities・unknown probability・型付き判断・abstentionがbit一致。prefixは32層全token hidden＋72 state array、質問実行は31層全token hidden＋終層最後token hidden＋48 state arrayを比較した。失敗/replay0。

| 条件 | token数（suffix） | 従来query | 新query | 新handler命令 | 命令削減率 | 新Candid bytes | 通信削減率 | 単回時間 秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix準備 | 45 | 242 | 194 | 173,781,523,510 | 0.1625% | 246,693,116 | 15.5029% | 23.9704 |
| 主問題 | 132（87） | 265 | 193 | 321,911,152,912 | 0.4659% | 371,329,222 | 19.2618% | 40.1721 |
| 情報不足 | 125（80） | 265 | 193 | 279,881,218,179 | 0.4578% | 345,843,820 | 19.0480% | 38.6988 |
| 最大変更 | 134（89） | 265 | 193 | 332,008,679,890 | 0.4537% | 378,624,781 | 19.3171% | 63.4646 |
| prefixなし | 132 | 438 | 294 | 464,623,602,703 | 0.9865% | 580,794,410 | 22.5793% | 63.5643 |

主問題は72 query（27.1698%）・Candid88,588,192 bytes・handler1,506,737,946命令を削減。初回prefix込み507→387 query。prefixなしは144 query（32.8767%）・169,385,792 bytes・4,629,004,972命令を削減。50/32 queryは未達。handler総命令を5Bで割り切り上げるだけでも主問題65・初回100・prefixなし93が下限で、CDK・通信・依存を無視している。次は整数dot自体の演算方式を実測する必要がある。

命令はCDK Candid encode/decodeを除外するhandler counter。通信はCandid request+replyでHTTP/CBOR/signatureを除外。時間はquery cache未制御の単回実測で、速度改善の保証とはしない。元公式参照との差は残り、最大変更gold=yesに対する従来noも同じである。今回の一致は既存INT8実装から追加劣化がないことの確認で、一般判断精度の改善を証明しない。

最大queryは主問題3,873,386,783、最大変更4,006,293,787命令。最大観測heap4,119,920,640 bytes。query limit5B/heap4GiBは変更していない。

採用module `cfa16f56a8de31fbeeaf7a60650a87fe33a6974f93e05b4ad62f94c072288f6e`、主local canister `4caro-hl777-77775-aaaba-cai`。Wasm/native/build/probeは`artifacts/delta-projected`、全層生記録`artifacts/delta-projected-v1-*`、比較`docs/delta-projected-v1-*-results.json`と`docs/delta-projected-v1-summary.json`。`artifacts/delta-projected-v1-cache-checks/report.json`で37 implementation hashの前後一致、固定721 tensor cache 4,065,416,192 bytes・名前一覧の不変、認証module bookendsを確認。global release Wasmも同hashである。

固定重み準備は別途721 update・15,027,997,708命令・230.0523秒、Candid request47,473/reply14,915,249 bytes。準備のpack status1/cache status2 query・認証module read2、全層前後cache status2 query・認証module read2は推論数から分けた。推論中にupdateで状態を保持せず、モデル変更/upgrade後の固定重み準備だけupdateする。

Wasm/native・重み・測定JSON・全中間状態はgitignore対象。Layaのソース・Git・canisterは変更していない。

部分probeの比較値数fieldは保存Wasmとの比較を明示するため`differences_to_saved_wasm_values`へ改名した。元の生`report.json`とSHA256を保持し、`report.corrected.json`へmetadata訂正を記録。測定request/replyや命令数は変更していない。Wasm側の0は候補Wasm対保存Wasmが同じという意味で、native対Wasm一致を意味しない。
