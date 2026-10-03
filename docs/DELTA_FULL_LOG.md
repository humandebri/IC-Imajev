# 32 head Deltaの通常query統合

2026-10-03。全5条件の実query検証後に採用。以後の固定活性化表版は[PREPARED_ACTIVATION.md](PREPARED_ACTIVATION.md)を参照。従来[Column32版](COLUMN32.md)の主91→67 query、prefix込み初回181→133 query。50 queryには未達。

## 採用済みの全5条件実測

旧採用Column32 v1との比較。全5条件の保持hidden、prefix72/各質問48の状態配列、判断・確率がビット一致。prefixのDelta log 24層は検証専用helperでF32へ復元し、旧Wasm保存状態の全bitと比較した。推論側には復元結果を渡さず、実際の通常queryがlogを復元して後続hiddenを生成する。失敗/replay0。

| 条件 | query | handler命令数 | Candid通信bytes | 単回秒 | 最大handler命令数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix45 token | 90→66 | 156,313,227,846→151,805,703,029 | 145,902,403→71,105,691 | 20.633→17.913 | 2,697,997,814 |
| 主 suffix87 / 全132 token | 91→67 | 279,591,007,008→279,999,674,508 | 183,432,765→113,709,005 | 32.945→31.929 | 4,869,416,555 |
| 情報不足 suffix80 / 全125 token | 91→67 | 259,182,867,349→259,814,890,111 | 173,060,522→106,668,202 | 31.508→29.222 | 4,520,882,576 |
| 重大変更 suffix89 / 全134 token | 122→98 | 305,987,641,245→306,318,620,639 | 267,767,242→197,070,502 | 37.812→35.243 | 4,767,482,917 |
| prefixなし132 token | 292→292 | 447,558,332,080→447,558,340,370 | 580,229,781→580,229,781 | 58.083→57.558 | 2,882,723,905 |

主問題は24 query（26.374%）、69,723,760 byte（38.0105%）削減。一方、prefix状態の復元が追加されるため408,667,500命令（0.1462%）増。情報不足は0.2439%、重大変更は0.1082%増、prefixなしは8,290命令増。prefixは4,507,524,817命令（2.8836%）減。prefix込み初回のhandler合計は435,904,234,854→431,805,377,537命令（0.9403%減）。query削減を計算量削減とは扱わない。

主問題の単回31.929秒、prefix込み49.842秒。5条件の単回時間はすべて以前より短いが、cache/負荷を統制した反復実験ではないため速度改善を保証しない。時間にはオフライン状態比較helperを含めない。通信はrequest/replyのCandid合算でHTTP/CBOR/署名を含めない。

最大handler命令4,869,416,555（MLP）、全query実上限内で完走。統合Deltaの12境界caseの最大4,361,168,668命令。handler counterにはCDKのCandid decode/encodeを含めない。最大終端heap観測は4,122,083,328 byteで4 GiB設定以内だが、実際の一時peakを保証するものではない。固定weight cache4,065,416,192 byteとRoPE131,072 byteは以前と同じ。

今回の固定準備は721 update、15,041,482,761命令、214.754秒、request47,473/reply14,927,506 byte。別にpack status1 query、cache status2 query、認証module read2回。全5検証のcache確認2 query、認証module read2回も推論queryとは分けて記録。

採用Wasm SHA256: `8eb3e3b93a67f8e26bde3ce3e98806783553c680000f2c644619dc36a611b20b`。pack SHA256: `364e3aa3f69a61a69e6df62e3e2a71d4f55457f1a8e1c16934aefc6130f04c84`。全モデルproofの49 hashが現在のsource/保存archiveと一致し、実行前後の認証module hash・固定cacheも一致。

prefix状態比較helper SHA256: `20a8120378c06d43b06fb183921471b2402337bc2b078fe28d51c2ab8e714617`。検証専用helperのソースも49 hashに含める。

raw証拠: `artifacts/delta-full-log-v1-cache-checks/report.json`、各`artifacts/delta-full-log-v1-*/report.json`、比較`docs/delta-full-log-v1-*-results.json`、集計`docs/delta-full-log-v1-summary.json`。境界12 query比較は`artifacts/delta-full-log/partial/report.json`（checkerを含め50 source hash）。native4 case比較とsource/Wasm/archiveも同directoryへ保存。生成物はignore対象。

元BF16公式実装との既存差と、重大変更gold=yes/model=noの誤判定は残る。型安全な出力と判断の正しさは別に評価する。今回のビット一致は以前からの劣化を検出しないことを示し、一般精度改善を示さない。

## 次の削減対象

主67 queryの内訳はDelta24、MLP31、Attention8、embed/norm/終端MLP/readout等4。handler命令はMLP150,307,111,120（53.68%）、Delta97,439,705,183（34.80%）、Attention31,887,960,801（11.39%）。さらに17 queryを減らすには演算削減と分割境界の再配置が必要。handler合計だけで5B×50を超えるため、統合だけで50へ到達したとは主張できない。次は、BF16の有限入力が65,280種類に限られることを使い、毎要素のsigmoid/SiLUのexp・除算を固定準備の完全なlookupへ置き換える案を検証する。F32入力では元の演算へfallbackし、Wasm自身で表を生成して元の丸めbitを保持する。本検証時点ではlookup最適化は未実装だった。後続の[固定活性化表版](PREPARED_ACTIVATION.md)で実装し、全5条件の比較後に採用した。

## 実装

`delta_full_log_integer`は入力量子化・元F32 QKV/Z LoRA A積・両gateを一度だけ計算する。8192行のQKV、4096行のZ、全8192 conv channel、32 head再帰、2560行の出力投影を同じ通常queryへ接続する。conv重みとnormも全headで共有し、旧2 queryのprepared input・中間gated output・状態の返信／再送／復元を除く。

継続入力には従来のF32 Delta状態の代わりに、prefixのK、元F32 innovation、元F32 decayを含める。状態はquery内で可逆復元する。[DELTA_INNOVATION_LOG.md](DELTA_INNOVATION_LOG.md)の144通常queryによる全24層比較と通信上限の測定を前提にする。

prefixでは再帰の更新量をその場で記録する。元のSIMD recurrenceを基にした`delta_recorded_simd.rs`が、元のinnovationを保存しながら同じ状態とQ出力を計算する。新しい内積や2回目のrecurrenceは行わない。隣接value headのQ/K正規化を共有し、Kは元BF16丸めを、innovationとdecayは元F32 bitを維持する。INT8 base block256、元F32 adapter/readout、calibrationは変更しない。

通常queryで質問状態を保持せず、クライアントがprefixの`delta_log`とconv履歴を保持する。固定重みとRoPE準備にだけupdateを用いる。

## 境界とcache形式

新opのdimsは`[現在token数,32,prefix log token数,keep]`。現在token数1〜90、prefix logは最大132 token、keepは0/1。keep=1は新規prefixだけ（既存log token数0）に限定する。900,000 logical floatsと2 MB frameは変更しない。keep=1の返信は事前の保守的byte boundで検証し、大きすぎるprefixを重み取得前に拒否する。今回のprefixは45 token。

クライアントのprefix cache version2はDelta層に`conv`とflatな`delta_log`を持つ。version1の従来F32 cacheも旧経路で利用できる。layerの状態形式・version・設定が一致しない場合は拒否する。logを使う継続は90 token以内が候補の対応範囲。prefixなし132 tokenは従来の分割経路を維持する。token数やlayerを省いてquery数を減らすものではない。

クライアントはprefix準備前に、最大16,384 byteのheader、bitmap、BF16出力/K、元F32 innovation/decayを含む保守的返信サイズと900,000 logical floatsを確認する。`bf16-exact`では72 token、`bf16-block256-exact-v1`では75 tokenまでがlog保存対象となり、その境界を超えるprefixは全Delta層で従来の分割経路へfallbackする。新規の終端推論（keep=0）の90 token上限は変えない。

cache versionは保存済み全Delta層の状態形式から決定する。従来状態はversion1、logはversion2で、形式混在は拒否する。version1の継続では`--fuse-delta-full-log`を指定しても元の非ゼロF32状態を使う従来経路を選ぶ。version2を同フラグなしで読む場合は引き続き拒否する。session/reportの`delta_full_log_requested`、`delta_full_log_effective`、`delta_state_version`に要求と実際の経路・形式を記録し、再開時の一致確認にも含める。修正後はgraph hashが変わるため、prefix cacheを新しい実行directoryで再生成する。既存cacheの移行や書き換えは行わない。

## 検証の分離

Rust nativeでは従来nativeの2 queryを接続して比較し、Wasmでは保存済み従来Wasm出力と比較する。gateのexp等に既存のnative/Wasm差があるため、backendをまたいだbit一致で性能や精度を評価しない。

全モデル比較の際は、`compare_compact_heads.py`がprefixの新logを`delta_log_verify`でF32状態へ復元し、従来Wasmの保存状態と全bitを比較する。このhelperは検証専用であり、運用クライアント・readout・後続layerへ計算結果を渡さない。追加の推論queryも発行しない。helper SHA256と復元層数を比較結果に記録する。実際に推論で使う状態はlogのまま。

候補core 49 source hashを`artifacts/delta-full-log/source-hashes.json`へ固定し、source archiveも保存。生成Wasm、pack、実験ログ、reply等はignore対象。

Rust runtime 63、INT8 integration 3、F32 integration 2、compile-fail 3 tests通過。クライアント11 tests通過。45/80/87/89 tokenのlayer0のnative境界比較ではhidden/conv/prefix状態がbit一致。全モデルの全5条件と実query上限も上記の通り検証した。

## 手順

buildは既存の全最適化featureへ`experimental-delta-full-log`を追加する。runtimeのnew opはこのfeatureだけで有効となり、canisterではByteBufとprojected Deltaも同時に有効にする。

```sh
.venv/bin/python scripts/check_delta_full_log.py \
  --canister 4caro-hl777-77775-aaaba-cai \
  --wasm artifacts/delta-full-log/full.wasm \
  --directory artifacts/delta-full-log/partial --layers 0,22,30

.venv/bin/python scripts/validate_prepared_weights.py \
  --canister 4caro-hl777-77775-aaaba-cai \
  --run-name delta-full-log-v1 --baseline column32-v1 \
  --wasm artifacts/delta-full-log/full.wasm \
  --preparation artifacts/delta-full-log/preparation \
  --reuse-projection-inputs --frame-checksum blake3 \
  --fuse-attention --fuse-attention-full \
  --fuse-delta-projected --fuse-delta-finish \
  --fuse-delta-full-log --fuse-mlp-pipeline --fuse-mlp-full \
  --require-prepared-rope
```
