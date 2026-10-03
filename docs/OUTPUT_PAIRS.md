# INT8固定配置とquery内入力の再利用

2026-10-03。毎回の処理を準備時または同じquery内の一度だけの処理へ移し、全5条件の通常queryで検証したV2を命令数削減の採用版とした。主問題のhandler命令は2.3933%減。query数・通信量は同じで、50/32 queryは未達。単回の時間は主問題・prefixなし等で増えており、速度改善を断定しない。

## 採用済みV2の全モデル実測

[固定活性化表版](PREPARED_ACTIVATION.md)との比較。全5条件の保持hidden・状態配列がbit一致し、判断・raw logits・確率も一致。prefixのlog自体と独立した検証用復元も一致。失敗attempt/replay0。実行前後の認証module hash、固定cache、55 core source hashを確認した。

| 条件 | 通常query | handler命令: 元→V2 | 削減率 | Candid byte | 単回秒: 元→V2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix45 | 66 | 147,852,713,772→136,863,422,956 | 7.4326% | 71,105,691 | 17.359→17.623 |
| 主 suffix87 / 全132 | 67 | 272,452,026,264→265,931,336,448 | 2.3933% | 113,709,005 | 30.549→33.596 |
| 情報不足 suffix80 / 全125 | 67 | 252,882,014,904→244,255,114,705 | 3.4114% | 106,668,202 | 28.559→28.092 |
| 重大変更 suffix89 / 全134 | 98 | 298,608,130,562→275,371,260,925 | 7.7817% | 197,070,502 | 33.959→35.770 |
| prefixなし132 | 292 | 436,125,574,914→405,740,371,751 | 6.9671% | 580,229,781 | 55.562→71.551 |

初回prefix+主問題は133 query、402,794,759,404 handler命令（元420,304,740,036から4.1660%減）、184,814,696 Candid byte、今回単回51.219秒（元47.907秒）。handler counterはCDK Candid decode/encodeを含まず、通信はHTTP/CBOR/署名を含まない。反復・負荷統制をしておらず、この測定で時間短縮は確認していない。

主問題の最大queryは4,723,869,983→4,591,163,261命令。全5条件の最大終端heap観測は4,123,656,192 byteで4 GiB内。一時peakを保証する値ではない。cacheの重みpayloadは元と同じ4,065,416,192 byte、うち変換済みweight byteは3,565,158,400。RoPE131,072 byteと固定活性化表1,048,576 byteは別計上。

今回の一度限りの準備は721 owner update、60,994,167,929 handler命令、247.593秒、Candid request47,473/reply14,949,136 byte。元の準備15,077,999,700命令より45,916,168,229命令増。準備1回＋prefix1回＋同じ主問題N回のhandler命令合計ではN=6から新方式が少ない。この比較はupdate/queryの料金比較ではない。pack/cache statusと認証module readは推論query数とは別に記録する。

採用canister `4caro-hl777-77775-aaaba-cai`、Wasm SHA256 `36c04a57e9501eb9d32163c92d7725171191fa46a45a880ad03465993fceed73`、7,542,462 byte。INT8 block256・元F32 adapter/readout・tokenizer/calibration・モデル組を維持した。重大変更gold=yes/model=noという既存の見逃しと公式BF16との差は残る。型安全な出力と判断精度を分けて評価し、bit一致を一般的な精度改善と扱わない。

証拠は`artifacts/output-pairs-v2-cache-checks/report.json`、各`artifacts/output-pairs-v2-*/report.json`、`docs/output-pairs-v2-summary.json`と各比較JSON。`artifacts/output-pairs/v2-validated-source.zip`に55 core source＋新しいintegration testを保存し、core/test hashを別記録した。生成物はGit ignore対象。Layaは変更していない。

## 繰り返し処理の除去

対象の200 INT8 tensor、3,565,158,400 weight byteを、準備updateでK4/2出力の配置へ並べ替える。元INT8値・row scale・payload容量を維持し、元配置の全コピーを同時保持しない。固定cacheの合計は従来と同じ4,065,416,192 byteを期待する。質問に依存する中間状態は従来どおりクライアントが保持し、通常queryで進める。

量子化済み入力の4要素を2組へ複製する処理は、変更不能な`QuantizedRows`の寿命内で`OnceCell`へ一度だけ保存する。同じ入力を使う投影や出力行tileはこれを共有する。実際のtoken行だけを用意し、87 tokenを88行へ、89 tokenを96行へ埋めて余分な積和を行う処理を避ける。45は45、80は40+40、87は44+43、89は45+44、132は44+44+44で計算する。入力の精度は変えない。

固定配置のreaderは整数kernelへ型付きのviewを渡す。元byte配列への復元は必要になった場合だけ行い、固定配置を使える投影では復元を呼ばない。scale footerは元の連続byteを直接参照する。旧byte reader・未対応の行範囲には元配置を正確に復元するfallbackを残す。scaleの適用とblock256のF32加算順を維持する。

## これまでの検証

V1時点でruntimeの67 unit test、5 integration test、3 compile-fail test、固定cacheの4 testとWasm type checkが通過。符号付き極値、複数blockのscale、1/7/8/32/45/48/64/80/81/87/88/89/132 tokenで元kernelとのbit一致を確認した。readerのtestでは、整数fast pathが元byte復元を呼ぶとpanicするreaderを使い、非zero開始行も確認する。未対応の奇数開始行には元kernelへfallbackする。V2修正後は該当3 unit testと、512/2560列・複数block・8/16/24/40/56出力行の新しいintegration testが通過した。

採用条件は全5条件の通常query実測で、命令数・query数・通信量・終端heap、保持hidden/state/判断/確率の一致を確認すること。nativeとWasmの既存の数学関数差は、独立した検証用`delta_log_verify`で同じ方式へ復元して比較する。検証用復元値を推論入力に使わない。

最初の全モデル候補V1（module `aee7028d…`）では全5条件の出力・状態がbit一致したが、prefixなし132 tokenで不採用となった。主問題は272,452,026,264→265,915,967,624命令、67 queryのまま、prefixは147,852,713,772→136,855,190,132命令。しかしclientの分割行数は8の倍数で、固定配置のviewは32の倍数だけを許していた。132-token経路の一部で元byteへ復元するfallbackが走り、31失敗attempt、成功354 query・586,461,725,069命令へ悪化した。失敗attemptの消費命令はこの成功query集計へ含めない。検証scriptは失敗logを検出して採用判定を拒否した。`artifacts/output-pairs/v1-rejected.json`とexact V1 source/Wasmを保存した。

V2は8出力行単位の分割viewを許し、32未満の最後の出力tileだけ新配置のまま小さなscratchへ0詰めする。元byte配置へ全体を読み戻さない。実token行のpaddingは追加しない。元の分割境界を変えず、非zeroの開始行8/24・末尾8/24行などをnative reader testへ追加し、bit一致と元byte復元を呼ばないことを確認した。V2の全5条件のWasm実測は上表のとおり完了し、この失敗を解消した。V2は本番候補と同じruntime featureで67 unit test、6 integration test、3 compile-fail testも通過した。

実装は`crates/imajev-runtime/src/output_pairs.rs`、生成kernelは`output_pairs_simd.rs`、生成scriptは`scripts/generate_output_pairs.py`。cache接続は`canisters/inference/src/weight_cache.rs`。featureは`experimental-prepared-output-pairs`、準備・検証scriptの必須確認は`--require-output-pairs`。55 source hashとexact source archiveは`artifacts/output-pairs/`へ保存した。生成物はGit ignore対象。

## ローカル実測手順

既存のImajevローカルcanisterだけを候補Wasmへupgradeし、sealed packを維持したままheap cacheを準備し直す。Layaのソース・Git・稼働canisterを変更しない。

```sh
.venv/bin/python scripts/prepare_weight_cache.py \
  --canister 4caro-hl777-77775-aaaba-cai \
  --wasm artifacts/output-pairs/v2-full.wasm \
  --directory artifacts/output-pairs/v2-preparation \
  --include-f32 --require-prepared-rope --require-prepared-activation \
  --require-output-pairs

.venv/bin/python scripts/validate_prepared_weights.py \
  --canister 4caro-hl777-77775-aaaba-cai \
  --run-name output-pairs-v2 --baseline prepared-activation-v1 \
  --wasm artifacts/output-pairs/v2-full.wasm \
  --preparation artifacts/output-pairs/v2-preparation \
  --reuse-projection-inputs --frame-checksum blake3 \
  --fuse-attention --fuse-attention-full --fuse-delta-projected \
  --fuse-delta-finish --fuse-delta-full-log \
  --fuse-mlp-pipeline --fuse-mlp-full \
  --require-prepared-rope --require-prepared-activation --require-output-pairs
```

32/50 queryへの到達は、この実装だけから推測しない。5B命令/query、通信frameと4 GiBの制約を維持し、実測の合計命令と分割境界から判断する。

## 残るボトルネック

主問題はMLP31 query・142,293,974,009命令（全体の53.51%）、Delta24 query・92,386,643,043命令、Attention8 query・30,965,590,940命令。主の265.931B handler命令は5B×50を超える。50 queryにはさらに少なくとも5.99%、32 queryには39.83%のhandler削減が必要で、CDKと分割境界の余裕も別途必要。MLP89 tokenのprepare/down合計は約4.8Bだが、現行のfull MLP上限87を変更した実queryの検証はまだ行っていない。単純に上限を上げた結果としてquery削減を計上しない。F32 LoRA入力の並べ替えを同一入力の複数投影で共有する案も、未実測の候補として扱う。
