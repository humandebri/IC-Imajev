2026-10-03：本書は前段階の記録。現行の採用版は [MLP pipeline](MLP_PIPELINE.md) を追加し、主138・初回277 queryへ削減した。

# F32 LoRAの端数tokenで重みを共有する

2026-10-02。採用済みINT8 kernelの48-token共有に続き、元F32 LoRAの固定重みロードを減らす候補を実装した。現行の32-token groupの後に残るtokenを16/8/4 groupへまとめ、最後の1〜3 tokenは独立したSIMD laneに入れる。追加laneはゼロで初期化し、返信へ含めない。各出力の列の積和順序と、乗算・加算の分離を維持する。BF16境界・量子化・LoRA A/B・scaleは変えない。

`experimental-matrix-tail`は直接比較API、`experimental-adopt-matrix-tail`は通常の`matrix`への接続を有効にする。元kernelは`matrix_baseline`として同じWasm内に保持する。全層検証後、下記の形状選択付きWasmを採用した。

## native検証

1/3/4/5/7/8/15/16/17/31/32/33/45/63/64/65/80/87/89/127/128/132 token、1/15/16/17/64出力行、1/7/64列で、符号付きゼロ・極小値・有限値を含むscalar oracleと全出力bit一致した。不正・overflow寸法は配列確保前に拒否した。`artifacts/matrix-tail/tests.log`、2 tests。

## 専用Wasmのv1診断

canister `5pova-id777-77775-aaagq-cai`、module `b299149d485b1fa39eb9898a73e6191f195ae342db1412cffd66b22036425cb6`。第3層gate/downの元F32 LoRA A/Bの4形状を、5条件の保存済み入力で測る。gate Aは保存済みAttention正規化入力を使う代表shape診断であり、同一層MLPの実入力とは扱わない。B入力は独立native scalarで算出したA積。normal gate Bは900K出力上限に従い4096行、normal downは元の実行と同じ最初の96-token chunkを使う。

28固定重み準備update、40通常query、module status read2 update。重みは4形状ごとに準備し、元/候補のqueryは同じcache・入力を使う。全20条件で両Wasmの全出力digestが独立native scalarと一致。nativeは元/候補の全出力値もbit比較した。counterはF32 matrix本体で、入力復元・digest・Candid encodingを除く。全層の精度・query数・時間の改善へ外挿しない。

| 条件 | gate A | gate B | down A | down B |
| --- | ---: | ---: | ---: | ---: |
| prefix45 | −15.1131% | −17.8090% | −15.1616% | −17.7128% |
| 617・87 | −26.3321% | −27.6660% | −26.4804% | −27.5089% |
| insufficient80 | +2.3297% | −1.3357% | +2.2830% | −1.2134% |
| maximum89 | −8.4968% | −11.0100% | −8.5768% | −10.8896% |
| normal gate132/down96 | +4.4598% | −0.6799% | +4.2872% | −0.7816% |

A投影で一部が増えたためv1を全面採用しない。入力配置helperが全groupでpadding用分岐を通り、上限の広いbufferへのindex計算も繰り返していた。v2はfull groupとpadding groupをconst parameterで分離し、入力/packedの範囲を先に絞る。v2は再測定後に採否を決める。

v1記録`artifacts/matrix-tail/check/report.json`、30ソースhash/archiveは`v1-source-hashes.json`/`v1-source.zip`、Wasm`v1.wasm`。候補はローカル診断だけで、mainnet・Laya環境を変更しない。生成物はgitignore対象。50 queryは未達。

## v2の再測定と形状ごとの選択

| 条件 | gate A | gate B | down A | down B |
| --- | ---: | ---: | ---: | ---: |
| prefix | -18.2417% | -18.5911% | -18.2885% | -18.5461% |
| 617 | -32.2200% | -28.0790% | -32.3863% | -28.0186% |
| insufficient | -5.5394% | -1.3847% | -5.6391% | -1.3893% |
| maximum | -14.7322% | -11.0190% | -14.8445% | -11.0015% |
| normal | +0.7504% | -1.1002% | +1.1074% | -0.8517% |

v2も全20条件・40通常queryで独立scalarの全出力digestと一致。input配置を範囲の狭いsliceにし、full/padded groupをconst parameterで分けたことでA80の回帰は解消した。ただしA132/96には0.75/1.11%増が残る。

全層候補の`matrix`は、出力64行以下・token数が4の倍数・主tileの端数が0/4の場合は従来kernelを使う。これらは従来から同じgroup分割をしており、input packingの比率が高い。主tileは従来と同じ、128 token以上かつ64出力行以上なら64、それ以外の32 token以上かつ64出力行以上なら32、残りは16。その他はv2のgrouped kernelを使う。値・回答・ケース名を条件にしない。

診断v2は未選択のkernelを測った記録で、selector自体のWasm測定とは扱わない。selectorとfull group/末尾paddingをnative scalar oracleで検査し、全層Wasm比較を次に行う。`experimental-matrix-tail`をcanister buildへ加える。query count・全層命令・判断精度の改善は全層結果が出るまで未証明。

v2 module `f5c4ef3f3e7ba0b5237b891ad9a514815a51cdd669a41696bf77026f39fe2234`、記録`artifacts/matrix-tail/v2-check/report.json`、v2ソース`v2-source.zip`/`v2-source-hashes.json`、Wasm`v2.wasm`。

## 全層Wasmの比較と採用

| 条件 | query | handler命令 | 削減命令 | 削減率 | Candid bytes | 実測秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix | 170 | 162,421,088,833 | 4,104,980,304 | 2.4651% | 237,812,108 | 41.8413 |
| 617 | 169 | 301,253,091,458 | 14,244,756,967 | 4.5150% | 354,192,982 | 47.1793 |
| insufficient | 169 | 269,218,648,714 | 834,922,317 | 0.3092% | 330,090,604 | 39.4505 |
| maximum | 169 | 314,639,254,393 | 4,769,352,877 | 1.4932% | 361,100,413 | 44.9474 |
| normal | 294 | 457,345,556,170 | 281,384,526 | 0.0615% | 580,794,410 | 63.2147 |

全5条件で保持hidden・状態・判断・確率が直前の48-token INT8版とbit一致し、失敗/replay0。prefixは全32層の全token hidden・72状態配列、質問は31層の全token hidden＋終端層の最後のtoken・48保持状態配列を比較した。全32層・全tokenを処理し、終端の不要状態省略は従来仕様。公式BF16との差と、最大変更gold=yesをnoとする既存誤判定は残る。型安全性と判断精度を区別する。

主問題は14,244,756,967命令（4.5150%）減。prefixは4,104,980,304命令（2.4651%）減。主169、初回prefix込み339、prefixなし294 queryと通信量は同じ。50/32未達。主handler合計だけでも5B/queryで少なくとも61 query、初回prefix込みでは93 queryが必要であり、実際の依存関係・通信・CDKを含む下限は別に必要になる。

最大queryは全条件3,780,687,864命令、最大観測heap4,119,986,176 bytes。query5B/heap4GiB/通常float codec logical900K/frame2MB上限を維持した。counterはCDK Candid encode/decodeを、通信はHTTP/CBOR/signatureを除外。heapは終端page数で瞬間ピークではない。時間はquery cache未制御の単回測定。prefix・主・情報不足は前回より遅く、最大変更・prefixなしは速かったため、速度改善を保証しない。

全層構成の56 runtime tests、2 column16 integration tests、2 matrix-tail integration tests、2 compile-fail doctestsが通過。selector・groupの境界・padding・符号付きゼロ・極小値・不正寸法をnativeで検査した。記録`artifacts/matrix-tail/full-tests.log`。

固定721 tensor・4,065,416,192 bytes、RoPE131,072 bytesをfresh heapへmanifest順で準備。準備は721 update、15,041,482,761命令、232.8894秒、Candid request47,473/reply14,927,506 bytes。準備時pack status1/cache status2 query・認証read2と、全層前後cache status2 query・認証read2は推論と別に記録した。モデル変更/upgrade時の固定準備だけupdate、推論は通常query、中間状態はclient-heldを維持する。Laya環境は変更していない。

採用module `2fc01ab1a0e0e6fae1a1410890970ad5b23c6fc6696b9e4219214a071182e665`、専用local canister `4caro-hl777-77775-aaaba-cai`、Wasm `artifacts/matrix-tail/full.wasm`。認証module/cache bookendsと39実装hashは`artifacts/matrix-tail-v1-cache-checks/report.json`、検証時ソース`artifacts/matrix-tail/validated-source.zip`。全層記録`artifacts/matrix-tail-v1-*`、比較`docs/matrix-tail-v1-summary.json`。生成物はgitignore対象。

診断Wasmはglobal `RUSTFLAGS='-C target-feature=+simd128'`、全層Wasmは通常のbuildで既存のSIMD関数属性を用いた。診断の削減率を全層へ外挿せず、上表の実測を採用判断に使った。

```sh
cargo build --offline --release --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-projection-reuse,experimental-full-weight-cache,experimental-blake3,experimental-attention-fusion,experimental-delta-projected,experimental-prepared-rope,experimental-column16-token48,experimental-delta-finish,experimental-matrix-tail
# 専用local canisterへupgrade後:
.venv/bin/python scripts/prepare_weight_cache.py --canister <local-id> \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --directory artifacts/new-matrix-tail-preparation --include-f32 --require-prepared-rope
.venv/bin/python scripts/validate_prepared_weights.py --canister <local-id> \
  --run-name new-matrix-tail --baseline column16-token48-v1 \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --preparation artifacts/new-matrix-tail-preparation --reuse-projection-inputs \
  --frame-checksum blake3 --fuse-attention --fuse-delta-projected \
  --fuse-delta-finish --require-prepared-rope
```

## 次のMLP query統合に向けた確認

`scripts/size_prepared_mlp.py`は4条件×31層の実packet形状から、準備済みのINT8 q値・元F32 scales/LoRA A積・BF16残差を独立したtyped segmentへ入れる保守的なframe上限を計算した。主87 tokenは1,298,472 bytes、最大変更89は1,327,944 bytes。各segmentは既存の個別900K上限内だが、全体をflat floatへ復元すると900Kを超える。通常codecの上限を上げず、専用の検証済みdecoderとopaque QuantizedRowsを使う実装が必要になる。新しい量子化やhost推論は行わない。

現在のMLP gate/up最大counterは主3,610,868,768、down＋次層norm1,749,362,055。全体を単に1 queryへ連結すると最大5,360,230,823で5Bを超える。次は最初のqueryでdown入力と元A積を一度だけ準備し、後のqueryでdownと残差/normを完了する案を検証する。prefixは全体2,843,737,473、情報不足は4,771,545,104で5B未満だが、実際の統合query counterとCandid負荷の測定が必要。最大変更は5,614,017,917で単純連結不可。

これらは容量の保守的計算と既存handlerの合計で、統合kernelの実測・新query数の達成証明ではない。記録`artifacts/matrix-tail/prepared-mlp-sizing.json`。50 queryは引き続き未達。
