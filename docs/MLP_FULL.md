現行の採用結果は [BYTE_BUFFER.md](BYTE_BUFFER.md)。以下は元の80-token MLP full v2の記録であり、検証ソースarchiveを保存している。

# MLP の準備状態を query 内で down へ直接渡す

2026-10-03。全Attention版の保存済みcounterで、MLP準備＋down/normの合計はprefix45最大2,748,503,090、情報不足80最大4,607,854,918命令だった。主87は最大5,189,269,065、最大変更89は5,421,276,878で5Bを超える。全MLPの１query化は80 token以下に限定する。これらは別handlerの合計であり、融合後の上限を実queryで確認する。

`mlp_full_integer` は hidden と Attention出力を受け取り、residual/norm、gate/up＋SwiGLU、元のblock256入力量子化と元F32 LoRA A、INT8 down＋LoRA B、residualと次のnormを続けて実行する。旧２query経路の準備済みINT8値を巨大F32配列へ展開・返信・再送・復元する境界をなくし、privateな `PreparedMlp` 内のopaque `QuantizedRows` を直接使用する。

元のINT8重み、activation block256、BF16境界、F32 LoRA A/B、scale2と積和順序、readoutとcalibrationは変更しない。内部準備関数を旧２query経路でも共有し、80 token超は旧codecで同じ準備状態を返信する。

通常のlossless BF16 codecを使い、要求dims `[tokens,2560]`、scalars `[2,1e-6]`、tensorは当該層のpost-attention norm、auxは次層のinput norm。prefix最終層31では `model.language_model.norm.weight` を使用する。１query経路のみ最終層も許可し、旧２query codecのlayer0–30制限は維持する。全32層の実行、1–80 tokenに適用し、終端の専用MLP経路は維持する。87/89 tokenは従来２query、prefixなし132 tokenは従来分割。

通常float900K/frame2MB/query5B/heap4GiBの上限は変えない。中間状態をcanisterへ永続化せず、query終了時のhidden/normalizedをクライアントへ返す。固定重みとRoPEだけを準備updateで一度用意する。

native runtime60 tests、INT8 integration2、F32 matrix integration2、compile-fail doctest3が通過した。旧codec/final layerの適用範囲、不正token数・欠損weight・非有限入力を検査した。新client2 testsと旧pipeline2 testsで最終normの選択、形状、codec維持、設定前提を確認。`artifacts/mlp-full/{tests,client-tests}.log`。

build時 `experimental-mlp-full`、client実行時 `--fuse-mlp-full` を追加する。既存flagも維持する。実queryと全５条件の比較後、下記v2を採用した。50/32 queryは未達。Laya、mainnet、Git remoteは変更していない。生成物はgitignore対象。

## v1全層検証と修正

v1 module `ef97d3ad984e02b5b47e14ed3d3248109b371af4530d323559ece3ebde67979d` は全５条件で保持hidden/state・判断・確率bit一致、失敗/replay0。prefix123→90 query、情報不足122→91 queryになった。ただし適用対象外の主問題で27,873,011命令（約0.0095%）増えたため、この版は採用しない。準備関数が正規化配列の残差部分をto_vecで複製し、さらにwire配列へコピーしていた。要求metadataの複製も旧２queryには不要だった。

v2は元の配列をtruncateして残差部分を保持し、旧２queryでのコピーを従来と同じ１回に戻す。内部準備の共通型にはrequestを含めず、１query経路でだけ束縛済みrequestを作る。v1の検証済みsource43 hash/archiveは `artifacts/mlp-full/v1-source-hashes.json` / `v1-source.zip`、Wasm `v1.wasm`、全層記録 `artifacts/mlp-full-v1-*`。v2の全層検証結果は下記。

## v2全層比較と採用

| 条件 | query（旧→新） | handler命令 | 旧版との差（増加は＋） | Candid bytes | 実測秒 | 最大query命令 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix | 123→90 | 156,348,756,976 | -1,603,635,918 | 145,902,403 | 20.1592 | 2,699,508,850 |
| 617 | 122→122 | 293,581,856,802 | +28,303 | 262,955,428 | 35.2185 | 4,582,737,601 |
| insufficient | 122→91 | 259,168,806,473 | -2,659,972,959 | 173,060,522 | 32.6211 | 4,522,112,981 |
| maximum | 122→122 | 305,919,377,719 | +29,791 | 267,767,242 | 54.1315 | 4,767,254,113 |
| normal | 292→292 | 447,475,675,371 | +0 | 580,229,781 | 60.2260 | 2,882,723,911 |

prefixは33 query、1,603,635,918命令、43,733,651 Candid bytes減。情報不足は31 query、2,659,972,959命令、73,127,239 bytes減。主問題は122 queryのままで初回prefix込みは90＋122＝212 query（旧245）。主問題は28,303命令（約0.00001%）、最大変更は29,791命令の微増が残る。v1で生じた余分なコピーによる約2787万命令の増加は除去したが、適用対象外の命令も減ったとは主張しない。prefixなし132 tokenはquery/命令/通信すべて同じ。50/32 queryは未達。

全５実行で直前の全Attention版と保持hidden/state・判断・確率がbit一致、失敗/replay0。prefixは32層の全token hiddenと72状態配列、質問は31層の全token hidden＋最終層の最後のtokenと48保持状態配列を比較。全32層を実行し、専用終端経路は維持した。共通prefix45＋主suffix87、情報不足45＋80、最大変更45＋89、prefixなし132 tokenを維持。公式BF16との差と最大変更gold=yesをnoとする既存誤判定は残る。型安全性と判断精度を分け、一般的な精度改善は主張しない。

最大MLP queryは情報不足で4,522,112,981命令、全体最大はAttentionの4,767,254,113命令。最大観測heap4,119,986,176 bytesは旧版と同じ。query5B/heap4GiB/frame2MB/通常float900Kを維持。counterはCDK Candid encode/decode、通信はHTTP/CBOR/signatureを除く。heapはquery終端page数で瞬間ピークではない。時間はquery cache未制御の単回測定で、最大変更・prefixなしは遅くなったため速度改善を保証しない。

v1の境界診断は５成功通常query・４不正通常query拒否・認証module read2、記録 `artifacts/mlp-full/partial/report.json`。v2は全層検証で全適用層・最終normと旧分割経路を検証した。native60 tests・2 INT8 integration・2 F32 matrix integration・3 compile-fail doctestを再実行し通過、`artifacts/mlp-full/v2-tests.log`。

固定721 tensor・4,065,416,192 bytesとRoPE131,072 bytesのv2準備は721 update、15,041,482,761命令、234.9709秒、request47,473/reply14,927,506 bytes。準備時pack status1/cache status2 query・認証read2、全層前後cache status2 query・認証read2は推論とは別計上。質問の中間状態をcanisterへ永続化しない。

採用module `4323859e9b17b8d54529fa033ff4a5cd2ee1adad0f03fca400319a88ee8089ce`、専用local canister `4caro-hl777-77775-aaaba-cai`、Wasm `artifacts/mlp-full/full.wasm`。認証module/cache bookendsと43実装hashは `artifacts/mlp-full-v2-cache-checks/report.json`。保存hash・現在ソース・検証source archiveの一致を再確認済み。`artifacts/mlp-full/source-hashes.json` / `validated-source.zip`、全層記録 `artifacts/mlp-full-v2-*`、比較 `docs/mlp-full-v2-summary.json`。生成物はgitignore対象。

## 再実行

```sh
cargo build --offline --release --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-projection-reuse,experimental-full-weight-cache,experimental-blake3,experimental-attention-fusion,experimental-attention-full,experimental-delta-projected,experimental-prepared-rope,experimental-column16-token48,experimental-delta-finish,experimental-matrix-tail,experimental-mlp-pipeline,experimental-mlp-full,experimental-dot-scale
# 専用local canisterへupgrade後、固定モデルだけを一度準備:
.venv/bin/python scripts/prepare_weight_cache.py --canister <local-id> \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --directory artifacts/new-mlp-full-preparation --include-f32 --require-prepared-rope
.venv/bin/python scripts/validate_prepared_weights.py --canister <local-id> \
  --run-name new-mlp-full --baseline attention-full-v1 \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --preparation artifacts/new-mlp-full-preparation --reuse-projection-inputs \
  --frame-checksum blake3 --fuse-attention --fuse-attention-full \
  --fuse-delta-projected --fuse-delta-finish --fuse-mlp-pipeline --fuse-mlp-full --require-prepared-rope
```

主handler合計を5Bで割っても59 query、初回prefix込みは90 queryが必要。CDK・通信・依存関係を含む下限とは別に扱う。主87 tokenのMLPは旧２query合計5.189Bで１query化できず、次は整数kernelの反復処理を減らす必要がある。87 tokenはpadding込み88を48＋32＋8 tileに分け、同じweightを３回展開する。44＋44などの均等なtileと、出力行間の入力共有拡大を次の計測候補にする。これは未実装・未測定の候補で、削減率や１query化達成は主張しない。
