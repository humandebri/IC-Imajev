2026-10-03：本書はMLP pipeline採用時の記録。現行は [整数dot/scale融合](DOT_SCALE_FUSION.md) を追加し、同じquery数でさらに命令を削減した。

# MLP の中間表現を次の通常 query へ持ち越す

2026-10-03。毎 query の再処理と通信を減らすため、MLP の residual/norm・gate/up・down・次層 norm を２ query の pipeline にした。固定重みと RoPE は準備 update で一度だけ用意し、質問依存の状態はクライアントが保持する。

第１ query は residual 加算、正規化、元の gate/up と SwiGLU を実行し、down 用の元の block256 INT8 入力・scale と元 F32 LoRA A の結果を準備する。第２ query は準備状態を型付きで復元し、元の INT8 down と LoRA B、residual 加算、次層正規化を実行する。元の SwiGLU BF16 配列を返信して次 query で展開・全域有限値走査・量子化する境界をなくし、INT8 整数列を直接使用する。量子化と A 積は計算位置を移したもので、元の１回を０回にしたとは主張しない。

INT8 重み、activation block256、BF16 境界、LoRA の F32 精度・scale2、積和順序、専用 readout と calibration は変更しない。画像や host inference は追加しない。

## 型と適用範囲

`experimental-mlp-pipeline` と `--fuse-mlp-pipeline` を追加する。32層の全層実行で、layer0–30、1–89 token にのみ適用する。layer31 と132 tokenの prefix なし実行は従来経路を使用する。MLP/norm 融合と INT8 経路が前提。

専用 codec `mlp-down-state-exact-v1` は BF16 residual、INT8 qx、F32 scale、F32 A 積を独立 segment で扱う。最大89 tokenでは論理値数1,056,964になるため、専用型に限り通常 float 配列の900K制限から別扱いにする。各 segment は900K以内、frame全体2 MB以内。通常 codec の制限は変えない。第２ query は qx を巨大 F32 配列へ展開せず、opaque な `QuantizedRows` を直接使用する。要求の model/pack/hash と metadata は既存 envelope で検証し、layer/shape/scale/epsilon/方向・有限値・予約INT8値−128も検証する。クライアント保持状態の実行者由来を認証する機構は追加していない。

## 検証

native runtime 58 tests、INT8 kernel integration 2 tests、F32 matrix integration 2 tests、compile-fail doctest 3 tests、client 2 testsが通過。`artifacts/mlp-pipeline/tests.log`。

専用ローカルcanister `4caro-hl777-77775-aaaba-cai`、module `cbd1aac04872ce6675950a1641c5d5015da7199024fa93349d36270c07f0469e`。キャッシュ準備前の prefix45・主87・情報不足80・最大89 tokenのlayer0/30、８か所を従来Wasmの保存済み出力と比較し、全出力bit一致。16成功通常query、不正状態７通常queryを拒否、module確認read2回。最大4,154,424,303命令。`artifacts/mlp-pipeline/partial/report.json`。これは境界検証であり、全層比較や warm 性能改善の証明とは扱わない。

全５条件の全層比較を完了し、この実装を採用した。50/32 query は未達。

core41ソースhashは `artifacts/mlp-pipeline/source-hashes.json`、検証ソースは `validated-source.zip` に保存した。生成物はgitignore対象。Laya、mainnet、Git remoteは変更していない。

## 全層比較と採用

| 条件 | query（旧→新） | handler 命令 | 削減率 | Candid bytes | 実測秒 | 最大 query 命令 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix | 170→139 | 161,346,803,520 | 0.6614% | 198,872,822 | 23.5239 | 2,028,852,954 |
| 617 | 169→138 | 299,110,295,633 | 0.7113% | 278,946,124 | 39.3735 | 3,819,543,092 |
| insufficient | 169→138 | 267,297,632,957 | 0.7136% | 260,895,070 | 36.7413 | 3,404,337,610 |
| maximum | 169→138 | 312,446,664,202 | 0.6969% | 284,124,623 | 45.4339 | 3,996,383,975 |
| normal | 294→294 | 457,345,547,339 | 0.0000% | 580,794,410 | 82.3415 | 2,975,764,131 |

主問題では31 query（18.3432%）、2,142,795,825命令（0.7113%）、75,246,858 bytes（21.2446%）減。prefix139＋主138で初回277 query（旧339）。prefixなし132 tokenは適用範囲外なので294 query、通信量は同じ。8,831命令の差はwrapper/dispatch差の範囲であり、MLP演算削減とは扱わない。主問題は共通prefix45＋suffix87、情報不足45＋80、最大変更45＋89、cold132のtoken数を維持している。

全５実行で保持hidden/state・判断・確率が直前のmatrix-tail版とbit一致。prefixは全32層の全token hiddenと72状態配列、質問は31層の全token hiddenと最終層の最後のtoken、48保持状態配列を比較した。失敗/replay0。公式BF16との差や最大変更gold=yesをnoとする既存誤判定は残る。型安全な返信と判断精度は別に評価し、一般的な精度改善は主張しない。

最大観測heap4,119,986,176 bytesは旧版と同じ。query上限5B・heap4GiB・frame2MBを維持。counterはCDK Candid encode/decodeを除き、Candid通信はHTTP/CBOR/signatureを含まない。heapはquery終端のpage数で瞬間ピークではない。時間はquery cacheを制御していない単回測定で、最大変更・coldは遅くなったため速度改善を保証しない。

固定重み準備は721 update、15,041,482,761命令、234.3341秒、request47,473/reply14,927,506 bytes。cacheは721 tensor・4,065,416,192 bytesとRoPE131,072 bytes。準備のpack status1/cache status2 query・認証read2、全層比較前後のcache status2 query・認証read2は推論と別計上する。query中に質問の状態をcanisterへ永続化しない。

認証module/cache bookendsと41実装hashは `artifacts/mlp-pipeline-v1-cache-checks/report.json`。保存した41 hashとの一致を再確認済み。全層記録 `artifacts/mlp-pipeline-v1-*`、比較 `docs/mlp-pipeline-v1-summary.json`。各queryの命令・request/reply bytes・時間は各reportのqueriesと保存packetで確認できる。

## 実行手順

```sh
cargo build --offline --release --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-projection-reuse,experimental-full-weight-cache,experimental-blake3,experimental-attention-fusion,experimental-delta-projected,experimental-prepared-rope,experimental-column16-token48,experimental-delta-finish,experimental-matrix-tail,experimental-mlp-pipeline
# 専用local canisterへupgrade後、固定重みを一度だけ準備:
.venv/bin/python scripts/prepare_weight_cache.py --canister <local-id> \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --directory artifacts/new-mlp-pipeline-preparation --include-f32 --require-prepared-rope
.venv/bin/python scripts/validate_prepared_weights.py --canister <local-id> \
  --run-name new-mlp-pipeline --baseline matrix-tail-v1 \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --preparation artifacts/new-mlp-pipeline-preparation --reuse-projection-inputs \
  --frame-checksum blake3 --fuse-attention --fuse-delta-projected \
  --fuse-delta-finish --fuse-mlp-pipeline --require-prepared-rope
```

## 残る制約

主handlerの合計を単に5Bで割っても60 query、初回prefix込みは93 queryが必要。依存関係・通信・CDKを含む実際の下限は別であり、query統合だけで50回に達するとは扱わない。大きい整数dotと元F32 LoRAの積が残る主コストで、今回のboundary除去の削減率は約0.7%に留まる。次は同じ入力の検査や変換を内部で二重に行う箇所、固定重みのロード/展開をtoken/output間で共有できる箇所を測る。外部から届く状態の検証は必要なので、内部で検証済みと分かる型を通して重複だけを除く。
