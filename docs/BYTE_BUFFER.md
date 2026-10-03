2026-10-03後続：32出力で入力ロードを共有する版を採用し、87 token全MLPも1通常queryで全31層完走。主122→91 query、全5条件bit一致。現在の採用版は[COLUMN32.md](COLUMN32.md)。以下はByteBuf v3採用時の実測履歴。

# query の byte 復元と MLP 内部の重複処理を減らす

2026-10-03。`step`・`profile_step`・`decision` の入力と `Measurement.state` を、feature `experimental-byte-buffer` で `serde_bytes::ByteBuf` に替える。Candidの `vec nat8` / blob型は同じ。通常のvector要素復元の代わりに、Candidの `deserialize_byte_buf` が入力を一括コピーする。送信側の元Vecも固定primitiveの一括書き込みに対応していたため、送信全般が従来逐次だったとは主張しない。

固定しているCandid 0.10.35とserde_bytes 0.11.19の実装を確認した。nativeテストでは0/1/256/900,000 byte、全byte値の入力について、旧VecとCandid型・送信バイト列が一致し、旧バイト列をByteBufで復元できた。canisterの４テストが通過。Candid interface、query/update区分、owner認可とframe検証を維持する。

MLP fullの内部では、準備段階で検証したroot/次normと残差を非公開型 `VerifiedFinish` に引き継ぐ。後半で同じrequestをJSON化して再比較し、同じ固定11tensorを再走査する処理を省く。元のresidual/norm配列の容量を保持してdown結果を追加し、残差のcloneと再配置を省く。クライアントから受け取る分割状態は、この型を作れず従来のidentity・metadata・値検証を維持する。

元INT8重み、block256 activation量子化、積和順序、BF16境界、F32 LoRA、readout/calibrationは変更しない。全MLPは80 token以下、87/89は従来の２query、132は従来分割にする。query5B/frame2MB/通常float900K/heap4GiBを維持する。中間状態はクライアント保持、固定モデル準備だけupdate。

## 87 token統合の上限を実測

44＋44kernelだけの87-token全MLPは、主問題layer0でIC0522（single message 5B命令超過）。ByteBufを追加したv1はprefix３件・情報不足２件・主22層でbit一致して完走したが、主layer22で同じ上限超過。二重検証とコピーを除いたv2も、主22層でbit一致したがlayer22で上限超過した。主でのv2 handler削減は約110万〜140万命令/MLPで、全面統合には足りなかった。これらの87-token版は不採用。

記録は `artifacts/balanced44-mlp87/failure-report.json` と `artifacts/byte-buffer/{partial-failure-report,v2-partial-failure-report}.json`。対応Wasm・43 source hash/archive・ログを保存した。分割handler合計はCandid decode/encode等を除き、実query完走の保証にはならない。ByteBuf部分だけのdecode命令削減は単独計測していないため数値を断定しない。

## v3全層比較と採用

80-token上限へ戻したv3を採用。全５実行で元のMLP full v2と保持hidden/state・判断・確率がbit一致、失敗/replay0。prefixは32層の全token hiddenと72状態配列、質問は31層の全token hidden＋最終層の最後のtokenと48状態配列を比較した。全32層、共通prefix45＋主suffix87、情報不足45＋80、最大変更45＋89、prefixなし132 tokenを維持する。

| 条件 | query | handler命令 | 元MLP full v2との差 | Candid bytes | 実測秒 | 最大query命令 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix | 90 | 156,313,221,966 | -35,535,010 | 145,902,403 | 20.7483 | 2,697,997,769 |
| 617 | 122 | 287,186,913,161 | -6,394,943,641 | 262,955,428 | 35.1484 | 4,515,331,871 |
| insufficient | 91 | 259,182,862,243 | +14,055,770 | 173,060,522 | 30.8509 | 4,520,882,537 |
| maximum | 122 | 305,987,635,365 | +68,257,646 | 267,767,242 | 37.0795 | 4,767,482,855 |
| normal | 292 | 447,558,326,622 | +82,651,251 | 580,229,781 | 78.2867 | 2,882,723,905 |

主問題は6,394,943,641命令（2.17825%）減、prefixは35,535,010命令（0.02273%）減。情報不足0.00542%、最大変更0.02231%、prefixなし0.01847%の微増を記録した。query数と通信量は５条件とも同じ。主122 query・262,955,428 bytes、初回はprefix90＋主122＝212 query。50/32未達。整数dotの支配的な費用と、Delta状態を渡すquery境界が残る。

最大query4,767,482,855 handler命令、最大観測終端heap4,119,134,208 bytes。実queryは５B上限内で全５条件を完走した。counterはCDK Candid decode/encodeを除き、ByteBuf入口の削減量はこの数値に直接含まれない。通信はHTTP/CBOR/signatureを除く。heapは終端page数で瞬間ピークではない。時間はcache未制御の単回実測で、prefixなしは60.2260→78.2867秒へ増加した。速度改善を保証しない。

公式BF16との差、最大変更gold=yesをnoとする既存誤判定は残る。追加量子化はなく、型安全性と判断精度を別々に評価する。一般精度改善を主張しない。

固定721 tensor・4,065,416,192 bytesとRoPE131,072 bytesの準備は721 update、15,041,482,761命令、214.2733秒、request47,473/reply14,927,506 bytes。準備時pack status1/cache status2 query・認証read2、全層前後cache status2 query・認証read2は推論と別計上。

採用module `bcacce7aa638ab737d2dcd1d6dddd529c8cab987ac373ba0e813f989cf1ba7cb`、専用local canister `4caro-hl777-77775-aaaba-cai`、Wasm `artifacts/byte-buffer/full.wasm`。認証module/cache bookendsと43 source hashは `artifacts/byte-buffer-v3-cache-checks/report.json`、hash/archiveは `artifacts/byte-buffer/{source-hashes.json,validated-source.zip}`、全層記録 `artifacts/byte-buffer-v3-*`、比較 `docs/byte-buffer-v3-summary.json`。現在ソース・検証hash・archiveの一致を確認済み。生成物はgitignore対象。Laya、mainnet、Git remoteは変更していない。

Rust runtime60・INT8 integration2・F32 integration2・compile-fail doctest3、canister4、client新旧MLP4テストが通過した。`artifacts/byte-buffer/{v3-tests,candid-tests,v3-client-tests}.log`。

追加探索では、元の主問題31層のINT8 down入力に含まれるゼロ率は3.39〜9.11%だったが、８要素が全ゼロの群は最大0.00799%、256要素の全ゼロblockは０だった。ゼロ群を飛ばすkernelは実装せず、速度改善を主張しない。読み取りだけの記録 `artifacts/byte-buffer/zero-groups.json`。

## 再実行

`experimental-mlp-full` はcanisterのbyte bufferとruntimeの44＋44kernelを依存featureとして有効にする。クライアントの `--fuse-mlp-full` は維持する。

```sh
cargo build --offline --release --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-projection-reuse,experimental-full-weight-cache,experimental-blake3,experimental-attention-fusion,experimental-attention-full,experimental-delta-projected,experimental-prepared-rope,experimental-column16-token48,experimental-delta-finish,experimental-matrix-tail,experimental-mlp-pipeline,experimental-mlp-full,experimental-dot-scale,experimental-balanced44,experimental-byte-buffer
# 専用local canisterへupgrade後、固定モデルだけを一度準備:
.venv/bin/python scripts/prepare_weight_cache.py --canister <local-id> \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --directory artifacts/new-byte-buffer-preparation --include-f32 --require-prepared-rope
.venv/bin/python scripts/validate_prepared_weights.py --canister <local-id> \
  --run-name new-byte-buffer --baseline mlp-full-v2 \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --preparation artifacts/new-byte-buffer-preparation --reuse-projection-inputs \
  --frame-checksum blake3 --fuse-attention --fuse-attention-full \
  --fuse-delta-projected --fuse-delta-finish --fuse-mlp-pipeline --fuse-mlp-full --require-prepared-rope
```
