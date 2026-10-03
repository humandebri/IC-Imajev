# 繰り返すframe処理の削減

2026-10-02。クライアントの同一request二重encodeを除去し、frame checksumだけをBLAKE3 SIMDへ切り替える候補を実装した。量子化、重み、浮動小数点演算順序、readout、calibrationは変更しない。

## 二重encodeの除去

`JournalTransport.run`はチェックポイント照合のためrequestをencodeした後、`Transport.run`で同じ入力をもう一度encodeしていた。共通の`_run_encoded`に既存のframeを渡し、serialize・codec変換・hashを1回にした。既存requestとのbyte照合、返信identity/step検査、有限値検査、atomic保存、retry、再開時の検査を維持した。requestのbyteは同じなので、この変更だけでcanisterの命令数や通信量が減るとは扱わない。

新しいjournal試験でrequest encodeが1回、下位transportで0回となることを確認した。version1/2とも再開時に検証済みreplyを再利用し、checksum方式変更で入力不一致を拒否する。

## version2 checksum

frame version1は従来SHA256、version2はBLAKE3 unkeyed 32-byte digestとする。header/payload/32-byte footerの構造、2,000,000-byte上限、header16,384-byte上限、codec名は維持する。bounded headerだけを先にparseし方式を選び、全frame digestを確認してからpayloadをdecodeする。対応featureなしのruntimeはversion2を拒否する。model lock、pack、input identity、checkpointファイル、module・sourceのprovenanceはSHA256のまま。

BLAKE3はRust `=1.8.2`、Python `==1.0.8`を固定。Wasmは公式`wasm32_simd` featureを使用する（[固定tagのCargo.toml](https://github.com/BLAKE3-team/BLAKE3/blob/1.8.2/Cargo.toml)）。旧frameを読めるようversion1も残す。CLIは互換性のためSHA256が既定で、`--frame-checksum blake3`を明示する。sessionへ方式を記録する。checksumは認証ではなく、クライアントが保持するstateの破損検出である。

## 独立hash測定

モデルを持たない専用local canister2個で、35公開vector・6実request frame・2 MB patternの42入力をSHA256/BLAKE3、portable/SIMDの両方で確認。168通常queryのdigestがnativeと一致し、35vectorは[固定tagの公開値](https://github.com/BLAKE3-team/BLAKE3/blob/1.8.2/test_vectors/test_vectors.json)とも一致した。公開vectorファイルSHA256は`dcb91ea8accc77e6d6e632af7cdc1a99a9f3ae78cf648da595c7d064db32f624`。

| 入力bytes | SHA256命令 | BLAKE3 SIMD命令 | BLAKE3 portable命令 |
| ---: | ---: | ---: | ---: |
| 411,281 | 31,069,168 | 6,752,496 | 14,711,488 |
| 1,365,973 | 103,177,946 | 22,393,482 | 48,858,095 |
| 1,800,805 | 136,020,142 | 29,519,467 | 64,411,599 |
| 2,000,000 | 151,068,275 | 32,774,508 | 71,536,594 |

大入力のhash部分は約78.3%減。これは全モデルの削減率ではない。小入力の比率は別で、実時間やquery数の削減とも同一視しない。生記録`artifacts/hash-bench/probe/report.json`、入力/native/Wasm/module bookendsも同directory配下。module bookendsはmanagement statusによるread-only update4回であり、認証read-stateではない。初回手動empty query1回は168回の外。

全モデル用canisterでも旧新frame6通常queryがnative replyとbyte一致し、破損header/payload/digest・downgrade・誤ったdigest方式・未知version・過大header・truncationの8通常queryを拒否。前後認証module read2回。記録`artifacts/blake3/frame-check/report.json`。

## 重み処理の追加確認

INT8 weightのロードと符号拡張を毎回別に行うように見えるソースを、生成Wasmでも確認した。主要なdot tile7関数すべてでLLVMが`V128Load8x8S`へ統合済み（各256静的命令）。この2操作を手作業で統合しても追加削減の根拠はない。静的auditは動的命令数やspeedの測定ではない。記録`artifacts/blake3/dot-audit.json`。

固定weight/F32復元は準備updateへ移動済み、質問間で共通のprefixはclient-held cache、分割projectionの量子化とLoRA A結果もclient-held capture/reuse済み。固定weight scale復元の準備移動候補は全5条件で命令が微増したため不採用（[PREPARED_SCALES.md](PREPARED_SCALES.md)）。中間状態をcanister updateへ保存してquery数を減らす方式は使っていない。

## 実行

```sh
cargo build --release --offline --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-projection-reuse,experimental-full-weight-cache,experimental-blake3
# Imajev専用local canisterへinstall/upgrade後、固定packがsealedであることを確認。
.venv/bin/python scripts/prepare_weight_cache.py --canister <local-id> \
  --wasm artifacts/blake3/full.wasm --include-f32 \
  --directory artifacts/blake3/full-preparation
.venv/bin/python scripts/validate_prepared_weights.py --canister <local-id> \
  --wasm artifacts/blake3/full.wasm --preparation artifacts/blake3/full-preparation \
  --run-name blake3-v1 --baseline mlp-reuse-v1 \
  --reuse-projection-inputs --frame-checksum blake3
```

## 全32層の実測と採用

主local canister `4caro-hl777-77775-aaaba-cai`で、固定prefix45 tokenを新規準備し、主問題・情報不足・最大変更・prefixなしの5実行を比較。全5条件で保持hidden/stateと、質問4実行の終端hidden・raw logits・probabilities・unknown probability・型付き判断・abstentionが従来`mlp-reuse-v1`とbit一致。prefixは32全token hiddenと72 state array、質問実行は31層全token hidden＋終層readout用最後token hiddenと48 state arrayを比較した。各実行失敗0、replay0。通常推論queryのみで完走した。

| 条件 | token数（suffix） | query | 従来命令 | 新命令 | 削減率 | Candid bytes | 単回時間 秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix準備 | 45 | 282 | 193,719,805,173 | 174,864,378,808 | 9.7333% | 318,918,016 | 26.8822 |
| 主問題 | 132（87） | 305 | 354,704,765,427 | 324,764,764,252 | 8.4408% | 506,366,104 | 44.1175 |
| 情報不足 | 125（80） | 305 | 310,182,042,285 | 282,391,439,207 | 8.9594% | 469,941,494 | 40.2832 |
| 最大変更 | 134（89） | 305 | 365,457,606,400 | 334,900,680,606 | 8.3613% | 516,788,871 | 48.6687 |
| prefixなし | 132 | 485 | 519,494,180,139 | 470,936,269,789 | 9.3472% | 821,148,886 | 69.9028 |

query数とCandid bytesは全条件で同じ。Candid通信量はHTTP/CBOR/signatureを含まない。命令数はhandler counterであり、CDK Candid decode/encodeを含まない。query cacheを制御しておらず、時間は単回実測で、速度改善の保証とはしない。公式参照との量子化由来の差は残る。最大変更問題のgold=yesに対し従来版と同じnoを返し、判断精度改善を主張しない。今回確認したのは既存INT8実装から追加劣化がないことである。

最大query handler命令は主問題3,873,386,762、最大変更4,006,293,766、prefixなし2,979,088,604。観測heap最大は4,115,005,440 bytes。selected localのquery limit5B・heap4GiBは変更していない。主問題305 query・初回prefix込み587・prefixなし485で、50/32 query未達。現在のhandler総命令を単純に5Bで割って切り上げた下限も主問題65・初回100・prefixなし95であり、これは通信・依存・CDK費用を無視した数で、実現可能query数の予測ではない。

採用moduleは`412c565ed2a06e76c59de1612a7bda56c28193a1eacc8dbfeaf1c7020b1d64db`。新native/Wasmとビルドlogは`artifacts/blake3`、全層生記録は`artifacts/blake3-v1-*`、比較・summaryは`docs/blake3-v1-*-results.json`と`docs/blake3-v1-summary.json`。`artifacts/blake3-v1-cache-checks/report.json`で35 implementation hashの前後一致、固定721 tensor cache 4,065,416,192 bytesと名前一覧の不変、前後認証module hashを確認した。全ordinary step requestがversion2であることも検査した。

固定重み準備は別途721 update・15,027,997,708命令・223.0710秒。Candid request47,473/reply14,915,249 bytes。pack status1/cache status2 query、認証module read2は準備の管理計測として分けた。全層前後cache status2 query・認証module read2も推論query数に含めない。重みの準備は質問依存ではなく、model変更/upgrade時に必要である。質問ごとに重みupload/準備を繰り返す運用にはしていない。

Rustはdefault44＋doctest1、projection-reuse/blake3有効48＋compile-fail doctest2、canister cache3試験が通過。Pythonはframe checksum3、journal4、scheduler13、wire4、block codec2、INT8 wire4の計30試験が通過。生成物・重み・測定JSON・native/Wasm・nested hash benchmark targetはgitignore。Layaのソース・Git・canisterには変更していない。
