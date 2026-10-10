# 1024-token対応のレビュー指摘への対応

2026-10-09。検証先は前回作成した専用ローカルcanister `2vn5i-k3777-77775-aaaua-cai` とrelay `2sm34-hd777-77775-aaauq-cai`。mainnetと既存の別canisterは変更していない。

## 保存先の例外

`PaidTransport` はリポジトリ内の証拠を従来どおり相対パスで記録し、リポジトリ外は絶対パスで記録する。update完了後の `relative_to(ROOT)` 例外を解消した。外部保存先・空白を含むパス・relay経由の応答保存をmockで検査し、1回だけ送信されることも確認した。

## receiptの容量とupgrade

- 容量は128件から4096件へ拡張。完了・失敗の記録をstable memoryの64 KiB slotsに格納し、heapにはSHA-256のcaller/request ID索引と小さな位置情報を置く。
- 24時間のreplay保証を維持する。未返金のPending/InFlightはTTLで削除しない。retryによるrefund状態更新もstable recordへ反映する。
- 処理中のreceiptはcallbackが終わるまでheapに残す。別の要求によるpruneが完了callbackを壊さないよう保護した。
- upgrade直前に退避し、metadataにはarchiveのbase/slot数だけを保存する。upgrade後にslot headerから索引を復元する。旧metadataにはarchiveがなくても読み込める。
- immutableなmodel packの末尾から4 MiBをupgrade metadata用に確保し、その後にarchiveを置く。metadata footerは実際のstable memory末尾へ書く。archiveの拡張とfooterが衝突しないよう追加pageを残す。
- slotは必要時に増やし、TTL満了のterminal recordのslotを再利用する。全4096件が保持対象なら、従来どおり受領前にReceiptCapacityで拒否する。無制限保存ではない。
- 失敗理由はUTF-8境界で8 KiBに制限する。platformの長いreject文がrecordのslot上限を超えて受付を妨げないようにする。

新しいarchiveを読めない旧WASMへのrollbackは保証しない。実行上限を小さくする場合も、archive対応のソースからビルドする。

## 実WASMの数値参照

検証専用feature `experimental-attention-reference` で独立したscalar Attentionを用意した。scoreの積和、softmax、BF16確率、valueの積和をproductionのGQA/views/SIMD helperを使わずに計算する。ownerだけが、推論非実行中に切り替えられる。通常ビルドには参照featureと切替APIを含めない。

実WASMの7ケースでビット一致を確認した。1024位置・57-token・4 heads、512位置・8 heads、符号付きゼロ、端数幅を含む。

全32層の照合では同じ1024-token入力を通常経路とscalar Attention経路で処理し、32層のhidden hash、記録対象state hash、最終hidden、logit、確率、判定を比較する。Deltaのstate hashはconv、AttentionはKVに対応する。参照はprojection/Delta/RoPE/層のschedulerを共有するので、全モデルを別実装した参照や自然言語の精度評価ではない。

実モデルでの比較は全項目で一致した。通常経路は75 workers・580.055秒・最大33,993,340,744命令、scalar経路は157 workers・699.393秒・最大38,748,846,679命令だった。最大heapは両方4,238,278,656 bytes。証拠は `artifacts/paid-1024-fixes-20261009/full-proof/report.json`。

診断版ではscalar参照用にworker予算20B・上限256を使う。通常経路は30B・上限128のまま。各workerの40B上限とheap 4 GiBを検査する。

## 検証記録

- Python: transport 3件、paid prefix 6件、API命名8件、prefix builder 2件を通過。
- runtime: 77件を通過。attention-viewsと独立scalar参照を有効にして検査。
- canister: 診断構成32件、chunking無効の構成29件を通過。130件の完了receipt、upgrade後replay、callerの分離、PendingのTTL超過保護、二重返金guard、処理中receiptの保護を含む。
- SIMD/scalar実WASM kernel: 7ケース通過。
- 診断版module: `993a1dc8d7cfa8e604143570ce26edc343565d1945851b9178796200ac7c6dd7`
- 通常版module: `979e92426a4e6905747df958f424425dbbf8be702db02eee8b58fde62306937d`

- 実APIで130件を受領。stage 0の診断trapで処理を短縮し、最初の1件は返金失敗を模擬してPending、残りはDoneとし、relay残高で返金を確認。130件の実モデル推論を走らせた試験ではない。`receipt-proof-v2/report.json` がcomplete。
- 同じcanisterを通常版へ実upgradeし、最初・最後のレシートを取得。Pendingの再返金でrelay残高が料金分増え、再度retryで増えないことを確認。同一requestのreplayはFailed/Doneを返し、添付cyclesは全額未受領で返る。`restored-proof/report.json` がcomplete。
- 初回の受付試験は、受領済み料金の返金を未受領cyclesのカウンタで見る誤ったassertionで停止した。観測をrelay残高へ修正しv2で通過。初回に残したPendingも `preliminary-refund-cleanup/` の記録どおり返金済み。
- 診断版での数値試験の後、TTL超過中のactive receipt保護と失敗理由の長さ制限を追加した。通常版はこの最終修正を含む。nativeテストで境界を検査し、実upgrade・返金試験には通常版を使用した。

通常ビルドの1024-token推論も完走した。

| 測定 | 通常版WASM |
| --- | ---: |
| workers | 75 |
| 経過秒 | 558.506 |
| 合計計測命令 | 2,346,464,175,999 |
| 最大worker計測命令 | 33,991,570,982 |
| 最大heap bytes | 4,236,181,504 |
| 4 GiBまでの余裕 | 56 MiB |

通常版のlogit・確率・判定も診断版の通常経路とビット一致した。同一request IDの再送は同じ結果を返し、料金を受領しない。1025 tokensはquote・updateともInvalidで、添付cyclesを受領しない。証拠は `normal-proof/report.json`。

専用modelとrelayは試験終了後に停止。共有ローカルnetworkと別canisterは維持した。全phaseの完了状態とmodule hashは `artifacts/paid-1024-fixes-20261009/summary.json`、停止状態は同ディレクトリの `model-final-status.json` / `relay-final-status.json` に保存した。

## ビルド経路とテキスト上限の追加修正

`build_paid_contract.py` に課金ソース3ファイルのコピーとupgrade metadataの整合処理をまとめた。ブラウザ版・非chunk版・1024版で共用し、`receipt_archive.rs` のコピー漏れと、stable memory拡張後のfooter保存位置のずれを修正した。旧形式・修正済み形式を認識し、想定外または重複した保存処理はコンパイル前に拒否する。helperも各ビルドのsource hashに含める。

診断APIテストに `setPaidInferenceReference` のfeature条件を追加した。`TextPreparer` は `paid_prefix.MAX_TOKENS` を使い、実テキストで1024 tokensまで準備できる。ブラウザの96-token上限は維持する。

- Python: builder・prefix・transport・API・テキスト境界・ブラウザ例の31テストを通過。実promptをtokenizeし、512・513・1024は受付、1025は拒否を確認した。
- Rust: 通常34件・診断34件・非chunk32件、計100件を通過。既存incremental cacheを変更せず、`artifacts/review-fixes-20261009/cargo-target` を使った。
- フロント: query計画・tokenizer parity・全91計画のbudget検査を通過。
- 実Wasm: 3経路ともビルド、20件の公開API、34個のprojection body一致、Candid整合性検査を通過。成果物は `artifacts/review-fixes-20261009/{browser,unchunked,chunked}-build/`。

専用ローカルcanisterで1-byteのモデルデータと合成の完了・Pendingレシートを保存し、stable memoryが68 pagesへ拡張されたことを確認した。生成したブラウザ版→非chunk版→1024版へ実upgradeし、各版でレシート復元と追加課金なしの完了・失敗replayを検証した。Pendingを未返金のまま3版に引き継ぎ、最後に返金成功と再試行時の二重送金なしを確認した。保存・復元の試験であり、合成レシートの料金を実推論で受領した試験やモデル精度の試験ではない。

最終記録は `artifacts/review-fixes-20261009/upgrade-proof-v2/report.json`（`complete:true`）。検証先は `http://localhost:8001/`。各試験で作成したmodel canisterとrelayは終了後に削除済み。既存canister、本番、frontend release pinは変更していない。
