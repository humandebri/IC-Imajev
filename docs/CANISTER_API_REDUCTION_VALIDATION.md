# API 削減の検証記録（2026-10-09）

実装の一覧・認可・移行手順は [CANISTER_API_NAMES.md](CANISTER_API_NAMES.md) を参照する。本番デプロイは実施していない。

## 検証できた範囲

| ビルド | 実際の WASM method export | 結果 |
| --- | ---: | --- |
| 凍結ソースの最適化本番 | 20（query 9 / update 11） | 名前・件数・区分を検査 |
| 同じ凍結ソースの診断版 | 27（query 11 / update 16） | 名前・件数・区分を検査 |
| 通常 Cargo 本番 | 18 | 名前・件数・区分を検査 |
| 通常 Cargo 診断版 | 25 | 名前・件数・区分を検査 |
| Cargo feature 無効 | 9 | 名前・件数・区分を検査 |

allowlist と完全一致させており、削除5件・旧公開名・本番の診断7件は export に存在しない。最適化本番 WASM の SHA-256 は `91893b177580bc7b528394b7901a5ba30a0d539ed711adc9c87c87cdd225ccba`。34個の projection body は検証済み donor と同一。診断版の export 検査は、同じ凍結ソースを診断 feature 付きで再コンパイルした未パッチ WASM に対して行った。

Python のソース変換・公開 query 構成テスト11件が成功。Rust の default / diagnostics は各21単体テストと Candid テスト1件が成功。ただし prefix27 を前提とする `registration_rejects_voting_bank_for_attention_and_delta` は、同時進行する prefix5 移行との不一致により1件除外した。feature 無効は6単体テストと Candid テスト1件が成功。client の Cargo 検査も成功。

専用 local canister `3neil-e3777-77775-aaaqa-cai` を新規作成し、最適化本番 WASM と合成3.6MB pack で検証した。チャンクの不正 checksum・不正 offset、順序を変えた送信、同一重複・競合重複、全体 hash 不一致、prepare からの回復、未完了 upload の upgrade・再送、CLI 逐次 upload の再開・並列数1、検証完了済みモデルの upgrade を確認した。匿名 query、owner 制限、worker の self-only、receipt / refund の caller 単位の参照、匿名課金要求拒否について16件を確認した。検証後に専用 canister を停止・削除した。

詳細の新規成果物は `artifacts/api-reduction-20261009/` に保存した（Git ignored）。既存の測定成果物は変更していない。stable memory・receipt の形式を変更せず、未完了 upload の upgrade 後に受付表とカウンタがリセットされる従来の挙動を確認した。

## 未完了の検証

同時進行する prefix5 移行の途中で、frontend の `prefix5-v1/manifest.json` が未生成、query 件数・計画の期待値が不一致、TypeScript の `release.prompt_layout` が不一致となった。ブラウザ用 Candid subset の6 query 検査は成功したが、ブラウザ全体の検証成功は主張しない。

今回の専用 canister には実モデルを投入していない。最適化 WASM の実モデルでの最終結果、query 計算量上限と応答サイズ、課金推論の全行程の再測定は未完了。既存の paid 単体テストでは重複要求・返金・upgrade の状態遷移を確認しているが、この API 削減後の実モデル end-to-end 検証とは区別する。prefix5 移行の整合が取れた版で実施する必要がある。

逐次・prefix・統合 query runner は単独判断経路を削除し、全文推論には最終判断を含む query を要求する。長文の paid token proof は融合できない reference graph では hidden/state 一致のみを報告し、独立した保存済み判断 reference がない場合に確率比較済みとは報告しない。

## レビュー指摘の修正後

prefix 定義の混在を修正した。通常 Cargo の登録・readiness・Delta stream・長文 carry と、凍結ソースの登録・stream・carry が課金入口の共通 prefix 定義を参照する。ビルド時に配列宣言と token 数を照合する。default / diagnostics の Rust テストは各22単体テストと Candid テスト1件が成功し、以前除外した prefix 登録テストも成功した。

最適化本番を再ビルドし、34 projection body の一致と20 method export を検査した。修正版 SHA-256 は `2c873a02d6ca3493920d776364e869ee70e0edddfa975d942a0366422fe49e42`。その WASM 内の Candid getter から全サービス定義を抽出し、`canisters/inference/optimized-inference.did` を生成した。Candid と WASM の method 名・件数・query/update 区分が一致する。`ic-wasm check-endpoints` でも一致を検証した（34 kernel patch exports・lifecycle・Candid getter の補助 export を hidden 一覧に指定）。

CLI は `reset:true` により検証未完了の pack を明示的に再準備できる。検証済みモデルは従来どおり再準備を拒否する。未指定では既存チャンクを保存して再開する。不正 hash の upload 検証ツールも明示的 reset を使う。

この追記は API と prefix 定義・アップロード回復の修正検証であり、上記の実モデル全行程やブラウザ移行に関する未完了事項は解消済みと扱わない。本番デプロイは実施していない。

修正後の専用 local canister で21 API 呼び出しに加え、CLI の通常再開・不一致拒否・検証済みモデルの再準備拒否・未完了 upload の不正 hash 後の明示的 reset 回復を検証し、成功した。結果は `artifacts/api-reduction-20261009/review-fixes/upload-recovery.json` に保存した。課金モジュールを含まない単独 `experimental-update-inference` feature のビルドも検査した。長文の上限を全512 token と共通 prefix 長から導出し、scheduler テストを suffix 507 token まで拡張した。
