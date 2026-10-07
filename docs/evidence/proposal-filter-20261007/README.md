# proposalフィルター評価の共有用根拠資料

2026-10-07の保存済み結果から作成した公開用の写しです。新しい推論や判定変更は行っていません。全161件の処理結果、18種類のモデル入力、集計、条件付きレビュー、検証記録を含みます。人間の正解ラベルに対する精度評価ではありません。

- [全161件の判定・処理経路・理由](DECISIONS.md)、[機械可読の判定一覧](decisions.json)
- [新規実行の集計](fresh-summary.json)、[実行検証](verification.json)
- [入力token列とprompt](inputs.json)、[prefix計画](plan.json)、[実行方針](execution-policy.json)
- [元proposalとの根拠照合と閾値比較](REVIEW.md)、[各件の旧新値とレビュー](review.json)
- [128-token入力の実行記録例](run-000.json)
- [最適化の測定](OPTIMIZATION.md)、[照合結果](optimization-verification.json)、[build記録](build.json)
- [packの構成](pack-manifest.json)、[固定版のrelease specification](RELEASE-SPEC.md)、[モデルカード](MODEL_CARD.md)
- [原本と出力のSHA-256・変換範囲](provenance.json)

ファイル内の原本・ソース等のパス文字列はリポジトリ基準です。参照先の重み、Wasm、全snapshot、中間tensor、生ログはこの共有資料には含みません。別checkoutで元実験をそのまま再実行できるbundleではありません。検証記録は保存済み実験についての記録であり、この写しだけで独立した実行証明になるものではありません。

元のローカル証跡がそろった環境では、`python3 scripts/export_proposal_filter_evidence.py`で再出力、`--check`で原本からの変換一致を確認できます。原本の`artifacts/`と`checkpoints/`全体をGit対象へ変更する必要はありません。
