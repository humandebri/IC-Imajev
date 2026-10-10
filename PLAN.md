# 設計と検証の入口

Imajevの推論をInternet Computer上で実行する。モデル、tokenizer、量子化設定、数値の加算順序を固定し、実Wasmと独立参照の出力を比較して変更を検証する。

## 構成

- 推論canister：重み・prefix cacheを保持し、入力の検査、料金見積もり、推論、receiptの再取得を提供する。
- ランタイム：INT8基底投影、F32 LoRA、Attention／Delta／MLPの演算を実装する。
- クライアントとfrontend：入力準備、canister呼び出し、結果表示を行う。

## 公開資料

- [利用とビルド](README.md)
- [ランタイムの構成と数値契約](docs/runtime/README.md)
- [canister API](docs/CANISTER_API_NAMES.md)
- [prefix5への移行](docs/PREFIX5_MIGRATION.md)
- [prefix5の検証結果](docs/PAID_PREFIX5_VERIFICATION.md)
- [モデルrevisionとhash](MODEL_LOCK.json)
- [ライセンスと出典](NOTICE.md)

測定値には入力、module、比較対象、計量範囲を付記する。ローカルでの時間・命令数を本番subnetの性能保証として扱わず、単体kernelの改善率を全体推論の改善率へ換算しない。
