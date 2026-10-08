# 推論ランタイムと効率化の整理

2026-10-05。再利用する部品、モデル実装、検証記録の入口をこのフォルダに集約する。既存の測定文書とartifactは元のパスに残し、参照を維持する。

| 読みたい内容 | 文書 |
|---|---|
| Laya由来実装の置き換えと計算量・数値のA/B | [INDEPENDENT_RUNTIME.md](INDEPENDENT_RUNTIME.md) |
| crateの役割・依存方向・完成条件 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 名前と意味、数値契約 | [CONTEXT.md](CONTEXT.md) |
| 他モデルへ持ち出す知見と不採用案 | [LESSONS.md](LESSONS.md) |
| llama_cpp_canisterとの比較 | [comparison/LLAMA_CPP_CANISTER.md](comparison/LLAMA_CPP_CANISTER.md) |
| IC上のINT8投影の対向実測 | [comparison/KERNEL_BENCHMARK.md](comparison/KERNEL_BENCHMARK.md) |
| 投影実測の数値と証拠hash | [comparison/kernel-results.json](comparison/kernel-results.json) |
| 比較先のcommitと取得ファイルhash | [comparison/sources.json](comparison/sources.json) |
| 抽出後の検証コマンドと範囲 | [VALIDATION.md](VALIDATION.md) |
| 全体実装の最新実測 | [STATUS.md](../STATUS.md) |
| queryとreplicated ingressの同一要求比較 | [REPLICATED_INFERENCE.md](../REPLICATED_INFERENCE.md) |

共通演算は [inference-core](../../crates/inference-core/README.md)、モデル演算とframe処理は `imajev-runtime`、ICの入口は `canisters/inference`、クライアントのgraphとjournalは `client/` が担当する。
