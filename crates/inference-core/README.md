# inference-core

モデル名・tensor名・Candid・ストレージに依存しない演算部品。Imajevの既存実装から抽出し、`imajev-runtime` が直接利用する。nativeとWasm SIMD128を対象とする。外部依存はない。

| モジュール | 契約 |
|---|---|
| `linear` | token-major入力、output-major重み。独立tokenをSIMD laneに配置し、各dotの列順と乗算→加算を保存 |
| `bf16` | BF16のbit判定・格納・展開。`pack` は上位16bitの格納なので、可逆に使う呼び出し元は `all_bf16` を確認 |
| `block256` | 256要素単位のBF16/F32可逆payload。有限値・canonical形式を検査し、符号付きゼロも保存 |

`MAX_FLOATS=900000` とblock codecの2MB frame予算は抽出元の上限を保持する。汎用のモデル寸法を示す値ではない。行列積はshapeを検査するが有限性の全走査は行わず、入力検査は呼び出し元が担当する。`block256::append` が失敗した場合、書きかけの出力は破棄する。

```sh
cargo test --offline -p inference-core
cargo run --offline -p inference-core --example linear_codec
```

APIの使用例は [linear_codec.rs](examples/linear_codec.rs)、抽出の境界と残る作業は [ランタイム設計](../../docs/runtime/ARCHITECTURE.md) を参照。GGUF読込、tokenizer、自動生成、モデルgraph、schedulerはこのcrateには含まれない。公開配布は第二モデルの検証後に判断する。
