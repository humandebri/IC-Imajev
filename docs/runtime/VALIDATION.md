# 共通演算抽出後の検証

2026-10-05。以下は成功した。ログと共通crateのsource hashは `artifacts/runtime-extraction/report.json` に保存した。

```sh
cargo test --offline -p inference-core -p imajev-runtime
cargo test --workspace --offline --all-features
cargo fmt -p inference-core -- --check
cargo run --offline -p inference-core --example linear_codec
cargo build --offline --release --target wasm32-unknown-unknown -p inference-core --example wasm_contract
cargo build --offline --release --target wasm32-unknown-unknown -p imajev-inference --all-features
cargo check --offline --manifest-path scripts/codec_scan_bench/Cargo.toml --target wasm32-unknown-unknown
```

`wasm_contract.wasm` はNodeのWebAssembly engineで実行し、export `check_contract()` が0を返した。24形状のSIMD/scalar投影・codecのbit一致と65,536 BF16 bitパターンの分類を確認する。公開APIのnative契約試験は260形状と、不正shape・overflow・出力上限・codec要素上限を対象にした。PythonスクリプトのAST構文確認、文書のローカルリンク、`git diff --check` も通過した。

作業ツリーの実験用update経路でWasm専用のheap計測をnativeビルドから分離し、全featureテストを再実行した。Wasmのheap計測は従来と同じ式を使う。

canisterのreleaseビルドはraw Wasmで、既存の実験kernel用patch・install・新moduleの全モデル命令測定は含まない。性能比較先の実行も含まない。
