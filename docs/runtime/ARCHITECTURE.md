# 共通演算とモデル実装の境界

`imajev-runtime` からF32基底投影・scalar参照・BF16 codec・block256 codecを `inference-core` へ抽出した。既存のRust入口はre-exportで維持する。数値順序・wire形式・既存の実行上限を保持し、canisterのCandidは変更しない。

依存は `canisters/inference → imajev-runtime → inference-core`。`inference-core` は外部crate・モデルmanifest・tensor名・IC APIを参照しない。nativeとWasm SIMD128を実装する。モデル固有のLoRA入力共有から使うraw SIMD関数はunsafeで、必要なshapeとメモリ範囲をAPIに明記する。

## 現在の配置

| 場所 | 責務 | 状態 |
|---|---|---|
| `crates/inference-core/src/linear.rs` | F32投影、token lane共有、scalar参照 | 抽出・既存ランタイム接続済み |
| `crates/inference-core/src/bf16.rs` | BF16判定・格納・復号・有限性検査 | 抽出済み |
| `crates/inference-core/src/block256.rs` | canonicalなBF16/F32可逆payload | 抽出・既存frame接続済み |
| `crates/imajev-runtime/src/` | INT8、prepared weights、LoRA、Attention、Delta、MLP、frame、dispatch | 既存実装を維持 |
| `canisters/inference/` | 認可、固定モデル準備、stable/heap、IC入口 | 既存実装を維持 |
| `client/` | モデルgraph、client-held carry、予算、journal | モデル依存のscheduler |

古いcodecファイルは互換用re-exportとして残す。測定スクリプトのruntimeソース探索には共通crateのソースとmanifestを加え、新しい測定の証跡に依存先を含める。過去のartifactのhashは書き換えない。

## 次に切り出す条件

INT8 kernelのfeature群はprepared配置と演算方式に結び付いている。まず `QuantizedRows`、重み配置、block256 scaleの契約を第二モデルで検証し、その後に移す。prepared weightsは固定重みの読込とレイアウト契約、carryはmodel/pack/input/progressの束縛を含むため、単にファイルを移すだけでは共通化できない。

層境界の行数・head数を固定した50query配分はImajev専用の計測結果である。他モデルの標準schedulerへコピーせず、命令・payload・メモリ予算に基づく配分を実測で作る。

## 現在の優先事項

現在はImajevの数値契約と検証済みPrefix27経路を対象にする。別モデルのadapter・classifier・比較実装はこのリポジトリでは管理しない。INT8互換経路の置き換えは、同一入力のbit一致とローカルIC命令数で検証する。

## 完成条件と今回の検証範囲

共通演算crateの抽出は実装済み。モデル読込から推論完走まで扱う汎用ランタイムの完成には、第二モデル、モデル仕様の表現、重み読込、scheduler、異常系、同一条件のWasm実測が必要となる。GGUF loader・汎用tokenizer・自由生成は未実装。

共通crate抽出時の検証はworkspaceの標準/全featureテスト、共通crateの公開API契約試験、Wasmビルド、Python構文確認を対象とする。共通crateのWasm契約試験を実行し、24形状の投影/codec bit一致と全65,536 BF16 bitパターンの有限性分類も確認する。共通crate抽出後の本番推論用canisterはinstallしていない。過去の別モデル投影比較は診断専用canisterでの記録であり、全モデルの命令数の証拠には使わない。crate境界変更は最終Wasmの最適化に影響し得るため、過去の性能値を新moduleの実測値には読み替えない。
