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

目的はLaya側の命令数削減。共通runtimeへの移植は保留し、個別の演算最適化を検証する。[探索と実測結果](comparison/LAYA_INSTRUCTION_SEARCH.md)。以下の二モデル対応は設計候補で、現在の実装目標ではない。

## 第二モデルはLayaで検証する

共通runtimeの第二モデルには既存のLayaを使う。固定pack・参照結果・IC実装があり、Imajevのdecoderとは異なるencoder構成なので、モデル固有の前提を見つける対象になる。既存Layaのソース・Git・network・canisterは変更せず、このリポジトリ内のadapterと診断用環境で検証する。

Layaの既存INT8演算は出力行単位の重みscaleとtoken単位の入力scaleを使う。Imajevのblock256入力契約へ無条件に変換せず、整数dot、scale適用、出力の再量子化を分けて契約を定義する。入力幅2624は256の倍数ではないため、端数列・paddingの扱いも検証対象になる。[既存の比較分析](../LAYA_COST_ANALYSIS.md)。

実重みの先頭512出力行を使った初回投影比較は完了し、全10条件でLaya側が少ない命令数だった。[条件と結果](comparison/LAYA_RUNTIME.md)。今回の候補は実験用token-scale kernelで、採用済みblock256/S1の比較ではない。

次は実activationと出力再量子化を含めて参照出力を照合し、その後にモデルadapter・重み読込・schedulerを接続してLayaの全推論を再現する。命令数・通信・準備費用と判定出力を記録する。この節は検証対象の決定であり、Layaの共通runtime接続が完了したという記録ではない。

Layaでの二モデル対応と、llama_cpp_canisterとの同一モデルA/Bは別の検証になる。Layaを相手側でも実行するにはモデルarchitectureとdecision headの対応を確認する必要がある。

## 完成条件と今回の検証範囲

共通演算crateの抽出は実装済み。モデル読込から推論完走まで扱う汎用ランタイムの完成には、第二モデル、モデル仕様の表現、重み読込、scheduler、異常系、同一条件のWasm実測が必要となる。GGUF loader・汎用tokenizer・自由生成は未実装。

共通crate抽出時の検証はworkspaceの標準/全featureテスト、共通crateの公開API契約試験、Wasmビルド、Python構文確認を対象とする。共通crateのWasm契約試験を実行し、24形状の投影/codec bit一致と全65,536 BF16 bitパターンの有限性分類も確認する。共通crate抽出後の本番推論用canisterはinstallしていない。後続のLaya投影比較は別の診断専用canisterで実施し、全モデルの命令数は再測定していない。crate境界変更は最終Wasmの最適化に影響し得るため、過去の性能値を新moduleの実測値には読み替えない。
