# アダプター統合版の32 query経路での比較


32 queryの最適化経路でも、アダプター統合によって命令数が **13.21〜13.52%減った**。速度改善は確認できなかった。3入力をそれぞれ各モデル3回ずつ測定し、18回すべて32 query、replay・fallbackなしだった。

| 入力 | prefix / suffix | 統合前の命令数 | 統合後の命令数 | 削減率 | 統合前の時間中央値 | 統合後の時間中央値 |
|---|---:|---:|---:|---:|---:|---:|
| 617 | 38 / 56 | 144,992,032,811 | 125,670,728,353 | 13.33% | 33.54秒 | 50.31秒 |
| 620 | 38 / 48 | 125,258,970,072 | 108,707,143,535 | 13.21% | 34.14秒 | 23.44秒 |
| 653 | 27 / 57 | 145,133,023,643 | 125,514,896,362 | 13.52% | 22.52秒 | 20.66秒 |

命令数は、各入力・各モデルについて3回とも完全一致した。すべてのhandlerが50億命令未満で、Candid requestとreplyは各2 MB未満だった。両モデルの判定は3入力とも `likely` で一致した。統合・再量子化で確率は変化し、最大差は **3.51 percentage points**（620）だった。これは3入力の比較であり、全入力の品質維持を示すものではない。

所要時間の生データ（秒）は次の通り。

| 入力 | 統合前の3回 | 統合後の3回 |
|---|---|---|
| 617 | 40.92, 27.70, 33.54 | 51.47, 20.47, 50.31 |
| 620 | 39.20, 34.14, 20.79 | 45.94, 23.44, 17.74 |
| 653 | 22.11, 49.11, 22.52 | 44.18, 20.66, 20.21 |

同じモデル・同じ入力でも最大/最小が約1.5〜2.6倍に広がった。別の検証が同じローカルICで動いており、時間は実行待ち・ホスト負荷の影響を受ける。617では中央値が悪化し、620/653では改善したため、今回の時間から統合による速度改善を断定しない。`host-load.jsonl` に各測定前後のload averageも保存した。

## 測定条件

2026-10-06のローカルICで、既存の固定prefix・演算融合・32 queryスケジュールを使い、元のINT8＋F32 LoRAと、LoRAを取り込んで再量子化したINT8 packを比較する。

入力は `artifacts/text-short-v2/inputs.json` の617・620・653。617/620は38トークンの固定stem、653は27トークンの共通prefixを使う。prefixのトークン列は同じでも、hidden・KV・Deltaの状態は重みに依存するため、統合版については実際のcanister queryで再計算する。元のモデル用のprefix状態を統合版に渡さない。

元の最適化済みWasm（SHA-256 `2cfe5d3a1d2da9481d9446dbd0a877617de8db140f8347058b02a1dbb2c63a05`）の保存済みRustソースと6個のWATカーネルから、実験用Wasm（`6ce099e1f128b835b153bb28cde9f27dee719711be3822a38976cb4688613de4`）を作った。両モデルを同じWasmで実行する。既存の融合処理はLoRAの存在を前提にしているため、統合packのhashに限定して、内部で「アダプターなし」の記述子を生成し、A/B行列積とAの列方向継続演算を省く。実際のpackにはA/Bテンソルを追加せず、読み込みも行わない。

同じスケジュール・通信形式で比較するため、rank64の通信欄はゼロのpaddingとして残す。アダプターに関する通信欄まで取り除いた実装の最小コストを測ったものではない。統合版用の内部記述子生成などの追加コストも測定値に含まれる。

測定対象は専用のローカルcanisterのみ。

- 元のモデル: `2hlkr-gl777-77775-aaaxa-cai`
- 統合モデル: `2akmf-lt777-77775-aaaxq-cai`
- endpoint: `http://localhost:8001/`

モデルのアップロード・重みの準備・prefixの計算とpacket化・固定状態のowner updateは、質問ごとの命令数と時間から除外する。質問ごとの推論は通常queryだけで行う。元のモデルの721テンソル、統合モデルの321テンソルについて、prepared cacheの名前とbyte数、pack hash、Wasm hashを照合する。

入力ごとに各モデルを3回、順序を交互にして実行する。各回は新しいjournalを使用し、query引数のstepを変えて応答キャッシュのヒットを避ける。replicaのquery cache設定そのものは変更しない。replay・fallback・失敗したqueryを含む実行は成功した測定として数えない。

命令数は既存handlerのperformance counterの合計であり、CDKのCandid decode/encodeの一部を含まない。所要時間はrunner内のwall clockで、module照合・ローカルHTTP・bridge・serialization・ファイル保存を含む。mainnetのレイテンシを示すものではない。

## 数値検証

元のモデルについては、保存済みの32 query検証と3入力すべての最終hiddenがビット単位で一致し、logitsも一致した。実験用の対応によって元のモデルの数値結果を変えていない。

統合版については、prefixを使わずtoken 0から全32層を計算する汎用queryスケジュールを別に実行した。3入力すべてで、32 query経路の最終hiddenがビット単位で一致し、logits・確率・判定も完全一致した。この追加照合の時間・命令数は上の比較表に含めない。結果は `artifacts/merged-query32-v1/verification.json`、入出力と生の計測は各runの `report.json` と `queries/` に保存した。

Candid requestの最大値は1,599,354 bytes、replyの最大値は1,306,249 bytes。質問ごとの推論にupdateは使っていない。既存runnerの一部のprecisionラベルは共通の旧形式のままなので、統合の有無は保存したpack hash・実際のテンソル一覧・実験用Wasmの分岐で判断する。統合版のpack/cacheにA/Bテンソルは存在しない。

## 再現用コードと生データ

- `scripts/build_merged_query32.py`: 保存済みの32 query実装から実験用Wasmを作成。
- `scripts/make_merged_query32_runner.py`: 既存runnerから、実験専用のmodule・prefix照合を備えたrunnerを生成。
- `scripts/run_merged_query32.py`: 実験用runner。既存の通常runnerは変更しない。
- `scripts/measure_merged_query32.py`: prefix計算・固定状態準備・計測・集計・独立した全体計算との照合。
- `artifacts/merged-query32-v1/`: ソースhash、Wasm、重み準備記録、prefix、各回のquery入出力、集計と検証結果。
