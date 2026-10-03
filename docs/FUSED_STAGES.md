# Delta演算の統合と短い入力のMLP幅

2026-10-02。32 queryを目標に、数値演算の順序を変えずに通常queryの境界を減らす。中間状態はclient保持、モデル・INT8量子化・F32 LoRA・専用readout・calibrationは固定する。

## 実測結果

専用local canister、通常query、rotations=1。下表の推論は共通45-token prefix準備済み。初回準備費用を除外している。

| 入力 | tokens（追加処理） | query 前→後 | handler命令 | Candid bytes | 単回秒 | 最大query命令 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BOOM DAO 617 | 132（87） | 603→**379** | 465,519,016,515 | 614,956,898 | 62.711 | 4,769,185,854 |
| 情報不足 | 125（80） | 571→**379** | 413,291,335,820 | 570,419,806 | 56.168 | 4,197,250,132 |
| 最大lock変更 | 134（89） | 603→**411** | 487,285,585,730 | 643,210,897 | 65.834 | 4,582,503,214 |

prefix準備は570→**354 query**、248,365,025,622命令、366,601,934 bytes、40.458秒。617の初回は準備込み733 queryとなる。cacheはclientが保持し、準備時も全演算は通常queryで実行する。

617ではquery **37.15%減**、命令 **4.09%減**、通信 **23.80%減**。handler終端heapの最大観測は43,450,368 bytes。通信はCandid request+replyでHTTP/CBOR/署名を含まず、命令はCDK decode/encodeを除くhandlerカウンタ。単回時間で速度向上を断定しない。

3入力すべてで32層hidden・48個の保持state配列・最終hidden・logit・確率・unknown・ラベルが従来INT8版とbit一致。prefix準備も32層hiddenと72 state配列がbit一致。最終設定は失敗query、journal replayとも0。最大lock変更の既存誤判定（gold=yesに対しno）は改善していない。型安全な出力は3入力とも検証済み。一般的な判断精度を保証する評価ではない。

Wasm SHA256: `d76b23df9194c27ec75bcc750f097561f05569d9359f3f1a01f402d98f7afe86`。生結果は `artifacts/fused-stage-{prefix,617,insufficient,maximum}/report.json`、各queryのrequest/reply/metricも同じdirectory配下に保存。bit比較は `docs/fused-stage-*-results.json`、失敗を除外していないことの検証・演算別集計は `scripts/summarize_fused_stages.py`。

cacheなし132-tokenも全層完走。**700→508 query**、703,573,958,783→683,369,700,235命令、1,182,062,887→914,378,679 bytes、単回83.057秒、最大query 4,395,987,590命令。32層hidden・48保持state・最終hidden・判断はbit一致。失敗/replayは0。生測定は `artifacts/fused-stage-normal/report.json`。

Rust 27 tests、Python 34 testsが通過。channel配置、複数token chunk間のstate/conv継続、終端stateの省略、89-token MLPの幅制限、不正なshape、未対応adapterの拒否を含む。

## 32 queryへの残る制約

617の射影 `lora_integer` が2105億命令、MLP gate/upが1520億命令で、合計 **77.87%**。Delta統合段階が628億（13.48%）。今回の改善は通信とquery境界が中心で、整数dotそのものは変えていない。

現行の4655億命令をそのまま詰め直しても、5B/queryで最低94回が必要になる（通信・依存関係・CDK分の余裕を無視した下限）。32回の総予算1600億に入れるには、さらに**65.63%以上**の総命令削減が必要。32回の達成、あるいは不可能性を示した結果ではない。

次の大幅削減には行列積の方式変更が必要。回転W8A8は別の探索で局所RMSE改善を観測しているが、全層の判断精度・命令数は未検証で採用していない。最終層の不要token計算省略も未実装。出力レーン方式の失敗を踏まえ、方式変更は実Wasm命令と全層判断を併せて比較する。

## 実装

`--fuse-delta` はconv、Q/K RMSNorm、Delta再帰、gated normを `delta_stage_bf16` に統合する。Q/Kの共有関係を保ち、head群のconv重みだけをstable memoryから読む。入力には3行のconv履歴、QKV、z、decay、beta、F32状態を含める。出力はtoken-major gated hiddenと、継続時のみF32状態。conv履歴は入力の末尾3行をclientでそのまま保持する。prefix準備時にはDelta状態を省略しない。

INT8経路では非LoRAの `in_proj_a` / `in_proj_b` とゲート関数も `delta_gates_integer` に統合する。各射影のBF16丸め境界は維持する。固定adapterと異なりA/BにLoRAを持つmanifestは拒否する。

`--delta-head-cap 16` でもblob 2 MB、900,000 floatsの上限に合わせて偶数head群へ縮小する。32 headsを一括送信する方式ではない。

`--wide-mlp` は88 tokens以下のMLP gate/upの結合work予算を4.0Gから4.5Gへ広げる。ICのquery命令上限を引き上げる変更ではない。87-token実入力は全9216行をまとめられた。89-tokenでは上限超過を実測したため、この条件では従来の4.0G予算を使う。予算はMAC見積もりであり、実行命令数とは別。

## 不採用候補

整数dotのSIMD laneを出力列へ割り当てる重み配置を試したが、成功したshapeでも命令数が増加し、10 shapes中4つはquery上限を超えた。元のkernelへ戻した。生測定は `artifacts/output-lanes/probe/report.json`、再現用probeは `scripts/probe_projection_shapes.py`。レーン配置だけから高速化を推定していない。

89-token MLP全幅の試行には失敗queryが含まれるため、中間版 `artifacts/fused-delta-maximum` の成功query集計を最終性能として使わない。

## 再現

既に重みをupload・sealしたImajev専用local canisterだけを使う。Laya環境は触らない。

```sh
cargo test --offline -p imajev-runtime
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
cargo build --offline --release --target wasm32-unknown-unknown -p imajev-inference
icp canister install 4caro-hl777-77775-aaaba-cai --mode upgrade --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm --network local --identity imajev-local
.venv/bin/python scripts/validate_fused_delta.py --canister 4caro-hl777-77775-aaaba-cai
```

検証scriptは専用の新しいprefix cacheを作り、3入力の全32層hidden、保持state、判断を従来compact版とbit比較する。既存の測定directoryを使い回してreplayを新規実行と数えない。生成物はgitignore対象。
