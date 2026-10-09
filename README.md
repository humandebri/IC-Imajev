# IC-Imajev

Imajev-4Bのテキスト判断をInternet Computerのcanister上で実行するプロジェクトです。Qwen3.5-4B＋判断用LoRA＋専用readoutを、INT8基盤・F32 LoRA/readoutのRust/Wasmランタイムで全32層計算します。入力の抽出・token化と実行制御はclientが担当し、モデルの数値演算は通常queryで行います。

## Imajevとは

Imajevは、文章・記録・写真などの証拠と、質問・選択肢を受け取り、あらかじめ定義した答えを直接スコア化する判断モデルです。商品情報と写真の矛盾検査や、問い合わせの振り分けなどを想定し、選択肢ごとの確率と、証拠不足を表すunknownを返す仕組みを持ちます。

このリポジトリは固定したImajev-4Bの**テキスト判断経路**を扱います。画像encoderや自由な文章生成を今回のIC評価には使っていません。基盤はQwen3.5-4B、LoRAはrank 64・alpha 128、decision readoutはF32の256×2,560行列です。最後の判定位置のhiddenから必要な選択肢codeを評価するため、文章生成とその後の分類を挟みません。配布元の全機能やbenchmark性能を、IC上で確認したという意味ではありません。

基盤・adapter・tokenizerのrevisionとSHA-256は[MODEL_LOCK.json](MODEL_LOCK.json)で固定します。Imajevの用途、学習済み版、LoRAとreadoutの役割は[詳しい説明](docs/PROPOSAL_FILTER_REPORT_20261007.md#imajevとは何か)を参照してください。

## IC上の実行構成

モデルの重みをcanisterへ導入し、固定重みのcache・配置・lookup表等を準備updateで作ります。入力ごとの推論は通常queryに分割し、hidden・attention K/V・DeltaNet状態・演算途中の値をclientが保持して次のqueryへ渡します。query間の状態を別入力へ誤って使わないよう、model・pack・input・進捗・checksumを照合します。

共通する先頭token列はprefixとして全層で事前計算し、その状態からsuffixを継続します。prefixは入力を短く要約する処理ではなく、同じ先頭部分の計算を再利用する仕組みです。準備費用は入力ごとの推論query数と分けて数えます。

都度払いのupdate `infer`も別経路として実装しています。利用者canisterの外部update 1回に対して、検証した3入力の内部workerは5／4／5回でした。この値は通常queryの32／50回と別の測定です。[update推論の実測と利用手順](docs/PAID_UPDATE_INFERENCE_MEASURED.md)。

## IC向けの主な最適化

|対象|行ったこと|
|---|---|
|重み容量・読み出し|テキスト経路のINT8 packを約4.70 GBへ縮小。固定INT8/F32重みの復元・検査・配置を準備時へ移し、queryでは借用|
|基盤・LoRAの積和|block256のINT8整数dot、Wasm SIMD、出力/token間のロード共有、終端1 token用kernel|
|重複計算|同じ入力の量子化・scale・LoRA A積、RoPE等を共有。queryをまたぐ途中結果も再利用|
|通信とprefix|BF16/F32の可逆codec、prefix状態のlog・圧縮packetで、数値を保ちながら転送量を削減|
|query配置|MLP・Delta・Attention・残差/normを融合し、層をまたぐ仕事量を命令・通信予算内へ配分|
|終端処理|判定に不要な最終token以外の演算・中間返信を省き、final normとreadoutまで接続|

INT8化は数値計算を変える最適化です。一方、通信圧縮・再利用・融合は採用済み整数方式に対する数値一致を検証しています。両者を混ぜて、元の未量子化モデルとbit一致すると説明しません。最適化で同じ誤判断が再現されることもあり、query削減を判断精度の改善とは数えません。

[最適化の仕組みと検証範囲](docs/PROPOSAL_FILTER_REPORT_20261007.md#icで実行するために行った最適化)、[手段ごとの測定履歴](docs/COMPUTE_REDUCTION.md)、[従来READMEの最適化履歴](docs/README_OPTIMIZATION_HISTORY.md)に詳細があります。

## 現在の検証結果（2026-10-07）

BOOM DAO proposal #500〜#660の数値参加条件ハーネスで、モデル対象50件を同一入力18種類にまとめ、新規推論を実行しました。全入力は質問・選択肢・chat特殊記号込みで67〜128 tokensです。

|実測項目|結果|
|---|---:|
|32-query経路|7入力|
|50-query経路|11入力|
|18入力の推論query合計|774回（保存済みの同じ入力では3,703回）|
|今回のprefix・codec準備|11 bank、990 query|
|準備＋成功推論|1,764 query|
|入力token合計|18種類で1,861、50件を個別に数えると4,799|
|再現性|18種類すべてで前回とraw logitsがbit一致|

32／50 queryはprefix準備後の1入力の呼び出し数です。重みupload・固定準備update・module照合・探索費用は含みません。今回の#617はprefix 38＋suffix 59＝97 tokensで32 query、#505は48＋80＝128 tokensで50 queryでした。固定prefix 27＋suffix 59＝86 tokensは以前の構成で、Imajev全体の一律上限ではありません。128 tokensも今回のハーネスの予算で、任意の入力を32 queryで処理できる保証ではありません。

測定はlocalhostのICで行い、各queryのhandler命令50億未満、Candid要求・返信各200万bytes未満、記録されたheap 4 GiB未満を照合しました。mainnetの遅延・throughput・費用や応答のcertificationはこの評価の対象外です。使用した凍結Wasmとdriverの識別情報、全18入力のprefix/suffix表、時間・資源使用量は[評価報告書](docs/PROPOSAL_FILTER_REPORT_20261007.md)にあります。下記の基礎build手順だけで、同じ32／50-query構成が再現されるという意味ではありません。

## 危険なproposalを通さないフィルター

今回はapproveを通過、rejectとholdを停止という方針で評価しました。全161件の処理結果は次の通りです。

|判定|件数|扱い|
|---|---:|---|
|approve|2|数値参加条件のフィルターを通過|
|reject|10|停止|
|hold|61|停止。必要に応じて追加確認|
|対象外・除外|88|未検査。通過とは扱わない|

モデル対象50件のうち、大幅なstake・最低投票lockの制限強化とレビューした32件は、reject 10件・hold 22件ですべて停止する方針になります。この32件はエージェントによる条件付きレビューであり、人間の正解ラベルに対する危険検出率ではありません。通過した#656・#659は数値上の参加条件を改善しますが、目的・支配権・投票力配分等の確認は残ります。

holdは危険の確定ではなく、低スコア・証拠不足・判定範囲外・入力予算超過を含みます。内部unknownを直接採用するのではなく、raw A/B差の未校正スコア閾値0.6とTool側のgateで最終判定を出します。後続処理への接続・自動停止は未実装で、投票は実行していません。

[結果・根拠照合・フィルター評価](docs/PROPOSAL_FILTER_REPORT_20261007.md)を参照してください。

## 詳細資料

- [Proposal filter evaluation report (English)](docs/PROPOSAL_FILTER_REPORT_20261007.en.md)
- [Local path portability (English)](docs/PATH_PORTABILITY.en.md)
- [ローカルパスの移植方針](docs/PATH_PORTABILITY.md)
- [proposalフィルターの判定一覧・集計・検証結果](docs/evidence/proposal-filter-20261007/README.md)
- [全層推論・重み導入の基礎](docs/FULL_INFERENCE.md)
- [32-query経路の配分と検証](docs/QUERY32_PROGRESS.md)
- [ランタイムの構成と再利用知見](docs/runtime/README.md)
- [初期の最適化記録](docs/EXACT_OPTIMIZATION.md)

以下は環境構築・基礎検証の手順です。個別の最適化経路は、対応する文書のmodule・feature・準備条件を確認してください。

## 環境

2026-10-01、Apple Silicon / 32 GiB RAM、Python 3.12.14、MLX 0.32.3、mlx-vlm 0.7.1、icp-cli 1.0.2、ic-cdk 0.20.3。Python依存を `requirements.lock`、Rustを `Cargo.lock` で固定します。参照Pythonソースは固定commitを `vendor/` に取得します。重み、ログ、秘密鍵はGit管理しません。

```sh
cd IC-Imajev
python3 scripts/bootstrap.py
UV_CACHE_DIR="$PWD/.cache/uv" uv venv --python 3.12 .venv
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python -r requirements.lock
python3 scripts/acquire.py --weights
.venv/bin/python scripts/inventory.py
```

`acquire.py` は既存のMODEL_LOCKを書き換えません。約10 GBを取得します。検証入力は`benchmarks/cases.json`に保存しており、外部の比較用checkoutは不要です。

## 公式ホスト推論

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python scripts/reference.py \
  --first-only --output artifacts/reference-first.json
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python scripts/reference.py \
  --orders --output artifacts/reference-orders.json
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python scripts/reference.py \
  --serving --orders --output artifacts/reference-serving.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/capture_delta.py
HF_HUB_OFFLINE=1 .venv/bin/python scripts/capture_projection.py
```

出力が既にある場合、参照ツールは上書きを拒否します。再実験には別の `--output` を使ってください。capture用出力が既にある場合も移動・保存してから再実験してください。全forwardと公式サーバーのshared-prefix経路を分けます。すべて1質問・画像なし・rotations=1・calibration.jsonです。`--orders` は独立した選択肢並べ替えを評価し、平均しません。

## RustとWasm

```sh
cargo test --workspace --locked
cargo build --release --locked --target wasm32-unknown-unknown -p imajev-inference
cargo build --release --locked -p imajev-client -p imajev-runtime
cargo run --example export_did -p imajev-inference --locked > canisters/inference/inference.did
.venv/bin/python scripts/export_readout.py
.venv/bin/python scripts/export_projection.py
```

runtimeはF32積和、BF16丸め、RMSNorm、SiLU、causal depthwise conv、causal attention、Gated DeltaNet recurrence、専用readout/softmaxを実装します。partial RoPE、層接続、全言語pack、client-held再開も実装済みです。実行可能なprimitiveは`target/release/primitive request.bin response.bin [manifest.json pack]`でnative比較できます。

## 専用ローカルcanister

`icp.yaml` はこのプロジェクトだけのnetworkを8001に置きます。起動前にポートが空いていることを確認してください。ほかのnetworkを停止しないでください。実endpointを `status` で確認します。

```sh
icp network start -d
icp network status --json
icp identity new imajev-local --storage plaintext --output-seed artifacts/imajev-local.seed
icp identity principal --identity imajev-local
icp canister create imajev-inference --identity imajev-local --json
icp canister create --detached --identity imajev-local --json
```

各新規canisterを、上で得たowner principalをinit引数にして `icp canister install <id> --identity imajev-local --mode install --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm --args '(principal "<owner>")'` でinstallします。既存canisterに `install` や `reinstall` を実行しないでください。

```sh
icp identity export imajev-local > artifacts/imajev-local.pem
chmod 600 artifacts/imajev-local.pem artifacts/imajev-local.seed
.venv/bin/python scripts/check_canister.py --url http://localhost:8001/ --canister <readout-id> --upload
.venv/bin/python scripts/check_projection.py --url http://localhost:8001/ --canister <projection-id> --upload
```

`--upload` は初回だけです。以後は省略し、同じimmutable packを使います。prepare/upload/sealはupdate、計算はowner専用の通常queryです。クライアントが中間状態と再送用binaryを保持します。重みのSHA256をchunkごととpack全体で検証します。状態checksumは認証の代替ではありません。

初期のreadout／projection検証で作成したnetworkのURLは `http://localhost:8001/`、readoutは `4caro-hl777-77775-aaaba-cai`、projectionは `4fbx2-kt777-77775-aaabq-cai`。ローカルcanister IDは別networkで同じ文字列になり得るため、URLとセットで指定してください。

`check_contracts.py` は初期検証の専用URL/IDを使用する回帰確認です。型付きchoice、unknown、pack違い、checksum破損、重複選択肢、非ゼロ行タイル、同じqueryの再送を確認します。`scripts/summarize.py` が生ログからGit管理する集計を作ります。

mainnet、push、PR作成は行っていません。

初期のupload検証用の追加canisterは `4qggx-l3777-77775-aaaca-cai`（同じ8001）。`check_upload.py` がpack全体hash拒否と再準備を確認した。既にseal済みのcanisterではこの初回検証を再実行せず、別の新規canisterを指定してください。

Canister の公開 method 名と互換名の対応は [API の命名](docs/CANISTER_API_NAMES.md) を参照してください。
