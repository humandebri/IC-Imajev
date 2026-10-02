# IC-Imajev

**最新の追加改善（2026-10-02）：INT8直接ロードとtile変更で、cacheなし708 queryの命令数を7393億→7110億へ3.82%削減。共通45-tokenのclient cache利用時は611 query・4926億命令・0.873 GB。ただし初回cache準備は別に578 query・2557億命令・0.464 GB。3入力で全32層hidden/state/判断が以前のINT8版とbit一致。50 queryは未達、速度向上は未確認。** [複数方向の実装・不採用候補・初回費用](docs/DIRECTIONS.md)。

Imajev-4Bのテキスト判断をInternet Computerの通常queryで全32層実行する実験です。INT8 base＋F32 adapter/readoutを固定し、中間状態はclientが保持します。

前段階では整数積和を全モデルへ接続し、BOOM DAO 617で総命令数を当初から72.24%削減した。 132 tokens・rotations=1は1.022兆命令、932 query、Candid通信1.671 GB、単回107.274秒。50 queryは未達です。23件のホスト診断で同じpackのF32演算とラベル一致、元順序の正解付き7問は両方6/7。ただし確率差は最大約8.48ポイントで、一般精度・校正の同等性を保証しません。[整数方式・実測・再現](docs/INTEGER_ARITHMETIC.md)。

数値を変えないF32演算経路も残しています。最新の可逆通信版は1.762兆命令・1,116 query、全層/state/logit/確率が以前の可逆版とbit一致。[追加削減と50-query予算](docs/FIFTY_QUERY_ANALYSIS.md)、[全query比較](docs/multi-token-compact-lossless-results.json)。整数方式は明示した場合に使い、通信は可逆のまま演算差を評価します。

既存INT8通信は可逆BF16通信版から候補確率が最大約0.121変わる別の量子化です。[通信INT8の条件](docs/ACTIVATION_INT8.md)。全モデルの構成・重み準備は [docs/FULL_INFERENCE.md](docs/FULL_INFERENCE.md)。旧最適化の測定履歴は [docs/EXACT_OPTIMIZATION.md](docs/EXACT_OPTIMIZATION.md)、[docs/LAYA_COST_ANALYSIS.md](docs/LAYA_COST_ANALYSIS.md)。

測定と残作業は [docs/STATUS.md](docs/STATUS.md)。採用revision、ファイル容量、SHA256、tensor形状は [MODEL_LOCK.json](MODEL_LOCK.json)。`PLAN.md` の当初計画を残しています。

## 環境

2026-10-01、Apple Silicon / 32 GiB RAM、Python 3.12.14、MLX 0.32.3、mlx-vlm 0.7.1、icp-cli 1.0.2、ic-cdk 0.20.3。Python依存を `requirements.lock`、Rustを `Cargo.lock` で固定します。参照Pythonソースは固定commitを `vendor/` に取得します。重み、ログ、秘密鍵はGit管理しません。

```sh
cd /Volumes/KINGSTON/ICP/IC-Imajev
python3 scripts/bootstrap.py
UV_CACHE_DIR="$PWD/.cache/uv" uv venv --python 3.12 .venv
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python -r requirements.lock
python3 scripts/acquire.py --weights
.venv/bin/python scripts/make_benchmark.py
.venv/bin/python scripts/inventory.py
```

`acquire.py` は既存のMODEL_LOCKを書き換えません。約10 GBを取得します。`make_benchmark.py` はLayaの保存済み結果を読み取るだけです。別マシンではベンチ入力をリポジトリに保存したまま使い、このコマンドを省略してください。

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

今回作成したnetworkのURLは `http://localhost:8001/`、readoutは `4caro-hl777-77775-aaaba-cai`、projectionは `4fbx2-kt777-77775-aaabq-cai`。ローカルcanister IDは別networkで同じ文字列になり得るため、URLとセットで指定してください。

`check_contracts.py` は今回の専用URL/IDを使用する回帰確認です。型付きchoice、unknown、pack違い、checksum破損、重複選択肢、非ゼロ行タイル、同じqueryの再送を確認します。`scripts/summarize.py` が生ログからGit管理する集計を作ります。

mainnet、push、PR作成は行っていません。

upload検証用の追加canisterは `4qggx-l3777-77775-aaaca-cai`（同じ8001）。`check_upload.py` がpack全体hash拒否と再準備を確認した。既にseal済みのcanisterではこの初回検証を再実行せず、別の新規canisterを指定してください。
