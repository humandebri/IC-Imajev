# 固定adapterを準備時に統合するホスト候補

2026-10-06には統合したフルtext packを保存し、再検証した。最新の結果は末尾の「2026-10-06：統合重みを保存して再検証」を参照。以下は2026-10-02時点の実験記録。

2026-10-02。毎質問でF32 LoRA A/Bの積和を行う代わりに、固定baseとadapterを準備時に統合する候補をホストで実装・評価した。checkpointは変更せず、公式MLXモデルをメモリ上で読み、200射影の `original BF16 base.astype(F32) + 2 * (B @ A)` をF32で作り、対称per-row INT8へ量子化する。元のembedding INT8、norm、F32 readout、tokenizer、prompt、calibration.jsonは同じ。入力のbase算術は従来と同じper-token block256 A8で、重み変更と新しい中間状態量子化を同時に導入していない。

これは元の二段のBF16境界・F32積和を保存する方式ではない。現在の採用canisterは元のF32 LoRAを別に計算する経路のまま。新たなweight packやWasm全層・命令数の測定は未実施で、この候補をquery数の改善には計上しない。

`scripts/host_integer_reference.py --merge-adapter`を追加した。元の200 LoRALinearの出力を、準備済みの統合INT8射影へ置き換える。元のsource model、adapter、readoutは同じ固定revisionのまま。未統合時の既存CLIも保存した。MLX roundのhalf境界6値をNumPy RNEと比較し、ties-to-even一致を確認した。derived digestはフルpack hashではないため、候補reportの `weight_pack_hash` をnullとし、元packと200射影のdigestを別に記録する。

同じtoken列・input hash・選択肢・rotations=1・校正versionの23条件を、以前の未統合ホストINT8版と比較した。

| 項目 | 実測 |
| --- | ---: |
| 判断一致 | 23/23 |
| 元順序のgold付き問題 | 7 |
| 元順序の正解数：未統合→候補 | 6/7→6/7 |
| 校正済み確率の最大絶対差 | 0.0763903119 |
| host型付き出力・finite・確率和の検査 | 23/23 |
| 統合した固定adapter射影 | 200 |

最大lock問題の誤判定は残った。順序依存・unknownの挙動や確率も、一般に同じと保証する結果ではない。7.64ポイントの確率変化があるため、bit一致、校正精度維持、未知問題に対する精度保証とは報告しない。型の妥当性を判断の正しさとして扱わない。この候補のホストnonlinear graphとWasm間の差も未検証である。

最初の主問題は同じlikelyで、候補likely=0.8058936、比較する未統合host版=0.8079622。まず1質問で確認し、次に全23条件を実行した。単独実行と23条件実行で、200射影のdigestが一致した。メタデータは正式な固定model lock、元pack SHA256、script hashと併せて保存した。

200射影digest: `89744c9fd0995012cdd01c900efe7563cca8ac65bdb54cf4bde5b976d5ab78cf`。生データ `artifacts/merged-adapter/{first,orders,comparison}.json`。元packは `364e3aa3f69a61a69e6df62e3e2a71d4f55457f1a8e1c16934aefc6130f04c84`。生成物はgitignore。主canister・Laya・mainnet・Git remoteをこの候補で変更していない。

次は独立したderived packと診断canisterで実Wasm算術を比較するか、元の精度を保つLoRA Aのclient-held cacheを比較する。現在の採用版の305 query / 全bit一致の結果は [ROPE_REUSE.md](ROPE_REUSE.md)。50/32 queryは未達。

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python scripts/host_integer_reference.py --arithmetic int8 --merge-adapter --first-only --output artifacts/<未使用first名>.json
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python scripts/host_integer_reference.py --arithmetic int8 --merge-adapter --orders --output artifacts/<未使用orders名>.json
```

## 2026-10-06：統合重みを保存して再検証

`scripts/export_merged_adapter.py`で、元のBF16射影とF32 LoRAの200組をF32で統合し、per-row INT8へ量子化したフルtext packを別途保存した。統合対象以外のtensorは採用INT8 packからそのままコピーする。元の2つのpackのSHA256・model lockを検証し、生成packも書き込み後に再ハッシュした。元checkpointを上書きしない。

生成物は `artifacts/merged-adapter-v2/model.{pack,manifest.json,provenance.json}`。pack SHA256は `dd00c307ff65f2ae03ce5ee935d1dcfc7e298b6bd496e0aae2be1d81cd04cad6`。保存版にはLoRA A/B tensorがなく、`host_integer_reference.py --merged-pack`で200個のLoRA経路を迂回して、保存したINT8射影を実際に読み込む。

| 項目 | 今回の結果 |
| --- | ---: |
| pack容量：未統合→統合 | 4,702,451,200→4,214,863,360 bytes |
| 削減容量 | 487,587,840 bytes（約488 MB） |
| 統合準備時間（source hash検査・コピー込み） | 201.21秒 |
| 省けるLoRA積和/token | 121,896,960 |
| 密な射影の積和削減率 | 3.3026% |
| 23条件の判断一致 | 23/23 |
| 元順序・gold付き7問の正解数 | 6/7→6/7 |
| 校正済み確率の最大絶対差 | 0.0763903119 |
| 既存native Rustでの部分射影と整数oracle | bit一致、最大差0 |

積和削減率は248個の密なbase射影の3,569,090,560 MAC/tokenとLoRAの合計に対するもの。attention・非線形演算・通信を含む推論全体の削減率や、Wasm命令数の削減率ではない。

判断の比較は、今回改めて実行した `before.json` と `after.json` を同一token列・prompt・選択肢・校正versionで照合したもの。10問を選択肢の循環順序で23条件に増やしており、独立した23問ではない。最大lock問題の誤判定は残った。一般の判断精度や校正精度を維持する保証はない。

Rust検査は実入力から4 token、layer-0 QKVの64出力行（開始行8）を取り、既存 `linear_integer_bf16` を統合packで実行した。block256の整数積和・F32 scale・BF16丸めを独立NumPy oracleとbit比較した。全層Wasm検証ではない。現在のLoRA専用の融合query経路へ適用するには別途対応が必要。

```sh
# 最初のexportはMetalにアクセスできる環境で実行する。
.venv/bin/python scripts/export_merged_adapter.py --output artifacts/<未使用directory>/model
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python scripts/host_integer_reference.py --arithmetic int8 --merged-pack artifacts/<directory>/model --orders --output artifacts/<directory>/after.json
.venv/bin/python scripts/check_merged_adapter_native.py --directory artifacts/<directory>
.venv/bin/python scripts/compare_merged_adapter.py --directory artifacts/<directory>
```

`compare_merged_adapter.py`は同じdirectoryの `before.json` と `after.json` を比較する。両variantの `*-timing.json` がある場合は、初回forwardを除いた同一入力の測定も集計する。今回のraw検証と比較結果は `artifacts/merged-adapter-v2/{native-check,comparison}.json`。

速度はGPU処理を重ねず、同じ132-token主問題を各variantで5回実行し、初回を除いた4回を集計した。未統合のforwardは6.303/7.744/3.904/3.217秒、統合版は2.981/2.745/2.586/2.526秒。中央値は5.103→2.665秒だったが、未統合側の最大/最小比が2.41と大きく、今回の測定から改善率を断定しない。`comparison.json`でも `timing_inconclusive_due_to_spread=true` を記録した。

続いて同日のローカルICで全32層を各3回実行し、handler命令数は602,002,923,942→529,050,777,852（12.1182%減）、query数は双方772回だった。通信込み時間の中央値は178.275→186.222秒で、速度改善は確認できなかった。判断は全回likelyで、確率差は最大3.47ポイント。条件と生データは [MERGED_ADAPTER_IC.md](MERGED_ADAPTER_IC.md)。採用canisterには適用していない。

その後、32 queryの最適化経路でも同じ3入力を各モデル3回ずつ測り、命令数が13.21〜13.52%減った。統合版の固定prefixは実際のcanisterで再計算した。所要時間は入力ごとに改善・悪化が分かれ、速度改善は断定していない。詳細は [MERGED_ADAPTER_QUERY32.md](MERGED_ADAPTER_QUERY32.md)。
