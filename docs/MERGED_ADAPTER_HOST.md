# 固定adapterを準備時に統合するホスト候補

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
