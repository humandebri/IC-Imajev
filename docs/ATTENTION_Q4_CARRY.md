# Q head分割と共通A積の引き継ぎ

2026-10-04。レビュー修正 `1d02d62` の後、MLP/attention連結APIに `mlp_complete_attention_kv_q4` と `attention_finish_mlp_front_q4` を追加した。既存の2演算も維持する。実験canisterだけをupgradeし、moduleは `596a350763fb31e38944ecebe3dabe84a6511efe7de743598b57756286aa0f4d`。

前queryでMLP完了、normのblock256量子化、K/V、Qの元F32 LoRA A積、最初4 Q headを計算する。後queryは量子化済みnormとA積を使い、残り12 head、元順序のhead連結、out projection、次MLP前半を実行する。専用readout/adapterの精度、F32加算順、BF16境界は変更しない。attention gateの出力は元演算が既にBF16へ丸めているため、その値を可逆に2 bytesで転送する。A積はF32のまま保持する。

後queryに載せられるMLP行数が増えた。以前は主87 tokenで3328行、最大変更89 tokenで3072行までだった。新経路は全7 attention箇所（layer3/7/11/15/19/23/27）で5120/5376行が成立した。主87、情報不足80、最大変更89の42条件すべてで、K/V・MLP準備状態・完了後hidden/normが固定INT8出力と全ビット一致した。

最初のMLPを1792行まで済ませた状態からの計測。次表は各入力の全7層/2幅におけるhandler counter最大値で、CDKのCandid decode/encodeを含まない。全呼び出しも成功した。

|suffix tokens|MLP完了+KV+Q4 最大命令|Q12+out+次MLP前半 最大命令|観測heap最大 bytes|
|---:|---:|---:|---:|
|87 主617|4,651,933,969|4,724,838,643|4,139,319,296|
|80 情報不足|4,243,557,681|4,282,518,657|4,137,484,288|
|89 最大変更|4,761,099,690|4,846,296,355|4,142,202,880|

profileの42条件すべてで、前queryの `bridge_q_A_once`・`bridge_q_first4`・`bridge_attention_quantize_once` は各1回。後queryはQ Aを再計算せず、通常 `lora_matmul_A` はout projectionの1回だけで、次MLPのgate/up Aは別の `stream_input_A_once` として実行される。合成tensorでも4+12 headと一括16 headの全ビット一致を確認し、共有した各Q分割は独立計算に比べA重み読み出し655,360 bytesを省いた。

最初のsweepではMLP前半1280行を使った。主/情報不足の4352〜5120行は8条件とも成立したが、最大変更89 tokenは前queryでIC0522となった。失敗requestのbyte・SHA・エラーを保存した。前queryへ回す仕事を減らすため最初のMLPを1792行まで先に進め、上記42条件で成立を確認した。主の5120行で後queryは4,644,095,002命令だった。

これは単体連結経路の配分改善で、全体graphには未接続。54query全体の削減や総命令・総通信の改善を示したとは扱わない。元の一括16 Q headもA積は1回なので、A共有の削減効果は独立したQ分割との比較である。carryは大きくなり、主87 tokenの後query要求は約1.86MBになる。2MB frame検査を維持する。50/32queryは未達で、次にMLP/Deltaの隣接境界も連結して全体graphへ接続する必要がある。

Rustは候補featureで123 unit・15 integration・9 compile-fail doc test通過（外部fixture1件ignore）。Python codec3件、再開6件、境界dispatch3件も通過。Wasm buildでsource前後一致、4 kernel patchのvalidator確認。固定準備は721 update・4,065,416,192 bytes・78,739,929,194命令・260.586秒。質問推論は通常query、質問状態はclient-held。

## 再現と証拠

```sh
.venv/bin/python scripts/check_attention_mlp_bridge.py \
 --canister 6eydd-o3777-77775-aaama-cai \
 --wasm artifacts/prefix_codec/full-build-attention-q4-v1/full.wasm \
 --directory artifacts/prefix_codec/NEW-q4-proof \
 --q4 --front 1792 --layers 2,6,10,14,18,22,26 --begins 5120,5376
.venv/bin/python scripts/check_attention_q4_profiles.py \
 artifacts/prefix_codec/NEW-q4-proof --all-layers
```

- build/source/patch: `artifacts/prefix_codec/full-build-attention-q4-v1`。
- 固定cache準備: `artifacts/prefix_codec/attention-q4-preparation-v1`。
- 初回境界・失敗request: `artifacts/prefix_codec/attention-q4-check-v1`。
- 全42条件・source ZIP: `artifacts/prefix_codec/attention-q4-all-v2`。
- profile回数/coverage検証: `scripts/check_attention_q4_profiles.py`。

生成物はignore。Laya・保護対象の主canister・mainnet・remoteは変更していない。固定INT8との一致をBF16判断精度の同等性と混同せず、既存の重大変更見逃しが改善したとも扱わない。

候補moduleで既存全6条件の全体graphも保存hidden/state/final hidden/型付き判断/logits/確率が全ビット一致した。主の標準経路は62query、coldは290query。全体連結graphの3条件も再実行し、主54query・241,643,966,063命令・133,896,992 Candid bytes・36.917秒、情報不足54query、最大変更は標準62query。全条件で失敗0件・replay0件。分岐追加に伴う数千命令のcounter差を削減成果として扱わない。compact tailで返さないlayer30 hiddenを比較したとは扱わない。

証拠とsource ZIPは `artifacts/prefix_codec/full-attention-q4-proof-v1` と `artifacts/prefix_codec/rolled-attention-q4-regression-v1`。別ディレクトリ `artifacts/prefix_codec/attention-q4-main-replay-v1` で保存54件を再利用し、追加の推論query・命令・通信が0、final hidden/判断が同じことを確認した。module/cache確認の管理用readは推論query数と分離している。主canisterはRunning、module `36c04a57e9501eb9d32163c92d7725171191fa46a45a880ad03465993fceed73` のまま。

旧Q一括bridgeも `artifacts/prefix_codec/attention-q4-legacy-v1` で3入力・3072行を再検証し、K/V・MLP準備状態・完了後hidden/normが全ビット一致した。新opを指定しない呼び出しの互換性も維持した。

旧bridgeの3072行query counterは以前のmoduleより約0.8M命令増えた（約0.02%）。現在の54query graphではこの未接続APIを使わない。旧形式のdecodeでprefixの一時配列を組み直す箇所も、次の削減対象とする。
