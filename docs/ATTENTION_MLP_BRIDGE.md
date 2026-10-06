# Full attentionをまたぐMLP継続

2026-10-04。50queryへ向けて、専用feature `experimental-attention-mlp-stream` とclient-held通常queryの2つの演算を追加した。実験canisterは `6eydd-o3777-77775-aaama-cai`、module `778a85cf338729b1c9a2af39a556661ac9463a98b4238953bad5e94d9d64beb6`。この新経路は全体graphへ未接続。全体50queryの成立を示す成果ではない。

`mlp_complete_attention_kv` は入力/product整数・scales・F32 A部分和を保持したMLP streamを完了し、次層attentionのK/Vと入力のblock256整数・scalesを作る。MLPのBF16 hidden/norm・K/V・量子化済みnormを返す。`attention_finish_mlp_front` はそれを受け取り、量子化を再実行せずQ/GQA/gate/out projectionを完了、次層MLPの指定行数までを実行する。残りMLPは既存stream finishへ渡せる。prefix K/Vはクライアントが保持し、元のhead/position順序でqueryへ送る。追加の中間量子化は行わない。

実canister生成の保存入力と独立MLP/attention出力を使い、K/V・MLP継続状態・finish後hidden/normをビット比較した。最初のMLPは1024行まで準備済み、続くqueryは残り8192行とK/Vを処理する。

|suffix token|MLP完了+KV命令|次attention+MLP前半の成立例|次query命令|
|---:|---:|---:|---:|
|87 主617|4,077,185,607|3328行|4,868,399,973|
|80 情報不足|3,726,625,281|3328行|4,400,090,372|
|89 最大変更|4,168,699,719|3072行|4,916,256,716|

主は2560/2816/3072/3328行、情報不足はそれに加えて4096〜5120行、最大変更は2560/2816/3072行で成立。主と最大変更の4096〜5120行、最大変更3328行はIC0522を記録した。最初の層3のsweepだけで別層も同じ境界が成立すると仮定しない。

全7箇所（MLP layer2/6/10/14/18/22/26→attention layer3/7/11/15/19/23/27）の4352/4608行も測った。情報不足の14成立条件は全出力一致、主・最大変更の28条件は命令上限超過。失敗要求は個別保存して次のqueryで上書きしない。最初のsweep v2は失敗SHA/エラーを記録したが、同じindexの次要求がファイルを置き換える問題があり、診断scriptを修正してv3/v4では失敗byteを別保存する。

候補と同じruntime featureで121 unit・15 integration・9 compile-fail doc testが通過（外部carry fixture1件ignore）。Python codec2件が通過し、1/80/87/89 tokenのframe境界・負ゼロ・整数/scaleを検査した。4つの整数/F32 kernel patchをvalidatorで確認し、実験canisterだけをupgradeした。固定モデルcacheの再準備は721 update・4,065,416,192 bytes・78,739,929,194命令・261.905秒で、質問推論と分離して記録した。

新moduleの既存全6条件が全層保存出力・保持state・最終hidden・型付き判断・logits/確率でビット一致、失敗/replay0。既存の最大変更の見逃しも同じ。compact tailの非返却layer30 hiddenを比較したとは扱わない。新APIだけで判断精度改善・全体query削減は主張しない。

## 再現と証拠

```sh
.venv/bin/python scripts/check_attention_mlp_bridge.py \
 --canister 6eydd-o3777-77775-aaama-cai \
 --wasm artifacts/prefix_codec/full-build-attention-bridge-v1/full.wasm \
 --directory artifacts/prefix_codec/NEW-bridge-measurement \
 --begins 2560,2816,3072,3328
```

- source/build/patch：`artifacts/prefix_codec/full-build-attention-bridge-v1`。
- 固定cache準備：`artifacts/prefix_codec/attention-bridge-preparation-v1`。
- 低い境界の3条件：`artifacts/prefix_codec/attention-bridge-small-front-v4`。
- 全7箇所の診断：`artifacts/prefix_codec/attention-bridge-check-all-v3`。
- 既存全6条件とsource ZIP：`artifacts/prefix_codec/full-attention-bridge-proof-v1`。

次にQの一部も前のqueryへ移す案を検証する。K/Vだけを移す現在の配分では主のMLP前半が3328行までで、次のMLP finish/Delta融合へ十分な空きを作れない。QのLoRA Aも一度だけ計算して引き継ぎ、元のF32加算順・BF16境界・GQA位置を維持する必要がある。単にquery平均命令数を5Bで割る計画は実際の境界・2MB/frameの成立を証明しない。

生成物はignore。Laya・保護された主canister・mainnetに変更はなく、push/PRも行わない。

新moduleで既存の連結graphも3条件を新規通常queryで再検証した。主54query・241,643,971,840命令・133,896,992 bytes、情報不足54query、最大変更は標準62query。全保存出力・判断・確率ビット一致、失敗/replay0。証拠とsource ZIPは `artifacts/prefix_codec/rolled-attention-bridge-regression-v1`。新APIが全体へ未接続であることと、現行54query経路がこのmoduleでも成立することを分けて報告する。

表の命令数はhandler内counterで、CDKのCandid decode/encodeを含まない。単体の成功境界を別層・別入力へ一般化せず、50queryは通常query全体完走・各queryの上限・全出力一致を実測してから判定する。
