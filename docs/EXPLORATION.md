# 追加探索：回転、整数dot、状態とGQAの通信

2026-10-02。採用版の全推論コストは引き続き132-token通常実行711,017,272,458命令／708 query、45-token prefix準備済み492,550,510,942命令／611 query。今回は新しい全層高速化を達成した記録ではなく、候補を実測で絞った記録である。

## 最新カーネルの内部計測

専用local `http://localhost:8001/` の `4caro-hl777-77775-aaaba-cai` で、保存済み実入力の整数射影10形状を再実行し、全312射影queryの出現数でspanを展開した。すべて保存済み出力とbit一致。以下は標本からの推定であり、別々のspanの親子を加算しない。

| stage | 推定命令 | 通常実行全体に対する比率 |
| --- | ---: | ---: |
| integer dot（ロード・内部処理を含む） | 374,353,358,080 | 52.650% |
| LoRA A F32積和 | 25,730,064,040 | 3.619% |
| LoRA B F32積和 | 31,551,598,200 | 4.438% |
| 射影の入力＋出力SHA-256 | 38,503,010,304 | 5.415% |
| block scale/sum | 14,216,101,120 | 1.999% |
| base stable read | 9,737,944,160 | 1.370% |
| activation量子化 | 5,141,872,112 | 0.723% |

計測用buildの通常stepは、同じ標本を使った採用buildに対して加重命令数+0.0244%。profile呼出し自体はさらに+2.6763%だった。spanは原因探索に用い、全モデル改善として計上しない。原記録は `artifacts/bottleneck/explore-current-profile/report.json`、集計は `docs/exploration-results.json`。

INT8ロード＋符号拡張を明示的な `i16x8_load_extend_i8x8` に置換した候補も同じ10形状で比較した。**Wasm SHA256が採用版と完全に同じで、命令削減は0。** コンパイラが既に融合していた。候補のソースは採用せず復元した。

両実験後に元のsourceとWasmを復元し、証明付きmodule hashを確認した。採用hashは `806c1006b7effc726b0cb5ab77e7f4735fdbdb69308e6000d26856704e1f9664`。Layaは変更していない。

## 回転を実入力で試す

[QuaRot](https://arxiv.org/abs/2404.00456)と[SpinQuant](https://arxiv.org/abs/2405.16406)の考え方を参考に、256列ごとの符号付きHadamard変換を入力と元BF16重みの両方へ適用した。base射影単位なので、DeltaNet全体やLoRA/readoutの座標変換を仮定していない。量子化前の直交変換による積の保存も確認した。

617、情報不足、重大変更の3入力。層0/3/12/15/24/27/31の各46射影、最大16 token行×32出力行を間隔を空けて抽出し、固定seed 0/1/2で各138試行。学習やgoldによるseed選択は行っていない。評価基準は元BF16重みと固定実activationをF64で積和したbase出力。adapter・非線形・誤差の層間伝播は含まない。

| 方式 | 617：改善試行／138、RMSE比中央値 | 情報不足 | 重大変更 |
| --- | ---: | ---: | ---: |
| 回転W8＋A8 block256 vs 現方式W8A8 | 138、0.538 | 138、0.548 | 138、0.562 |
| 回転W8＋元精度activation vs 未回転W8＋元精度activation | 116、0.848 | 115、0.850 | 120、0.870 |
| 回転W8＋A8 token単位scale vs 現方式W8A8 | 103、0.801 | 106、0.772 | 111、0.788 |

通常のblock256 W8A8では414/414試行でRMSEが減少した。しかし重みだけの量子化では悪化例があり、token単位scaleでは94/414試行が悪化した。後者はF32 scale/sumを減らす候補だが、現方式より誤差が大きい例があるため採用していない。

これは**射影の標本誤差**であり、判断精度・確率校正・canister命令削減の証明ではない。入力は従来経路で生成した固定値で、回転版を全層伝播させた入力ではない。全32層・専用readoutまでのA/B、未使用問題、重大変更・unknown・選択肢順の評価が必要。INT8のまま回転しても積和数は同じで、オンライン回転の費用も生じる。INT4へ変更していない。

## 通信の不要分を実データで数える

保存済みqueryをdecodeし、候補の値配列を既存lossless BF16 codecで再encodeした。新形式をcanisterへ実装した測定ではなく、同じheaderを仮置きしたframe差分であり、将来のCandid費用や命令削減とは区別する。

| 候補 | prefix利用時の省略bytes | 通常実行 |
| --- | ---: | ---: |
| 終端DeltaNet stateの返送を省略 | 51,903,564 | 51,903,564 |
| ゼロ初期stateの入力を省略 | 対象外（prefix stateが必要） | 26,738,688 |
| GQAの4 query heads間でK/Vを共有 | 13,787,136 | 13,787,136 |
| 合計 | **65,690,700（元通信の7.53%）** | **92,429,388（7.41%）** |

DeltaNetは全24層の各headが1 token chunkで完了することを確認した。終端stateは今回の単発判断では以後の数値計算に使わないが、prefix準備、継続生成、複数chunkには必要。省略モードを実装する際は用途を明示し、state比較用の経路も保持する。

F32終端stateのzlib圧縮は50,331,648→47,245,802 bytes（約6.1%減）に留まった。Wasm codec費用は未測定で、圧縮より返送省略を優先する。

GQAでは重複K/Vが全bit一致することを確認した。132-tokenの全16 query headsを、16Q＋4K＋4Vとして格納すると811,008 floatsとなり、現900k-float制限内へ収まる。8層のAttentionを各2 queryから1 queryへまとめる候補になるが、新APIの入力検証・実際の命令上限・出力比較は未実装。

prefix Attentionの捨てる45 query位置は、全score pairsの11.79%。現在のAttention本体はcache-hit全体の3.87%なので、これ単独で大幅削減は見込まない。最終層全体も約3.02%であり、最後のtokenだけ計算する改善の上限はその内側にある。

## 次の優先順位と再現

1. 数値保存の通信改善として、終端state省略・GQA共有・演算融合を実装して通常queryで比較する。
2. 命令数の主対象はinteger dot。明示load融合は既に最適化済みなので、同じ書換えを繰り返さず、事前重み配置と実際のload/local操作を調べる。
3. 回転W8A8は精度面で有望な候補。全層A/Bとオンライン変換費用を測り、量子化簡略化の悪化例を解消できるか評価する。

```sh
.venv/bin/python scripts/explore_fused_load.py --canister <専用local ID>
.venv/bin/python scripts/explore_fused_load.py --canister <専用local ID> --profile
.venv/bin/python scripts/explore_rotation.py
.venv/bin/python scripts/explore_rotation.py --source artifacts/prefix-hit-diagnostics/record-19 --output docs/rotation-insufficient-screen.json
.venv/bin/python scripts/explore_rotation.py --source artifacts/prefix-hit-diagnostics/record-11 --output docs/rotation-maximum-screen.json
.venv/bin/python scripts/explore_query_payloads.py
.venv/bin/python scripts/explore_query_payloads.py --source artifacts/layout-full-617 --output docs/query-payload-standalone-exploration.json
.venv/bin/python scripts/summarize_exploration.py
```

実験は既存の固定モデル・実query journalを前提とする。CLIのWasm実験は指定local canisterを一時upgradeし、finallyで元へ復元する。JSON・raw journal・候補Wasmはgitignore対象。新規全層推論、50/32 query達成、判断精度の改善を今回の結果として報告しない。
