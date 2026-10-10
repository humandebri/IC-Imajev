# update推論の命令数診断（履歴）

2026-10-06のprefix27／38構成に対する診断記録。3入力のworker handler命令合計をそれぞれ100,000,000,000以下にする試験の初期測定であり、下表では目標未達。現行prefix5構成の性能値として扱わない。

現行構成の検証は[PAID_PREFIX5_VERIFICATION.md](PAID_PREFIX5_VERIFICATION.md)、ランタイムの構成と検証範囲は[runtime/README.md](runtime/README.md)を参照する。

## prefix27／38構成の診断結果

投票38-token/common27-token prefix、6個の検証済みkernel、34Bで区切るupdate schedulerを使い、診断用runtimeを別にビルドした。各update直後のinclusive spanを取得し、保存Candidの診断返信を再decodeして集計した。基底投影とその内側の区間を重複加算しない。

|入力|診断handler命令合計|update|INT8基底|F32 LoRA|GQA head|その他|
|---|---:|---:|---:|---:|---:|---:|
|617|145,511,362,568|5|68.58%|11.77%|1.18%|18.47%|
|620|125,613,705,623|4|68.52%|11.71%|1.10%|18.66%|
|653|146,530,098,494|5|69.42%|11.89%|1.01%|17.68%|

各入力1回の診断実行で、profileの命令も含む。通常版の削減実測ではない。基準は[都度払い推論の実測](PAID_UPDATE_INFERENCE_MEASURED.md)。その他にはDelta再帰、norm、活性化、hash、コピー、量子化、読み出し、scheduler等が残る。

別の分類で、617のMLP段は82.130B（56.44%）、Delta段は50.701B（34.84%）、Attention段は12.647B（8.69%）。これらは基底投影・LoRAを含むため、上表へ足さない。

全3入力で31層の保存suffix hidden、32層のconv/suffix KV、最終norm hidden、判断・logits・校正確率・unknown確率が参照とbit一致。参照にないlayer30 hiddenと未exportのDelta密状態は直接比較していない。

653の診断合計から目標へは約46.53B（31.76%）の削減が必要。INT8基底だけで達成するならその区間を45.74%減らす必要がある。

ビルド・測定・再検証は `build_update_instruction_profile.py`、`prove_update_instruction_profile.py`、`measure_update_instruction_profile.py`、`report_update_instruction_profile.py`。証跡は `artifacts/update-instruction-profile-v1/{build-v3,proof-v1,summary.json}`。moduleは `f1d74339…`。固定準備の721 update×2 bank、prefix cache準備、prefix登録、境界拒否試験は推論合計から除外する。
