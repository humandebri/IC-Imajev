# 最終MLP chunkとdown projectionの同一query実行

2026-10-04。レビュー修正を `597206a` にコミット後、`mlp_stream_complete` を追加した。主graphにはまだ接続せず、全体50 query達成とは扱わない。

`mlp_stream_next` の最後のchunkを処理してからcarryをflat化・encode・返送し、次の `mlp_stream_finish` で復元する往復を省く。所有権付きのPartsを直接共通finish関数へ渡す。元の量子化・F32 LoRA Aの列加算順序・BF16丸め順序は同じ。固定重み準備だけupdateを使い、質問状態はclient保持、推論は通常queryのまま。

codecは `mlp-stream-exact-v1`。dimsは `[tokens,既存の行数,最後の行数]` で、両行数は正の256倍数、合計9216、tokensは1〜89。request identity、元の層と次のnorm、整数・scale・有限値の検査を維持する。完了済みcarryを入力にして最終列を二重処理することはdecode時に拒否する。出力はhiddenとnormのBF16値であり、不要な完了carryを返さない。

## 実canister測定

実験用 `6eydd-o3777-77775-aaama-cai`、module `c5ad3b17a1577af66456ab444225d323a3a08e3e41c6d71af837f39c4d84d8ea`。同じmodule上の従来prepare/next/finishと比較した。保存済み実canisterの主87・情報不足80・最大変更89 token、layer0/1/3、2048+7168・3840+5376・4096+5120・4608+4608・2048+2048+5120の45条件で、最終hidden/normは元の一括MLP出力とビット一致した。融合後の全product carryを外部へ返して比較したとは扱わない。

全45条件で従来の分割より通常queryが1つ減り、命令数は92,412,713〜104,934,552減、Candid通信量は2,856,836〜3,178,105 bytes減。最大queryは3,414,638,011命令。2分割は3 query→2 query、3分割は4→3。これは診断のMLP単体比較であり、一括MLPからのquery削減ではない。一括MLPは元々1 queryで、単独の2-query経路をその代替として採用しない。

|layer0、4608+4608|token|融合2 query合計命令|最後のquery命令|Candid合計bytes|分割比命令差|分割比通信差|
|---|---:|---:|---:|---:|---:|---:|
|主|87|4,143,256,290|2,570,326,616|4,075,778|−102,642,371|−3,106,713|
|情報不足|80|3,787,284,790|2,348,083,477|3,748,030|−92,750,824|−2,856,841|
|最大変更|89|4,233,787,737|2,627,659,270|4,169,420|−104,902,666|−3,178,105|

各入力は84通常診断queryと6 profile診断query、計252+18 query。通常には3つの一括準備比較queryと従来の分割queryを含む。profile返却値も通常出力とビット一致。推論全体と別計上する。

Rustは候補featureでruntime106/canister9 unit・integration9・compile-fail doc8通過。Python5件と315保存payloadのbyte一致が通過。境界・完了carryの再投入・最大89 token carryとplain返却のlossless codecを検査した。offline Wasm buildと4つの演算kernel patchが成功した。

証拠：`artifacts/f32_k_continue/complete-check-{617,insufficient,maximum}/report.json`、各validated-source.zip・保存request/reply・profile出力。固定準備は `artifacts/prefix_codec/full-mlp-complete-preparation/report.json`。build/source/kernel hashは `artifacts/prefix_codec/full-build-mlp-complete-v1`。再実行は `scripts/check_mlp_stream.py --canister … --wasm … --directory … --source-case 617 --complete`。生成物はGit対象外。

次はMLP完了の所有権付き状態から次層Deltaの前半へ直接渡し、out projectionの列継続を同じquery内へ置く。個別命令数の合計だけで5B命令・2MB frame上限や全体50 queryを満たしたとは判断しない。

## 同じ候補の全体回帰

レビュー修正後の同じWasmで全6条件を完走し、返却hidden・保持state・型付き判断・logits・確率が既存INT8版とビット一致した。compact tailで非返却のlayer30 hiddenを直接比較したとは扱わない。失敗・replayは0。主87は62 query、235,201,573,732命令、最大4,716,852,899命令、Candid 123,281,081 bytes、返却時heap最大4,130,144,256 bytes、単回34.369秒。直前のrange-v2版とquery数・命令数・通信量は同じ。単回の時間差を高速化の証拠にしない。最大変更の見逃しも既存どおり。

全体証拠は `artifacts/prefix_codec/full-mlp-complete-proof/report.json` とvalidated-source.zip。`scripts/archive_mlp_complete_proof.py` は45診断条件の保存byte・source/build hashと全体6条件のreport hashを検査する。主canisterのmodule hashは読取で従来の `36c04a57…` と一致を確認した。Layaを変更していない。
