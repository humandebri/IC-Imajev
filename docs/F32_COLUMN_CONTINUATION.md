# F32の加算状態を引き継ぎ、分割のたびの再計算を避ける

2026-10-03。元のINT8 baseとF32 adapterの境界を変えず、LoRA Aの列方向の計算を通常query間で継続する `matmul_k_continue` を実装した。列範囲とF32の部分和をクライアントが保持する。質問の状態をcanisterへ保存するupdateは使わない。

## 実装と制約

`dims=[tokens,output_rows,total_columns,begin,count]`。入力はtoken-majorの列chunkと、その直前のtoken-major F32部分和。列範囲は64の倍数に限定する。最初の部分和は+0のビットでなければ拒否する。有限値、manifestのF32 tensor shape、model/pack、request identityを検証した型を使い、検証後に別requestへ差し替えられない。

固定output32配置の重みを直接使い、既存の検証済みK64 Wasm kernelを呼ぶ。入力の転置と重みの元配置への復元をしない。各列のmultiply/addを元の順序で継続する。部分和をBF16/INT8へ丸めない。列chunkを独立して計算して最後に足す方式は加算順序を変えるため使わない。

Rust nativeテストは列数128/2560/9216、token数1/7/45/87、出力の部分view、chunk64/256/1024を検証。非有限値・-0の初期状態・範囲外・request identityの変更を重みの読み出し前に拒否する。型のprivate fieldはcompile-failテストもある。

## ローカルcanisterでの実測

専用実験canister `6eydd-o3777-77775-aaama-cai`、module `9a5483b34e46426965ea50b1e17381891edb8df24663fa33918c3071075076d2`。

実際の主87-token入力からlayer0/1/3のMLP積をcanisterで生成し、固定packの元F32 Aを使う独立native scalar参照と比較した。layer0は1/7/87 token、layer1/3は87 token。各々256/1024/3072列分割すべてが最後までビット一致した。診断251通常queryは推論query数とは別。

layer0・87 tokenのF32 Aだけを測ると、256列分割36 queryは175,741,645命令、1024列分割9 queryは143,885,760命令、3072列分割3 queryは136,289,857命令。分割にはframe/transport等の追加コストがある。これを単体で置き換えても高速化やquery数減にはならない。層をまたぐ計算を詰めるための部品であり、現在の主graphへの接続は未実施。

全6条件の既存graphも完走し、返却hidden・保持state・判断・確率は既存INT8版とビット一致。compact tail内部の非返却layer30 hiddenを直接比較したとは扱わない。主87は62 query、235,201,564,012命令、Candid 123,281,081 bytes、最大4,716,852,665命令で前段候補と同じ。cold132は290 queryで+744命令。50/32 queryは未達。既存の最大変更の見逃しも残る。

証拠は `artifacts/f32_k_continue/check/report.json`、`artifacts/prefix_codec/full-f32-k-continue-proof/report.json` と `before-after.json`。コードとkernelのハッシュ付きZIPを保存。固定model準備721 update・4,065,416,192 bytes・78,739,929,194命令は推論と別計上。主canisterのmoduleは `36c04a57e9501eb9d32163c92d7725171191fa46a45a880ad03465993fceed73` のまま確認済み。

## 残る繰り返し

分割MLPの `mlp_gate_up_capture` は、準備済みF32 Aを読み出した後で旧 `matrix` を直接呼んでいた。そのため重みを元配置へ戻し、入力を転置する処理が残っていた。`matrix_loaded` 経由へ修正し、準備済みoutput配置を直接使う。nativeの1/7/87/132 tokenで出力一致と重み復元未実行を検証した。実canisterの全6条件がビット一致し、cold132の全推論で1,447,149,192命令（0.400601%）減。主87は同じ。[分割MLPの実測](MLP_PREPARED_CAPTURE.md)を参照。この修正は主87のfull MLP経路を改善したという意味ではない。

次の層間分割では、gate/up入力のblock256量子化とF32 Aを一度計算して保持し、各product chunkのdown A部分和を今回の方式で引き継ぐ。productも元のblock256境界で一度だけ量子化する。通信上限・継続状態・queryごとの命令数を実測してからgraphへ採用する。
