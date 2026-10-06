# 分割MLPで固定重みの復元と入力転置を繰り返さない

2026-10-03。`mlp_gate_up_capture` 内のF32 A演算が、準備済み重みを受け取った後で旧 `matrix` を直接呼んでいた。queryごとにoutput32配置を元のrow-major配置へ復元し、入力を転置する処理が残っていた。

`matrix_loaded` へ変更し、既に準備したoutput32配置を既存F32 kernelで直接読む。weight pack・INT8量子化・F32 adapter・丸めと加算順序・通信frameを変更しない。質問ごとの入力とLoRA Aの計算自体は必要であり、省いたのは重みの復元と入力転置である。通常queryのみで実行し、保持する量子化入力・scale・F32 Aはクライアントへ返す。

## ビット一致と削減の検証

nativeで1/7/87/132 tokenを比較し、MLP積が元の経路とビット一致した。準備済み重みのfallback復元bufferが作られていないこともprivateなテスト用観測で確認。候補と同じfeatureの全workspaceテストはruntime97・canister9 unit、integration9、compile-fail doc6を含めて通過した。

専用実験canister `6eydd-o3777-77775-aaama-cai` の候補moduleは `d3350dbeb5c1575b64769e5fbca3178314a5fdee9a8c5f3014aecd2a63692608`。

保存済みcold132の実入力を使い、layer0〜30の31 capture通常queryを測定した。各入力は `[132,4120,2560,0,64]`。MLP積だけでなく、次の分割queryで使う整数入力・F32 scale・両F32 Aも含む返却値全体がビット一致した。

layer0は2,066,266,626→2,019,584,365命令。31 query合計で1,447,153,439命令削減。queryごとのframe stepの長さが元実行と異なるので、この診断合計をそのまま全推論の削減として報告しない。証拠は `artifacts/f32_k_continue/capture-check/report.json` とhash付きsource ZIP。

## 全推論への影響

全6条件の回帰検証が完走し、返却hidden・保持state・型付き判断・logits・確率は既存INT8版とビット一致した。失敗とreplayは0。compact tailの非返却layer30 hiddenは直接比較しない。証拠は `artifacts/prefix_codec/full-capture-prepared-proof/report.json`、`before-after.json`、hash付きsource ZIP。

|条件|query|合計命令|Candid送受信bytes|最大query命令|時間・単回秒|
|---|---:|---:|---:|---:|---:|
|prefix45|66|123,228,153,297|71,105,691|2,154,257,815|19.527|
|旧log・主87|64|236,883,729,739|112,344,191|4,716,852,665|34.469|
|主87|62|235,201,564,012|123,281,081|4,716,852,665|34.431|
|情報不足80|62|215,010,915,283|116,455,579|4,334,000,137|31.600|
|最大変更89|62|240,511,135,258|125,231,193|4,816,561,177|35.230|
|cold132|290|359,797,067,808|580,207,902|2,354,057,277|58.417|

cold132の全推論は361,244,217,000→359,797,067,808命令、1,447,149,192命令（0.400601%）減。診断の合計とは4,247命令異なり、実際の全推論のcounterを採用する。query数・通信・観測heap終了最大4,131,258,368 bytesは同じ。時間は直前57.887→58.417秒で、速度向上は主張しない。

87-tokenの主ケースは元からfull MLPが準備済み配置を使っており、今回の修正では主62 query、命令、通信は同じ。50/32 queryは未達。cold132の分割経路が今回の削減対象。既存の最大変更の見逃しも残り、判断精度の改善とは扱わない。

固定モデル準備updateは721回、4,065,416,192 bytes、78,739,929,194命令。質問ごとのqueryとは別計上。Layaと主canisterを変更せず、mainnet・push・PR・commitは行わない。
