# DeltaとMLPを層境界で連結する通常query

2026-10-04。実験canister `6eydd-o3777-77775-aaama-cai`、module `aa499039a8aaeebe210f30bbb2cbe7244454cf2fcbf5f812e125731a3ad80cf8`。重み・adapter・readout・tokenizer・calibrationは既存MODEL_LOCKの組合せを維持した。通常queryで推論し、途中状態と固定prefix packetはクライアントが保持する。

`--roll-start` は実験オプション。共通prefix45 tokenの後、主問題87 tokenと情報不足80 tokenで連続するDelta層を次の4queryへ連結する。

1. Delta全headとMLP最初4096行。最初の層だけtoken IDsからembedding/normも実行する。
2. MLP残り5120行を完了し、次層Deltaの入力整数/scale・元F32 LoRA A・gateを一度だけ準備、18 headとout projectionの部分和を計算する。
3. 準備を再計算せず残り14 headとout projectionを完了し、次層MLPのproduct整数/scale・down Aと768出力行まで進める。
4. 残りdown行と次層Deltaを完了する。その後は当該MLPとfull attentionを既存経路で実行する。

未丸めのF32部分和は4 byte planeの可逆codecで送る。中間状態に追加の量子化は行わない。Rust内部では検査済み状態を直接MLPへ移し、内部wire生成・復元を省く。

|条件|suffix token|既存query|今回query|総命令数|Candid bytes|単回秒|
|---|---:|---:|---:|---:|---:|---:|
|主617|87|62|54|242,510,873,150|134,518,983|37.565|
|情報不足|80|62|54|221,969,301,829|125,701,450|33.353|
|最大変更・既存経路への切替|89|62|62|240,511,230,474|125,231,193|35.378|

主問題の同module既存経路は235,201,659,242命令・123,281,081 bytes・36.980秒。今回queryは12.90%減だが総命令3.11%増・通信9.12%増。速度・命令削減の採用結果とは扱わず、既定経路を置き換えない。主の最大queryは4,922,066,093命令、返却時heapの観測最大4,144,562,176 bytes。handler counterはCDKのCandid decode/encodeを含まず、通信はCandid要求+返信でHTTP/CBOR/署名を含まない。単回時間だけで速度改善は断定しない。

3条件で保存対象の全層hidden・Delta conv・attention K/V/positions・最終hidden・型付き判断・logits・確率が既存INT8版とビット一致した。compact tailが返さないlayer30 hiddenは比較していない。最大変更の見逃しは以前と同じで、判断精度改善はない。主・情報不足とも54query、50/32は未達。

89 tokenの入口は3840行でもIC0522となったため、88/89 tokenは確認済みの標準query経路へ切り替える。標準経路89 tokenの全体完走を今回確認した。88 tokenの分岐は単体試験までで、実canister完走の測定対象ではない。87 tokenでも4608/4864/5120行の入口は5B超過し、4096行の成立を実測した。失敗要求とエラーは保存している。未測定の入力・prefix長へ同じ境界が成立するとは扱わない。

## レビュー修正と検証

- 終端attentionを外から渡して再開すると、終端融合の分岐がMLPを省く問題を修正。attentionを再実行せずMLPを完了する回帰試験を追加した。
- raw部分finishでは、callerのdimsへアクセスする前にprepared requestとの完全identityを検査。不一致・空dimsをweight read前に拒否する。
- rollのchunk/head/down行数を開始前に検査。89 tokenで超過する入口を実行せず標準経路へ切り替え、例外後にもdispatch modeを復元する。
- 54queryの全体再開は保存frameを再利用し、実行した推論query0・判断一致を確認した。module hash確認のqueryは推論数に含めない。

同じRust sourceのruntime119 unit（外部fixture1件ignore）、integration15、compile-fail doc9が通過。Pythonはjournal4・開始codec3・pair codec7・tail/replay8・pipeline4・境界分岐3件が通過。Wasm rawは以前の検証済み候補とbyte一致し、4 kernel patchを含む最終moduleも同一。クライアント変更を含むsourceはbuild/proofごとに保存した。

同moduleの既存全6条件（prefix、hybridなし617、主、情報不足、最大変更、prefixなし132 token）も全出力一致・失敗/replay0を確認した。これはv7 sourceの回帰で、最終v8は報告項目と境界dispatchだけを変更し、3条件を新規通常queryで再測定した。

raw `mlp_down_norm_partial_prepared` は87 token/1600行で499,734,957命令、89 token/1280行で633,331,111命令。圧縮finish比それぞれ181,883,637・185,395,869命令減、要求約0.27MB増、hidden/normビット一致。全体4query経路の最後は次Deltaも含むためこのraw APIへ置き換えていない。

## 再現と保存証拠

既にupload/preparationが完了した上記実験canisterに対し、以下は推論の通常queryだけを実行する。model upload/installはこの検証scriptに含まない。

```sh
.venv/bin/python scripts/check_roll_graph.py \
  --canister 6eydd-o3777-77775-aaama-cai \
  --wasm artifacts/prefix_codec/full-build-roll-review-v8/full.wasm \
  --base-proof artifacts/prefix_codec/full-roll-review-base-proof-v7 \
  --directory artifacts/prefix_codec/NEW-roll-proof
```

- 最終3条件とsource ZIP：`artifacts/prefix_codec/full-roll-review-graph-v8`。
- 既存全6条件とsource ZIP：`artifacts/prefix_codec/full-roll-review-base-proof-v7`。
- build/source/module：`artifacts/prefix_codec/full-build-roll-review-v8`。raw moduleをそのままinstallしない。
- 89 tokenの入口失敗：`artifacts/prefix_codec/full-roll-review-graph-v7/maximum/queries`。
- 先頭4queryの独立比較・profile：`artifacts/f32_k_continue/roll-ids-check-617-v2`。
- raw finish比較：`artifacts/f32_k_continue/raw-partial-finish-check-v2`。

生成物はGit対象外。Laya・保護された主canister・mainnetは変更しない。push/PRも行わない。

次は、連結経路で復活したprefix log復元と圧縮carry展開、full attentionの前後に残るquery境界を測る。現在の主54queryは命令数の下限ではなく、5B/queryと2MB/frameの制約下で実測した境界である。
