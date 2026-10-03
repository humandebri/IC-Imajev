# F32 LoRA重みを全tokenで再利用する

2026-10-03。通常query内でF32 inputをtoken群ごとに並べ替え、同じ固定LoRA B weightをtoken群ごとにloadする処理を省いた。演算のF32精度と元のcolumn加算順を維持する。INT8本体、元のadapter/readout/calibrationを変更しない。固定配置だけをowner updateで準備し、質問・中間状態はクライアントが保持する。

## 実装

`experimental-f32-output-reuse`はLoRA B（cols64、32行単位）の固定F32をcolumn/output32配置へ並べ替える。配置は元と同じbyte容量で、元配置を二重常駐させない。`PreparedF32`のprivate layout情報を保持し、型付きreaderから直接投影する。F32 columnごとのmul/addを別々に行い、正のゼロから64 columnを元の順序で加算する。FMAや近似、再量子化を使わない。

新WATは最初のtokenで512 weight vectorsをloadし、その後全tokenでlocalを再利用する。入力F32は元のrow-major bufferからload32_splatで直接使う。queryごとの入力転置bufferを作らない。output32 tileの最後に元のtoken-major出力へ戻す。未patchのstubはNaNを書いて有限値検査で失敗する。

元のbyte/F32読み取り経路も保つ。非整列の部分viewは必要な範囲だけ元のF32列へ復元する。最適化された投影はこの逆変換を使わない。2行ずれ、tileをまたぐview、1/7/45/80/87/89/132 token、符号付きゼロ等をnativeで検証した。

実装は`prepared_weights.rs`、`f32_output.rs`、`evaluate_integer`のLoRA B dispatch、canisterの`weight_cache.rs`。全モデル候補のruntime unit83、integration9、compile-fail doc3、canister unit7件を通過。

## LoRA Bの単体診断

固定layer0 down-B2560×64と、以前のcanisterが生成して保存した実down-A出力を使用。4実入力と7境界の22通常queryでnative scalar・既存matrix・候補のdigestが一致。準備6 owner updates、固定配置655,360 bytes。診断だけは旧配置も保持するため計1,310,720 bytes。counterはF32 input decodeとprojectionを含み、digest/Candidを含まない。

|入力|既存命令|候補命令|差|
|---|---:|---:|---:|
|prefix45|26,616,904|20,461,639|−23.1254%|
|主87|49,884,760|39,355,796|−21.1066%|
|情報不足80|45,007,809|36,221,300|−19.522%|
|最大変更89|51,423,691|40,242,292|−21.744%|

1-tokenの境界も削減した。132 tokenはsynthetic境界であり、単体の実cold推論として数えない。証拠は `artifacts/f32_reuse/check/report.json` / `validated-source.zip`。単体baselineは固定した旧compiled runtimeの採用matrix-tail kernelで、依存hash/feature/compilerを記録した。

## 全モデル実測

専用canister `6eydd-o3777-77775-aaama-cai`、module `4e4072ef214c39d61f699737ae972a53c29c4aeaded96d7ddffd461ac047aeda`。直前の元容量S1候補と比較した。

|条件|query|handler合計命令|直前候補比|Candid bytes|秒（単回）|最大query命令|
|---|---:|---:|---:|---:|---:|---:|
|prefix45|66|132,635,768,588|−2.0334%|71,105,691|14.434|2,325,844,310|
|主87・旧log対照|66|252,795,996,489|−1.8154%|113,697,747|21.678|4,360,554,347|
|主87・packet再利用|64|250,996,660,791|−1.8125%|124,634,637|43.255|4,360,554,347|
|情報不足80|64|229,087,135,594|−1.6921%|117,701,589|20.689|3,978,231,309|
|最大変更89|64|256,790,234,460|−1.9159%|126,615,477|21.410|4,461,358,702|
|prefixなし132|291|387,986,256,815|−1.0655%|580,218,522|64.732|2,380,336,886|

主で4,633,251,334命令減。5条件すべての全32層hidden/state、判断4条件の最終hidden・value・abstain・logit・確率が既存INT8版とbit一致。型付き出力検査も通過し、失敗/replay0。既存の最大変更の見逃しは残る。query数と通信は直前候補と同じで50/32未達。

固定cache721 tensors / 4,065,416,192 bytes、INT8 pair3,569,090,560 bytesを維持。RoPE131,072 bytes、activation1,048,576 bytesも準備した。準備721 updates、75,407,015,126命令、261.848秒を推論と分けた。観測heap終了値最大4,144,103,424 bytesはピークではない。prefix packetの2回目の準備はquery/命令/Candid bytesとも0。

命令はhandler counter、通信はCandid送受信でHTTP/TLSを含まない。時間は単回で、別canisterのLoRA A診断と一部並行したため速度改善率として使わない。主canisterのmodule `36c04a57…`が変わっていないことをreadだけで確認し、Layaは変更していない。

証拠：`artifacts/prefix_codec/full-f32-output-proof/report.json`、`validated-source.zip`、`codec-second.json`、`full-f32-output-preparation/report.json`、`main-unchanged-f32-output/report.json`。

## 削減後のprofile

実87-token MLPのlayer0/1/3を同moduleの通常queryでprofileし、保存済み返信byteと一致。layer0全体4,358,183,447命令、INT8 base投影3回3,346,322,970、LoRA B積3回319,437,514。LoRA Bは直前407,553,860から88,116,346（21.62%）減。base inclusiveは約76.78%、B積は約7.33%。残りをLoRA Aだけのコストとは断定しない。inclusive/checksumのspanを重ねて加算しない。

証拠：`full-f32-output-profile/report.json` / `validated-source.zip`。

## 次の候補：LoRA AのK64 block再利用

別診断に固定gate-A64×2560を準備し、実normalized MLP inputを全モデルcanisterの`add_norm_bf16`通常queryで生成した。F32 partial sumをK64間で保持し、40 blocksを元のcolumn順で計算する。column間の並べ替えや近似はしない。各blockで全tokenが同じweightを再利用し、query inputの転置を省く。

|実入力|既存比|
|---|---:|
|prefix45|−25.0931%|
|主87|−18.7383%|
|情報不足80|−16.7108%|
|最大変更89|−19.7700%|

4実入力と7境界の22測定queryでnative digest一致。これとは別にnormalized input生成4通常queryを行った。主は45,205,791→36,735,008命令。固定候補655,360 bytes。synthetic132境界も一致したが、全モデルでのLoRA A適用・判断・query数削減は未検証。

`scripts/f32_block_bench`、`generate_f32_block.py`、`check_f32_block.py`に実装した。証拠は `artifacts/f32_block/check/report.json` / `validated-source.zip`。reportのtensor/shapeはgate-A64×2560で、使用重み・実入力・native oracleを明示している。全モデルへ接続する次の候補であり、現在の全推論性能へ足し合わせない。

## 全モデル候補の再現

```sh
.venv/bin/python scripts/generate_f32_reuse.py
.venv/bin/python scripts/build_full_prefix_candidate.py \
  --directory artifacts/prefix_codec/full-build-f32-output \
  --target-directory artifacts/prefix_codec/full-target \
  --direct-input --terminal-attention --prefix-start \
  --delta-state-layout --strassen-raw --f32-output-reuse --instruction-profile
```

rawのpair body、S1 raw body、F32 bodyの3つを順番にpatch・validateする。最後は以下。前段2つのコマンドは [STRASSEN_RAW_REUSE.md](STRASSEN_RAW_REUSE.md) と同じで、出力ディレクトリをこの候補に合わせる。

```sh
artifacts/wasm-audit-target/release/imajev-wasm-patch \
  artifacts/prefix_codec/full-build-f32-output/raw-patched.wasm \
  artifacts/f32_reuse/build/kernel.wat \
  artifacts/prefix_codec/full-build-f32-output/full.wasm \
  __imajev_f32_output64
```

固定された`source.zip`、`extra-kernel-source.zip`とpatch metadataを使う。変更時は新しいディレクトリを指定する。生成物はignore。
