# 固定重みと入力の再利用

2026-10-03。標準SIMDのINT8 pair kernelで、繰り返す読み出しを減らす候補を実装した。量子化、整数dotの加算木、F32 scale乗算・蓄積順序は変えていない。

- 同じ重みを最初のtokenのdotで読み出してINT16へ展開し、後続tokenはlocalの値を使う。
- 各tokenの入力を最初のoutput pairで読み出し、後続の15 pairsは同じlocalを使う。
- 初回の読み出し結果を`local.tee`で演算stackと再利用localへ渡す。先に保存して直後に読み直す命令を省く。
- 既存出力と整数dotをoperand stackに保持し、一時出力の保存・読み直しを省く。
- F32 input scaleの読み出しとsplatを標準SIMDの`v128.load32_splat`にまとめる。

生成コードは `scripts/generate_pair_reuse_wat.py`。9個のI32引数を持つ明示ABIのbodyだけを置換し、Wasm validatorを通す。raw stubは出力をNaNにするため、未置換moduleを推論に使わない。

## 独立kernelの実測

専用canister `5tkpr-7d777-77775-aaaeq-cai`、実固定Q weight、11形状×旧/新の22通常query。全出力digestはnative・既存採用kernel・候補で一致。量子化とqueryごとの入力準備を含む命令数の変化は次のとおり。

|入力|token数|命令数の変化|
|---|---:|---:|
|prefix|45|+0.0613%|
|主問題suffix|87|−0.5210%|
|情報不足suffix|80|−0.6015%|
|最大変更suffix|89|−0.5002%|
|cold入力のQ部分|132|−0.6980%|

境界1/7/8/32/64/88 tokenもbit一致。これはbase Q projectionだけの測定であり、全モデル・判断精度・4 GiB配置の証明ではない。記録は `artifacts/prefix_codec/full-wat/check-reuse/report.json`、検証時sourceは同directoryの `validated-source.zip`。

重みloadとINT8→INT16変換だけを融合した前候補は主問題で+1.8441%となり不採用。基準Wasmは既に同じ標準SIMD融合loadを使っていたため、融合そのものを基準からの改善とは数えない。

## 本体への接続

実験feature `experimental-pair-wat` の本体では、87 tokenの44+43分割などをやめ、最大132 real tokenの全行を1回のK-block呼び出しへ渡す。固定重みの展開はその呼び出しで共有する。各token内のblock蓄積順序は同じ。中間stateはクライアントに保持し、query永続化を追加しない。

`experimental-paired-only` はWasmで未準備の旧INT8 projectionを明示errorにし、使わない大型fallbackの特殊化をビルドから除く。既存デフォルトとnative比較経路は保持する。固定重みcacheが全対象を準備済みであることが前提。

full candidateは通常opt3/thinLTO/cgu1で68秒でビルド成功。module `8ca6e7468b761d2e87e92fa4a69ca20583c8f39928d4f78eaa5ed04a05ff7fcc`、1,470,912 bytes。保存先 `artifacts/prefix_codec/full-build-paired-only`。検証専用full canister `6eydd-o3777-77775-aaama-cai` へupgrade済み。固定INT8 packは保持され、cache準備後にprefix hybridを含む全推論を検証する。主canisterは変更していない。

初回full推論は32-row gate48個が旧byte cacheに残るため準備必須errorで停止した。修正候補は `artifacts/prefix_codec/full-build-paired-all`、module `44640eb4c3f3cfc513002281eef618da2e428fc9852dd342425b735cdecbef80`。全248 INT8 tensorsをpair化する。固定payload総量4,065,416,192 bytesは同じで、pair weight部分は3,569,090,560 bytesとなる。専用full canisterをこの修正候補へupgrade済み、初回の失敗を成功した全推論結果として数えない。

追加探索として、保存済み主suffixの31層down入力を同じblock256 quantizerで調べた。個別ゼロは3.39〜9.11%だが、全4要素がゼロのK4 groupは0.0025〜0.1472%、K256全ゼロは0%。この測定ではgroupを省くbranchの追加を正当化できず、実装を見送った。`artifacts/prefix_codec/zero-group-probe/report.json`。hostのゼロ数測定であり、IC命令数の改善を示さない。

## 元の量子化bufferを直接読む

入力を一度複製してから読む経路も省いた。`experimental-pair-direct-input` はWasmで`q.values()`を渡し、対応するWATは元I16 stride・8-byte offsetの`v128.load64_splat`で同じduplicated vectorを初回に作る。旧`experimental-pair-wat`のABI/layoutは保持し、featureを区別する。

独立11形状×2の22通常queryは全てnativeとbit一致。主Q投影−0.6152%、prefix−0.0334%、情報不足−0.6977%、最大変更−0.5958%、cold Q部分−0.8857%。query入力準備counterは主入力で1,004,372→218。`artifacts/prefix_codec/direct-input-check/report.json` / `validated-source.zip`。これを全モデルへ接続し、`FULL_PREFIX_REUSE.md`の最終候補で全5条件のbit一致まで確認した。

再生成・ビルド（既存のRust/wasm target環境から、対象project内で実行）：

```sh
.venv/bin/python scripts/build_full_prefix_candidate.py \
  --directory artifacts/rebuild-direct-input \
  --target-directory artifacts/rebuild-direct-input-target --direct-input
cargo build --release --manifest-path scripts/wasm_patch/Cargo.toml \
  --target-dir artifacts/wasm-audit-target
artifacts/wasm-audit-target/release/imajev-wasm-patch \
  artifacts/rebuild-direct-input/raw.wasm \
  artifacts/prefix_codec/direct-input-build/direct-input.wat \
  artifacts/rebuild-direct-input/full.wasm > artifacts/rebuild-direct-input/patch.json
```

feature一覧をsource内へ固定し、WAT generatorも自動実行するため、旧ignored build logやdiagnostic runtime rlibをfull再ビルドの前提としない。buildはofflineなので初回の依存取得は既存READMEの環境構築に従う。既存canisterを上書きするcommandをこの手順に含めない。新規検証環境で`prepare_weight_cache.py --require-all-output-pairs`後、`check_full_prefix_hybrid.py`へ専用canister/codec/完成Wasm/新directoryを渡して検証する。
