# W4A8テーブル参照の実装とローカル実測

2026-10-05。4bitの重みと8bitの入力を使う投影カーネルを独立したローカルcanisterに実装した。現行の最適化済みINT8より命令数が多く、重みの量子化誤差もあるため、推論本体へは採用しない。32queryへの改善は達成していない。

これはT-MACと同系統のテーブル参照という発想をWasm SIMDで試した独自実装であり、公式T-MACカーネルの移植・性能評価ではない。今回の失敗から、すべてのINT4方式が遅いと結論しない。

## 実行条件と数値表現

- 固定したモデルの`layers.3.self_attn.q_proj.weight`、8192×2560。既存INT8 packのtensor SHAとMODEL_LOCKを照合した。
- BOOM DAOの保存済み実入力4条件、suffix87/45/80/89 token。境界入力は1/3/8 token。cold132は対象外。
- 重みだけを既存INT8から対称W4へ再量子化。K256ごとの最大値/7でスケールを作り、ties-to-evenで[-7,7]へ丸める。BF16から直接量子化した結果ではない。
- 入力は従来のA8/block256と同じ丸め。過去のINT8 native digestとも一致を確認した。
- K4の入力部分集合16通りを符号付きI16の表にする。範囲は[-508,508]。表自体を量子化しない。
- 4bit重みのbit planeを4bitの表番号に変換し、2番号を1byteへ格納する。SIMD swizzleで参照し、byte shuffleでI16を復元する。
- 整数dotはI32、各K256のスケール乗算・加算はF32。通常dotとLUTで同じ順序にする。中間状態・LoRA・readout・calibrationの追加量子化は行っていない。

## 探索した実装

最初の単純なW4 dotは全8192出力で50億命令上限に達した。失敗ログを`artifacts/w4-lut/check-v1.log`へ保存した。その後は出力1024を8queryに分け、同じW4重みのdot対照とLUTを比較した。updateで固定重みを準備し、進捗はクライアント側のtile番号で管理する。queryは状態を保存しない。

|版|変更|主87 tokenのLUT・8query合計命令|
|---|---|---:|
|V3|低/高byteをshuffleで復元|5,284,248,112|
|V4|K256内の64ステップ展開、pointerと定数offset|4,345,334,936|
|V5|出力128を共有、weight番号の展開を全tokenで再利用|3,731,036,096|
|V6|V5へ全8192出力の単一query経路を追加|3,859,663,336|

V6は全投影の1queryも実際に完走した。分割経路と単一経路の両方を独立nativeと照合した。V5の準備を一度にする見積もりとV6の実測は異なるため、最終比較にはV6の実測を使う。生成・コンパイルされたbodyが変わると分割経路にも命令差があり、V5の数値をV6へ転用しない。

## 全出力・単一通常queryの結果

|実入力|現行INT8 S1/128・命令|W4 LUT V6・命令|倍率|W4投影の相対L2誤差|
|---|---:|---:|---:|---:|
|BOOM DAO617・87 token|952,043,488|3,468,910,892|3.644倍|4.7825%|
|prefix・45 token|507,814,831|1,799,375,992|3.543倍|4.5419%|
|情報不足・80 token|868,593,200|3,190,681,746|3.673倍|4.7706%|
|最大変更・89 token|973,248,348|3,548,410,648|3.646倍|4.7611%|

counterはF32入力復号・A8量子化・入力表準備・投影を含み、出力digestとCandid処理を除く。現行INT8値は保存済みの別moduleの同じtensor・入力・測定境界による比較。同moduleのW4 dot/LUT対照も別に残した。境界1 tokenは50,381,570命令、3 tokenは129,901,110命令、8 tokenは328,650,501命令。

相対L2はQ投影のtensor誤差であり、判断正解率の低下率ではない。主の最大絶対誤差は1.331821、cosineは0.998862。全モデルの判断・確率・選択肢順序依存は未検証なので、精度維持を主張しない。計算の正しさは「同じW4重みのnative/dot/LUTが一致」という別の性質である。

## 容量と検証

このtensorの重みとscaleは21,004,288→10,813,440 bytes、48.518%減。内訳はW4重み10,485,760、group scale327,680 bytes。これはtensorの配置容量で、全モデルのheapや通信削減の実測ではない。診断canisterは比較用の元INT8・dense W4も保持するため、診断heapを製品のメモリとして使わない。

V3/V4/V5/V6の合計455通常queryと260準備updateが完走した。全7条件のW4結果について、分割の各1024出力とV6の全出力digestが独立NumPy参照とbit一致。nativeの追加3テストで、全符号付き4bit値、token端数、複数出力tile/K256境界、丸めとNaN拒否を検証した。実入力のINT8参照digestも過去の採用S1記録と一致する。

V6診断canisterは`3wbuo-6d777-77775-aaasq-cai`。既存の`4caro-hl777-77775-aaaba-cai`と`6eydd-o3777-77775-aaama-cai`は前後のmodule hashが不変。Layaには操作していない。mainnet・push・PRは行っていない。

実装は`scripts/w4_lut_bench/src/`、128出力生成は`scripts/generate_w4_lut_wide.py`、ビルドは`scripts/build_w4_lut.py`、実測は`scripts/check_w4_lut.py`、証跡検査は`scripts/analyze_w4_lut.py`。Wasm、重み、入力、返信、比較JSON、生成されたwide sourceは既存の`/artifacts/` ignore内へ保存した。各buildのsource.zipと各checkのchecker-source.zipをSHAで検査した。

## 再実行

既存の固定pack、保存入力、native Candid helper、コンパイル済みCDK依存が必要。builderは既存ビルド記録の依存を明示的に使い、source・依存・compiler・WasmのSHAを記録する。過去版の正確なsourceは各ZIPを使う。

```sh
python3 scripts/generate_w4_lut_wide.py --directory artifacts/w4-lut/retry-source --full-query
python3 scripts/build_w4_lut.py --directory artifacts/w4-lut/retry-build --source-directory artifacts/w4-lut/retry-source
icp canister create --detached --network local --identity imajev-local --json
icp identity principal --identity imajev-local
# 上の新規IDとownerを指定してinstallする。既存推論canisterへinstallしない。
icp canister install <new-id> --network local --identity imajev-local --mode install --wasm artifacts/w4-lut/retry-build/diagnostic.wasm --args '(principal "<owner>", 8192 : nat32, 2560 : nat32)' --yes
.venv/bin/python scripts/check_w4_lut.py --canister <new-id> --directory artifacts/w4-lut/retry-check --build-directory artifacts/w4-lut/retry-build --full-lut
rustc --edition=2021 --test scripts/w4_lut_bench/tests.rs -o artifacts/w4-lut/property-tests
artifacts/w4-lut/property-tests
```

## 採否と次の候補

正確なI16表を使うW4 LUTでは、4bit planeそれぞれの参照・I16復元・加減算・I32拡張が残る。重みと入力の共有、ループ展開でも、現行S1のI16 dotとtoken pair共有より命令が多かった。今回は容量削減と演算削減が両立しないため不採用とする。現行推論の50query、234,521,052,335命令と既存の判断は変更していない。

次は以下を別実験として扱う。

- **INT8を維持する演算改善**：既に単体で効果のあった奇数末尾S1省略を全層へ接続し、LoRAのoutput64共有を測る。追加量子化誤差がなく、今回のLUTより採用しやすい。
- **LoRA A/BのINT8化**：F32 LoRAの11.5%を狙う。全モデル・判断回帰を必要とし、単独で32queryへ届くとは主張しない。
- **BF16起点の校正付きW4や外れ値対策**：今回のW8再量子化より品質を改善できるか調べる。低bit LUT自体の量子化や回転には別の誤差があり、速度も再計測が必要。

中間状態の量子化はこの実験に混ぜていない。判断品質を維持したまま大幅削減できた、という結果ではない。
