# 整数Strassenの重複水平和と初回読み直しを省く

2026-10-03。通常queryで進めるINT8本体の追加診断。元のINT8 weight、block256 activation quantization、K-block内の整数dot、F32 input scale/row scale/加算順を保つ。重み準備だけowner update。Laya・主canister・全モデル実験canisterは変更しない。

## 実装

既存S1は7つの整数積それぞれでSIMDの隣接laneを水平加算し、その後4つの出力を整数線形結合していた。水平加算と整数線形結合が交換できることを記号的に確認し、7積のlaneを残したまま出力を再構成し、最後の2出力vectorでのみ水平和を取るようにした。元のI32中間値上界は維持する。F32への変換後の順序は変更しない。

加えて、固定weightを先にlocalへ保存して最初のdotで読み直す処理、各token pairのinputを先にlocalへ保存して最初のoutput quartetで読み直す処理を省いた。初回dotのloadをlocal.teeで演算stackと後続再利用へ渡す。1792 weight vectorsは最初のtoken pairのみ初期化し、224 input vectorsは各token pairの最初のoutput quartetだけで初期化する。入力scaleのload/splatもv128.load32_splatへまとめた。

新generatorは `scripts/generate_wat_s1_late_reduce.py` と `scripts/generate_wat_s1_first_use.py`。旧generatorとkernelはそのまま維持し、新WATをartifactsへ生成する。9×I32 ABIの単一bodyだけpatchし、他section/bodyの同一性とWasm validatorを検証した。診断raw moduleは以前から固定されたcompiled runtime/dependencyのものを使う。現在の全モデルCargo buildと同じcompiler出力として扱わない。

## 実測

専用診断 `5tkpr-7d777-77775-aaaeq-cai`。実layer3 Q weight8192×2560、実入力prefix45/main87/情報不足80/最大変更89/cold132（coldは4096出力行）。独立native scalar・通常INT8・採用済みprepared pair・各候補のdigestが一致。境界1/7/8/32/64/88も一致。各候補22通常query、準備65 update。counterは量子化・入力準備・投影を含み、digest/Candidは含まない。

|入力|通常INT8 pair命令|旧S1命令|水平和を最後へ|初回loadも融合|通常pair比|
|---|---:|---:|---:|---:|---:|
|prefix45|543,136,819|577,479,027|563,614,067|545,709,427|+0.4737%|
|main87|1,044,316,360|1,071,090,884|1,044,753,604|1,014,699,204|−2.8360%|
|情報不足80|962,099,208|978,090,260|954,333,460|926,590,740|−3.6907%|
|最大変更89|1,067,862,040|1,094,645,440|1,067,714,240|1,037,081,280|−2.8825%|
|cold132/Q半分|798,455,508|803,436,280|783,836,920|762,444,280|−4.5101%|

主Qで29,617,156命令減。旧S1から56,391,680命令減。prefixは通常pairより2,572,608命令多く、小tokenの境界も悪化するため一律採用しない。これはQ投影だけの改善で、LoRA・全層判断・50 query達成の検証ではない。既存graphの全モデル最新完走値は64 query / 約260.70B命令のまま。

固定S1係数は元INT8の3.5倍容量（対象Qで73,400,320 bytes）。全モデルを現行4GiBへ常駐できることも証明していない。命令削減だけを根拠に全モデルへ接続しない。

## 次の配置候補

現在は7個のI16係数arrayを固定準備で保持する。一方、各係数は4個の元INT8 quadrantから導ける。4 quadrantをI8で並べ替えて保持すれば元weightと同じ容量になる。queryの最初のtoken pairでI8→I16展開と5つの加減算を行い、後続token pairへlocalで再利用する候補を調べる。全token・全dotごとに展開しない。元のPreparedPairsもI8→I16展開を最初のtokenで行うため、この追加コストと係数load数の減少を同条件で比較する。

各quadrant arrayは(rows/4)*cols bytes。K256 blockでは2つのK128部分を分け、同じ4-output quartetのrow0/2とrow1/3をそれぞれ8-byte groupへ並べる。rawならweight pointerのrow-quartet strideはcols bytes、block offsetはstart bytes、load offsetはg*8 bytes。旧I16係数では各値が2 bytesで、それぞれcols*2/start*2/g*16。inputの7つのquery-local I16係数は旧配置のまま。この配置・実測は未実装で、性能や全モデル容量を断定しない。

## 証拠・再現

- 水平和版：`artifacts/wat_s1_late/build/generator.json` / `patch.json`、`check/report.json` / `validated-source.zip`。
- 初回load版：`artifacts/wat_s1_first/build/generator.json` / `patch.json`、`check/report.json` / `validated-source.zip`。
- 元S1：`artifacts/wat_s1/check/report.json`、raw：`artifacts/wat_s1/cached-build/diagnostic.wasm` / `report.json`。compiled dependenciesのhashをこのreportに固定済み。

```sh
.venv/bin/python scripts/generate_wat_s1_first_use.py --directory artifacts/<fresh-build>
artifacts/wasm-audit-target/release/imajev-wasm-patch artifacts/wat_s1/cached-build/diagnostic.wasm artifacts/<fresh-build>/kernel.wat artifacts/<fresh-build>/diagnostic.wasm
# 専用診断canisterにowner,8192,2560でinitした後、固定重みを準備し通常query比較。
.venv/bin/python scripts/check_wat_s1_first.py --canister <diagnostic-id> --directory artifacts/<fresh-check>
```

現check scriptは専用buildパスを固定しているため、別buildへ適用するときはpathを明示変更し、patch/build/hashの照合を維持する。生成物はignore。mainnet/push/PR/commitは行わない。
