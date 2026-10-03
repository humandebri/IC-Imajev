# 固定変換を共有する Strassen1 / 直接 Wasm の診断

2026-10-03。2 token × 2出力の8整数内積を7内積へ置き換え、2組の出力行をSIMD内に詰める。固定係数はupdateのsealで一度だけ作り、入力変換はqueryにつき一度だけ作る。元のINT8・block256 scale・F32積和順序を保ち、近似と追加量子化を使わない。

## 実測

専用canisterで65準備update・22通常query。全11形状でnative scalar・採用済みpair・新kernelの出力bit一致。準備の合計は 2,604,570,185 命令。表は入力量子化・入力変換・射影を含むhandler計測。digestとCDK Candidを除く。LoRA、全層推論、判断精度を評価した測定ではない。

| 入力 | tokens | 従来命令数 | 候補命令数 | 変化 |
|---|---:|---:|---:|---:|
| prefix | 45 | 543,136,819 | 577,479,027 | +6.3229% |
| 617 | 87 | 1,044,316,360 | 1,071,090,884 | +2.5638% |
| insufficient | 80 | 962,099,208 | 978,090,260 | +1.6621% |
| maximum | 89 | 1,067,862,040 | 1,094,645,440 | +2.5081% |
| normal | 132 | 798,455,508 | 803,436,280 | +0.6238% |

5実入力すべて命令数が増えたため不採用。主Qは2.5638%増。固定係数は元INT8の3.5倍容量で、全モデルの4GiB配置も未解決。内積回数の削減だけでは、SIMDのlane復元・データロード・変換費用を吸収できなかった。主推論canisterは変更していない。

## 再現と証拠

`generate_wat_s1.py`は2×2積のsymbolic恒等式とI32中間値上限を検査。nativeは256/512列、n=1/7/8/32/45/64/80/87/88/89/132、複数scale block、非ゼロstrideの出力prefixを検査し2 tests通過。`build_wat_s1_cached.py`で固定runtime/depsへリンクし、`wasm_patch`で9×I32 ABIのkernel本体だけ置換。完成Wasmをvalidatorで検証してからinstallした。

完成module SHA: `5f48451ae82a4b4e5cced3dca81db21c2a7ef3f782f792333ca5d9766d81506d`。raw `artifacts/wat_s1/check/report.json`、`patch-report.json`、`cached-build/report.json`、測定時source `validated-source.zip` / `validated-source-hashes.json`。`raw-build-source.zip`はRustリンク時点のgeneratorとWATを保管する別段階の証拠。最初のWAT生成はstack不足でvalidatorが拒否し、修正前はinstallされていない。修正版のWAT hashとkernel body hashはpatch reportに保存。

測定に使った専用診断canisterは、測定終了後にprefix準備codecへ戻した。上記module hashは測定時の前後一致を保存した履歴で、現在の稼働moduleとは異なる。生成WAT・バイナリ・実測結果はGit ignore対象。
