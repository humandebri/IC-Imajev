# S1 kernel内の共通アドレスを再利用する

2026-10-03。元容量INT8配置、入力量子化、整数dot・F32 block256積和と丸め、F32 LoRA/readoutを維持する。

## 共通オフセットを一度計算

S1の7入力係数は同じtoken pairとK256 blockのbyte offsetを使う。従来は各係数について`(t*cols)+(start<<1)`を計算していた。`generate_s1_address_reuse.py`は保存済みraw S1 WATの同一式14箇所（初回pairと後続pair）を、各pairの冒頭で一度計算したlocalへ置換する。operand表の参照範囲とinteger/F32の全演算式は同じ。入力tableはimmutableで、出力accumulatorへaliasしない。

独立診断は同じコンパイル済みRust moduleの1 bodyだけをpatchして比較した。主Q8192×2560、87tokenは1,004,100,935→1,000,271,175命令、3,829,760（0.38141%）減。5実入力＋6境界入力、33通常queryの全digestがnative参照・前候補と一致。全11入力で減少した。固定準備65 update、推論33通常queryを分離した。

証拠`artifacts/s1_address_reuse/check/report.json` / `validated-source.zip`。generator input/output hashとbody patchのhashを保存。module前後・source前後の一致を検査した。Rustの再コンパイルに伴う削減ではない。

## 全モデル接続

実験canister6eyddのmodule2cf9074a…で全5条件＋旧log対照を完走。全保持hidden/state/最終hiddenと判断value/abstain/logits/確率/unknownが旧INT8出力と一致、型付き出力有効、graph失敗/replay0。

|条件|query|handler命令|前候補から削減|Candid bytes|最大query命令|秒・単回|
|---|---:|---:|---:|---:|---:|---:|
|prefix45|66|129,971,420,253|340,326,400 (0.2612%)|71,105,691|2,276,638,228|13.052|
|主87・旧log対照|65|249,017,361,866|632,813,440 (0.2535%)|113,687,128|4,288,836,126|22.981|
|主87|63|247,184,414,792|632,813,440 (0.2554%)|124,624,018|4,288,836,126|21.392|
|情報不足80|63|226,009,453,723|575,323,520 (0.2539%)|117,690,970|3,919,922,516|19.505|
|最大変更89|63|252,749,475,623|647,185,920 (0.2554%)|126,604,858|4,384,713,587|29.706|
|cold132|290|384,398,282,063|950,399,280 (0.2466%)|580,207,902|2,379,472,026|52.159|

主は632,813,440命令（0.25535%）減、247,184,414,792命令。63 queryで50/32目標は未達。通信量・固定cache4,065,416,192 bytes・観測heap終了値最大4,131,258,368 bytesは同じ。瞬間peakは未測定。主132tokenは45tokenのprefixを再利用し87tokenを処理する。最大変更を見逃す既存判断は残り、判断精度が改善したという意味ではない。

counterはhandler内で、CDK Candid処理を含まない。時間は単回、速度改善率には一般化しない。固定cache721 update /78,739,929,194命令 /269.983秒を推論から分離。prefix packetの二回目準備query/命令/通信0、bridge binary不変、主canister36c04a57…不変を確認。Layaを変更していない。生成物はignore。

証拠`artifacts/prefix_codec/full-address-reuse-proof/report.json` / `before-after.json` / `validated-source.zip`、`codec-second.json`、`client-binary-check.json`、`main-unchanged.json`。準備は`full-address-reuse-preparation/report.json`。build source・extra kernel source・3body patch logは`full-build-address-reuse/`。

全モデルRustは前候補と同じraw module SHA08ffdfb2…であり、前候補のruntime86 unit＋9 integration＋3 doctest、canister8 unitの検証済みRust sourceと一致する。今回変更したWATは実際のWasm出力と全推論で検証した。

## 固定入力ポインタの再利用・未採用

次候補`generate_s1_pointer_reuse.py`は7planeへのimmutable pointerをkernel呼び出し冒頭で読み、後続tokenで再利用する。同じRust診断moduleを使い、同じ11入力33queryで全digest一致。主Qでさらに734,720命令減（address版から0.07345%減）。1tokenは35,840命令増えるため、全モデルへはまだ接続していない。全モデルの削減率やquery削減とは扱わない。

証拠`artifacts/s1_pointer_reuse/check/report.json` / `validated-source.zip` / `before-after.json`。比較する3moduleは全て同じコンパイル済みinput module SHA717c790d…から1 bodyだけを置換している。

## 50 queryへ向けた次の融合候補

実測した主問題の最後のMLP30と最終Attention31＋MLP31＋判断は、2query合計5,017,845,117命令。5Bからの超過は17,845,117命令で、input norm/hiddenのquery間再送とcodecを除く融合が次の候補になる。情報不足は4,609,996,651、最大変更は5,124,058,405命令で条件差がある。

これは既存counterの和であり、融合後の測定ではない。新たなコピー・検査・Candid費用、layer30 hiddenとlayer31 KVの保持を含め、通常query上限以内で完走してからquery削減を認定する。62 queryもまだ未実装・未検証。token削減や中間状態破棄では解決しない。

監査script `analyze_tail_fusion_budget.py`、根拠`artifacts/tail_fusion_budget/report.json`。50 queryの目標は維持する。

## 再現

```sh
.venv/bin/python scripts/generate_s1_address_reuse.py --directory artifacts/s1_address_reuse/build
```

[前候補](OPERAND_WRITE_ONCE.md)と同じfull build flagsと固定cache準備を使い、S1のbody patchだけを`artifacts/s1_address_reuse/build/kernel.wat`にする。pair direct-inputとF32 output64の2 patchはそのまま。保存先をfresh directoryにして`check_full_prefix_hybrid.py --terminal-attention --terminal-decision --prefix-start`で全5条件を検証する。
