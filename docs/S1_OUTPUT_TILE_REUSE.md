# 128出力行で同じINT8入力ロードを共有する

2026-10-03。通常queryの整数S1で、出力tileを32行から128行へ広げる候補。元容量のINT8 quadrant、元block256量子化とscale、I32 dotと再構成、F32 block加算順・BF16境界、元F32 adapter/readout/calibrationを保つ。質問状態はクライアント保持。

## 省いた反復処理

7面のINT8入力係数は既にquery内で一度作って共有していたが、出力32行ごとに再びロードしていた。128行版は各token pairの最初のoutput quartetで読み込んだ224 input vectorsを、32個のoutput quartetsへ再利用する。入力ロード・token/scale/address処理を32行版の4倍の出力へ共有する。固定weightのI8→I16と係数加減算は、そのK256 blockの最初のtoken pairでだけ行い、後続pairへlocalで再利用する既存方式のまま。

`generate_wat_s1_wide.py`は元S1の整数恒等式とI32上界を検証し、output tileだけを広げる。header/operand・K256 span・F32演算順は維持する。9×I32の専用bodyをWasm validator付きでpatchする。128行のweight spanとscaleが完全に初期化されたviewだけに使い、小さなview/部分tile/4行境界に合わないstartは従来32行版へ戻る。全モデルのfeatureは`experimental-strassen-output128`で、既存の別候補`experimental-strassen-wide`とは区別した。

## 同一module診断

同じcompiled dependencyとRust sourceのmoduleに32行対照とwide bodyを入れ、診断専用canister `5tkpr-7d777-77775-aaaeq-cai`で測定した。layer3 Q8192×2560の実INT8 weightと実入力prefix45/主87/情報不足80/最大変更89/cold132（4096出力行）を使う。各版は5実入力＋境界1/7/8/32/64/88の22通常query、固定準備65 update。全digestが独立native scalar/INT8参照と一致。

|入力|32行命令|64行命令|128行命令|128行削減率|
|---|---:|---:|---:|---:|
|prefix|533,109,250|516,232,911|507,814,831|4.7447%|
|617|1,000,271,187|968,119,328|952,043,488|4.8215%|
|insufficient|912,510,883|883,232,368|868,593,200|4.8128%|
|maximum|1,022,566,223|989,687,580|973,248,348|4.8230%|
|normal|754,212,807|730,125,268|718,100,628|4.7881%|

単体counterは量子化・入力係数準備・投影を含み、digest/Candidを除外する。主87-tokenは64行で3.2143%、128行で4.8215%減。全モデルへの同率改善とは扱わない。128行では単体wall timeが対照より増える入力もある。命令数の改善だけを確認し、単回時間から高速化を一般化しない。

256行版も生成・validator確認したが、ローカルICは関数のlocals18,748が上限10,000を超えるためinstallを拒否した。256行の精度・性能は未測定。128行版を選ぶ。`artifacts/s1_wide/install256.log`と`rejected256.json`に実エラーを保存。診断sourceとnative/runtime snapshot、body patch、request/replyのraw Candidとdigestは`check64`/`check128`のreport/validated-source.zipと`build64`/`build128`に保存。全model採用済みcanister・Layaはこの診断で変更しない。

## 全モデル検証

候補build `artifacts/prefix_codec/full-build-s1-wide-v2` は直前host-checksum候補と同じfeature構成に`--strassen-output128`を追加し、専用128行bodyを4つ目のpatchとして挿入する。固定cache準備は従来と同じで、事前weight容量を増やさない。全6条件の結果を以下に記録する。

|条件|query|handler命令|削減命令|Candid bytes|最大query命令|秒（単回）|
|---|---:|---:|---:|---:|---:|---:|---:|
|prefix|66|123,228,153,297|4,264,150,888|71,105,691|2,154,257,815|25.845|
|baseline-617|64|236,883,729,739|7,888,129,764|112,344,191|4,716,852,665|35.146|
|617|62|235,201,564,012|7,888,349,165|123,281,081|4,716,852,665|35.584|
|insufficient|62|215,010,915,283|7,181,679,991|116,455,579|4,334,000,137|35.491|
|maximum|62|240,511,135,258|8,080,981,361|125,231,193|4,816,561,177|35.936|
|normal|290|361,244,216,256|6,732,261,884|580,207,902|2,354,057,253|57.946|

主問題は243,089,913,177→235,201,564,012、7,888,349,165命令減（3.2450%）。62 query・Candid123,281,081 bytesは同じ。最大変更89-tokenは最後の2層が1queryで完走して63→62 query、Candid1,373,665 bytes減。compact tail87/80/89は4,716,852,636 / 4,334,000,100 / 4,816,561,148命令でbit一致。clientのtail上限を89へ広げ、mock routing/replay7 testが通過。

全6条件の返却hidden・保持state・最終hidden・typed判断・logits・probabilities/unknownは既存INT8版とbit一致、失敗/replay0。最大変更89もlayer30 hidden非返却に変わったため、未返却hiddenを直接比較したとは扱わない。元の最大変更の見逃しも残り、判断精度向上の主張はしない。観測heap終了値最大は同じ。cold最大queryは2,353,243,151→2,354,057,253と814,102命令増だが、合計は減った。

時間の悪化も測定した。主は20.221→35.584秒、prefix11.555→25.845秒、cold43.797→57.946秒。単回だが、一律に高速化したとは扱わない。命令数削減を狙う128行版は実行時間とのトレードオフがあり、CPU時間重視なら32行候補を維持する代替案がある。64行は単体約3.21%命令減の中間案で、全モデル時間は未測定。

同一moduleのMLP layer0/1/3も再profileして旧INT8出力とbit一致。layer0 handler4,213,170,480→4,052,542,935、base3,319,242,044→3,158,614,499。A/Bは同じで、入力共有によるbase投影の削減を切り分けた。nativeはruntime93＋canister9 unit、integration9、compile-fail doc5が通過。

専用全モデルmoduleは75cc59252c9ed919ff72f736661b2761d3a4b3c9c0c3a7e512a1b653c6cb2cc4、canister6eydd。固定cache準備721 update / 78,739,929,194命令 / 263.449秒 / 4,065,416,192 bytes、推論と別計上。prefix packet再準備は2回目query/命令/通信0。主canister4caro/codec7huk module不変、Layaは未変更。

証拠は`artifacts/prefix_codec/full-s1-wide-proof/report.json`、`before-after.json`、`validated-source.zip`、`codec-second.json`、`main-unchanged.json`と`artifacts/s1_wide/profile/report.json`。build-v2のsource.zipに対するclient routingだけの変更は`client-routing-override.json`/diff/ZIPに明示し、compiled Rust/canister sourceの不変をassertした。未patch raw moduleはfail-closed。生成物はignore、mainnet/push/PR/commitなし。

## 50queryの次の分割

主62queryを8層ごとに分けると16/16/16/14query、実handler命令は60,696,271,782 / 60,632,792,497 / 60,638,666,037 / 53,233,833,696。仮に13/13/13/11queryへ詰めると計50で、各群のnominal余裕は4,303,728,218 / 4,367,207,503 / 4,361,333,963 / 1,766,166,304。これは単純counter和であり、50query完走を証明しない。query境界、IC全体counter、追加carryの2MB上限・復号/連結費、全状態bit一致を実測する必要がある。

監査script `analyze_query_packing.py` と証拠`artifacts/s1_wide/packing-budget/report.json`を追加。既存512行のMLP→Delta carryだけでは2queryを詰め替えるだけなので、query削減とは扱わない。8層をまたぐ途中演算の継続と実測が次の実装対象。主50/32queryは未達、目標は維持する。

診断後のharness修正は`check_s1_wide.py`の出力先指定を必須化（古い診断を上書きしないため）と、直接Cargo buildのtile既定値128だけ。実測時のsourceは各validated-source.zipに保存しており、64/128 builderの明示cfgに数値変更はない。
