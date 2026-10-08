# update推論を1,000億命令以下にする目標

2026-10-06。BOOM DAO 617/620/653の3入力すべてで、都度払い推論のworker handler命令合計を100,000,000,000以下にする。入力・重み・量子化・既存prefix bankを維持し、保存参照に対する出力bit一致を条件にする。目標は継続中。F32投影、Delta再帰、固定prefixの準備、LoRA最終BF16変換とhashを改善し、3入力の全体命令を約7.9〜9.0%削減した。全入力で目標未達。

## 現在のupdate経路の内訳

投票38-token/common27-token prefix、6個の検証済みkernel、34Bで区切るupdate schedulerを使い、診断用runtimeを別にビルドした。各update直後のinclusive spanを取得し、保存Candidの診断返信を再decodeして集計した。基底投影とその内側の区間を重複加算しない。

|入力|診断handler命令合計|update|INT8基底|F32 LoRA|GQA head|その他|
|---|---:|---:|---:|---:|---:|---:|
|617|145,511,362,568|5|68.58%|11.77%|1.18%|18.47%|
|620|125,613,705,623|4|68.52%|11.71%|1.10%|18.66%|
|653|146,530,098,494|5|69.42%|11.89%|1.01%|17.68%|

各入力1回の診断実行で、profileの命令も含む。通常版の新しい削減実測ではない。基準は[都度払い推論の実測](PAID_UPDATE_INFERENCE_MEASURED.md)。その他にはDelta再帰、norm、活性化、hash、コピー、量子化、読み出し、scheduler等が残る。

別の分類で、617のMLP段は82.130B（56.44%）、Delta段は50.701B（34.84%）、Attention段は12.647B（8.69%）。これらは基底投影・LoRAを含むため、上表へ足さない。

全3入力で31層の保存suffix hidden、32層のconv/suffix KV、最終norm hidden、判断・logits・校正確率・unknown確率が参照とbit一致。参照にないlayer30 hiddenと未exportのDelta密状態は直接比較していない。

653の診断合計から目標へは約46.53B（31.76%）の削減が必要。INT8基底だけで達成するならその区間を45.74%減らす必要がある。

ビルド・測定・再検証は `build_update_instruction_profile.py`、`prove_update_instruction_profile.py`、`measure_update_instruction_profile.py`、`report_update_instruction_profile.py`。証跡は `artifacts/update-instruction-profile-v1/{build-v3,proof-v1,summary.json}`。moduleは `f1d74339…`。固定準備の721 update×2 bank、prefix cache準備、prefix登録、境界拒否試験は推論合計から除外する。

## 二段分解の試験

rank49の二段整数分解で2つの出力quartetを同じvectorへ入れ、K部分和の水平加算を出力再構成後へ移した。64出力tileで入力を共有し、固定重みを最初のtoken groupで展開する。INT8重みの容量、block256量子化、F32 scaleとblock加算の順序を維持した。localsは7,168。

|投影のtoken数|現行S1 pair-bounds|二段分解候補|命令増加|
|---:|---:|---:|---:|
|57|624,841,752|715,556,877|14.52%|
|59|645,897,524|719,313,125|11.37%|
|67|730,257,252|804,551,665|10.17%|
|87|941,097,246|1,017,586,699|8.13%|

layer3 Qの8192×2560実重み、5保存実入力・3サイズ切出し・11境界の19条件で、同一moduleのS1/S2両方が独立native整数oracleとdigest一致。計38 queryの保存返信を再decodeし、counterの区間合計とhashも照合した。サイズ切出しは投影サイズの試験で、現在の短文推論そのものではない。量子化・入力準備・投影を含み、digest/Candidを除く。

命令増加のため不採用。全モデルへ接続していない。初版はlane復元順の誤りで境界試験が停止した。修正版build-v3/check-v2だけを成功結果に使用し、失敗版build-v2/check-v1も保存している。

生成・測定・再検証は `build_s2_pair_late_probe.py`、`check_s2_pair_late_probe.py`、`report_s2_pair_late_probe.py`。証跡は `artifacts/s2-pair-late-v1/{build-v3,check-v2,summary.json}`。診断canister `zm54s-at777-77775-aaa5q-cai` は測定後に停止した。

## 二段分解の変換・復元共有

継続試験では、入れ子の共通部分を使って重み変換の加減算を95→55、出力復元の加減算を128→88へ減らした。記号式が元の49係数と16復元式へ一致すること、共有する中間値のI32上限を確認した。localsは7,216。

同じ19条件・38 queryでnative oracleと全digest一致し、保存返信の再decode・counter合計・hashも再検証した。57/59/67 tokenは前候補より1.27/1.40/1.31%減ったが、現行S1より13.06/9.81/8.73%多い。87 tokenも現行S1より6.87%多く、不採用。共通変換だけでは二段分解の周辺負荷を解消できないことが分かった。

証跡は `artifacts/s2-pair-factored-v1/{build,check,summary.json,frozen-workflow.zip}`。生成・測定・再検証は `build_s2_pair_factored_probe.py`、`check_s2_pair_factored_probe.py`、`report_s2_pair_factored_probe.py`。moduleは `34e0070c…`。全体推論への接続は行わず、診断canisterは停止した。

## S1内部のcounter試験

現行S1へ4つのcounterを追加し、初回pair、後続pair、奇数tail、整数復元の読み値を保存した。19条件・38 queryで独立native oracleとdigest一致し、保存返信の再decode、入力・ソース・依存物・moduleのhashも照合した。証跡は `artifacts/s1-inner-profile-v1/{build-v6,check,summary.json,frozen-workflow.zip}`。moduleは `d624ab9c…`。

この方式は内部のコスト配分には採用しない。counterによって通常版より約40〜81%命令が増え、dotと復元の配分にもずれがある。Wasmの区間入口での先行計量が境界をまたぐ可能性があり、読み値の合計が全体以下であることだけでは各演算への帰属を証明できない。`cost_attribution_validated=false` として生値を残した。全体推論へ接続しておらず、削減実績には含めない。

生成・測定・再検証は `build_s1_inner_profile.py`、`check_s1_inner_profile.py`、`report_s1_inner_profile.py`。重み準備と保存入力は従来の単体投影試験を使用するため、現行3入力のworker合計を直接測る試験ではない。

## 継続する調査

INT8基底のdot・係数変換・入力ロード・整数復元について、計量境界の影響を避けて比較し、積の削減が周辺命令の増加を上回る候補を探す。その他の約18%もDelta内部から分解する。単体候補は実Wasmと独立参照で検証し、改善した候補だけを都度払い経路へ接続して3入力を再測定する。

全体比較canister `6eydd` はsnapshotから元のmodule `6052cc94…`、721項目・4,065,416,192 bytesの重みcache、pack状態へ復元して照合した。今回のsnapshotは削除済み。保護対象・mainnet・frontendは変更していない。

## F32出力共有とスタック部分和の候補

既存の64出力kernelを残し、出力行数が128の倍数のときだけ128出力kernelへ進む候補をビルドした。F32の列順、各mul/addの順序、固定重みのpacked output32配置は維持する。moduleは `2af4d17c…`、証跡は `artifacts/update-f32-output128-v1/{build,proof,summary.json}`。3入力各1回の全体update検証が完了し、保存返信のhidden/state/最終norm/判断を再検証した。従来の3反復中央値に対し617/620/653は145,369,317,910 / 125,495,416,760 / 146,383,993,119命令、0.0959/0.0956/0.0978%減。元のmodule・cache・packへ復元済み。これは大幅な改善ではなく、全入力で目標未達。

別の単体試験では、F32部分和を列ごとにlocalへ保存する代わりにWasm operand stack上で保持した。64列の積と加算を元の順序で行い、入力を列ごとのlocalへ一度loadする。layer0 down-Bの同じ重み・入力を用いた15条件・30 queryでnative scalarと全digest一致し、保存Candid返信も再decodeした。既存128出力候補に対し57/59/67/87 tokenで19.236/19.249/19.278/19.313%減。投影単体の結果であり、全体worker合計の削減率ではない。

単体moduleは `574be2bd…`、証跡は `artifacts/f32-stack-v1/{build,down-B,summary.json,frozen-workflow.zip}`。`build_f32_stack_probe.py`、`check_f32_stack_probe.py`、`report_f32_stack_probe.py`で生成・測定・再検証した。診断canister `zm54s` は停止済み。

64/128両出力幅へスタック部分和を適用した全体候補の3入力検証が完了した。moduleは `544f19b1…`、証跡は `artifacts/update-f32-stack-v1/{build,proof,summary.json,workflow-hashes.json,frozen-builder.py}`。617/620/653は142,029,525,141 / 122,645,484,197 / 142,998,868,511命令、全3入力で保存参照とbit一致。従来3反復中央値に対し2.391/2.364/2.408%減。617はworkerが5→4回になり、620/653は4/5回。保存worker返信と参照を再検証した。元のmodule・cache・packへの復元とsnapshot削除も照合済み。目標は全入力で未達。

128出力のみのsnapshot `00000000000000197fffffffffa000180101`、スタック投影のsnapshot `000000000000001a7fffffffffa000180101` は復元・削除済み。次の比較は下記の投影/Deltaの組合せ。

## 整数係数の事前展開（不採用）

rank343の重み変換を質問前のowner updateで準備する試験を実施した。変換後INT16重みは元のINT8重みの343/32（約10.72）倍。独立native整数oracleと19条件・38 queryすべてでdigest一致し、保存Candidとcounter、入力・source hashも再検証した。

ただし57/59/67/87 tokenは現行S1より48.75/44.23/41.69/32.32%多い。固定変換を移しても入力準備、読み出し、復元の負担が残る。元重量の展開と既存対照を合わせたseal結果は266,764,288 bytes、準備1,828,212,278命令。追加容量を伴っても推論改善に至らず不採用。moduleは `c934bb01…`、証跡は `artifacts/s3-prepared-v1/{build,check,summary.json,frozen-workflow.zip}`。全体へ接続しない。

## Deltaスタック部分和の候補

128×128のregister stateを維持し、出力列ごとの2つの昇順key sumをoperand stack上で計算する。状態のdecayと更新をlocal.teeへまとめる。同じ出力列内のF32 mul/add/subの順序を維持し、異なる列の独立処理だけをまとめ直した。

現行register対照と同一moduleで、1/8/57/59/67/87/132 tokenの7条件・14 queryを実施した。出力に加え、全16384要素の最終key-major状態もnativeの明示的F32参照とbit一致。保存返信を再decodeしてhash/counterを照合した。57/59/67 tokenのkernelは20.657/20.660/20.616%減。これは合成入力の1head試験で、全体推論の削減実績ではない。

moduleは `948a8c19…`、証跡は `artifacts/delta-stack-v1/{build,check,summary.json,frozen-workflow.zip}`。診断canisterは停止済み。

F32投影と組み合わせた全体候補 `107ca325…` の3入力検証と保存返信の再検証が完了した。証跡は `artifacts/update-f32-delta-stack-v1/{build,proof,summary.json,workflow-hashes.json,frozen-workflow.zip}`。

|入力|worker命令合計|従来3反復中央値からの削減|worker回数|
|---|---:|---:|---:|
|617|140,950,540,437|3.1327%|4|
|620|121,720,640,165|3.1006%|4|
|653|141,884,930,315|3.1682%|4|

各1回の実行で、全31保存hidden、32conv/KV hash、最終norm、判断・logits・確率がbit一致。全体経路で未exportのDelta密状態を直接比較した結果とは区別する。653でもworkerが5→4回になった。snapshot `000000000000001b7fffffffffa000180101` から元のmodule `6052cc94…`、721項目・4,065,416,192 bytesのcache、packへ復元したこととsnapshot削除を再検証した。全入力で目標未達。

## INT8の160/128/32出力切替

以前の160出力候補では端数を128出力へpaddingして命令が増えた。残り出力数に応じて160、128、32へ切り替え、8192/4096出力のpaddingを除いた。同じraw INT8配置、block256量子化、整数演算、F32 block加算順を維持した。

単体19条件・38 queryで独立native整数oracleと全digest一致し、保存返信・counter・source hashを再検証した。57/59/67/87 tokenの8192×2560投影は現行128出力より0.2941/0.2913/0.2951/0.2966%減、132 tokenの4096出力は0.1899%減。単体moduleは `4c0d2af2…`、証跡は `artifacts/s1-adaptive160-v1/{build,check,summary.json,frozen-workflow.zip}`。

F32/Delta改善と組み合わせた全体候補 `24ce2a7a…` を `artifacts/update-adaptive160-v2/build` へビルドした。最初の全体builderはsource置換の二重anchor検出でcompile前に停止し、失敗版v1も保存した。v2は通常投影だけを置換し、8個のkernelを個別にpatch・Wasm検証した。`prove_update_adaptive160.py` の3入力検証と `report_update_f32_candidate.py` の保存返信・hash再検証が完了した。617/620/653は140,660,146,855 / 121,472,507,094 / 141,589,972,780命令、各4 workerで全保存参照とbit一致。従来3反復中央値に対する削減は3.332/3.298/3.370%。snapshot `000000000000001c7fffffffffa000180101` から元のmodule・721項目cache・packへ復元し、snapshot削除まで再検証した。全入力で目標未達。診断canisterは停止済み。


## 都度払いAPIを含む候補ビルド

組合せ候補を都度払いAPIへ接続するmodule `5d896014…` も `build_paid_f32_delta_stack.py` でビルドした。7個のkernelを別々のfunctionへpatchしてWasm検証し、課金・schedulerの3ファイルが現在の `canisters/inference/src` とbyte一致することを照合した。証跡は `artifacts/paid-f32-delta-stack-v1/{build,workflow-hashes.json,frozen-builder.py,build-verification.json}`。ビルド・source検証だけであり、この都度払いmoduleでの推論・課金試験は未実行。

## 入力ゼロ領域の確認

保存されたlayer3 MLP product（87×9216）を、現在のblock256のF32 max/127・nearest-even方式で量子化して調べた。ゼロ要素は約7.10%だが、連続4要素のゼロ領域は約0.020%。S1の7入力planeでは完全なtoken pairのゼロ4要素が0〜0.026%で、dotごとの分岐を増やす根拠にはならない。証跡は `artifacts/int8-sparsity-v1/report.json`。1つの保存入力の調査で、全層や現在の3短文入力の性質を保証しない。疎kernelは採用していない。

## F32の出力幅・列幅512の単体候補

スタック版を512出力へ拡張し、固定重みのpacked output32配置と64列内の昇順F32 mul/addを維持した。kernelは8,275 locals。layer0 down-Bの15条件・30 queryでnative scalarとbit一致し、保存返信・counter・hashを再検証した。現在のスタック128出力版と同じ入力・重みの比較で57/59/67/87 tokenは1.9929/1.9982/1.9245/1.9647%減。moduleは `d0164da1…`、証跡は `artifacts/f32-stack512-v1/{build,down-B,summary.json,frozen-workflow.zip}`。

別の候補は64出力を維持し、呼び出しごとのF32列範囲を64→512へ拡張した。kernelは8,709 localsで、全列を元の昇順に計算する。layer0 gate-Aの15条件・30 queryでnative scalarとbit一致し、同一moduleの対照WATが現在のスタック64版とbyte一致することも照合した。57/59/67 tokenは0.9743/0.9749/0.9750%減。moduleは `c9974069…`、証跡は `artifacts/f32-columns512-v1/{build,gate-A,summary.json,frozen-workflow.zip}`。

両方とも単体投影の実績で、全体worker合計ではない。診断canisterは停止した。INT8のadaptive160とDelta改善へ接続した全体候補を `build_update_f32_shapes.py` でビルドし、10個のkernel・source/dependency/workflow hashとWasmを検証した。moduleは `041b3925…`、証跡は `artifacts/update-f32-shapes-v1/{build,workflow-hashes.json,frozen-builder.py}`。出力512はcols64・rows512倍数、列512はrows64・cols512倍数だけへ接続し、残る形状は既存の128/64版を使う。`prove_update_f32_shapes.py` の3入力検証と保存返信・source hashの再検証が完了した。617/620/653は140,471,089,641 / 121,308,227,380 / 141,397,477,577命令、各4 workerで全保存参照とbit一致。従来3反復中央値に対し3.4622/3.4289/3.5009%減。snapshot `000000000000001d7fffffffffa000180101` から元のmodule/cache/packへ復元し、snapshot削除も再検証した。全入力で目標未達。

## 登録済みprefix状態を使う候補

2026-10-06継続。全体update経路の `server_delta_hybrid_input` は、各推論で既存packetとBF16値を一度payloadへ作り直してdecodeする。固定状態cacheは27-token専用で、投票38-token packetへ一致せず復元を繰り返す。登録時に元のchecked server入力経路で状態を一度decodeし、privateな型へ保持する候補を作った。prefixのtoken列・conv history・packet・量子化は維持する。

推論時はversion/model/pack/tensor/op/encoding/aux/scalars、head数、prefix長、keep値とBF16入力のshape/finiteを照合し、固定状態から新しいworking Vecを作る。suffix状態は固定prefixへ保存しない。登録時に以前の24項目prefix cacheをクリアし、同じ48 MiB相当をserver graphへ置き換える。最終常駐状態のサイズを増やさない意図だが、全体Wasmでのheap実測は未完了。

module `0fb9364a…`、証跡は `artifacts/update-bound-prefix-v1/{build,workflow-hashes.json,frozen-builder.py,prefix-helper.rs,prefix-api.rs,identity/report.json}`。10個のkernel、source/dependency/workflow hashとWasmを検証した。実helperをnative harnessへincludeし、5 token数、15種類のrequest変更、7 input境界、3 state境界、working stateの独立コピーを検証した。packet decodeと全体Wasm参照一致の証明とは区別する。`prove_update_bound_prefix.py` を用意した。`prove_update_bound_prefix.py` の3入力検証と保存返信・source hashの再検証が完了した。617/620/653は138,287,216,546 / 119,156,695,525 / 140,610,292,642命令、各4 workerで保存参照とbit一致。F32形状候補から2,183,873,095 / 2,151,531,855 / 787,184,935命令減。snapshot `000000000000001e7fffffffffa000180101` から元のmodule・721項目cache・packへ復元し、snapshot削除も照合した。最大heap実測は617が4,204,920,832 bytesで、直前候補より655,360 bytes増。固定prefix状態をgraphへ移したため721項目の重みcacheの内容とサイズは維持する。全入力で目標未達。

## 残る処理の診断ビルド

prefix候補を基に、Rust関数の呼び出し境界でDelta head、executeのconv/norm各種、LoRAのBF16最終変換、MLP活性化、prefix入力、exported hashをinclusive計測する別moduleをビルドした。実WATの10 kernelは変更しない。moduleは `4a9cad68…`、証跡は `artifacts/update-other-profile-v1/{build,workflow-hashes.json,frozen-builder.py}`。診断命令が入るため改善実測には扱わない。`measure_update_other_profile.py`、`prove_update_other_profile.py`、`report_update_other_profile.py` を用意した。profileは未実行で、内訳とbit一致は未検証。source・10個のWAT patch・workflow hashは再照合した。現在進行中のprefix改善の結果を先に確定する。

## float hashの一括更新

server-held graphのhidden/state digestはF32を4 bytesずつSHA-256へupdateしている。同じinitialized float byte viewを一度updateする診断を、独立hashlibの入力byte列と比較した。実保存LoRA入力とsigned zero/subnormal/NaN/Infinityを含む7条件・14 queryでSHA値が一致し、保存返信とcounter・hashを再検証した。大きな4096/24576/146432/196608要素はhash本体13.97/14.02/14.03/14.03%減。1/3要素は1.097/0.016%増。

最初のcheckerは返信のbyte listをhex文字列と比較して停止した。失敗版checkを残し、list同士へ修正したcheck-v2だけを成功証跡に使う。moduleは `ae512a39…`、証跡は `artifacts/float-hash-v1/{build,check-v2,summary.json,frozen-workflow.zip}`。診断canisterは停止した。

4096要素以上かつlittle-endianのgraph hashだけを一括更新する全体候補を `build_update_prefix_hash.py` でビルドし、10 kernel・source/dependency/workflow hashとWasmを検証した。moduleは `28ebfe01…`、証跡は `artifacts/update-prefix-hash-v1/{build,workflow-hashes.json,frozen-builder.py}`。`prove_update_prefix_hash.py` を用意したが、全体測定は未実行。小さい配列とbig-endianは元の逐次updateを使う。hash単体の削減率であり、全体worker命令の削減率ではない。全体検証前に採用済みとは扱わない。

## 診断時の重み準備を再利用する候補

全体診断では2種類のprefixを比較するため、従来は同じ721個の重みをbankごとに再準備していた。診断専用のowner update `reset_update_prefix` を追加し、完了済みgraph/prefixだけをclearして固定の重み・activation/rope cacheを保持する候補を作った。activeな推論があればresetを拒否する。測定前のcache statusを照合し、2つ目のbankのwarm処理は既存721項目を再利用する。準備命令はworker推論合計に含めない。

初版v2はRust closureのResult error型の推論に失敗してcompileで停止した。失敗artifactを残し、明示的な型を与えたv3のビルドとsource/dependency/workflow hash・10 kernel・Wasm検証が完了した。moduleは `04d0faf5…`、証跡は `artifacts/update-other-profile-v3/{build,workflow-hashes.json,frozen-builder.py}`。`build_update_other_profile_reuse.py`、`measure_update_other_profile_reuse.py`、`prove_update_other_profile_reuse.py`、`report_update_other_profile_reuse.py` を用意した。現行prefix候補の全体検証・復元が完了するまで、診断用moduleを比較canisterへinstallしない。各入力の最初のworker後にactive reset拒否を試し、2つ目のbankの準備updateが0回で同じ固定cacheへ一致することを照合する verifierも追加した。現行prefix候補の3入力検証・復元と保存返信の再検証後、v3の全体診断を開始した。snapshotは `000000000000001f7fffffffffa000180101`。prefix切替・warm cache再利用・profile・診断版のbit一致と復元は未検証。


## 診断のreset応答と再試験

v3最初の試験は617/620の推論を完了後、prefix切替の応答がCandid型推論で `variant {17_724}` と数値ラベルになり、checkerの `Ok` 照合で停止した。reset自体の失敗とは扱わない。失敗したproof・measurement-source・frozen workflowを `.failed-reset-label` へ保存し、元のmodule/cache/packへの復元とsnapshot削除を確認した。明示DIDを指定するcheckerへ修正し、snapshot `00000000000000207fffffffffa000180101` の再試験を開始した。3入力完了・保存返信の再検証までは成功証跡に使わない。

## LoRA最終BF16変換のSIMD候補

LoRA出力の `bf(bf(base)+bf(scale*z))` を4要素ずつSIMDで計算する単体候補を作った。F32 mul/addを別々に実行し、各BF16丸めとscalar tailの順序を維持する。signed zero、subnormal、tie前後、0/-0/正負scale、実保存入力の値を算術のoperandとして使う9条件・18 queryで、同一Wasmのscalarと独立Pythonの明示F32/BF16計算へ全出力bitが一致した。保存返信を再decodeし、source/dependency/helper hashとcounterを検証した。4096/24576/196608要素のfinalization本体は71.70/71.76/71.77%減。1/3要素は悪化し、全体候補は4要素未満で元のscalar式を使う。実LoRAの途中結果をexportして比べた試験ではなく、全体推論の削減率とも区別する。

単体moduleは `321f629a…`、証跡は `artifacts/lora-finish-v1/{build,check,summary.json,frozen-workflow.zip}`。診断canisterは停止済み。opaque prefix、F32 shape512、Delta stack、INT8 adaptive160とlarge-slice bulk hashへ組み合わせた全体module `03e3ddd4…` をビルドし、10 kernel・Wasm・source/dependency/workflow hashを検証した。証跡は `artifacts/update-lora-finish-v1/{build,workflow-hashes.json,frozen-builder.py}`。全体3入力はまだ未実行。既存のinstruction-profile cfgはbuilderから継承されるが、counter開始とprofile exportを追加した診断版ではなく、CLOCKは未開始の経路である。


診断v3の再試験も、voting重み・prefix準備後にローカルcanisterのcycle残高が30日分の凍結reserveを割り、workerと自動snapshot restoreがIC0207で拒否されて停止した。ローカルへ50兆cyclesを補充し、停止済みcanisterをsnapshot `0020…` から手動復元・起動して、元module・721項目cache・pack一致をTransportで検証し、snapshotを削除した。`proof/manual-restored.json` と失敗ログを保存した。診断の3入力完了を主張しない。追加の同じ診断warmを繰り返す前に、検証済みSIMD算術を含む全体候補の3入力測定を開始する。


## SIMD最終変換・hash候補の全体検証完了

全体module `03e3ddd4…` の617/620/653各1回の推論と、保存返信・参照・source/dependency/workflow hashの再検証が完了した。証跡は `artifacts/update-lora-finish-v1/{build,proof,summary.json,frozen-workflow.zip}`。

|入力|worker命令合計|従来3反復中央値からの削減|worker回数|
|---|---:|---:|---:|
|617|134,577,055,962|7.5128%|4|
|620|115,968,851,190|7.6795%|4|
|653|136,835,967,932|6.6140%|4|

全31保存hidden、32conv/KV hash、最終norm、判断・logits・校正確率・unknown確率がbit一致。layer30 hiddenと未exportのDelta密状態は直接比較していない。これらの数値はSIMD最終変換とbulk hashを合わせた全体結果であり、片方だけの全体寄与を断定しない。2つ目のprefix bankは追加の重み準備updateが0回で、同じ固定cacheを維持した。推論中のreset拒否はこの非診断版の測定では試していない。診断v3の3入力内訳試験の完了も主張しない。

snapshot `00000000000000217fffffffffa000180101` から元module `6052cc94…`・721項目/4,065,416,192 bytes cache・packへ復元したこととsnapshot削除を検証した。診断canister `zm54s` は停止済み。現在の最大は653の136,835,967,932命令で、1000億へさらに36,835,967,932命令（現在値の約26.92%）の削減が必要。goalは継続中で、全入力で未達。都度払いAPI・課金を含むmoduleへの今回の組合せ導入と実行はまだ行っていない。


## rank49の88出力共有試験

固定重みの二段変換を64→88出力で共有する候補を作り、9,574 localsで10,000上限内に収めた。raw INT8を保持し、固定重みの不足末尾は88の倍数へzero pad、scaleの不足末尾は1で補い、要求した出力幅だけ返す。K256ごとのF32 scale・加算順とrank49 identity・I32 boundsは既存の検証を維持した。

module `c26add02…`、証跡は `artifacts/s2-pair88-v1/{build,check,summary.json,frozen-workflow.zip}`。19条件・38 queryで独立native整数oracleへ全digestが一致し、保存Candid返信・counter・source/dependency/workflow hashを再検証した。結果は以下。

|条件|現行S1比の命令削減率|旧64出力版比の命令削減率|
|---|---:|---:|
|size-67|-8.1379%|0.5388%|
|617|-6.2620%|0.5707%|
|normal|-3.8957%|0.6038%|
|size-57|-12.4677%|0.5232%|
|size-59|-9.2313%|0.5207%|

負の削減率は悪化を示す。同一moduleの現行S1 controlより遅く、全体候補へ接続しない。旧64出力版からは改善したが、出力幅拡大のみで変換コストを吸収できるという仮説は採用の根拠にならなかった。診断canisterは停止済み。全体の最良値134,577,055,962 /115,968,851,190 /136,835,967,932と1000億goalは維持する。


## conv_stateの4 channel SIMD候補

k=4の畳み込みを4 channelずつ計算し、重みを4 tapのvectorへ転置して各tokenで共有する候補を作った。各channelのF32 multiply/addは元のj=0,1,2,3の順で、+0から開始する。BF16丸めとbf_silu、最後の3-token履歴を保持する。元のchecked conv_stateのshape条件を満たしたWasm k=4かつchannelが4の倍数だけを切り替え、その他とnativeは元のscalar経路を使う。

最初のprobeは二重extern指定でcompile失敗、v2は2つの依存directoryのserde/serde_jsonがStableCrateId衝突してcompile失敗した。各失敗artifactを残し、現行全体候補の一貫した依存セットを使用したv3をビルドした。小型診断module `e1b69948…`、証跡は `artifacts/conv4-simd-v3/{build,check,summary.json,frozen-workflow.zip}`。

10条件・40 queryで、元runtime execute(conv_state)の活性化後digestへ一致し、活性化前のscalar/SIMD両digestが独立NumPyの明示F32 multiply/add oracleへ一致した。保存Candid、counter、source/dependency/workflow hashを再照合した。signed zero/subnormal/tie値、layer0の実INT8重みを元と同じrow scaleでF32へdecodeし、保存activationの値をoperandとして使用した。実畳み込みの途中入力をexportした試験ではない。

n=48/56,c=8192のkernel本体は70.504/70.525%減。n=57/67,c=2560は70.404/70.413%減。固定activation表のprepareと入力decodeはkernel counterから除外され、全体命令の削減率ではない。診断canisterは停止済み。既存SIMD BF16/hash改善へ組み合わせた全体候補をビルド中。3入力全体検証の完了前に採用済みとは扱わない。


全体conv4候補のビルド・10 kernelのWasm検証・source/dependency/workflow hash照合が完了した。moduleは `331aa2df…`、証跡は `artifacts/update-conv4-v1/{build,workflow-hashes.json,frozen-builder.py}`。`prove_update_conv4.py` で3入力測定を開始し、snapshot `00000000000000227fffffffffa000180101` を保存して比較canisterへinstallした。proverは各終了経路で元snapshotへのrestoreとmodule/cache/pack照合、snapshot削除を行う。固定重み準備・3入力推論・復元は進行中で、全体の改善値は未確定。reporterは `report_update_conv4.py`。最後に検証済みの全体最良値と1000億goalは維持する。


## conv4の固定活性化表参照をまとめる候補

固定BF16 SiLU表をconv4処理ごとに一度immutable borrowし、各要素でRefCellを再参照する処理を除いた。BF16丸めを行った有限値だけをそのhigh16 bitでindexし、表がない場合と非有限値は元のbf_silu_originalへ戻す。質問のactivation/stateは保存しない。k=4/channel4倍数のshape guardとF32 tap加算順、履歴tailは維持する。

単体module `04b7ea7c…`、証跡は `artifacts/conv4-cached-v1/{build,check,summary.json,frozen-workflow.zip}`。同じ10条件・40 queryで元scalar runtimeへの活性化後digest一致と、独立F32 oracleへの活性化前digest一致を検証し、保存返信・counter・source/dependency/workflow hashを再照合した。n=48/56,c=8192はscalar比75.065/75.093%減、先行SIMD版からkernel本体を約15%追加削減した。表なしfallbackはこのprobeでは直接試しておらず、準備済み表の実行経路の証跡である。全体命令の削減率ではない。診断canisterは停止済み。

全体builderの初版は入れ子source文字列の改行escapeで構文エラーになり、compile前に停止した。失敗版 `artifacts/update-conv4-cached-v1` を残した。APIを別sourceファイルとして保存するv2でビルド、10 kernelのWasm検証とsource/dependency/workflow/entry hash照合が完了した。moduleは `2fe19846…`、証跡は `artifacts/update-conv4-cached-v2/{build,workflow-hashes.json,frozen-builder.py,activation-table-api.rs}` とentry source/hash。3入力prover/reporterを用意したが、先行conv4全体試験の完了と復元までこのmoduleを比較canisterへinstallしない。


## conv4 SIMD全体検証完了

全体module `331aa2df…` の3入力各1回の全体測定と、保存返信・参照・source/dependency/workflow hash再検証が完了した。証跡は `artifacts/update-conv4-v1/{build,proof,summary.json,frozen-workflow.zip}`。

|入力|worker命令合計|従来3反復中央値からの削減|worker回数|
|---|---:|---:|---:|
|617|132,797,091,978|8.7361%|4|
|620|114,443,701,350|8.8937%|4|
|653|135,024,152,180|7.8505%|4|

31保存hidden・32conv/KV hash・最終norm・判断/logits/校正確率/unknown確率がbit一致。layer30 hiddenと未exportのDelta密状態は直接比較していない。2つ目のprefix bankの重み準備updateは0回で、固定cacheを維持した。snapshot `00000000000000227fffffffffa000180101` から元のmodule `6052cc94…`・721項目/4,065,416,192 bytes cache・packへ復元したこととsnapshot削除を確認した。最初のworkerはcoldなcompile等でwall timeが長く、命令数の改善と時間の改善は同一視しない。

現在の最大は653の135,024,152,180命令で、1000億へさらに35,024,152,180命令（約350億）必要。全入力で目標未達。都度払いAPI・課金を含むmoduleへの今回の組合せ導入と実行はまだ行っていない。固定活性化表のborrowをまとめる次候補の3入力測定を開始し、snapshot `00000000000000237fffffffffa000180101` を保存した。次候補の結果と復元は未確定で、先行候補の成功証跡と区別する。


## conv4固定表参照版の全体検証完了

全体module `2fe19846…` の3入力各1回の推論、保存返信・参照・source/dependency/workflow hashの再検証が完了した。証跡は `artifacts/update-conv4-cached-v2/{build,proof,summary.json,frozen-workflow.zip}`。

|入力|worker命令合計|従来3反復中央値からの削減|worker回数|
|---|---:|---:|---:|
|617|132,665,218,146|8.8267%|4|
|620|114,330,701,886|8.9836%|4|
|653|134,889,919,052|7.9421%|4|

31保存hidden・32conv/KV hash・最終norm・判断/logits/校正確率/unknown確率がbit一致。layer30 hiddenと未exportのDelta密状態は直接比較していない。common bankの追加重み準備updateは0回。snapshot `00000000000000237fffffffffa000180101` から元module `6052cc94…`・721項目/4,065,416,192 bytes cache・packへの復元とsnapshot削除を検証した。先行conv4からの追加削減は617で131,873,832、620で112,999,464、653で134,233,128命令。最大134,889,919,052命令なので、1000億へさらに34,889,919,052命令（約349億）必要。全入力で未達、goalを継続する。都度払いAPI・課金を含むmoduleへの今回の組合せ導入・実行はまだ行っていない。


## rank343整数再合成の一度だけ使う中間値

744個の再合成DAG nodeの524個は一度しか使われていなかった。最終64 rootを除く460 nodeをWasm operand stack上で展開し、local保存と再取得を除いた。複数使用nodeと最終rootは保持し、leafの保存先を再利用せず独立したslotを割り当てた。I32計算とF32 blockのscale・加算順を維持し、既存symbolic identity・I16/I32 boundsの検証を残した。生成kernel9478 locals、固定重み準備後のkernel8885 localsで上限内。module `d6ef2fdb…`、証跡は `artifacts/s3-inline-v1/{build,generated,check,summary.json,frozen-workflow.zip}`。

21条件・42 queryで独立native整数oracleへ全digestが一致し、保存Candid・counter・source/dependency/entry/workflow hashを再検証した。48/56/57/67 tokenは現行S1比32.847/30.631/42.719/35.889%増。旧prepared rank343版の再合成より改善したが、現行S1より遅く、全体へ接続しない。固定重みの10.71875倍の展開容量も維持されており、全体cache予算を満たす候補ではない。診断canisterは停止済み。


## Winograd rank343の変換削減と全整数範囲検証

rank343を維持し、Winogradの再帰分解へ変更した。入力・重みの変換DAGはそれぞれ465から372 node、再合成は744から651 nodeになった。一度だけ使う非root中間値のstack展開も維持し、prepared kernelは8939 localsで上限内。module `81dc9bd8…`、証跡は `artifacts/s3-winograd-v1/{build,generated,check,integer-bounds.json,summary.json,frozen-workflow.zip}`。

64最終rootのbilinear係数を通常の8×8整数行列積と照合した。I16変換は入力8128・重み8192以内。変換済みleafを単純に足す保守的な上界は3,033,759,744となったが、各中間値の厳密なbilinear係数で相殺を計算すると、全343 leafの最大上界379,219,968、全651再合成nodeの最大上界337,084,416でI32内に収まる。最終dotは4,161,536以内。これはq[-127,127]・raw weight[-128,127]・leaf32列の全入力についての係数上界であり、測定した入力だけへの保証ではない。最初の`ring-proof.json`はmodulo2^32による恒等式を記録した履歴として保持し、追加の`integer-bounds.json`でwrapを必要としないことを検証した。F32変換・scale・block加算順は維持した。

21条件・42 queryの全digestが独立native整数oracleへ一致し、保存Candid・counter・source/dependency/entry hashと整数範囲証明を再検証した。48/56/57/67 tokenは現行S1比32.3823/30.1638/42.1951/35.3842%増。前のinline rank343版より約0.35〜0.37%改善したものの、現行S1より遅く、全体推論へ接続しない。固定重みは10.71875倍の展開容量のままであり、全体cache予算にも対応していない。診断canisterは停止済み。最良の全体測定値と1000億のgoalは変わらず、未達として継続する。


## 残差加算と正規化のSIMD候補

`add_norm_bf16`の残差加算・BF16丸め、平方、最終multiply/BF16丸めを4要素SIMD化した。平方和はlane0/1/2/3の順にscalar F32へ加え、行の左から右の総和順を維持した。`v * scale * weight`の2回のmultiply順とBF16丸めも維持し、端数はscalarへ戻す。既存のshape/weight/epsilon検査を通ったWasm実行で列幅4以上にだけ適用し、その他は従来経路を使う。

単体module `66107633…`、証跡は `artifacts/add-norm-simd-v1/{build,check,summary.json,frozen-workflow.zip}`。1/3/4/7/9列と2560/8192列、48/56/57 token相当、保存activationを算術operandへ用いた計11条件・22 queryで、残差と正規化後の全出力が同moduleの従来scalar版および独立したordered F32 oracleへbit一致。48/56/57×2560は処理本体50.2080/49.7661/49.7197%減。保存activationは実推論のadd_norm中間値そのものではない。保存Candid返信・counter・source/dependency/workflow hashを再検証し、診断canisterを停止した。

最良のconv4固定表版へ組み合わせた全体module `09c569af…` をビルドし、10 kernel patchとsource/dependency hashを検証した。snapshot `00000000000000247fffffffffa000180101`を保存して、3入力の全体測定を開始した。結果・参照一致・元module/cache/packへの復元はこの時点では未確定であり、全体削減の実績にはまだ含めない。


## 最新runtimeの都度払い版ビルド

`paid-add-norm-simd-v1`（module `b889e378…`）は最新runtimeの10 kernelをcanonical paid APIへ接続してcompileしたが、canonical schedulerへ置き換えることでbound prefixとbulk digestの改善を引き継がなかった。この版は測定・採用していない。

`paid-add-norm-simd-v2`（module `0a4da30a…`）ではcanonical schedulerの2 prefix bank、paid admin guard、start/continue公開関数、課金APIを維持し、Prefixのopaque prepared state、固定prefixの事前検査とworkerでのbound入力、large-slice bulk digestの3改善だけを適用した。`paid_types.rs`と`paid_inference.rs`はcanonicalとbyte一致。10 kernelをsource digestで照合してpatch・Wasm validationし、課金と組み合わせたcompileに成功した。証跡は `artifacts/paid-add-norm-simd-v2/{optimized-paid-scheduler.rs,frozen-builder.py,workflow-hashes.json,build/report.json}`。paid diagnostics cfgを有効にした診断版であり、実行・課金境界・worker合計・2 bankのheap容量・参照bit一致はまだ未検証。通常版の全体proof実行中には、このpaid版を同canisterへinstallしない。


## add_norm SIMD版の全体検証完了

module `09c569af…`の3入力各1回の全体測定と、保存返信・参照・source/dependency/workflow hashの再検証が完了した。証跡は `artifacts/update-add-norm-simd-v1/{build,proof,summary.json,frozen-workflow.zip}`。

|入力|worker命令合計|従来3反復中央値からの削減|worker回数|
|---|---:|---:|---:|
|617|131,845,214,400|9.3903%|4|
|620|113,633,955,391|9.5383%|4|
|653|134,072,983,316|8.4996%|4|

31保存hidden・32conv/KV hash・最終norm・判断/logits/校正確率/unknown確率がbit一致。layer30 hiddenと未exportのDelta密状態は直接比較していない。common bankの追加重み準備updateは0回。snapshot `00000000000000247fffffffffa000180101`から元module `6052cc94…`・721項目/4,065,416,192 bytes cache・packへの復元とsnapshot削除を確認した。先行conv4固定表版からの追加削減は617で820,003,746、620で696,746,495、653で816,935,736命令。最大134,072,983,316なので、1000億へさらに34,072,983,316命令（約341億）必要。全入力で未達。

通常版の復元確認後、paid diagnostics版`0a4da30a…`の実際のcaller canisterからの3入力、2 prefix bank、paid worker命令数・heap・保存参照・課金境界の検証を開始した。snapshotの保存・測定・復元結果はまだ未確定であり、都度払い版の成功実績には含めない。同時に同canisterへ別のmutationを行わない。


## RMSのordered SIMD単体検証

`rms`の平方と最終2段multiplyを4要素SIMD化し、平方和はlane0/1/2/3の順にscalar F32へ加える。BF16丸めを追加せず元のF32出力を保つ。単体module `51f9cbac…`、証跡は `artifacts/rms-ordered-simd-v1/{build,check,summary.json,frozen-workflow.zip}`。14条件・28 queryで従来scalar Wasmと独立ordered F32 oracleに全出力bitが一致した。1/3/4/7/9列・255/256列・2560/8192列、48/56/57 token相当、異なる桁の平方を交互に加える順序試験、保存activationを用いた算術operandを含む。保存operandは実推論のRMS中間値そのものではない。保存Candid・counter・source/dependency/frozen entry hashを再検証した。大きい形状の処理本体は約48〜50%減。診断canisterを停止した。

最良のadd_norm SIMD版に組み込んだ全体module `b61a94d7…`をビルドし、10 kernel patch・source/dependency hashを検証した。Wasmで同じ長さの入力・重みかつ4要素以上の場合にのみ新経路を使い、それ以外は元のrmsへ戻す。証跡は `artifacts/update-rms-ordered-simd-v1/{build,workflow-hashes.json,frozen-builder.py}`。都度払いproofの実行中なので、この全体候補はまだinstall・推論測定していない。


## 都度払いadd_norm版の3入力と競合試験の未完了

実caller canister経由の617/620/653は131,845,075,489 / 113,632,435,301 / 134,062,179,395命令、各4 worker。31保存hidden・32conv/KV hash・最終norm・判断/logits/確率の参照一致を通過した。saved Candidのforward/inner/paid debug返信を再decodeし、F32 bitsとcounterを再照合した。余分な付与cyclesの返却、duplicate、ID conflict、少額、quote version、token bounds、worker/status authorityを照合した。証跡は `artifacts/paid-add-norm-simd-v2/{proof,partial-summary.json,frozen-paid-partial-proof.zip}`。

追加の実推論とupgradeの競合試験では、Running receiptを観測してから送ったtiny moduleのupgradeが成功し、旧proverの「必ず拒否される」assertionで停止した。背景推論は約30秒で4 workerを完了し、競合した別推論はBusyで未課金だった。upgrade実行時点までactiveだったことは保存記録だけでは確定できず、競合順序の問題かguardの問題かは未確定。この試験を成功扱いしない。pause・same-version upgradeの後続検証も実行されていないためfull paid proofは未完了。snapshot `00000000000000257fffffffffa000180101`から元module・cache・packへの復元とsnapshot削除、caller停止は確認済み。最大heap4,287,168,512 bytesで4GiBまで7,798,784 bytesだった。1000億へ最大ケースで34,062,179,395命令が残る。

次のpaid ordered RMS候補`e25298c6…`は、検査済みopaque prefix生成後に利用されないraw NPF1 packetをPrefixから除いた。2 bankで保持していたraw packetは63,663,203 bytesで、不要な保持を除いてheapの余裕を確保する狙い。packetのshape・precision・stateの検査を削らず、渡されたpacketは検査完了まで保持する。runtimeは14条件でbit一致したordered RMS SIMD版。課金2ファイルはcanonicalとbyte一致し、10 kernelとcompileを検証したが、このmoduleの全体推論・heap改善は未測定。競合順序に依存しないowner-only active/refund fixtureで実pre_upgrade hookを検査する専用proofを開始した。


## ordered RMS版のdeterministic upgrade guard

module `e25298c6…`について、owner-only diagnostic fixtureでactive jobとInFlight refundを固定し、実IC pre_upgradeの拒否を検証した。Pending/Done failed receiptの同module upgrade前後での保持も検証した。4条件すべて成功し、元module/cache/packへのsnapshot復元・削除を確認した。証跡は `artifacts/paid-rms-ordered-simd-v1/{upgrade-guards/verified.json,upgrade-guard-entry-hashes.json,frozen-upgrade-guards.py}`。synthetic fixtureでのguard検証であり、このmoduleの推論成功や実推論中の競合順序を証明するものではない。

同moduleで実callerの3入力と保存参照・課金境界・receipt保持を検証するproofを開始した。実推論との競合upgradeは同moduleへのupgradeを使い、拒否ならactive理由を照合し、成功なら元の推論と同じjob/resultのCompleted receiptがupgrade後も保持されることを要求する。実行時までactiveであることを、古いRunning観測だけから仮定しない。active時の拒否自体は上の固定fixtureで別に検証する。この新proofの結果・復元はまだ未確定。


## ordered RMS都度払い版の3入力と後続試験

module `e25298c6…`の実caller 617/620/653は131,554,331,964 / 113,366,161,762 / 133,764,192,433命令、各4 worker。saved Candidを再decodeし、31保存hidden・32conv/KV hash・最終norm・判断/logits/確率のF32 bitsを再照合した。最大heap4,220,911,616 bytesで、前版から66,256,896 bytes減。基準との差分にはraw packetの保持削減とRMS SIMDの組合せが含まれ、片方だけの寄与を断定しない。最大ケースで1000億まで33,764,192,433命令（約338億）が残る。

競合した同module upgradeは成功し、その後のCompleted receiptが背景推論のjob/resultと一致した。別推論はBusyで未課金。後続のpause試験では、upgradeでcache/prefixが消えたためNotReadyが先に返り、Pausedを期待した旧順序のassertionで停止した。全額返却を確認した。この後のreceipt再upgrade/replay試験は未実行で、full paid proofは未完了。診断tracebackは旧templateの行を表示するためsource hash assertionの行に見えたが、保存frozen-proofの実際の停止行はpause assertionであり、source hashの変化は見つからない。snapshot `00000000000000277fffffffffa000180101`から元module/cache/packへ復元・snapshot削除・caller停止を確認した。証跡は `artifacts/paid-rms-ordered-simd-v1/{proof,partial-summary.json,frozen-paid-partial-proof.zip}`。pauseはcacheが準備された状態で先に検証し、upgrade後はNotReadyと返却を検証する次proverを準備する。

## owned gateを使うSwiGLU候補

MLPで保持済みのgate Vecをそのまま上書きして追加出力Vecを確保せず、immutable BF16 SiLU表のborrowを処理ごとに一度にまとめた。4要素ずつ最終multiplyとBF16丸めをSIMD化し、非BF16 input・table未準備・端数は元のBF SiLU算術へ戻す。単体module `f3e6fbf0…`、証跡は `artifacts/swiglu-cached-simd-v1/{build,check,summary.json,frozen-workflow.zip}`。

13条件・52 queryで、scalar/SIMD・tableあり/なしの4方法の全出力digestが独立BF16/F32算術oracleに一致した。48/56/57×9216の処理本体はtableあり48.0638/48.0791/48.0807%減、なし約18.4868%減。非BF16・signed zero・極小値・exp飽和・vector端数と保存activationをoperandにしたケースを含む。大きい形状は同じseed operandをquery内の入力準備で展開しており、推論そのものの入力とは区別する。保存Candid・source/dependency/workflow hashを再検証し、診断canisterを停止した。全体fused MLPの2経路へ接続したcandidateをビルド中で、全体改善はまだ未測定。


## SwiGLU全体・都度払いcandidateのビルド

全体module `5b4b719c…`とpaid diagnostics module `8708b47e…`をビルドした。fused MLPのroot経路とMLP reuse経路の2箇所だけをowned-gate helperへ接続し、generic executeの互換経路とfinite output検査は保持した。10 kernelのsource digest・Wasm validationとsource/dependency/workflow hashを検証した。paid_types/paid_inferenceはcanonicalとbyte一致し、prefix schedulerは先行RMS版と同一。証跡は `artifacts/update-swiglu-cached-simd-v1/{build,workflow-hashes.json}` と `artifacts/paid-swiglu-cached-simd-v1/{build,workflow-hashes.json}`。これらの全体推論はまだ未測定。

新paid proverは3入力のあと、cacheがある間にpauseと全額返却を検証してversionを上げて再開する。同version upgrade競合はCompleted receipt保持を要求し、upgrade後はcacheが消えた場合のNotReadyと全額返却を検証する。旧失敗proofを成功へ書き換えず、新candidateの証跡を分ける。新paid moduleのdeterministic upgrade guard検証を開始した。同canisterへの別mutationはguard検証・復元が終わるまで行わない。


新SwiGLU paid module `8708b47e…`のdeterministic active/refund upgrade拒否とPending/Done receipt保持の4条件が成功し、元module/cache/packへ復元してsnapshotを削除した。復元後に、試験順序を修正した同moduleの実callerによる3入力・課金・receipt proofを開始した。結果と最終復元はまだ未確定。


## Winograd leaf-major tile48候補

診断module `c15ac94c…`はprepared Winograd rank343の各leafについて8個の入力registerを読み込み、3組の出力へ共有する。leaf出力1029個と共有reconstruction scratchを保持し、局所変数は9637個でIC上限10000個を下回る。tile48の端数に対しweightをゼロで埋め、scale/outputを拡張し、戻り値へ元の幅だけコピーする。この費用も推論measurementに含める。既存のcanonical bilinear係数の整数範囲証明とF32 block順序を維持する。ビルド・Wasm validation成功、専用小型canisterでnative/S1と照合する21条件42 queryを実行中。初期測定ではS1より遅く、未採用。証跡は `artifacts/s3-leaf48-v1/{build,entry-hashes.json,kernel.wat,check}`。


## SwiGLU都度払い版の3入力（partial proof）

module `8708b47e…`の実callerによる617/620/653は130,857,976,156 / 112,769,228,162 / 133,055,408,849命令、各4worker。先行RMS版から696,355,808 / 596,933,600 / 708,783,584命令減。最大heapは4,220,911,616 bytes。1000億まで最大33,055,408,849命令（約331億）が残る。保存Candidを再decodeし31保存hidden・32conv/KV hash・最終norm・判断/logits/確率のF32 bitsを参照に照合した。layer30 hiddenとdense Delta stateの直接比較は未実施。証跡は `artifacts/paid-swiglu-cached-simd-v1/{partial-summary.json,frozen-paid-partial-proof.zip,proof}`。

cacheがある間のPausedと全額返却は成功した。その後の競合試験でsecondary requestのquote_versionが2のまま残り、現在version4に対する正しいQuoteChanged(current4)と268,000,000,000cycles全額返却が返った。Busy assertionで停止したため、競合upgrade後のreceipt照合とその後のsame-version roundtripは未実行。full paid proofは未完了。snapshot `00000000000000297fffffffffa000180101`から元module/cache/packへの復元・snapshot削除・caller停止を確認した。

旧proofを保存して別directory `artifacts/paid-swiglu-cached-simd-v2`に同じcandidateの再検証を開始した。secondary requestもquote_version4へ合わせた。元build・workflow・deterministic guard証跡をsymlinkで再利用しhashを検証する。再検証の結果・最終復元はまだ未確定。

Winograd tile48は21条件42 queryでS1と独立nativeの出力digest一致を通過し、saved Candidとinstruction sum/source/dependency/整数範囲証拠を再照合した。48/56/57/67 tokensでS1より30.1710/27.9370/39.6190/32.9247%命令増。prepared bytesは267,203,328。採用しない。証跡は `artifacts/s3-leaf48-v1/{summary.json,frozen-workflow.zip,check/report.json}`。小型診断canisterを停止した。


## rank49 Winograd DAG候補

module `0440a097…`は2段Winogradを使い、入力/weight変換それぞれ44個のDAG nodeと77個のreconstruction nodeを共有する。output tile88、locals9616、整数leaf49、reconstruction register95。canonical bilinear係数を展開し、49leafおよび77reconstructionを含むすべての中間のI32上限を確認した（最大84,271,104）。入力/weightのI16上限は2032/2048。16最終rootは元の4×4整数積に一致する。元weight/入力/scaleとF32のblock順序を維持し、weightの端数padding費用を含む。

21条件42queryで同module S1と独立nativeの全出力digest一致。保存Candidを再decodeし、instruction合計・入力hash・source/dependency/workflow hash・全canonical整数範囲を再照合した。48/56/57/67 tokensでS1より10.2959/8.8919/12.9890/8.5767%命令増。前rank49 tile88版に対して57/67ではさらに0.464/0.406%増で、採用しない。証跡は `artifacts/s2-winograd-v1/{summary.json,frozen-workflow.zip,integer-bounds.json,build,check}`。小型診断canisterを停止した。


## SwiGLU版のfull paid API proof再検証完了

同module `8708b47e…`の再実行 `artifacts/paid-swiglu-cached-simd-v2`は最後まで成功した。617/620/653は130,857,976,156 / 112,769,228,162 / 133,055,408,849命令で前回と完全一致、各4worker、最大heap4,220,911,616 bytes。saved Candidの全forward/inner/paid debug返信を再decodeし、31hidden・32conv/KV hash・最終norm・判断/logits/確率F32 bitsを再照合した。

超過cycles返却、duplicate/ID conflict/insufficient/quote version/token bounds/worker authority/status authority、cacheがある状態でPausedと全額返却、Busyと未課金、同module競合upgradeの成功時のCompleted job/result保持、次の同version upgrade前後のreceipt/config保持とduplicate replayの全額返却を検証した。実競合upgradeは推論完了後に成功しCompleted receiptを保持した。active/refund中の拒否は先に固定fixtureで同module4条件が成功済みの証跡を照合した。snapshot `000000000000002a7fffffffffa000180101`から元module/cache/packへ復元・snapshot削除・caller停止を確認した。

証跡 `artifacts/paid-swiglu-cached-simd-v2/{summary.json,frozen-paid-proof.zip,proof/report.json}`はpaid API/課金試験のcomplete=trueであり、1000億目標の完了を表さない。all_targets_met=false、最大残り33,055,408,849命令。layer30 hiddenとdense Delta stateの直接比較も未実施。目標は継続する。


## cached quantization候補の単体検証と組込み

量子化module `993a33dd…`は64個の入力v128をregisterで保持し、最大値走査後に同じ入力をメモリから再読込しない。通常scale（rounded scale >= F32 MIN_POSITIVE）の場合はdivide/nearestで絶対値が127.5未満になる上限を用い、結果を変えないclampを省く。subnormal scaleは元のclampを保持し、zeroのscale1とabsolute-bit peakのInf/NaN拒否を維持する。通常scaleの理論上限127*(1+2^-24)/(1-2^-24) <127.5をrationalで確認した。

27条件81queryでbaseline/cached/cached+normal-clamp省略の3経路を測定し、独立F32 divide/nearest-ties-even/clamp oracleの全I16値・全F32scale byteおよびInf/NaNのErr文字列が一致した。zeros/signed zero/ties/max finite/subnormalの1・63・127・190・191・255・1024・最大subnormal、normal scale境界前後、Inf/signaling NaN、48/56/57×2560の生成input、保存実Q projection input3件を含む。48/56/57形状でcachedだけは23.0949/23.2409/23.0446%本体減、clamp省略と合わせて39.6620/39.9126/39.5755%減。入力decode・返信serializationは本体外で比較し、同じ費用範囲を両方法へ適用した。保存Candidを再decodeし全byte・counter合計・source/dependency/workflow hashを再検証した。証跡 `artifacts/quantize-cached-v1/{summary.json,normal-bound.json,frozen-workflow.zip,build,check}`。小型canisterを停止した。

全体module `1f725738…`とpaid module `c396c789…`をビルドした。公開quantize blockのsignature・Resultを維持し、normal guardを常に有効にするhelperをruntime/quantize_simd.rsへ接続した。paid_types/paid_inferenceはcanonicalとbyte一致、schedulerは先のSwiGLU版とbyte一致。10kernelのWasm validation・source/dependency/workflow hashを検証した。全体gainはまだ未測定。同moduleのdeterministic upgrade guard proofを開始した。


最初のpaid quantize module `c396c789…`は診断featureを付けずにビルドされ、guard試験で`paid_upgrade_probe`が存在せず停止した。snapshotから元module/cache/packへ復元・snapshot削除を確認した。この試験は成功扱いせず、旧証跡を保持する。新directory `artifacts/paid-quantize-cached-v2`ではcompiler commandへ`paid-update-diagnostics`を明示してmodule `caad9994…`をビルドし、同moduleのguard試験を開始した。全体推論はまだ未測定。


新paid quantize diagnostics module `caad9994…`のdeterministic active/refund upgrade拒否とPending/Done receipt保持の4条件が成功し、元module/cache/packへ復元・snapshot削除を確認した。同moduleで実callerの3入力・課金・receipt proofを開始した。結果と最終復元はまだ未確定。証跡 `artifacts/paid-quantize-cached-v2/{upgrade-guards/verified.json,upgrade-guard-entry-hashes.json,frozen-upgrade-guards.py,proof}`。


## cached quantization版の全paid API proof

module `caad9994…`の実caller617/620/653は130,665,143,316 / 112,603,637,934 / 132,857,163,355命令、各4worker、最大heap4,220,911,616 bytes。先行SwiGLU版から192,832,840 / 165,590,228 / 198,245,494命令減。saved Candidの全forward/inner/paid debug返信を再decodeし、31保存hidden・32conv/KV hash・最終norm・判断/logits/確率F32 bitsを再照合した。元module/cache/packへの復元、snapshot `000000000000002d7fffffffffa000180101`削除とcaller停止を確認した。

超過付与cycles返却・duplicate・ID conflict・insufficient・quote version・token bounds・worker/status authority・cacheありPausedと全額返却・Busyと未課金・同module競合upgradeのCompleted receipt保持・さらにsame-version upgrade前後のreceipt/config保持とduplicate replay返却を検証した。固定active/refund fixtureとPending/Done receiptの4条件も同moduleで照合済み。証跡 `artifacts/paid-quantize-cached-v2/{summary.json,frozen-paid-proof.zip,proof/report.json}`。paid API proof complete=trueだがall_targets_met=false。最大残り32,857,163,355命令（約329億）。layer30 hiddenとdense Delta stateの直接比較は未実施であり、1000億目標は継続する。

## rank343 token-streaming tile512の候補

診断module `74775162…`は各leafのweight256v128を保持して全token groupへ適用し、入力8v128を512出力で共有する。整数leaf積をquery-local scratchへすべて書いた後、同じcanonical Winograd復元を行う。局所変数は1094個、scratchは343*32*ceil(tokens/8)*16 bytes。I32 boundsとF32 block順序は既存rank343と同じ。expanded weightのbytes比10.71875倍は単体Q projectionの診断だけで利用しており、全modelでのstorage/費用/heap対応は未実装。ビルド・Wasm validation成功、21条件42queryのnative/S1との照合を実行中。短い8tokenでは基準より遅い。全体へは接続していない。


rank343 token-streamwide tile512は21条件42queryのnative/S1出力digest一致を通過し、保存Candid・counter合計・source/dependency/整数範囲証跡を再照合した。48/56/57/67 tokensで基準より18.9988/16.6698/27.0795/20.8048%命令増となり不採用。scratchのwrite/readも測定内。証跡 `artifacts/s3-streamwide-v1/{summary.json,frozen-workflow.zip,layout.json,build,check}`。診断canisterを停止した。

次の候補はS1の7積のうち、入力/weightの片方だけが変換されるm1/2/3/4に対しI16 productの絶対値<=32512を使い、dot_i16x8をi16x8.mul+extadd_pairwise_i16x8へ置き換える。同じ整数結果とF32順序のまま、ICでの命令費用を実測する。m0/5/6はI16積が溢れる可能性があるため元のdotを保持する。全体への接続・性能改善は未確認。


## S1 bounded mul/pairwiseの費用測定

module `42e65695…`はadaptive160/128/32のkernel160と32に対して、m1/2/3/4だけをI16 mul+widening pairwise addへ置き換えた。A/B係数から全I16 productの絶対値上限32512を独立に再計算した。奇数tailではm0/2/4/6だけを計算する既存最適化を維持し、各productの置換数を記録した。残るm0/5/6は元のdotのまま。

21条件42queryでnative/S1との全出力digest一致を通過し、保存Candidの再decode、I16 bound、counter合計、source/dependency/workflow hashを再照合した。ICの実カウンタでは命令数が増え、採用しない。証跡 `artifacts/s1-mul-pairwise-v1/{summary.json,frozen-workflow.zip,integer-bounds.json,build,check}`。診断canisterを停止した。


## LoRAの実重み構造とdense Delta検査の準備

400個の実LoRA F32 tensor、121,896,960要素をpackから読んで検査した。numeric zero（±0を含む）、全ゼロ行・列・4出力vector、および保存F32 bitsが同じ行・列はすべて0件。正確な疎性・重複に基づく計算省略には使えない。各tensorのbytes SHAを保存し、source/manifest/model lock hashを再照合した。証跡 `artifacts/lora-structure-v1/{report.json,frozen-inspection.zip}`。命令数の改善は主張しない。

独立のcorrectness専用module `fda373c3…`を `artifacts/delta-capture-v2`へビルドした。dense Deltaの初期/最終stateとQ/K/V/G/BおよびBF変換前出力を取り出す。one-stage schedulerへ変え、算術kernelとF32演算順序は既存quantization版と同じ。検証済みのprepared prefix登録後は未使用raw packetを保持せず、capture用heap余裕を確保する。検査scheduler/captureの命令数をpaid推論の性能値に混ぜない。全3入力24 Delta layerを独立ordered F32 recurrenceで比較するsnapshot保護試験を開始した。結果と最終復元はまだ未確定。layer30 hiddenの歴史的参照欠落は別に残る。


## dense Delta stateの全実入力検証

correctness専用module `fda373c3…`で617/620/653を同じweight・token・voting38/common27 prefixのまま実行し、各24 Delta layer×32 headの最終dense state 37,748,736値とBF変換前出力を、独立ordered F32 recurrenceに対して全bit比較した。すべて一致した。F32 multiply/subtract/addを別々に実行し、key方向の128項の加算順序を保持する。initial state・Q/K/V/G/Bを保存Candidから取り出し、capture bytesと各返信のSHAを保存した。3入力とも31保存hidden・32conv/KV hash・最終norm・判断/logits/確率も既存参照と一致した。layer30 hiddenには歴史的参照がなく、この欠落は別に残る。

snapshot `000000000000002e7fffffffffa000180101`から元module/cache/packへ復元し、snapshot削除を確認した。全Delta kernelを含む10個のWAT sourceはpaid quantization版と同じ。検査用runtimeの追加はcapture moduleと2箇所のhookのみであり、演算sourceとの一致も別途検査する。証跡 `artifacts/delta-capture-v2/{proof/report.json,proof/restored.json,summary.json,frozen-workflow.zip}`と再照合スクリプト `scripts/report_delta_capture.py`。検査scheduler・capture費用は性能値へ混ぜない。命令数はpaid版の130,665,143,316 / 112,603,637,934 / 132,857,163,355のまま、1000億goalはactiveのまま継続する。


## 1段rank7 Winogradの実測

module `24613312…`は元S1と同じraw weightの配置・容量を保ち、Winogradの4 input node・4 weight nodeと共有reconstructionを使う。output128、locals7456。入力I16上限508、weight I16上限512、全canonical I32中間と元の行列積への一致を独立に検証した。FPのblock順序、量子化値、input/weight scaleは変更しない。odd tokenでも7積を計算する現構成であり、元kernelの4積tail最適化に性能上の利点がある。

21条件42queryで同moduleのS1基準および独立native oracleの出力digestが一致した。保存Candidを再decodeし、counter合計・input hash・source/dependency/workflow hash・全整数範囲を再照合した。48/56/57/67 tokenで命令数が7.4148/6.8508/8.1327/7.4375%増。採用せず全体へ接続しない。証跡 `artifacts/s1-winograd-v1/{summary.json,frozen-workflow.zip,integer-bounds.json,build,check}`。小型診断canisterを停止した。

初期のexport衝突・WAT宣言位置/括弧と、I16 input offsetのbyte換算漏れの失敗証跡は別directoryへ保持した。byte換算漏れ版 `45992f08…`は最初のboundary8でbit不一致となり成功扱いしない。修正版だけが上記全条件を通過した。dense Delta検査のworkflowは追記可能な共有docsへのhash依存を、検査時点のimmutable `progress-note.md`へ置き換え、全72 capture/保存Candid/独立F32 recurrenceを再照合した。1000億goalはactive、paid最大残り32,857,163,355命令のまま。


## 有限値チェックのSIMD64候補

有限値判定はIEEE F32指数255以外と同値である。sign/mantissaを変更せず、v128の指数mask/等値比較をORでまとめ、64要素単位で判定する。残る4要素とscalar tailも同じ条件。module `73c6a3b2…`でscalar/SIMD4/SIMD16/SIMD64を53条件212query比較した。全65536 BF16 patternをSIMD16各vectorへ配置し、signed zero、subnormal、任意F32 bits、Inf/signaling/quiet NaNの全64位置とscalar tail、および0・vector境界長を検証した。全boolが独立整数bit oracleと一致。保存Candid・counter合計・input/source/dependency/workflow hashを再照合した。通常の122880/143360/145920要素の本体はSIMD64で87.7358/87.7378/87.7380%減。入力decodeと返信serializationは両経路で測定本体外。証跡 `artifacts/finite-simd-v1/{summary.json,frozen-workflow.zip,build,check}`。小型診断canisterを停止した。

通常module `8c445b6a…`、paid diagnostics module `bd22be7b…`をビルドした。量子化版のruntimeで純粋なall-is-finite predicate82箇所を同じboolのSIMD64へ置き換え、他の条件やErr文字列は保持する。元の10kernel sourceと一致し、canonical課金fileとschedulerも同一。paid版のdeterministic active/refund upgrade拒否とPending/Done receipt保持の4条件が成功し、元module/cache/packへ復元・snapshot削除を確認した。同moduleで実caller3入力と課金/receipt proofを開始した。全体性能と最終復元はまだ未確定。


## 有限値SIMD64版の全paid API proof完了

module `bd22be7b…`の実caller617/620/653は126,639,551,937 / 109,117,656,487 / 128,772,417,067命令、各4worker、最大heap4,220,911,616 bytes。量子化版から4,025,591,379 / 3,485,981,447 / 4,084,746,288命令減。saved Candidの全forward/inner/paid debug返信を再decodeし、31保存hidden・32conv/KV hash・最終norm・判断/logits/確率F32 bitsを再照合した。82箇所の純粋なfinite predicate以外の算術source・10kernelを維持する。

超過cycles返却・duplicate・ID conflict・insufficient・quote version・token bounds・worker/status authority・cacheありPausedと全額返却・Busyと未課金・同module競合upgradeのCompleted receipt保持・same-version upgrade前後のreceipt/config保持とduplicate replay返却を通過した。固定active/refund fixtureとPending/Done receipt保持4条件も同moduleで照合済み。snapshot `00000000000000307fffffffffa000180101`から元module/cache/packへ復元、snapshot削除とcaller停止を確認した。証跡 `artifacts/paid-finite-simd-v1/{summary.json,frozen-paid-proof.zip,proof/report.json}`。archiveの重複summaryを最新entryへまとめ、全entry名の一意性と保存summary byte一致を確認した。

paid API scopeのcomplete=true、all_targets_met=false。最大残り28,772,417,067命令（約288億）。dense Deltaは前版で独立F32 recurrenceへ全37,748,736値が一致済みだが、layer30 hiddenの歴史的参照欠落はまだ残る。1000億goalを継続する。

欠落参照の再構成へ向け、保存queryのexact MLP carryを6件（3入力×layer26/30）取り出した。元requestのframe checksum・model/pack identityとsource/array hashesを確認した。n=67/59/57、済んだMLP product列は1280。残る7936列を元weightと同じI32 dot/F32 block/LoRA加算順序で完了するため、residual、入力quantized lanes/scales、gate/up LoRA-A、product prefixとscale、unrounded down-A partialを保存した。既存参照のあるlayer26で再構成経路を検証してからlayer30へ適用する予定。まだhidden30を再構成・比較した成功とは扱わない。証跡 `artifacts/terminal-mlp-reference-v1/{report.json,extraction-workflow.zip}`。

## 保存MLP carryからlayer30 hiddenを独立再構成

元packのSHAと全bytes数4,702,451,200を検査した。同じpackのINT8 weight/scale、LoRA F32 weightと保存exact carryを使い、残る7936列をhost側で計算した。各256列のI32 dotは最大4,161,536で溢れず、F32 scale積・block累積・LoRA累積は元の順序を保持する。BF16 RNE、sigmoid/SiLU、productの256列単位量子化を同じ定義で実行した。layer26は3入力の全171,520 / 151,040 / 145,920値が既存historical hiddenとbit一致し、同じ経路でlayer30を再構成した。

layer30も3入力すべてでpaid有限値SIMD64 module `bd22be7b…`の保存debug hidden hashと一致した。voting prefix38ではhistorical prefix27との差11tokenを除き、common prefix27では全suffixを照合する。再構成・抽出source、6carry、weight tensor bytes、MODEL_LOCK/manifest、既存参照と出力hashを再検査し、保存raw Candid paid_debugを再decodeして他31層も再照合した。これで現在のpaid版の全32層hiddenを確認した。layer30は保存carryの独立継続計算による参照であり、historical queryから直接exportされたhiddenではない。

証跡 `artifacts/terminal-mlp-reference-v1/{summary.json,frozen-workflow.zip,reconstruction/report.json}`、再照合 `scripts/report_terminal_mlp_reference.py`。初回full実行の件数guard誤りは失敗証跡を保存して修正し、全6件を再実行・再検査した。今回のhost検査から命令削減を主張しない。paid実測126,639,551,937 / 109,117,656,487 / 128,772,417,067、最大残り28,772,417,067命令で1000億goalはactive。

## shifted-bit unsigned-maxの有限値チェック候補

IEEE F32のraw bitsを1bit左shiftすると符号が消え、unsigned値が0xff000000以上であることと指数255が同値になる。SIMD4 lanesのunsigned maxを64要素単位または全sliceへ累積し、最後にthresholdと比較する。F32値に対する算術は行わない。module `e7949353…`でscalar/SIMD4/SIMD16/SIMD64/max64/maxfullを53条件318query比較し、全boolが独立整数bit oracleと一致した。全65536 BF16 pattern、ランダムF32 bits、全64位置とscalar tailのInf/signaling/quiet NaN、vector境界、signed zero/subnormalを含む。保存Candid・source/dependency/input hash・counter合計も再照合した。

通常の122880/143360/145920要素で、maxfullのチェック本体は既存SIMD64から28.53098/28.53675/28.53736%減、実617 activationでは28.54910%減。証跡 `artifacts/finite-max-v1/{summary.json,frozen-workflow.zip,build,check}`。小型診断canisterは停止した。初回はRust unsigned intrinsic名の指定誤りでcompile失敗し、別directoryに保持して修正した。

通常module `ece83ef5…`、paid module `def127c4…`をビルドした。既存runtimeとの変更はfinite_simd.rsだけであり、predicate82箇所と算術kernelは同じ。課金・scheduler sourceも同じ。paidのdeterministic upgrade guardを開始した。全体命令数・paid proof・最終復元はまだ未確定で、単体比率から全体性能を推定して成功扱いしない。

deterministic active/refund upgrade拒否・Pending/Done receipt保持の4条件が成功し、元module/cache/packへ復元・snapshot削除を確認した。同moduleで実caller617/620/653の全paid API proofを開始し、weight preparation中。最終結果と復元はまだ未確定。

次の候補の構造検査として、paid suffixのtoken ID重複は617の56token中19、620の48中12、653の57中19だった。同じtokenのembedding・norm0は同じ入力行になる。一方、保存exact MLP carryからlayer0/1/5/9/13/17/21/25/29のq lanesとscale bytesを比較した27条件では、全行が異なった。射影の共有を実装する場合は、まず位置・文脈処理前の最初の層を対象にする。まだ計算共有の実装・命令削減は行っていない。証跡 `artifacts/projection-row-reuse-v1/{report.json,frozen-inspection.zip}`。初回は旧reportのop表示でfilterしたためMLP carryを拾わず、別directoryへ保持し、実frame headerのop/encodingで再検査した。

## finite unsigned-max版の全paid API proof完了

module `def127c4…`の実caller617/620/653は126,443,145,807 / 108,947,230,669 / 128,573,078,502命令、各4worker、最大heap4,220,911,616 bytes。前版SIMD64から196,406,130 / 170,425,818 / 199,338,565命令減。全保存Candidを再decodeし、31historical hiddenと保存exact carryから独立再構成したlayer30、32conv/KV hash、最終norm・判断/logits/確率F32 bitsを照合した。全32hiddenが一致する。全算術kernelと他runtime sourceは同じで、finite_simd.rsだけが変わる。

超過cycles返却・duplicate・ID conflict・insufficient・quote version・token bounds・worker/status authority・cacheありPausedと全額返却・Busy未課金・同module競合upgradeのCompleted receipt保持・same-version upgrade前後のreceipt/configとduplicate replay返却が成功した。固定active/refund拒否とPending/Done receipt保持の4条件も成功。snapshot `00000000000000327fffffffffa000180101`から元module/cache/packへ復元し、snapshot削除・caller停止を確認した。証跡 `artifacts/paid-finite-max-v1/{summary.json,frozen-paid-proof.zip,proof/report.json}`。ZIP重複entryを最新へまとめ、全entry名の一意性と保存summary byte一致を確認した。workflow/source hashも再検査した。

paid proof complete=true、all_targets_met=false。最大残り28,573,078,502命令（約286億）。1000億goalはactive。

試験中に外付けdiskの空きが約240MiBまで減った。保存済みDelta検査証跡2.3GBを内蔵diskの 専用の`delta-capture-v2`保存ディレクトリ（実際の配置はローカルの移動記録に保持） へ移し、元のartifact directoryはsymlinkで維持した。2995ファイルを保持し、summaryのworkflow/reference manifest hash13件とファイル数を再検査した。証跡 `artifacts/delta-capture-relocation.json`。データは削除していない。

非使用中のRust incremental cacheも同じ内蔵diskの `rust-debug-incremental` へ移動した。元の `target/debug/incremental` をsymlinkで維持し、66,627ファイルの相対名とbytes数13,644,069,596を前後照合した。証跡 `artifacts/rust-incremental-relocation.json`。再生成は不要で、従来の参照パスを保持する。全proof完了後に行い、進行中のcompileはなかった。

## 最初の層のexact射影行共有候補

通常module `c9b324ba…`、paid module `4a33c354…`をビルドした。Delta0のcaptureで、入力q lanes、10block scaleのF32 bits、QKV/Z両方のrank64 LoRA-A結果のF32 bitsがすべて同じ行だけをまとめる。stable first occurrence順のunique q/QA/ZAを作り、QKV/ZのINT8+LoRA-B射影を一度計算して元のtoken順へscatterする。その後のconv、gating、Delta recurrence、state更新は元の全tokenで行う。gate projectionとLoRA-Aは元の全行計算を保持する。first-layer root以外とrestoreされたcarryには共有を適用せず、wire形式も維持する。

算術WAT kernelは前版と同じ。変更runtimeはdelta_head_continue.rsと追加の行対応helperだけ。行対応componentはn=1..89と全unique cardinalityの4005組、非隣接の重複、各入力planeに単独bit差がある場合、signed zero差、gather/scatterのbit roundtripを検証して通過した。初回はmodule用helperを直接crate rootとしてtest compileしたためpub(super)で失敗し、module wrapperのtest harnessに修正した。証跡 `artifacts/projection-row-plan-v1/{report.json,frozen-tests.zip}`、`artifacts/update-row-reuse-v1/{build/report.json,runtime-comparison.json}`。

paid moduleのdeterministic upgrade guardを開始した。射影の行groupingが変わるため、全3入力のhidden/state/最終出力と実命令数は別途検証が必要。まだ全体性能・bit一致の成功とは扱わない。1000億goalはactive、最後の確定実測は126,443,145,807 / 108,947,230,669 / 128,573,078,502命令。

同module `4a33c354…`のactive/refund upgrade拒否、Pending/Done failed receipt保持4条件が成功し、元module/cache/packへ復元・snapshot削除を確認した。証跡 `artifacts/paid-row-reuse-v1/upgrade-guards/{verified.json,restored.json}`。実callerの全3入力・課金・receipt試験を開始した。結果と試験後の最終復元はまだ未確定。

独立host側でも、元embedding INT8 row/scaleとnorm0 BF16 weightを読み、ordered F32 RMS・BF16 RNE・block256量子化・元LoRA-A/Bのordered積和・I32 dotとF32 block順序で最初のQKV/Z射影を計算した。q/scale/QA/ZAのbit keyで共有・scatterした結果が全行計算と一致した。617/620/653で56→37、48→36、57→38行となり、全688,128 / 589,824 / 700,416値（計1,978,368）がbit一致した。元の全token入力は維持する。

同じq/scale/QA/ZAを実装のRust RowPlanへ渡し、Pythonの独立byte-key対応表とfirst/indexが全3入力で一致することも検査した。証跡 `artifacts/row-reuse-native-v1/{report.json,frozen-tests.zip}` と `artifacts/row-plan-real-v1/{report.json,frozen-tests.zip}`。host計算と実装行対応の検査であり、Wasmの全体性能・bit一致とは区別する。paid proofでは721tensor、4,065,416,192 bytesのweight準備が完了した。実推論結果はまだ未確定。

### 行共有v1の実測と接続先の修正

paid v1 `4a33c354…`は126,443,159,247 / 108,947,244,109 / 128,573,091,942命令で、前版から全3入力とも13,440増え、削減にならなかった。実paid schedulerは `server_delta_bound_input` -> `PreparedDeltaHybrid::run` -> `delta_full_log::evaluate_from_state` を呼ぶ。v1で変更したdelta_head_continueはこの経路で呼ばれず、共有を適用できていなかった。全3入力・課金・receipt試験は完了し、snapshot `00000000000000347fffffffffa000180101`から元module/cache/packへ復元、snapshot削除とcaller停止を確認した。v1を性能改善として採用しない。

v2は前版finite-max runtimeから作り直し、delta_full_logのQKV/Z射影だけを同じRowPlanで共有する。元q/QA/ZAとgate計算は全行のまま、unique qとunique Axを射影してscatterした後のconv・recurrence・state処理も元の全tokenのまま。変更はdelta_full_log.rs、lib.rsのmodule登録と追加projection_row_reuse.rsのみで、delta_head_continueは元に戻る。通常module `a34609a1…`をビルドし、paid版をビルド中。全体性能とbit一致はまだ未検証。

paid v2 `0a9268c9…`をビルドした。同moduleのactive/refund upgrade拒否・Pending/Done failed receipt保持4条件が成功し、元module/cache/packへの復元とsnapshot削除を確認した。実callerの3入力・課金・receipt試験を開始した。証跡 `artifacts/paid-row-reuse-v2/{build/report.json,upgrade-guards/verified.json,proof}`。v1のsummaryは実paid経路に共有が入らなかったことと不採用を明示し、保存Candid・全32hidden・workflow hash・ZIP一意性を再検査した。v2の実測結果と最終復元はまだ未確定。


## 実paid経路の射影共有v2、全paid API proof完了

module `0a9268c9…`の実caller617/620/653は126,125,652,979 / 108,752,704,172 / 128,256,004,778命令、各4worker、最大heap4,220,911,616 bytes。finite unsigned-max版から317,492,828 / 194,526,497 / 317,073,724命令減った。実paid schedulerのbound DeltaHybrid → delta_full_log::evaluate_from_stateで、最初の層のexact q/scale/QA/ZAが同じ行のQKV/Z射影を共有する。scatter後は元の全token順でconv/gates/recurrence/stateを実行する。delta_head_continueに変更はなく、10算術kernelは元版と同じ。

保存Candidを再decodeし、31historical hiddenと保存exact carryから独立再構成したlayer30、32conv/KV hash、最終norm・判断/logits/確率F32 bitsを照合し、全32層hiddenが一致した。超過cycles返却・duplicate・ID conflict・insufficient・quote version・token bounds・worker/status authority・Paused全額返却・Busy未課金・競合upgradeのCompleted receipt・same-version upgradeのreceipt/config保持とreplay返却も成功。固定active/refund拒否とPending/Done receipt保持4条件も同moduleで確認済み。

snapshot `00000000000000367fffffffffa000180101`から元module `6052cc94…`、cache721、packへ復元し、snapshot削除とcaller停止を確認した。証跡 `artifacts/paid-row-reuse-v2/{summary.json,frozen-paid-proof.zip,proof/report.json,proof/restored.json}`。reporterの再照合後にもworkflow hashesとarchive entry名一意性・最新summary bytesを独立監査した。latest dense Delta直接captureは未実施で、現在のbit検証は全hidden/state参照による。paid proof complete=true、all_targets_met=false。最大残り28,256,004,778命令（約283億）で、1000億goalはactive。

## adaptive168のQ射影component候補

従来160幅kernelを168幅へ拡げたmodule `5fbcf48d…`は9767localsでplatform上限10000未満。168/128/32 dispatchを用い、rowsが160の倍数なら元160経路へ戻す。raw weight容量/layout、量子化とF32 block順序を保持する。実Q weight8192×2560と4096行の21条件42queryがnative digestに一致した。保存Candidを再decodeし、全native digestも再計算して、source/dependency/input hashes・counter sums・Wasm validator・locals上限・archiveを再検査した。48/56/57/67tokenの本体削減は0.07836 / 0.07865 / 0.07872 / 0.07895%。証跡 `artifacts/s1-adaptive168-v1/{summary.json,frozen-workflow.zip}`。

Q射影だけの小さい改善であり、全体へ未接続。9216行MLPでは168幅の端数paddingが増えるため、実gate weightの別componentを開始した。MLP性能・全体削減の成功とはまだ扱わない。


9216×2560の実layer3 MLP gate weightについて16条件32queryを完了した。48/56/57/67tokenで0.09138 / 0.08839 / 0.09089 / 0.08781%悪化し、全16条件で元160幅より遅かった。すべてnative digestに一致し、保存Candidの再decode、native digest再計算、source/input/workflow hashes、counter sums、ZIP一意性・summary bytesを監査した。入力activationは保存Q入力のサイズprobeとsynthetic boundaryであり、MLP activationそのものの全推論実測とは扱わない。証跡 `artifacts/s1-adaptive168-mlp-v1/{summary.json,frozen-workflow.zip}`。全サイズへの168幅適用は不採用。今後は改善が確認できた行列サイズだけに限定する。小型診断canisterを停止した。最新確定paid合計は126,125,652,979 / 108,752,704,172 / 128,256,004,778命令で、goalはactive。


## 改善行数に限定したadaptive168全推論候補

通常module `ad41dadc…`、paid diagnostics module `afc52c77…`をビルドした。strassen_raw.rsのprepared INT8射影でrows=4096/8192だけを168/128/32幅に変更し、他の行数は元のadaptive160を保持する。元10kernelのWAT SHAはすべて同一で、追加168kernelのみ増える。F32 block累積・量子化・weight容量とpack layout・scheduler・canonical課金sourceを維持する。11patchのWasm validatorとsource/wasm hashを監査した。

初回通常builderは生成Python内のnewline escapeで構文エラー、初回paid builderは旧10patch固定リストでIndexErrorになった。両失敗directoryを保存し、生成escapeと11patchの出力/件数を修正して再ビルドした。成功moduleだけでdeterministic active/refund upgrade拒否・Pending/Done receipt保持検査を開始した。全paid実測と復元はまだ未確定で、既存最良126,125,652,979 / 108,752,704,172 / 128,256,004,778命令を更新する証拠とはまだ扱わない。


module `afc52c77…`のactive/refund upgrade拒否とPending/Done receipt保持4条件が成功し、元module/cache/packへの復元とsnapshot削除を確認した。実caller617/620/653、全hidden/state/最終出力・課金境界・競合upgrade・receipt/replay検証を開始した。snapshot保存済みで、現在の全体実測と試験後復元はまだ未確定。


## 限定adaptive168、全paid proof完了

paid module `afc52c77…`の実caller617/620/653は126,103,177,759 / 108,733,198,985 / 128,233,276,148命令、各4worker、最大heap4,220,911,616 bytes。前版の実paid射影共有v2から22,475,220 / 19,505,187 / 22,728,630命令減少した。全保存Candidを再decodeし、31historical hiddenと独立保存carry再構成のlayer30、32conv/KV hash・最終norm・判断/logits/確率F32 bitsを再照合して全32層hiddenが一致した。

課金返却・duplicate/ID conflict・insufficient・quote version・token bounds・worker/status authority・Paused全額返却・Busy未課金・競合upgrade Completed receipt・same-version upgrade receipt/configとreplay返却、および固定active/refund拒否・Pending/Done receipt保持4条件が成功した。snapshot `00000000000000387fffffffffa000180101`から元module `6052cc94…`、cache721、packへ復元し、snapshot削除・caller停止を確認した。証跡 `artifacts/paid-adaptive168-v1/{summary.json,frozen-paid-proof.zip,proof/report.json,proof/restored.json}`。reporter後もworkflow hashとZIP entry一意性・最新summary bytesを独立監査した。original10算術kernelは同一で、168幅kernelのみ追加。最新dense Delta直接captureはまだ未実施。paid proof complete=true、all_targets_met=false。最大残り28,233,276,148命令（約282億）でgoalはactive。

追加の削減候補として、保存exact MLP input carryを256列ごとのq lanesだけで比較した。scale bitsが異なってもI32 dotを共有し、元F32 scale・K256累積順序を保持できる可能性を調べた。3入力×9層の27条件、5040 / 4320 / 5130 block-rowのすべてでq lanesが異なり、共有可能dot rowは0だった。完全行比較だけでなく部分block比較でも、この保存MLP inputでは共有による改善は見込めない。frame checksum/model/header・source hashes・ZIPとsummary bytesを再監査した。証跡 `artifacts/projection-block-reuse-v1/{report.json,frozen-inspection.zip}`。検査対象以外の層への一般化や命令削減を主張しない。


## S1出力保存のoperand stack最適化component

新しい広幅案の係数再計算は積ごとの整数加算を増やすため、先に一時出力保存を減らす候補を実装した。168幅kernelの210store siteで、ypを先にWasm operand stackへ残し、末尾のvalue一時local.set/getとypの取り直しを省く。各site2命令減。整数dot・整数再構成・shuffle・F32 scale/block累積のopcode順序は完全同一。9767localsとWasm validatorを確認した。module `2b32543a…`の21条件42queryがnative digestに一致した。48/56/57/67tokenのQ射影本体は0.36750 / 0.36974 / 0.36919 / 0.37132%減り、全境界条件でも削減した。

保存Candidの再decode、全native digest再計算、source/dependency/input hashes、counter合計、算術opcode順序、ZIP entry一意性とsummary bytesを再監査した。証跡 `artifacts/s1-stack-store-v1/{summary.json,frozen-workflow.zip}`。小型診断canisterを停止し、最新通常runtimeへ同じ168幅kernelだけを置き換える全体候補をビルド中。他のwidthとruntime sourceは維持する。全推論命令数と全paid proofは未測定であり、既存最良126,103,177,759 / 108,733,198,985 / 128,233,276,148命令をまだ更新しない。1000億goalはactive。


通常module `63bc8b37…`とpaid diagnostics module `89c91c42…`のビルドが完了した。前版とruntime sourceはすべてbyte一致し、11kernel中168幅のoutput store stack配置だけが変わる。他の10kernelは同一。11patchのWasm validatorとsource hashes、canonical billing sourceの一致を監査した。paid moduleのdeterministic upgrade guard4条件を開始した。全体命令数・paid検証・最終復元はまだ未確定。


paid stack store module `89c91c42…`のactive/refund upgrade拒否とPending/Done receipt保持4条件が成功した。元module/cache/packへの復元とsnapshot削除を確認した。同moduleで実caller3入力、全hidden/state/最終出力とpaid API境界・receipt/replay・競合upgrade・最終復元の検証を開始した。結果はまだ未確定。


## 160/128/32幅への出力stack保存拡張component

先行する168幅だけのpaid full proofがweight準備中の間に、160/128/32幅の200/160/40箇所（計400store site）へ同じ保存最適化を拡げた。元とcandidateを同moduleで比較できるよう3kernelを別exportへcloneし、同じadaptive160とraw weightで比較する。module `be3e21ad…`の21条件42queryがnative digestに一致した。48/56/57/67tokenのQ射影本体は0.37304 / 0.37532 / 0.37475 / 0.37691%減り、全境界条件でも減った。Wasm validator、locals上限（最大9501）、全算術opcode順序一致、保存Candid再decode・native digest再計算、source/dependency/input hashes・counter合計・ZIP一意性/summary bytesを再監査した。証跡 `artifacts/s1-stack-store-all-v1/{summary.json,frozen-workflow.zip}`。

初回builderでは既存post_upgradeを再追加してcompile失敗した。失敗directoryを保持し、追加を省いて再ビルドした。9216×2560 MLP gate weightのcomponentへ切り替えるupgradeは小型診断canisterのcycles不足で拒否された。ローカルの同canisterだけへ1Tcyclesを補充し、upgradeを再実行中。MLP性能や拡張候補の全推論への成功はまだ主張しない。


## 168幅stack保存版、全paid proof完了

paid module `89c91c42…`の実caller617/620/653は126,007,236,319 / 108,650,704,265 / 128,135,600,948命令、各4worker、最大heap4,220,911,616 bytes。前版限定adaptive168から95,941,440 / 82,494,720 / 97,675,200命令減少した。保存Candidを再decodeし、31historical hiddenと保存exact carryの独立再構成layer30、32conv/KV hash、最終norm・判断/logits/確率F32 bitsを照合し、全32層hiddenが一致した。

課金返却・duplicate/ID conflict・insufficient・quote version・token bounds・worker/status authority・Paused全額返却・Busy未課金・競合upgrade Completed receipt・same-version upgrade receipt/configとreplay返却、および固定active/refund拒否とPending/Done receipt保持4条件が成功した。snapshot `000000000000003a7fffffffffa000180101`から元module `6052cc94…`、cache721とpackへ復元し、snapshot削除とcaller停止を確認した。証跡 `artifacts/paid-stack-store-v1/{summary.json,frozen-paid-proof.zip,proof/report.json,proof/restored.json}`。reporter後にworkflow hashesとZIP entry一意性・最新summary bytesを独立監査した。runtime sourceは前版と同一で、168幅kernelの出力保存配置だけを変更する。latest dense Delta直接captureはまだ未実施。paid proof complete=true、all_targets_met=false。最大残り28,135,600,948命令（約281億）でgoalはactive。


160/128/32幅stack保存拡張module `be3e21ad…`の実layer3 MLP gate weight9216×2560について、16条件32queryを完了した。48/56/57/67tokenで0.37341 / 0.37570 / 0.37512 / 0.37729%減り、全境界条件も削減した。全native digestが一致し、保存Candidの再decode・全native再計算、source/dependency/input hashes、counter sums、算術opcode順序・locals上限、ZIP一意性とsummary bytesを監査した。入力はQ-sourceサイズprobeとsynthetic境界であり、MLP activationそのものの全推論実測とは扱わない。証跡 `artifacts/s1-stack-store-all-mlp-v1/{summary.json,frozen-workflow.zip}`。小型診断canisterを停止した。Q/MLP双方で拡張の採用候補を得たが、全推論moduleへの組み込みと全paid proofはまだ未実施。次は最新stack保存版を維持し、残る3widthへ同じ変更を追加する。


## 全INT8幅stack保存拡張の全体候補

通常module `244f414a…`をビルドした。最新168幅stack保存を維持し、160/128/32幅の400store siteにも同じ配置変更を適用する。runtime sourceは前版とすべてbyte一致。11kernel中の変更exportはs1_160、s1_wide、s1_rawだけで、168幅を含む他8kernelは同一。Wasm validator、source/wasm hashesと変更範囲を監査した。Q/MLPの21/16条件の保存native bit検証を依存として凍結している。paid版をビルド中で、実paid命令合計と全体bit一致・課金検証はまだ未測定。最良126,007,236,319 / 108,650,704,265 / 128,135,600,948命令をまだ更新しない。


paid diagnostics module `1503ff52…`もビルドした。canonical課金sourceとschedulerを維持し、11patch/source/wasm hashを監査した。固定patchリストを11kernel・変更128幅のpathへ更新し、実行するbuilder sourceを再保存してからworkflow hashを凍結した。full paid upgrade guard4条件を開始した。結果と最終復元はまだ未確定。


同paid module `1503ff52…`のactive/refund upgrade拒否・Pending/Done receipt保持の4条件が成功した。元module/cache/packへの復元とsnapshot削除を確認した。実caller617/620/653、全hidden/state/最終出力とpaid API境界・競合upgrade・receipt/config/replay・最終復元の全体proofを開始し、snapshot保存済み。全体性能と試験後の復元はまだ未確定で、最新確定最良値は126,007,236,319 / 108,650,704,265 / 128,135,600,948命令のまま。1000億goalはactive。


## 正のゼロ加算を保持した最初のblockによるoutput初期化候補

全幅stack保存版のpaid full proofが実行中の間に、追加のzero-seed componentを作成した。160/128/32幅の一時sumsをMaybeUninit<f32>で確保し、block0専用kernelでは全出力loadを正のゼロconstantに置き換える。元のF32加算は省かず、負のゼロの扱い・scale積・block累積順序を保持する。block0の全lane書き込み後、後続blockは既存stack保存kernelで累積する。全block終了後にだけinitialized Vec<f32>へ変換する。private Preparedのconstructorがcols>0かつ256の倍数を保証し、n>0のbounds検査からblock0が必ず実行される。paddingと奇数tokenのlaneもblock0で初期化する。

module `e8ff919c…`の21条件42queryがnative digestに一致した。48/56/57/67tokenのQ射影はstack保存候補から0.31948 / 0.32126 / 0.32076 / 0.32246%減。全条件で削減し、9patchのWasm validator・算術opcode順序・source/dependency/input hashesとcounter sums、保存Candid再decode・全native再計算・ZIPを監査した。tile32/128/160×n1..132の396address条件で全laneが一度ずつ書かれることを検査した。これはaddress/control構造の監査であり、実Wasm算術bit一致は別のnative比較で確認する。証跡 `artifacts/s1-zero-seed-v1/{summary.json,frozen-workflow.zip,initialization-coverage.json}`。MLP gate9216×2560の別componentも開始し、比較済み条件では約0.32%減。全条件の監査と全推論への組み込みはまだ未完了。最良paid合計は126,007,236,319 / 108,650,704,265 / 128,135,600,948命令のままでgoalはactive。


zero-seed module `e8ff919c…`の実layer3 MLP gate weight9216×2560は16条件32queryを完了し、全native digestが一致した。48/56/57/67tokenで0.31983 / 0.32162 / 0.32111 / 0.32281%減り、全境界条件でも削減した。保存Candidの再decode、全native再計算・source/dependency/input hashes・counter sums・算術opcode順序とlocals上限、396初期化address条件、ZIP entry一意性とsummary bytesを再監査した。入力はQ-sourceサイズprobeとsynthetic境界であり、MLP activationそのものの全推論実測とは扱わない。証跡 `artifacts/s1-zero-seed-mlp-v1/{summary.json,frozen-workflow.zip,initialization-coverage.json}`。小型診断canisterを停止した。全推論への組み込み・全paid proofはまだ未実施。


## 全幅stack保存版、全paid proof完了

paid module `1503ff52…`の実caller617/620/653は125,725,587,999 / 108,409,287,305 / 127,848,922,868命令、各4worker、最大heap4,220,911,616 bytes。前版168幅stack保存から281,648,320 / 241,416,960 / 286,678,080命令減少した。保存Candidの再decodeで31historical hiddenと保存exact carryの独立再構成layer30、32conv/KV hash、最終norm・判断/logits/確率F32 bitsを再照合し、全32層hiddenが一致した。

課金返却・duplicate/ID conflict・insufficient・quote version・token bounds・worker/status authority・Paused全額返却・Busy未課金・競合upgrade Completed receipt・same-version upgrade receipt/configとreplay返却、および固定active/refund拒否とPending/Done receipt保持4条件が成功した。snapshot `000000000000003c7fffffffffa000180101`から元module `6052cc94…`、cache721、packへ復元し、snapshot削除・caller停止を確認した。証跡 `artifacts/paid-stack-store-all-v1/{summary.json,frozen-paid-proof.zip,proof/report.json,proof/restored.json}`。reporter後にもworkflow hashesとZIP entry一意性・最新summary bytesを独立監査した。runtime sourceは前版と同一で、160/128/32幅の出力保存配置だけを変更し、既存168幅stack保存を維持する。latest dense Delta直接captureはまだ未実施。paid proof complete=true、all_targets_met=false。最大残り27,848,922,868命令（約278億）で、1000億goalはactive。次のzero-seed候補はQ/MLPでcomponent検査済みだが、全推論へ未組み込み。


## zero-seedを全推論へ組み込む候補

通常module `3a5ac54e…`をビルドした。strassen_raw.rsの160/128/32幅sumsはMaybeUninitを確保し、block0だけ初期化専用kernelを呼ぶ。positive-zero constantから元と同じF32加算を行い、後続blockは元kernelで累積する。bufferをF32 sliceとして読むのは全block実行後だけ。168幅は従来のzeroed Vecと元kernelを保持する。元11kernelのWAT SHAは同一で、初期化専用3kernelだけを追加し、14patchのWasm validatorとsource hashesを監査した。変更runtimeはstrassen_raw.rsだけ。Q/MLPの21/16条件のbit検証と全396address条件を依存として凍結した。

paid版をビルド中で、全体命令数・全hidden/state/最終出力・課金検証はまだ未測定。既存最良125,725,587,999 / 108,409,287,305 / 127,848,922,868命令を更新する証拠とはまだ扱わない。1000億goalはactive。


paid diagnostics module `f5e57df3…`のビルドも完了した。canonical課金sourceとschedulerを維持し、14patchのWasm validator・source/wasm hashesと凍結builderの14kernel一覧を監査した。deterministic active/refund upgrade拒否とPending/Done receipt保持4条件を開始した。全体実測と試験後復元はまだ未確定。


同paid zero-seed module `f5e57df3…`のactive/refund upgrade拒否とPending/Done receipt保持4条件が成功し、元module/cache/packへ復元・snapshot削除を確認した。実caller3入力の全hidden/state/最終出力・paid境界・競合upgrade・receipt/config/replayと最終復元proofを開始した。snapshot保存済みだが、最新全体命令合計と試験後復元はまだ未確定。既存最良125,725,587,999 / 108,409,287,305 / 127,848,922,868命令から1000億goalを継続する。


## 最終outputへ直接書き込むcomponent候補

先行zero-seed paid full proofの実行中に、次のdirect-output componentを作成した。160/128/32幅のkernelでinput scale strideをcols/256から導出し、既存stride引数をoutput行strideへ使う。第二tokenのyp1も出力行strideで導出する。block0のpositive-zero加算と後続blockの全整数/F32演算順序を保持し、tileごとのsums確保と最終outへのcopyを省く。最終out自体はzeroed Vecを保持する。rowsは32の倍数であり、adaptive tileが残り行数を超えないことをassertしてから直接書く。

module `3d3ed3e7…`のQ射影21条件42queryは全native digestに一致した。48/56/57/67tokenでzero-seed componentから0.38698 / 0.38797 / 0.38768 / 0.38534%減。15patchのWasm validator・最大9504locals・全算術opcode順序、保存Candid再decode・全native再計算、source/dependency/input hashes・counter sums、動的output strideのaddress intervalが全出力を一度ずつ覆うこと、ZIP一意性・summary bytesを再監査した。証跡 `artifacts/s1-direct-output-v1/{summary.json,frozen-workflow.zip,output-address-coverage.json}`。

初回builderは128/32幅のypがshift生成であることを見落としてassert失敗し、失敗directoryを保存して対応した。初回reporterはaddress検査のrがreport変数を上書きして失敗し、初回source/partial summaryを保存してblock_startに改名し、再実行・再監査した。MLP gate9216×2560のcomponentも開始し、比較済み条件では約0.39%減。全推論への組み込みはまだ未実施で、component性能から全体性能を主張しない。


## zero-seed版、全paid proof完了

paid module `f5e57df3…`の実caller617/620/653は125,546,244,654 / 108,255,506,248 / 127,666,383,523命令、各4worker、最大heap4,220,911,616 bytes。全幅stack保存版から179,343,345 / 153,781,057 / 182,539,345命令減少した。保存Candidを再decodeし、31historical hiddenと保存exact carryの独立再構成layer30、32conv/KV hash、最終norm・判断/logits/確率F32 bitsを再照合し、全32層hiddenが一致した。

課金返却・duplicate/ID conflict・insufficient・quote version・token bounds・worker/status authority・Paused全額返却・Busy未課金・競合upgrade Completed receipt・same-version upgrade receipt/configとreplay返却、および固定active/refund拒否とPending/Done receipt保持4条件が成功した。snapshot `000000000000003e7fffffffffa000180101`から元module `6052cc94…`、cache721、packへ復元し、snapshot削除・caller停止を確認した。証跡 `artifacts/paid-zero-seed-v1/{summary.json,frozen-paid-proof.zip,proof/report.json,proof/restored.json}`。reporter後にもworkflow hashesとZIP entry一意性・最新summary bytesを独立監査した。元11kernelは同一で、160/128/32幅のblock0初期化専用3kernelを追加する。168幅と後続blockは従来どおり。latest dense Delta直接captureはまだ未実施。paid proof complete=true、all_targets_met=false。最大残り27,666,383,523命令（約277億）で、1000億goalはactive。direct-output候補はQでcomponent検査済み、MLPは検査中、全推論へ未組み込み。


direct-output module `3d3ed3e7…`の実layer3 MLP gate weight9216×2560は16条件32queryを完了し、全native digestが一致した。48/56/57/67tokenでzero-seed componentから0.38906 / 0.39011 / 0.38919 / 0.38783%減り、全境界条件でも削減した。保存Candid再decode・全native再計算、source/dependency/input hashes・counter sums、算術opcode順序・locals上限、1394output address interval条件、ZIP一意性・summary bytesを再監査した。入力はQ-sourceサイズprobeとsynthetic境界であり、MLP activationそのものの全推論実測とは扱わない。証跡 `artifacts/s1-direct-output-mlp-v1/{summary.json,frozen-workflow.zip,output-address-coverage.json}`。小型診断canisterを停止した。全推論への組み込みと全paid proofはまだ未実施。最新最良125,546,244,654 / 108,255,506,248 / 127,666,383,523命令から1000億goalを継続する。


## direct-outputを全推論へ組み込む候補

通常module `5d2f20a6…`をビルドした。strassen_raw.rsでtile160/128/32かつ完全・4行整列・rows32倍数のviewだけに直接出力する。最終outは既存のzeroed Vecを保持し、block0はpositive-zero加算の初期化kernel、後続blockは動的output strideで元と同じ累積を行う。168幅・partial/unaligned viewは既存zero-seed経路を保持する。元14kernelのWAT SHAはすべて同一で、直接出力用6kernelのみ追加し、20patchのWasm validatorとsource/wasm hashを監査した。変更runtimeはstrassen_raw.rsだけ。Q/MLPの21/16条件のbit検査と1394address interval条件を依存として凍結した。

初回full builderは生成scriptのnewline escapeで構文エラーになった。失敗directoryを保存し、挿入patchを明示的なPython文字列として凍結して再ビルドした。成功通常moduleだけをpaid builderの依存に使用する。paid版をビルド中で、全体実測とpaid proofはまだ未測定。既存最良125,546,244,654 / 108,255,506,248 / 127,666,383,523命令を更新する証拠とはまだ扱わない。


paid diagnostics module `b62dbcf7…`のビルドも完了した。canonical課金sourceとschedulerを維持し、20patchのWasm validator・source/wasm hashesと凍結builderの20kernel一覧を監査した。deterministic active/refund upgrade拒否とPending/Done receipt保持4条件を開始した。全体実測と試験後復元はまだ未確定。


同paid direct-output module `b62dbcf7…`のactive/refund upgrade拒否とPending/Done receipt保持4条件が成功し、元module/cache/packへ復元・snapshot削除を確認した。実caller3入力の全hidden/state/最終出力・paid境界・競合upgrade・receipt/config/replayと最終復元proofを開始した。snapshot保存済みだが、最新全体命令合計と試験後復元はまだ未確定。既存最良125,546,244,654 / 108,255,506,248 / 127,666,383,523命令から1000億goalを継続する。


## 168幅のdirect-output component候補

通常direct-output paid full proof実行中に、既存168 stack-store componentを基準に、完全168幅だけblock0 positive-zero加算と最終output行strideへの直接書き込みを行う候補を作成した。128/32 tailは同じ旧経路のまま。module `0edc3fd6…`、7patchのfunction indexが一意、Wasm validator成功、direct2kernel各9770locals、元168kernelとの整数/F32算術opcode列一致、source/dependency hashesを確認した。21条件42queryでnative digest比較を実行中。全推論への効果は未測定。全体direct-output版の617/620/653命令値は125,337,949,778 / 108,073,979,146 / 127,447,276,173、保存参照・paid境界・復元検証は継続中。


## direct-output版、全paid proof完了

paid module `b62dbcf7…`は617/620/653で125,337,949,778 / 108,073,979,146 / 127,447,276,173命令、各4worker、最大heap4,220,911,616 bytes。zero-seed版から208,294,876 / 181,527,102 / 219,107,350命令減少した。保存Candid再decode、全32hidden（layer30は保存exact carryの独立再構成参照）、32conv/KV hash、最終norm・判断/logits/確率F32 bitsが一致した。paid境界・返却・duplicate/conflict・Busy・競合upgrade・receipt/config/replayも成功した。snapshot `00000000000000407fffffffffa000180101`から元module/cache721/packへ復元し、snapshot削除とcaller停止を確認した。証跡 `artifacts/paid-direct-output-v1/{summary.json,frozen-paid-proof.zip,proof/restored.json}`。reporter後のworkflow hashes・ZIP一意性・最新summary bytes・復元flagsを独立監査した。latest dense Delta直接captureは未実施。最大残り27,447,276,173命令（約274億）、1000億goalはactive。


168幅direct component `0edc3fd6…`の21条件42queryはnative bit一致、全境界でも削減した。48/56/57/67tokenで0.68565 / 0.69223 / 0.68820 / 0.69085%減。保存Candid再decode・全native再計算、source/dependency/input hashes・counter sums、元168の算術opcode順序、9770locals・7patch一意性、4096/8192行×tokenの241address interval条件、ZIP一意性・最新summary bytesを再監査した。証跡 `artifacts/s1-direct168-v1/{summary.json,frozen-workflow.zip,output-address-coverage.json}`。小型診断canisterを停止した。直接出力を完全整列168幅へ拡張する通常moduleのビルドを開始した。partial/unaligned viewは前経路、160/128/32は直前direct-outputを維持する。全体効果は未測定。


168 direct通常module `326548f6…`のビルドが完了した。元20kernelのWAT source SHAを維持し、168 direct/seed2kernelを追加した22patchのWasm validator・function index一意性とsource/wasm hashesを監査した。変更runtimeはstrassen_raw.rsだけ。paid moduleをビルド中。全体命令合計は未測定。


paid diagnostics module `43dcc12d…`もビルド完了した。22kernel WAT SHAが通常版と一致し、全patchのWasm validator・一意function index・source/wasm hashesを確認した。canonical課金source/schedulerを維持する。active/refund upgrade拒否とPending/Done receipt保持4条件を実行中。試験後復元と全体実測は未確定。


同paid168 direct module `43dcc12d…`のactive/refund upgrade拒否とPending/Done receipt保持4条件が成功し、元module/cache/packへ復元・snapshot削除を確認した。実caller3入力の全hidden/state/最終出力・paid境界・競合upgrade・receipt/config/replayと最終復元proofを開始した。snapshot保存とcandidate/caller install完了。caller `qgvly-63777-77775-aabma-cai` / `qbunm-td777-77775-aabmq-cai`。現行proofは進行中で、最新全体命令合計と試験後復元は未確定。既存最良125,337,949,778 / 108,073,979,146 / 127,447,276,173命令から1000億goalを継続する。次候補として、直接書き込みが全outputを初期化する場合の最終Vec zero-fill省略が考えられる。MaybeUninitとraw pointer copyのみで未初期化F32への参照を作らず、全tile/token coverage後にVecへ変換する必要がある。未実装・未測定。


## 最終output zero-fillを省くcomponent候補

168 directを基準に、出力allocationをVec<MaybeUninit<f32>>にし、完全168幅の初回kernel writesと128/32 tailの初期化済みsumsからのraw pointer copyで全outputを書いた後だけVec<f32>に変換する候補を作成した。未初期化F32 slice/referenceは作らず、cols>=256・K256倍数を明示検査する。module `539544ac…`の21条件42queryは全native digestに一致した。48/56/57/67tokenで0.30214 / 0.30400 / 0.30353 / 0.30529%減。7kernel WAT SHAは168 direct基準と同一。保存Candid再decodeとnative再計算のreporter実行中。

同方法を160/128/32 direct componentにも適用し、module `d92da6af…`をビルドした。15kernelは既存directと同一、全patchのvalidatorとfunction index一意性を確認した。全推論へは未組み込み。168 direct paid full proofは重み準備中でsession13557、最大確認済み値は127,447,276,173命令。


168 output-uninit module `539544ac…`の保存Candid再decode・全native再計算、source/dependency/input hashes・counter sums、241output address interval条件、ZIP一意性・最新summary bytesを再監査した。証跡 `artifacts/s1-output-uninit-v1/{summary.json,frozen-workflow.zip,output-address-coverage.json}`。160/128/32 output-uninit module `d92da6af…`はQ射影21条件42queryで比較中。実MLP gate9216×2560の16条件32query用checker/reporterも準備した。MLP入力はQ-source size probe/syntheticであり、本番MLP activationそのものとは扱わない。


168 direct paid full proofは617=125,163,078,896 / 620=107,919,153,857命令、各4worker、heap4,220,911,616 bytesを測定した。653と参照/paid/復元検証は進行中。output-uninit通常full builder `scripts/build_update_output_uninit.py`を用意したが、3componentのnative再検査が揃うまで実行しない。整列rows32/K256・非emptyのviewだけoutputのzero-fillを省略し、fallbackは初期化済みsumsからraw pointer copy、全output完成後にF32 Vecへ変換する。nativeは旧zeroed Vec、その他viewはzeroed MaybeUninitを維持する。全体効果は未測定。


## 168 direct-output版、全paid proof完了

paid module `43dcc12d…`は617/620/653で125,163,078,896 / 107,919,153,857 / 127,276,198,511命令、各4worker、最大heap4,220,911,616 bytes。前direct-output版から174,870,882 / 154,825,289 / 171,077,662命令減少した。保存Candid再decode、全32hidden（layer30は保存exact carryの独立再構成参照）、32conv/KV hash、最終norm・判断/logits/確率F32 bitsが一致した。paid境界・返却・duplicate/conflict・Busy・競合upgrade・receipt/config/replayも成功した。snapshot `00000000000000427fffffffffa000180101`から元module/cache721/packへ復元し、snapshot削除とcaller停止を確認した。証跡 `artifacts/paid-direct168-v1/{summary.json,frozen-paid-proof.zip,proof/restored.json}`。reporter後のworkflow hashes・ZIP一意性・最新summary bytes・復元flagsを独立監査した。latest dense Delta直接captureは未実施。最大残り27,276,198,511命令（約273億）、1000億goalはactive。output-uninit160/128/32 componentはQ21条件42queryでnative bit一致、reporter実行中。実gate9216×2560の16条件32queryも開始した。


160/128/32 output-uninit module `d92da6af…`のQ21条件42query、保存Candid再decode・全native再計算を完了した。48/56/57/67tokenで0.30168 / 0.30354 / 0.30307 / 0.30483%減。15kernel WAT SHAは既存direct-outputと同一、source/dependency/input hashes・counter sums、1394output address interval条件、ZIP一意性・最新summary bytesを再監査した。証跡 `artifacts/s1-output-uninit-wide-v1/{summary.json,frozen-workflow.zip,output-address-coverage.json}`。MLP16条件32queryはsession25522で実行中。全推論へ未組み込み。


output-uninit wide module `d92da6af…`のgate9216×2560 MLP16条件32query、保存Candid再decode・全native再計算を完了した。48/56/57/67tokenで0.30198 / 0.30385 / 0.30338 / 0.30514%減。15kernel WAT SHAは基準direct-outputと同一、source/dependency/input hashes・counter sums、1394output address interval条件、ZIP一意性・最新summary bytesを再監査した。証跡 `artifacts/s1-output-uninit-wide-mlp-v1/{summary.json,frozen-workflow.zip,output-address-coverage.json}`。小型診断canisterは停止済み。Q-source size probe/synthetic入力としての検査であり、本番MLP activationそのものとは扱わない。3componentを依存に、output-uninit通常full moduleのビルドを開始する。


output-uninit通常module `959f8ff8…`とpaid diagnostics module `04e285b7…`がビルド完了した。元22kernel WAT SHA維持、全patch validator・一意function index、source/wasm hashesを監査し、変更runtimeはstrassen_raw.rsだけ。paid active/refund upgrade拒否とPending/Done receipt保持4条件が成功し、元module/cache/packへ復元・snapshot削除を確認した。全paid proofを開始する。メモリ拡張余地の調査として全449 F32 tensorsの122,556,160valuesのうち、BF16へbit損失なく格納できる値は2850しかなかった。F32 weights全体の単純なlossless BF16化でメモリを空ける案は成立しない。未変更。


## 未適用だったpure finite scansのSIMD候補

output-uninit full proof実行中に、残っていたpure finiteのany6箇所とchain/slice13箇所を、既存の独立検証済みfinite SIMD unsigned-max predicateへ置き換える候補を作成した。chainは同じ左から右の短絡ANDで各sliceを検査し、演算結果やgate/scale等の追加条件は変更しない。通常module `67a04ecc…`、11runtime filesだけ変更、元22kernel WAT SHA維持、全patch validator・一意function index・source/wasm hashesを監査した。初回ビルドはdelta_logのVec statesに&を付け忘れてcompile失敗し、失敗directoryを保存して修正した。paid候補をビルド中。全体命令合計は未測定。output-uninit module `04e285b7…`のfull proofはsession69924で実行中。


finite-complete paid diagnostics module `3b0ba294…`もビルド完了した。通常版と元22kernel WAT SHA一致、全patch validator・一意function index・source/wasm hashesを監査した。全paid検証canisterを共有するため、先行output-uninit proofが復元を完了するまでupgrade guardを開始しない。F32 capacity調査を保存scriptで再実行し、full-int8.pack全SHAとMODEL_LOCK・manifest hashを確認した。449 tensorsの122,556,160values中2850だけがexact BF16、全体をlossless BF16格納できる24tensorの削減容量は1536 bytesに過ぎない。証跡 `artifacts/lora-exact-bf16-capacity-v1/verified.json`と`scripts/analyze_f32_exact_bf16_capacity.py`。候補精度やweight storageは変更していない。


output-uninit paid全推論の617/620/653は124,917,875,371 / 107,708,766,332 / 127,026,599,978命令、各4worker、heap4,220,911,616 bytesを測定した。前168 direct版から245,203,525 / 210,387,525 / 249,598,533命令減。先行paid proofは競合upgrade ordering/Completed receiptとBusy未課金を確認し、最終復元中。保存Candid再decodeとlayer30独立再構成参照を含む全32hiddenのreporterはまだ未実行。finite-complete module `3b0ba294…`は未インストール。


## output-uninit版、全paid proof完了

paid module `04e285b7…`は617/620/653で124,917,875,371 / 107,708,766,332 / 127,026,599,978命令、各4worker、最大heap4,220,911,616 bytes。保存Candid再decode、全32hidden（layer30は保存exact carryの独立再構成参照）、32conv/KV hash、最終norm・判断/logits/確率F32 bitsが一致した。paid境界・返却・duplicate/conflict・Busy・競合upgrade・receipt/config/replayも成功した。snapshot `00000000000000447fffffffffa000180101`から元module/cache721/packへ復元し、snapshot削除とcaller停止を確認した。証跡 `artifacts/paid-output-uninit-v1/{summary.json,frozen-paid-proof.zip,proof/restored.json}`。reporter後のworkflow hashes・ZIP一意性・最新summary bytes・復元flagsを独立監査した。latest dense Delta直接captureは未実施。最大残り27,026,599,978命令（約270億）、1000億goalはactive。finite-complete paid candidateのupgrade guardsを開始した。全runtime fileを独立比較し、11fileの指定したpure finite substitutions以外がbyte同一であることも確認した。証跡 `artifacts/update-finite-complete-v1/source-substitution-audit.json`。


同paid finite-complete module `3b0ba294…`のactive/refund upgrade拒否とPending/Done receipt保持4条件が成功し、元module/cache/packへ復元・snapshot削除を確認した。実caller3入力の全hidden/state/最終出力・paid境界・競合upgrade・receipt/config/replayと最終復元proofを開始する。最新全体命令合計は未測定。既存最良124,917,875,371 / 107,708,766,332 / 127,026,599,978命令から1000億goalを継続する。


finite-complete full proofの初回は、ローカルcycles ledger残高が1.99Tとなり8T caller作成で停止した。元module/cache/packへの復元・snapshot `00000000000000467fffffffffa000180101`削除を確認した。失敗proof directoryを保存し、明示network localのcycles mintで100Tを補充した（残高101.99T）。mainnetや実資産は使用していない。同じ候補・入力でproofを再実行する。


## Positive finite predicate component候補

量子化scales等のpure positive finite predicateを、unsigned IEEE bitsの範囲1..0x7f800000で表す候補を作成した。bits-1をwrap subtractionし、unsigned SIMD最大が0x7f7fffff未満なら全要素positive finite。zero/-zero/negative/Inf/NaNは閾値以上になる。scalar tailは元のfinite&&>0判定。module `ce837581…`でscalar・64要素early・fullmax3方式を比較する53条件159queryを開始した。65536 BF16 pattern全ての4vector位置、ゼロ/負数/Inf/NaNの全64/tail位置、random F32 bits、tail0..257・実activation・大きなpositive配列を含む。input oracleはunsigned区間、reporterは独立floating predicateで再照合する。初回component buildはfrozen-builder source保存が不足していたので、初回buildを保存し、source/hash凍結を追加して再ビルドした。module hashは同一。全推論へは未組み込み。


positive-finite component `ce837581…`の53条件159queryは全3方式でindependent unsigned範囲oracleに一致し、保存Candidを再decodeして独立floating finite&&>0 oracleにも一致した。全source/dependency/input hash、counter sums、ZIP一意性と最新summary bytesを再監査した。正の122880/143360/145920/393216要素ではearly64方式で94.54〜94.55%、full方式で94.71〜94.72%減。負数を含むreal activationはscalarが最初で拒否するためfull方式が大幅に遅い。量子化scaleの正常入力は全positiveだが、不正値の早期拒否も保つearly64方式を通常full候補へ選ぶ。14pure predicate箇所（うち1はnative unit-test）を置換するビルドを開始した。元のquantization・floating arithmetic・22kernelは維持する。component証跡 `artifacts/positive-finite-v1/{summary.json,frozen-workflow.zip}`。小型診断canisterを停止した。finite-complete full proofはローカルcycles補充後のsession71480で重み準備を完了し、prefix準備中。


positive-finite通常module `a231c1dd…`とpaid diagnostics module `5a79bbf8…`のビルドが完了した。元22kernelのWAT SHA維持、全patch validator・function index一意性・source/wasm hashesを監査した。14predicate箇所（native unit-test1を含む）だけを置換し、helperはcomponentで検査したearly64方式を選ぶ。全体効果は未測定。finite-complete全推論試験の617は122,869,817,980命令、4worker、heap4,220,911,616 bytes。output-uninit版から2,048,057,391命令（約20.48億）減り、残る2入力と参照/paid/復元検証は進行中。


finite-complete paid全推論の617/620/653は122,869,817,980 / 105,951,352,188 / 124,947,601,199命令、各4worker、heap4,220,911,616 bytesを測定した。前output-uninit版から2,048,057,391 / 1,757,414,144 / 2,078,998,779命令（約20.48 /17.57 /20.79億）減。保存参照・paid境界・最終復元proofは進行中で、reporterは未実行。次のpositive-finite版は通常/paidビルドとcomponent監査済みだが未インストール。次候補として、BF16 codecのall/classify/pack/unpackが現状4/8要素単位のSIMD loopなので、演算順序を変えず64要素単位にunrollしてloop overheadを減らせる可能性がある。classifyのinvalid activation errorとnonBF16 finiteのOk(false)を必ず維持する。未実装・未測定。


## finite-complete版、全paid proof完了

paid module `3b0ba294…`は617/620/653で122,869,817,980 / 105,951,352,188 / 124,947,601,199命令、各4worker、最大heap4,220,911,616 bytes。保存Candid再decode、全32hidden（layer30は保存exact carryの独立再構成参照）、32conv/KV hash、最終norm・判断/logits/確率F32 bitsが一致した。paid境界・返却・duplicate/conflict・Busy・競合upgrade・receipt/config/replayも成功した。snapshot `00000000000000477fffffffffa000180101`から元module/cache721/packへ復元し、snapshot削除とcaller停止を確認した。証跡 `artifacts/paid-finite-complete-v1/{summary.json,frozen-paid-proof.zip,proof/restored.json}`。reporter後のworkflow hashes・ZIP一意性・最新summary bytes・復元flagsを独立監査した。latest dense Delta直接captureは未実施。最大残り24,947,601,199命令（約249億）、1000億goalはactive。positive-finite paid candidateのupgrade guardsを実行中。全runtimeを独立比較し、14predicate substitutionsとhelper/module/export追加以外がbyte同一であることも確認した。証跡 `artifacts/update-positive-finite-v1/source-substitution-audit.json`。


同paid positive-finite module `5a79bbf8…`のactive/refund upgrade拒否とPending/Done receipt保持4条件が成功し、元module/cache/packへ復元・snapshot削除を確認した。実caller3入力の全hidden/state/最終出力・paid境界・競合upgrade・receipt/config/replayと最終復元proofを開始する。最新全体命令合計は未測定。既存最良122,869,817,980 / 105,951,352,188 / 124,947,601,199命令から1000億goalを継続する。


## BF16 pure predicateの64要素loop候補

既存inference-core BF16 all/classifyのSIMD4 bodyをcontrolとして、同じlow-bit ORとabsolute unsigned maximumを64要素単位にunrollするcomponentを作成した。classifyはnonfiniteがあれば常に同じinvalid activation error、finiteならBF16のtrue/falseを返す。pack/unpackはこの候補では変更しない。module `31ab3123…`を小型診断canisterに入れ、53条件212queryを開始した。all4/all64、classify4/classify64の4方式、全65536 BF16 patterns・randomF32・NaN/Inf全64/tail位置・tail/実activation/大きな配列を含む。独立bit oracleのbool/tri-stateと比較する。positive-finite full paid proofはsession86933で重み準備中。


BF16 predicate module `31ab3123…`の53条件212queryは元SIMD4と新SIMD64のall/classify全方式が一致した。保存Candid再decodeで独立low-bit/floating-finite oracleにも一致し、invalid activation errorをprobeで確認した。controlのall4/classify4/finish bodyはinference-core原本とwhitespace正規化後に完全同一。source/dependency/input hashes・counter sums・ZIP一意性と最新summary bytesを再監査した。122880/143360/145920/393216要素でall判定28.07〜28.11%、classify37.47〜37.49%減、real activationでも28.09/37.48%減。小型診断canisterを停止した。証跡 `artifacts/bf16-predicates-v1/{summary.json,frozen-workflow.zip,control-source-audit.json}`。Runtime wrapperのall_bf16/classify_finiteだけをSIMD64 helperへ向け、pack/unpackとnative pathを維持する通常full候補をビルドする。


positive-finite paid全推論の617/620/653は122,869,805,345 / 105,951,339,803 / 124,947,588,189命令、各4worker、heap4,220,911,616 bytesを測定した。前finite-complete版から12,635 /12,385 /13,010命令減。componentの94.5%減から全体への大幅効果は出ず、実handlerでの検査回数・scale長は小さい。保存参照・paid境界・最終復元proofは進行中でreporter未実行。BF16 predicate通常module `6de1ca4c…`は元22kernel WAT SHA維持、全patch validator・一意function index・source/wasm hashesを監査し、変更runtimeはbf16_codec.rs/lib.rsと追加bf16_predicates.rsだけ。paid版ビルド中。全体効果は未測定。


## positive-finite版、全paid proof完了

paid module `5a79bbf8…`は617/620/653で122,869,805,345 / 105,951,339,803 / 124,947,588,189命令、各4worker、最大heap4,220,911,616 bytes。保存Candid再decode、全32hidden（layer30は保存exact carryの独立再構成参照）、32conv/KV hash、最終norm・判断/logits/確率F32 bitsが一致した。paid境界・返却・duplicate/conflict・Busy・競合upgrade・receipt/config/replayも成功した。snapshot `00000000000000497fffffffffa000180101`から元module/cache721/packへ復元し、snapshot削除とcaller停止を確認した。証跡 `artifacts/paid-positive-finite-v1/{summary.json,frozen-paid-proof.zip,proof/restored.json}`。reporter後のworkflow hashes・ZIP一意性・最新summary bytes・復元flagsを独立監査した。latest dense Delta直接captureは未実施。最大残り24,947,588,189命令（約249億）、1000億goalはactive。

BF16 predicate paid diagnostics module `22413c80…`もビルド完了し、通常版と元22kernel WAT SHA一致、全patch validator・一意function index・source/wasm hashesを監査した。既存runtimeの変更はbf16_codec.rs/lib.rsのwrapper/module追加だけで、helperはcomponentとbyte同一。paid upgrade guardsを実行中。全体効果は未測定。次候補としてMLP carryのF32値を1要素ずつto_le_bytes/Vec.extendする箇所が残っており、Wasmのlittle endianを前提に安全に借用したF32 sliceをまとめてbyte appendできる可能性がある。owned Vecのdrop timingとerror/byte表現を維持する必要がある。未実装・未測定。


同paid BF16 predicate module `22413c80…`のactive/refund upgrade拒否とPending/Done receipt保持4条件が成功し、元module/cache/packへ復元・snapshot削除を確認した。実caller3入力の全hidden/state/最終出力・paid境界・競合upgrade・receipt/config/replayと最終復元proofを開始する。最新全体命令合計は未測定。既存最良122,869,805,345 / 105,951,339,803 / 124,947,588,189命令から1000億goalを継続する。


## F32 exact byte append component候補

MLP carry encoderにscalar F32 to_le_bytes/Vec.extendが残るため、Wasmのlittle endianとinitialized F32 representationを一括U8 sliceとしてappendするhelperを作成した。source/destinationは通常Rust借用で独立、F32にはpaddingがない。nativeは元の明示to_le_bytesを維持する。module `cfdecb7d…`でscalar extend・scalar extend_from_slice・bulk appendの3方式を53条件159queryで比較開始した。先頭のbyte headerを0..16byteに変え、unaligned outputも含む。全65536 BF16 patterns、全Inf/NaN符号/payload、random F32 bits、tail0..257・real activation・大きな配列を対象に、prefix+original little endian bytesのhost SHAと照合する。digest/Candid encodeは全方式同じくappend bodyの計測外。全推論へは未組み込み。BF16 predicate full proofはsession34943で重み準備中。


F32 byte append module `cfdecb7d…`の53条件159queryはscalar extend/scalar extend_from_slice/bulkの全方式が独立host SHAに一致した。保存Candid再decode・counter sums・source/dependency/input hashes・ZIP一意性と最新summary bytesを再監査した。122880/143360/145920/393216値でbulk append body84.51 /84.63 /84.68 /84.73%減、real activation84.63%減。先頭header0..16byteでunaligned outputと全IEEE edge/NaN payloadも含む。小型診断canisterを停止した。証跡 `artifacts/f32-byte-append-v1/{summary.json,frozen-workflow.zip}`。通常full候補では7 runtime filesの14scalar append loopsを20borrowed bulk copiesへ置換し、フィールド順とowned ax/scales/down Vecの明示dropを保つ。quantization・BF16 pack・validation・22演算kernelは維持する。通常full buildを開始する。


## BF16 predicate版、全paid proof完了

paid module `22413c80…`は617/620/653で122,867,538,329 /105,949,349,267 /124,945,286,613命令、各4worker、heap4,220,911,616 bytes。全32hidden（layer30は独立保存carry再構成）・32conv/KV・最終norm/判断/logits/確率bitsとpaid境界・競合upgrade・receipt/config/replayが一致した。元module/cache721/packへsnapshot004bから復元、snapshot削除とcaller停止を確認した。reporter後workflow hashes/ZIP一意性/最新summary bytes/復元flagsを独立監査した。最大残り24,945,286,613命令、goal active。latest dense Delta直接captureは未実施。

F32 byte append通常full module `0068aaab…`のsource hashes・Wasm hash・22patch validatorと一意function index・全22kernel source SHA維持を監査した。全runtimeを独立比較し、明示14scalar loops→20borrowed append calls、lib helper module/export、component byte同一helperだけの変更を確認した。証跡 `artifacts/update-f32-byte-append-v1/source-substitution-audit.json`。paid diagnostics版をビルド中。


F32 byte append paid module `ec7dffc5…`も22kernel source SHA・validator・一意function index・source/dependency/workflow/Wasm hashesを監査した。active/refund upgrade拒否とPending/Done receipt保持の4条件が成功し、元module/cache/packへ復元・snapshot削除を確認した。全3入力paid proofを開始した（session92963）、全体効果は未測定。逆方向のraw F32 byte decodeを一括コピーする小型component `17b1a845…`をビルドした。新Vecのraw pointerへbyte copyを完了してからset_lenし、未初期化F32参照を作らない。Wasm LE/all IEEE bit patterns/unaligned source header/trailing incomplete bytesを独立host hashで確認する予定。まだ未検証。


F32 byte decode component `17b1a845…`の53条件159queryはscalar collect/scalar push/bulk raw copyが全IEEE bits・全65536 BF16 patterns・random bits・tail・real activationで独立host SHAに一致した。source prefix0..16byteと不完全tail0..3byteを含み、chunks_exactと同じ不完全tail無視を確認した。保存Candid再decode/counter sums/source-dependency-input hashes/ZIP一意性/最新summary bytesを独立監査した。大配列のbulk decode bodyは元collectに対し58.55〜58.95%減、real58.77%減。小型診断canisterを停止した。通常fullでは6runtime filesの9scalar collect decoderをnew-Vec raw copyに置換し、shape/finiteチェックを保持する候補をビルド中（session98810）。全推論性能は未測定。


F32 byte decode通常full module `862c8548…`はビルド完了。全runtime独立比較で9scalar collect置換とhelper/module/export追加以外がbyte同一。component helperもbyte同一、source/dependency/workflow/Wasm hashes、元22kernel source SHA・patch validator・function index一意性を監査した。証跡 `artifacts/update-f32-byte-decode-v1/source-substitution-audit.json`。paid diagnostics版をビルド中（session83488）。先の記載session98810/14541はbuilder生成時のassert/parent pathの失敗でbuild開始前に終了、修正後session74458が成功した。失敗時のcanister変更なし。


F32 byte decode paid module `bef1ba12…`もビルド完了、source/workflow/Wasm hashesと元22kernel SHA・patch validator・index一意性、canonical billing source byte同一を独立監査した。write append full proof session92963が完了・復元するまで全体canisterへのdecode guards/installを待つ。読み戻し版全体性能はまだ未測定。


F32 byte append paid全推論の3入力は122,867,538,329 /105,949,349,267 /124,945,286,613命令で、BF16 predicate版と完全同一。component84.6%減の箇所が実3入力では命令減につながらなかった。現在全境界・復元proofが継続中（session92963）、完了前にreporterは実行しない。次のdecode版はビルド済みだがfull canister installはまだ行わない。

BF16 pack64 component `92ee6ed4…`をビルドした。元pack/unpack SIMD8 control bodyはcore原本とwhitespace正規化後token同一を独立監査した。pack64は元の2shift+narrowを1byte shuffleへ置換、64要素unroll、unpack64は同じextend/shift/storeを64要素unrollする。全高bit pack表現とhigh bits<<16 decode表現を53条件212queryで独立host SHAと比較中（session28588）。全65536 BF16 patterns、任意F32/NaN bits、tails、unaligned prefix0..16を含み、確保・入力準備・digest/Candidはbody計測外。小型診断canisterでのみ実行、全体候補へまだ未組み込み。


## F32 byte append版、全paid proof完了（速度改善なし）

paid module `ec7dffc5…`は617/620/653で122,867,538,329 /105,949,349,267 /124,945,286,613命令、各4worker、heap4,220,911,616 bytesでBF16 predicate版と同一。全32hidden（layer30独立carry再構成）・32conv/KV・最終norm/判断/logits/確率bitsとpaid境界/返却/duplicate/conflict/Busy/競合upgrade/receipt/config/replayが一致。snapshot004dから元module/cache721/packへ復元・snapshot削除・caller停止完了。reporter後workflow hashes/ZIP一意性/最新summary bytes/復元flagsを独立監査した。latest dense直接captureは未実施、goal active。decode paid `bef1ba12…`のupgrade guards4条件と復元を確認し、全3入力paid proofを開始した。

BF16 pack64 component `92ee6ed4…`の53条件212queryは元pack8/unpack8・新shuffle64 pack/unpack64が全host high-bit byte oracleに一致した。保存Candid再decode/counter sums/source-dependency/input hashes/ZIP一意性/最新summary bytesを独立監査。大配列のpack43.50〜43.52%、unpack26.94〜26.95%減、real43.51/26.94%減。checker progress reductionsは全methodをmethod0 baselineで表示するためunpackの表示19.4%はpack8との比較で、正しい同種unpack8→unpack64減はreporterの26.94%。小型diagnostic停止完了。core内部codecにも元pack/unpackがあるため、runtime wrapperだけでなく独立core overlayの2SIMD bodyだけを置換するfull候補を検討する。演算linear/block256とweights/inputは維持する。


BF16 pack64 full候補は通常runtime bytesを維持し、artifact内の独立inference-core overlayだけを作る。core原本4filesをcopyし、bf16.rsのWasm pack_simd/unpack_simd2bodyのみcomponent helperへdispatch、lib.rsにmodule追加、helperはcomponent source byte同一。linear.rs/block256.rs/native/public shapeチェック/finite判定はbyte同一を独立監査した。証跡 `artifacts/update-bf16-pack64-v1/core-substitution-audit.json`。workspace crate本体は変更せず、独立rlibをcompileしてfull runtimeのexternだけを差し替える。通常full build実行中（session18357）。F32 decode全3入力paid proof（session12731）は重み準備中。


BF16 pack64通常full module `ea41eea8…`はビルド完了。core overlay dependency/source/workflow/Wasm hashes・22kernel source SHA維持/validator/function index一意性、通常runtime全source byte同一を独立確認。runtime externとwrapper dependency searchは独立core rlibへ向き、原本core rlibは置換していない。build source ZIP一意性も確認した。paid版ビルドへ進める。decode full proof session12731は重み準備中、full canisterのinstallは完了・復元まで待つ。


BF16 pack64 paid diagnostics module `2d73913c…`もビルド完了。normal runtime rlib hash一致、独立core dependency source/hash、canonical billing byte同一、22kernel source SHA/validator/function index一意性、source/workflow/Wasm hashesを監査した。paid wrapperのdependency searchも独立coreを参照する。decode full proof session12731の完了と元module/cache/pack復元まで、pack64 guards/installは待つ。goal active、最大確定値124,945,286,613命令、1000億には24,945,286,613命令不足。


読み戻し版session12731は721weights/prefix voting/common準備を完了し、3入力測定へ進んだ。別BF16 core predicate full候補は既存独立all64/classify64 componentを独立core overlayへ適用する。元core4filesからpack/unpack/all/classifyのWasm4bodyだけをcomponent helperへdispatchし、linear/block256/native/publicvalidationを保持。helper2本はcomponent source byte同一。独立source置換監査 `artifacts/update-bf16-corepred-v1/core-substitution-audit.json` を保存した。通常full build実行中（session71688）。pack64 paid `2d73913c…`はビルド済み、decode full完了・復元までguards/installを待つ。


F32 decode paid3入力も122,867,538,329 /105,949,349,267 /124,945,286,613命令で速度改善なし。full境界/復元proof session12731が継続中で、完了前に次full candidateはinstallしない。BF16 core predicate通常 `52267c38…`・paid `5345f34b…`をビルドし、source/workflow/Wasm/dependency hashes・normal runtime byte同一・22kernel source SHA/validator/index一意性・canonical billing byte同一を監査した。

BF16 fused checked decode component `5c0f0c01…`はcoreのunpack_finite_simd8/unpack_f32_finite_simd4とfinish_simd原bodyをcontrolとしてtoken同一監査し、同じpeak/low bit summaryと全value書込みを64要素unrollする。decode/error/finite-BF16分類を返し、hostでexpanded raw bits+tri-state byteのSHAと照合する。全53条件212queryの準備、source/dependency/Wasm hashを確認した。Wasm error messageもinvalid activationであることをprobeでassertする。独立診断canisterにinstall中、まだcomponent検証未完了。


## F32 byte decode版、全paid proof完了（速度改善なし）

paid `bef1ba12…`は3入力122,867,538,329 /105,949,349,267 /124,945,286,613命令、4worker/heap4,220,911,616 bytesで旧版と同一。全32hidden（layer30独立保存carry再構成）・32conv/KV・最終norm/判断/logits/確率bitsとpaid境界/refund/duplicate/conflict/Busy/競合upgrade/receipt/config/replayが一致。snapshot004fから元module/cache721/packへ復元、snapshot削除/caller停止完了。reporter後のworkflow hashes/ZIP一意性/最新summary bytes/復元flagsを独立監査した。latest dense直接captureは未実施、1000億goal active。BF16 pack64 paid upgrade guards session99198を開始した。次full proofはこのguardsが復元完了してから開始する。fused component診断install session77338が継続中で、完了を待ってstart/checkする。


BF16 pack64 paid `2d73913c…`のactive/refund upgrade拒否とPending/Done保持4条件が成功、元module/cache/pack復元とsnapshot0050削除を確認した。全3入力paid proofを開始した。BF16 fused component診断のinstall/startは端末終了を確認し、53条件212queryを開始した。corepred paid `5345f34b…`はビルド/監査済み、まだfull installしていない。


BF16 fused component `5c0f0c01…`の53条件212queryは原SIMD8 BF16/原SIMD4 F32 checked decodeと新SIMD64が全expanded bits/finite分類に一致。保存Candid再decode、独立floating isfinite oracleとlow bit oracle、counter sums/source-dependency/input hashes/ZIP一意性/最新summary bytesを監査した。error messageもinvalid activationをprobeで確認。大配列BF16 checked decode26.96〜26.97%、F32 checked decode52.416〜52.422%減。real26.97/52.42%。checksum/確保/preparationはbody外で同条件。小型diagnostic停止完了。reporter field名をf32_gain_percentへ修正して再報告・archive監査した。次full候補はcore overlayのWasm6body（pack/unpack/all/classify/checkedBF16/checkedF32）のみ変更し、既存検証helper3本をbyte同一で使う。通常full buildを開始する。pack64 full proof session53762が重み準備中、corepred paidは未install。


BF16 fused通常full module `c551f1c6…`をビルド完了。core6body置換以外のsource byte同一、helper3本component source byte同一を独立監査済み。source/dependency/workflow/Wasm hashes、normal runtime全source byte同一、22kernel source SHA/validator/index一意性、独立core dependency searchとsource ZIP一意性を確認した。paid版ビルドへ進める。pack64 full proof session53762が継続中なので、次full candidateのguards/installは完了・復元まで待つ。


BF16 fused paid diagnostics module `d1f4aceb…`もビルド完了。normal runtime rlib hash一致、独立core source/dependency/workflow/Wasm hashes、canonical billing byte同一、22kernel source SHA/validator/index一意性、wrapper独立core searchを監査した。pack64 proof session53762は721weightsと2prefix bank準備完了、3入力測定へ進む。corepred/fused paidのguards/installはpack64の全proofと復元が完了するまで待つ。過去S3-prepared証跡を再読し、immutable重み変換を事前準備してもinput/reconstruction/loading費用でquery命令増だったことを確認した。単純な係数cache拡大は再採用しない。


BF16 pack64全3入力は122,867,558,756 /105,949,369,694 /124,945,307,040命令で全ケース+20,427の微増。全境界/復元proofはsnapshot0051復元・削除まで終了し、reporter実行中。採用しない。actual optimized paid schedulerのrunはevaluate/evaluate_decodedを使い、hidden/norm/attentionをdecoded Vecのまま渡しており、carry packet encode/decodeを行わないことをsourceで確認した。このため先のappend/decode/pack/unpack候補は重い推論経路に効かなかった。未installのcorepred/fused paid候補は速度改善候補から外す（artifact保存）。

actual schedulerのowned graph候補を実装し、通常runtime/22WATを最良BF16 predicate版のまま使う。両attention branchのnorm cloneをmem::takeへ、state hash後attention Vecをtruncateしてmove、MLP output Vecをsplit_offでnorm/hiddenに分けてhidden元capacityを次inputへ再利用する。全値順/shape/stage/stop/paid billingを保ち、源source置換数をassertする。paid build session76125を開始した。全体速度とsaved bitsは未検証。


BF16 pack64 paid全proofは全32hidden/32conv-KV/最終出力bits/paid境界/競合upgrade/receipt-config-replayが一致し、snapshot0051から元module/cache/packへ復元・削除/caller停止した。reporter後workflow hashes/ZIP一意性/最新summary bytes/復元flagsを独立監査した。全3ケース+20,427微増のため採用しない。goal active、最大最良124,945,286,613命令。

owned graph paid module `bdb84bb3…`ビルド完了。最良BF16 predicate通常runtime rlibと元22WATをbyte同一で使う。scheduler全source独立比較により8箇所の所有権移動/分割置換のみ、全値順/stage/stop/paid accounting保持。billing/source/dependency/workflow/Wasm hashes/22patch validator/index一意性を監査した。`scheduler-source-audit.json`保存。upgrade guards4条件とsnapshot0052復元・削除が成功。全proof session58003はlocal cycles残5.9893Tで8T caller作成失敗し、snapshot0053から元module/cache/packへ復元・削除を確認して終了。失敗証跡をproof-failed-local-cycles1へ保存した。localのみ100T mint、残105.9893Tを確認。全proofを再実行する。mainnet操作なし。


owned graph全3入力paid proofは122,796,648,377 /105,900,327,115 /124,867,848,550命令、各4worker、heap4,224,647,168 bytes。前最良から70,889,952 /49,022,152 /77,438,063命令減。全境界/復元までterminal完了、snapshot0054から元module/cache721/packへ復元・snapshot削除/caller停止。reporter実行中、保存Candidの独立再decode監査はまだ未完了。最大残り24,867,848,550命令（約249億）、goal active。

phase diagnostic v1通常 `51dae8bd…`・paid `80580e55…`はビルド/22WAT/source-hash/scheduler hook byte監査まで完了したがinstallしていない。既存LoRA_A/B labelだけではattention_fusion等のshared F32 A入口が漏れるため、v2はmatrix_loaded共通入口の全F32射影・base INT8 projection・activation quantize・outer Delta recurrenceの4spanだけにする。深いkernel/loop instrumentationは除外する。通常v2 build session34840を開始した。v1 artifactとscriptsを保存し、変更しない。typed profile decoderはsource/dependency hashを保存し、独立手製Candid Text（u64 maxを含む）で一致を確認した。最初のtools buildはreportにdependency_hashesがなくcompile前に停止、空artifactをfailed-manifest1へ保存し、実extern依存3fileをhash固定して成功した。profiling totalは最適化候補として扱わず、方向選定に使う。


## owned graph版、全paid proof完了

paid `bdb84bb3…`は122,796,648,377 /105,900,327,115 /124,867,848,550命令（4worker、heap4,224,647,168 bytes）。全32hidden（layer30独立保存carry再構成）/32conv-KV/final norm/判断/logits/確率bitsとpaid境界/refund/duplicate-conflict/Busy/競合upgrade/receipt-config-replayが一致。snapshot0054から元module/cache721/pack復元、snapshot削除/caller停止を確認し、reporter後workflow hashes/ZIP一意性/最新summary bytes/復元flagsを独立監査した。旧最良との差70,889,952 /49,022,152 /77,438,063命令減。最大残り24,867,848,550、goal active。latest dense直接captureは未実施。

phase-profile v2通常 `8c7326b7…`・paid `8b86882d…`をビルド完了。4coarse phases（base INT8/F32 matrix_loaded全入口/quantization/Delta recurrence）、深いloop instrumentationは除外。全runtime source独立比較でprofile filter/outer Delta span/matrix_loaded wrapperのみ、22kernel source SHA/validator/index一意性とsource-dependency-workflow-Wasm hashes/billing byte同一/scheduler clock hooksのみの差を監査した。Candid trace queryとtyped native decoder、全3入力paid proof hook/reporterを準備した。v2 guardsを開始する。診断のinstrumented totalは最適化結果とせず、支配費用を特定する。


phase v2 paid `8b86882d…`のactive/refund upgrade拒否・Pending/Done保持4条件成功、元module/cache/pack復元・snapshot0055削除を確認した。全3入力paid proofと4phase trace取得を開始した（session48670）。本診断はbase schedulerを使い、owned graphの値計算と同じだがcopy費用は旧型。profile totalを最適化成績にはしない。

profile-off通常候補は最良BF16 predicate runtime全sourceと22WATを保持し、runtime_commandのinstruction-profile cfgだけを外す。既存CLOCK Noneでも残る分岐をcompileから除く試験。実装bodyは同じ、native/paid結果は未検証。owner query profiling spansは空となる診断prototypeであり、mainnetへdeployしない。通常build session28184実行中。paid版は成功済みowned graph schedulerを組み合わせるためのbuilder/wrappersを準備した。


profile-off通常 `5936603e…`・owned paid `d97b6202…`のビルド完了。runtime全source byte同一、runtime compile commandはinstruction-profile cfg一組のみ除去、成功済みowned scheduler byte同一、22patch body SHA/validator/index一意性/source-dependency-workflow hashesを独立監査した。post-build-audit.jsonとscheduler-source-audit.json保存。reporterの比較dictを本候補に修正。診断full proof session48670は重み準備完了、prefix準備中なのでprofile-off installは復元後まで待つ。


phase v2 case617の最初の診断結果122,870,577,202命令（旧scheduler、profiling overhead含む）。rawCandid traceをnative decoderで再decodeし、base INT8 96840423057、F32 10452508721、Delta recurrence 4148928756、quant 99544987命令。base INT8が約78.82%で主対象。partial-phase-617.json保存、全proof/復元/最終reportは未完了。session48670継続中。


phase v2全3入力diagnosticは122,870,577,202 /105,955,635,895 /124,934,197,099命令。raw4phase Candidを再decodeし、base INT8は96,840,423,057 /83,612,977,569 /98,721,614,921（78.815/78.913/79.019%）。fullproof session48670 terminal成功、upgrade ordering/completed receipt/Busy unpaidも通過、snapshot0056から元canisterへ復元・削除。全report session84807開始。owned profile-off guards session55805開始（前proof復元後）。最大の削減対象はbase INT8と判明。

phase v2 reporter terminal成功。全32hidden/conv-KV/final decision-logits-probs bits/paid境界/upgrade全検証、4phase rawCandid再decode、全workflow hashes/ZIP一意性/最新summary bytes/復元snapshot削除flagsを独立監査した。診断onlyとして保存し最適化bestへ採用しない。profile-off guards session55805はactive upgrade拒否を通過し継続中。


owned profile-off paid guards session55805 terminal成功、active/refund拒否とPending/Done保持4条件・baseline復元済み。full proof session71240開始、snapshot/install/caller2体readyを確認。旧S2 rank49比較を再読、積削減のみでは変換・復元/局所変数費用を相殺できず回帰であり、単純再実行しない。


S2 Winograd hoisted試験を開始。数値/DAG/rank49/input/native保持、weight addressを16loadごとの再計算からj毎一度へhoist、出力storeaddressをstack保持して一時value set/getを除去。最初build54783はWAT local宣言をbody後へ追加した位置ミスでpatch失敗、artifactをfailed-local-placement1へ保存。宣言を全命令前に移してbuild54403再実行中。owned profile-off71240はweights準備中、両試験canisterは別。guards0057削除と4条件を独立監査した。

S2 hoisted module `afc1064f…` build54403 terminal成功、locals9632。独立逆変換で新kernelから16local追加/176pointer初期化/2816loadアドレス置換/176store stack保存を戻すと元WAT byte完全一致。全Rust src/DAG plan byte同一、source/dependency/upstream hashes/validatorを監査。小型canisterへupgrade88694 terminal成功→start terminal→check4594開始。途中bit一致だがS1比size67約6.20%増、617約4.75%増で改善未達、全21条件/42query終了後rawCandid再decode監査予定。full profile-off71240重み準備継続、全proofは未完了。

S2 hoisted全21条件42query/check4594/report terminal完了。全native bits/独立symbolic identity/INT32中間bounds/rawCandid再decode/input-source-dependency-workflow hashes/ZIP一意性/最新summary bytesを独立監査。109tokenは1,235,729,107命令、旧Winogradから約1.52%減だがS1対照1,173,054,552より5.34%増。採用せず、small canister stop terminal完了。full owned profile-off71240継続中。


NETWORK INCIDENT: profile-off proof71240 terminal通信失敗、finally stopも失敗、snapshot0058復元未検証。local8001 descriptor PID24039不在/TCP listenerなしを確認。対象networkのみrestart72518 terminal完了したが旧6eyddはDestinationInvalid、snapshotもnotfound。旧state復元を主張しない。既存state保持を期待した再起動が旧canisterを回収できなかった事前検証不足をuserへ明示。旧proof/保存bits/models/evidenceは保全。新detached goal-only canister4caro-hl777-77775-aaaba-caiをlocal8001に10Tでcreate、正確なbaseline6052cc94… install開始。モデルpackを同じhashで再uploadして比較環境を再構成する。mainnet操作なし。他project8000/8002には操作しない。incident.json保存。

新専用4caro baseline install77469/settings4GiB/topup50T terminal完了、管理statusでmodule6052cc94…/Running/controllerを検証した。local cycles mint100T成功、すべてlocalのみ。upload95024 live、128 chunks243MB送信まで確認。まだpack commit/hash/cache721/full再測定は未完了、旧0058復元を主張しない。


recovery new paid artifact `paid-owned-profile-off-recovery-v1`とguard/full-proof/reporter wrappersを作成。buildはhash同一d97b6202…へのsymlink、scheduler/billing/sourceそのまま、targetだけnew4caroへ切替。old failed proofと全entryhashesを保全し照合した。new guardはbaseline-ready.completeを前提とし、upload95024完了・pack/cache準備後のみ実行。snapshot download/upload CLIが存在することを確認し、baseline pack/cache準備後にinternalSSDへdurable snapshotを保存してからpaid再測定へ進む。現在upload1472chunks2,662MBまでlive。

recovery upload95024は全chunk送信を終了し、pack全体hash中（1,112/4,702MBまで確認）。new-target wrappersの旧workflow/entryhash/candidate module同一を独立監査済み。IC CLI snapshot download/upload対応を確認し、cache準備後のfull baseline snapshotをinternalSSDへ永続保存する。新baseline cache準備はupload terminal ready/hash確認後のみ。


recovery upload95024 terminal成功、4,702,451,200 bytes、3,201 updates、422.94秒。管理certifiedmodule6052cc94…を照合し、pack ready/hash/model/bytes/hashed/receivedが旧beforeと同一を独立確認。比較用721cache再構成 session81977開始、まだterminal未完了。portable snapshotはcache完了後のみ。S3prepared旧sourceを再読し、実際にB係数はtoken loop前に全load/cachedlocals（343×8×2=5488）していることを確認。ロード移動だけでは旧S3回帰を解消できないため同じ試験を再実行しない。


baseline cache81977 terminal成功721items/4,065,416,192bytes、237.29秒、準備78,949,726,126命令（worker性能値へ含めない）。checkpoint_goal_recovery_baseline.pyでcache全dictとpack critical fieldsを旧beforeに独立比較して一致。snapshot `00000000000000007fffffffffa000020101` create→start terminal後、internalSSD durable snapshot download session51737実行中。metadataのmodule3,123,573 bytes/heap4,112,187,392/stable4,702,470,144を観測。hash全file+download terminal+afterstate一致確認はまだ未完了。new paid proofはbaseline-ready.completeを待つ。

checkpoint51737 terminal成功。snapshot download全file完了、afterstate==before、cache/pack元状態同一を確認してbaseline-ready.complete=trueを保存した。metadata sizes==各file size、全file SHA256を独立再計算、wasm_module.binは6052cc94…原baselineとhash一致。durable-post-audit.json保存。旧snapshot0058復元ではなく、新4caroへ再構成した比較環境の永続snapshotである。

recovery paid upgrade guards session56407開始（durable baseline full audit後）。新target4caroのみをmutation対象とする。旧failed session71240はterminal失敗で再pollしない。元目標100B、保存最良124,867,848,550命令、最新dense直接captureも未実施、goal active。


recovery upgrade guards56407 terminal成功、4条件/entryhash/module/baseline復元/temporary snapshot削除を独立照合後にfull proof11516開始。proof11516 terminal成功、new4caroのみで3入力/4worker/課金境界/Busy/完成receipt/update順序を検証し、temporary snapshot00000000000000027fffffffffa000020101を復元・削除、元baseline6052cc94…/pack/cache同一を確認した。durable baseline snapshot00000000000000007fffffffffa000020101は保持。旧failed71240の復元成功を主張しない。

profile-off recovery reporter65003 terminal成功、全32 hidden（layer30独立saved-carry参照）/32 conv・KV/最終norm/decision/logits/probability F32 bitsを照合。independent audit_paid_profile_off_recovery.pyで31 raw Candid call再decode、全workflow/reference hashes、ZIP一意性・最新summary bytes、worker実数合計、4guards/復元flagを再確認。617=122,796,636,367、620=105,900,315,105、653=124,867,836,540命令、heap最大4,224,647,168bytes。旧owned graphより全case各12,010命令減で微小な改善。目標100,000,000,000未達、最大残り24,867,836,540（248.6783654億）。module d97b6202767ba1bc0f27484636bc4d8cbdb22db272b85907e95a62902ee716a2。runtime source/元22WAT/billing/scheduler bytes維持、runtime cfg instruction-profileのみ除外する診断用prototypeでowner query profiling spansは空。最新moduleのdense Delta直接capture未実施、goal active。

新候補: Yinqi Sun rank23/56-addition 3x3 scheme（https://arxiv.org/abs/2604.27645、MIT https://github.com/sunyinqi0508/3by3r23-56a）をcommit2917e6dedb624340a7a75fbb0214627ed545ea84で固定、LICENSE/README/verify.py全hash保存。author programは実行せずast.literal_evalでSIDES定数のみ読み、自作DAG/独立整数展開で3x3 rank23、2x2 rank7、6x6 rank161両順序、12x12 rank1127三順序を全Brent係数照合。6x6各46,656、12x12各2,985,984係数一致。元block256を保持しsegment43→SIMD44、segment22→SIMD24のpaddingをzeroとする。12x12最大変換input4064/4096（I16安全）、leaf/reconstruction保守bound160,219,136（I32安全）。DAGに明示単項negationを含むためnode数は論文56 additionsと同一指標ではない。独立scalar interpreter80条件（max/min/alternating/random×full/tail×5mixed orders）で元256項intdot/F32 conversion・scale・blockaddのbits一致。scripts/verify_rank23_mixed_symbolic.py、audit_rank23_mixed_native.pyとartifacts/rank23-symbolic-v1/report.json/native-audit.json保存。IC SIMD codegen/実測は未実施で、命令削減はまだ主張しない。


rank161 SIMD prepared diagnostic: build_rank161_prepared_probe.py作成。初回括弧ミスをfailed-brace1へ保全、修正build1774 terminal成功module2629960f…/locals9073。immutable transformed coefficientは元weight容量9.241973876953125倍、diagnostic-only。rawの元S1control source/WATを保持、rank161整数DAG/range bounds/全padding保持。新detached goal-only local4xhad-gd777-77775-aaacq-cai（8T）へinstall38517 terminal成功。checker14619は19条件38query nativebits全一致、55rawCandid再decode/workflow/dependency/source/input hashes/ZIP一意性を独立監査。109token1,590,123,131対S1 1,173,035,337=35.56%増、67token約41.87%増、617約35.76%増で採用しない。whole inference接続なし。small stop terminal。full4caro管理statusで元module6052cc94…/Runningを再確認。

rank161 vector-store版: integer/reconstruction/source変更を逆変換して元WATとbyte一致、scalar output gather/scale/storesのみSIMD化。lane-tag監査で6dot pairを元12column順に再構成しscale/blockadd演算順保持を確認。初回repeated output header replacement問題をfailed-replacement1へ保全、修正buildterminal成功67fe37b4…/locals9046/store72。small4xhadへupgrade64854 terminal→startterminal→check開始。全nativebit/性能は未完了、未採用。


rank161 vector-store check43442 terminal成功19条件38queries、全nativebits一致。source independent auditで元integer-DAG逆変換/9046locals/144vector scale mul/72vector blockadd/stores、source/dependency/archive hashesを確認。report_rank161_probe.pyで55rawCandid再decode・input/workflow/modules/native digest・両source ZIP一意性と全byteを監査し、frozen-audit.zipへ記録。109token1,500,471,251（scalar rank161から約5.64%改善）だがS1control1,173,035,337より27.91356%増。67token980,830,063対730,258,213=34.31277%増、617=1,206,052,097対941,098,207=28.15369%増、全case controlより遅いため不採用/whole inference未接続。small4xhad stop terminal完了。元同block256/int8/source・bit一致を保った新rank161算法のscalar output版とvector output版を実測して棄却した。最新全体最大はprofile-off recovery124,867,836,540命令（100B未達）のまま、goal active。


rank161 inline reconstruction診断版を作成。符号付き係数とは別に正しい参照回数をkeysで数え、single-use180個はstack式へinline、共有/出力123個のみ新rc localへ保存する。旧register reuse版と新stack版の双方を独立記号WAT-stack interpreterで全36root係数照合し、以前のnative bit検証を補強した。元intdot7084/F32 SIMDscale/block順/store72/weights/qprep/sourceは保持。build5e7470ad3f84e4b377876d67771a566c6e4d65ed95f7489044ca8d3fa121ffc6/locals9169、Wasm validator成功。small4xhad upgrade86647 terminal→start terminal後checker開始、実測/bit全条件は未完了、未採用。


rank161 inline check47534 terminal成功19条件38query全nativebits一致。report_rank161_probe.py再decode55rawCandid、全source/dependency/workflow/input/module hashesと2source ZIP一意性/byteを監査、frozen-audit.zip全file byte監査。109token1,453,685,651対S1control1,173,035,337（23.92514%増）、vector-store版1,500,471,251から約3.12%改善だが全case controlより遅く不採用。617=1,169,116,097対941,098,207=24.23%増、67token951,281,263対730,258,213=30.27%増。small4xhad stop terminal完了。

次の12x12 rank1127候補の8-K-lane packingを独立検証。4-Kを2output集合へ複製する6x6方式と違い、8Kを全て異なる位置として1output分のI32x4 partialへ積算、最後に整数水平和→F32 conversion/元scale/blockadd順を維持。segment22を24へpadし3vectors/leafとすることでA/B全cached localsは3381+3381=6762、P1127とsharedC/出力を加えても10000以下。scripts/audit_rank1127_simd_layout.pyで3composition orders×4patterns×4full/tails=48条件を独立native lane interpreter検証しintdot/F32 bits一致。全元256項はexactly once、dummyKはzero、次blockへreadしない。2-3-2 order retained772/locals上限8701、3-2-2 retained860/8789、2-2-3 retained933/8862。artifacts/rank23-symbolic-v1/rank1127-simd-layout-audit.json保存。IC codegen/実測未実施、命令改善はまだ主張しない。今後8-K vector load/input transform/weight transform、1127×3全A/B localcache、single-useC inline、12token×12outputs→4I32 partial横和して3float vectors/row storeを生成し、まずprepared lower-bound componentを比較する。元full worker最良124,867,836,540のままgoal active。


rank1127 prepared SIMD builder作成、session74295 live compile（rustc PID18257/100%CPUを確認）。8 distinct K/i16 lanes、3vectors/leaf、12token×12output、original256→segment22 padded24、共有772/単一使用Cはinline。生成integer-stack全144rootを元C-DAG係数と独立照合。source layout auditでlocals8685、A3381+B3381全load offset一意/完全、Bはtoken loop前、dot3381、F32scale mul72/blockadd36/outputstore36、memory ratio8192rows17.61797332763672倍（診断専用のpreparedcoef）。元S1control/body/sharedRustbilling保全、full inferenceへ未接続。まだcompile terminal/Wasmvalidate/install/native全19conditions/性能値未完了。small4xhadは旧inline版でStopped、full4caro復元済みbaselineのまま。


rank1127 build74295の最終確認はlive、rustc PID18257 elapsed05:43/CPU100%/RSS384080KiB。compiler warningは既存unused strideのみ、terminalなし。再起動や別buildは行わず同じhandleを待つ。source pointer/cache/scale/store auditは完了（Wasm validation pending true）、module install/checkはまだ実行していない。次は同session74295のterminal確認→build/report hash/module/Wasm validation独立監査→Stopped専用small4xhadへupgrade（owner+8192nat32+2560nat32）→start terminal→scripts/check_rank1127_prepared_probe.py --canister 4xhad-gd777-77775-aaacq-caiで全19条件を測定。proof全体100Bとbit/Dense直接capture要件は未達、goal active。


rank1127 expanded build74295 live（rustc18257 10min CPU100%、sample terminal9822はLLVM IndVarSimplifyPassを示す）。進行中compileは中止/restartせず、新しい独立artifact rank1127-generic-prep-v1を作成。A/B入力transform DAGのみconst table+SIMD interpreterへ置換し全node/1127leaf係数を独立展開して一致。query WATはexpandedとbyte完全同一、その他Rust src byte保持。generic builder14745 terminal成功f8448e59edaaf07d62b3f96d7204af12a717d84d373900d6d8c322a1f9fd5efd、8685locals、source/dependency/Wasm validator/ZIP一意性・byte監査完了。small4xhad install51224 terminal→startterminal→check開始。original74295引き続きlive、mainnet/full comparison canisterへ操作なし。並行探索で既存MLP block-reuse報告の3case全0 duplicates、およびfinish_lora_bf16が既にSIMDであることを再確認し、同じ候補を再実施しない。


rank1127 generic checker99566 terminal成功、全19条件38query bit一致。109token2,569,452,678対S1control1,173,034,381（119.043%増）、67token約121.719%増で不採用。report_rank161_probe.pyで55rawCandid再decode・全source/dependency/workflow/input/module/sourceZIPbyte監査しfrozen-audit.zip保存。small stop terminal後、準備とSIMD本体のコストを区別するcore profile版を作成。generic入力/weights/DAG/kernel WAT byte保持し、project_s3の各blockでperformance_counter前後を取り、method4の既存Measurement.input_prepare_instructions=入力変換合計、project_instructions=kernel rowloop合計へ診断意味を変更、totalは全query計測を保持。元control/method3未変更。profile81207 terminal成功77298314…/8685、source/module/Wasm validator/hash/WATbyte一致監査。smallinstall47417 terminal→startterminal→checker開始。originalexpanded74295はliveのまま、待ち時間だけを理由に中断/restartしない。core実測で原WATが既存全体より遅いと判明すれば、その証拠に基づいて展開版候補をretireする。


rank1127 core phase checker33614 terminal成功全19条件38query native bit一致。report_rank161_probe.pyは55rawCandid再decode、全source/dependency/workflow/input/module/sourceZIP一意・byteを監査、frozen-audit.zip保存。core-phase-audit.jsonで全19条件のkernel rowloop単独がS1control全query totalを上回ることを確認、quant+qprep+kernel <= total。109token: kernel2,485,162,180 / inputprep43,987,400 / quant12,007,142 / total2,569,466,118、S1total1,173,034,381。67tokenkernel1,567,524,360対S1total730,257,257。SIMD kernel WATは元expandedと完全byte同一なので、Rust準備を改善しても既存を下回れない実測証拠が得られた（profiling_only、full性能値への採用なし）。これに基づき不要となったowned rustc18257のargvを確認しSIGTERMで意図的終了、build74295 terminalexit1/SIGTERM15を確認。観測timeoutが理由の停止/restartではない。元展開ソース/ログ/sample/metadataは保全、retired.jsonとretirement-evidence.zip全byte監査済み。small4xhad stop terminal完了、live processなし。新rank1127候補も不採用、whole inference未接続。最良full124,867,836,540命令/100B未達、latestdense直接capture未実施、goal active。


最良paid d97b6202…のlatest Dense fidelity gapに着手。build_delta_capture_owned_profile_off.pyはnormal arithmetic counterpart5936603e…から同22WAT/runtime cfg/profileoffを保持し、同owned graph8 substitutions、capture begin/head/export、prefix packet discard、one-stage update限定を追加。build36701 terminal19346a38…/8,044,650bytes。runtime全fileをcapture hookのみ逆変換し元byte一致、scheduler全変更逆変換してnormalbyte一致、paid22patchsourcehash/owned substitutionsの全countを独立照合。source-audit.json: 完全なpaid binary自体ではなく、最新同一算術・owned schedulerの正確性専用counterpart、課金/STOP stage groupingは完了済みpaid proofで別検証、性能主張なし。

prove_delta_capture_owned_profile_off.pyはnew4caroへ限定、baseline-ready/durable backupとsource-auditを前提に旧72capture proofを凍結して開始74477 live。全3入力/voting38+common27prefix/同重み/24Delta層×32headsのordered independent F32 recurrenceとpre-BF outputを照合予定、temporarysnapshot restore/deleteと元module/cache721/pack完全一致をfinallyで確認する。source/dependency/hash/entry frozen proofを保全。report_delta_capture_owned_profile_off.pyを作成し、全22kernels/runtimehooks照合/全Candidchunks再decode/全72native recurrence、さらに独立保存carry hidden30で全32hidden照合と最新ZIPを追加予定。まだcapture測定/復元は未完了、reporter未実行、100B未達goal active。


latest dense capture proof74477 terminal成功。3入力/voting38 common27、24Delta層×32heads×3=72capture、37,748,736 final dense F32 valuesとpre-BF出力を独立ordered recurrenceで全bit照合。hidden31 historical/conv-KV32/final norm/decision/logits/prob一致、reporter98925で保存全Candid chunks再decode+72recurrenceを再実行、独立saved-carry layer30を追加して全32hidden verified。temporary snapshot00000000000000037fffffffffa000020101をrestore→start→module6052cc94…/cache721/pack全dict一致→delete terminal確認。管理statusRunning/元moduleも独立確認。durable baseline snapshot0保持。

新audit_latest_delta_capture_reference.py session14918 terminal成功、全72capture.binをSHAのみでなくbytes全体で旧frozen delta-capture-v2 referenceと直接照合して完全一致（initial/final dense、Q/K/V/G/B、pre-BF出力、headerも含む）。全22最新paid kernel source/owned substitutionsが同一のcorrectness counterpart19346a38…を使用した検証であり、paid binary d97b6202…の課金/4worker STOP grouping/性能検証は別の完成済みpaid proof。source/workflow/evidence全hash、最新ZIP一意性・summarybytes、restore flagsを独立監査しdirect-reference-audit.json/zip、frozen-latest-workflow.zipを保存。歴史paid summaryは改変せずgoal-fidelity-status.jsonでpaid性能と最新直接Dense/全32hidden証拠をhash付きリンク。latest Dense arithmetic fidelity gapは解消したが、worker3入力100B目標は未達。最良full最大124,867,836,540（残り24,867,836,540）、goal active、性能改善主張なし。

待機中のsource再調査でload_prepared_weightはcached F32をPreparedとして有限性再走査せず借用する早期returnを実装済みと確認。これを新しい削減候補として重複実施しない。現在live processなし、full比較target4caroは元baseline状態、small4xhadはStopped。

2026-10-07: 残る約112億命令を絞る診断v3を作成。通常88028af6…/paid cd927a6b…、元22WAT保持、既存4phaseへexecute_norm/conv/add、LoRA finalize、MLP activate、bound prefix input、exported hashを追加。normal sourceは追加wrapperを逆変換してcoarse-v2 runtime byte一致、paid schedulerはprofile前に最新owned-profile-off byte一致、billing canonical byte一致、source/dependency/hash/22patch validatorをaudit_phase_profile_v3_sources.pyで独立監査。最初の2 builder失敗はsource escape/record pathでcompile前、failed artifacts保全、canister変更なし。最終normal13498/paid83657はterminal成功。guards46855の4条件とsnapshot復元/削除を確認。専用full4caroでproof2095開始、まだ3入力計測/参照一致/最終復元未完了。診断のみで改善候補ではない。100B goal active、最良full最大124,867,836,540維持。

診断v3 proof2095 terminal成功、32raw paidCandid再decode/source-reference全hash/ZIP一意性・最新summarybytes/全32hidden独立carry30を監査。617/620/653 diagnostics122,806,616,568 /105,902,079,495 /124,878,674,943、4worker/heap4,224,647,168。snapshot5から元6052cc94/cache/pack完全一致復元・削除/caller停止。管理status Running/元moduleを独立確認。最大case: INT8 98,725,977,617、F32 10,634,544,735、Delta 4,222,039,190、norm1,934,276,159、hash1,830,651,515、LoRA finalize1,092,576,027、conv729,076,377、MLP activate696,677,802、prefix199,948,194、quant103,055,433、未帰属4,709,851,894。計測追加のみ、性能改善値に採用しない。reporter初回quote置換ミスcompile前/監査初回__file__root誤りread前を修正、canister変化なし。
新attention 4key-lane SIMD候補d1db86f0…はFrozen元ordered-dotをcontrolにし、transpose込みbodyの21条件42queryが独立ordered-K F32/BF oracleとbit一致。実サイズ15〜27%増、terminal1/prefix131は241.5%増で不採用。過去artifact/source保持。small4xhad停止後、bounded/unroll8版をbuild6744実行中。元順序とsigned-zero identity維持、検証済み全shapeからpointer spanを導く。fullへの接続・速度改善は未主張。100B goal active。

attention bounded/unroll8 component180d95c2… build6744 terminal成功。21条件42query全ordered-K独立F32/BF oracle bit一致、rawCandid全再decode、source/evidence/hash/ZIP一意・bytes監査。width1..256のK ascending exactly-once、total1..132/width境界のtranspose injective/source/target/vectorload spans、discard lanes maskingを独立監査。n45/48/56/57/67/87/132,width256ではtranspose込みscorebody51.69/55.80/57.01/56.42/57.33/58.74/58.83%減。一方n1,prefix131は124.40%増のためその形状には採用しない。small4xhad停止完了。
通常full候補e939bfa4…/8,039,099bytesはビルド88848 terminal成功。n>=16,width256,total<=132だけall-causal scoreを4keylaneで事前計算、元順序で各tokenへcopy、terminal/他shapeは元経路。attention_views/libを変更逆転してprofileoff runtime byte一致、新helperが単体検証helper byte一致、softmax/value projectionと元22WAT/cfg維持。paid e9c84a65…/8,411,845bytes build94594 terminal成功、owned scheduler/source byte同一、billing canonical byte同一、source/dependency hashes/normal rlib/22validator/source-audit.zipを独立監査。guards32153実行中、full3入力proof未実行、速度改善/最新fidelityをまだ主張しない。最良124,867,836,540/100B goal active。

attention keylane guards32153 terminal成功、4条件と元snapshot/module/cache/pack復元・削除完了。full paid proof4805開始、snapshot saved、まだweights/prefix準備と3入力測定/最終復元未完了。このhandleを継続して待つ。新full candidateの性能値未確定、100B目標active。

attention keylane paid proof4805 terminal成功、report80594 terminal全32hidden独立saved-carry30/32convKV/最終norm/decision/logits/probability F32 bits/paid API境界・返却・Busy・同version upgrade/receipt/config/replay一致。32raw Candid再decode・全source/workflow/reference hashes・ZIP一意性と最新summary bytesを独立監査。617=122,204,479,522、620=105,428,922,270、653=124,363,130,521命令、4workers/heap4,224,647,168。直前最良からgain={'617': 592156845, '620': 471392835, '653': 504706019}。最大残り24,363,130,521（243.63130521億）。snapshot7から元6052cc94/cache721/pack完全一致復元・snapshot削除/callers停止、管理status Runningを確認。最新Dense直接captureはこのattention候補について未実施、旧profileoffの完全直接capture証拠は保全、goal active。
INT8次候補: 旧S1 Winogradのfirst-pair weight initializationはraw4planesのアドレスを毎K再算出し、DAG/nodeからleaf cacheへcopyする。fused-v1では4raw row pointerをoutput quartetごとに一度計算、first-use weight load/transformをdotのoperandとしてlocal.teeし、m4/m5はm3の変換中に生成、m6=m5-m0の等価整数式とする。重み容量/量子化/F32/output順序は保持。最初の3 builder失敗（block位置/indentation/事前作成directory）はcompile前、失敗artifact保全、canister変更なし。修正後build85449実行中、component実測未完了/full未接続。


2026-10-07: attention keylane paid proof4805 terminal成功、617=122,204,479,522 /620=105,428,922,270 /653=124,363,130,521 instructions、4workers、heap4,224,647,168。最新最大1243.63130521億、1000億まで243.63130521億。全32hidden/conv/KV/finalnorm/decision/logits/prob bits、payment/refund/replay/conflict/upgrade guards確認、snapshot7復元削除。post-report-audit.jsonで32raw paidCandid再decode/hash/ZIP/restoredを独立確認。
最新同算術Dense capture095a975f…のproof6571 terminal成功、72capture/37,748,736 Dense values/pre-BF独立ascendingK recurrenceと全32hidden一致。snapshot8復元削除。report26508/direct-reference audit67501 terminal、全72capture.bin全bytes（initial/final dense/Q/K/V/G/B/header）旧frozen referenceと一致。goal-fidelity-status.json/zipで最新paid e9c84a65…とsource/capture evidenceをSHAリンク、fidelity complete/goal incomplete。歴史summary保持。
S1 Winograd first-use fused ab5ad8bf…は全21条件42query nativebit一致、109token3.745%増で不採用。row/output pointer hoisted c50f8903…も42query nativebit一致、109token1.18127%増、6171.2977%増、通常132token0.6706%増で不採用。savedCandid reporter complete、small4xhad停止完了。terminal single_quadは現在fullのfeatureに既に含まれ、重複候補にしない。
次のtail/stackstore候補1ae8b730…build87799 terminal/validator/source audit成功。後続奇数末尾はA21/A22=0でproduct3/4ゼロ、product6はfirstrow roots未参照なので省く。firstpair全7product/cache initializationは保持、float store一時localをstackへ変更、その他Rust source byte同一。最初のbuildはblock guard削除によるparentheses assertionでcompile前失敗、failed-parentheses1へ保全して修正。small4xhad install48668 terminal→startterminal、checker開始（新handleはtool state参照）。全条件測定未完了、full未接続、100B goal active。

Tail/stackstore checker37608 terminal全21条件42query native一致、report complete、109token0.97793%増で不採用。rowpair単位tail dispatch版b39d6869…build43981/install76200/check89022 terminal、42query native bit一致。109token0.15005%増、6170.10281%増、132token0.31325%増。48/56/57/59token僅かな削減0.17613/0.10487/0.01811/0.00637%だが現在全体のadaptive direct168等との比較ではなく、fullへ未接続、不採用。両saved reply reporter complete/small停止。
新gated-norm-finalize単体b97c04e3…build78162 terminal、latest arithmetic runtimeはappend helper以外byte同一。raw F32 SiLU table3を1row1borrow、bf(RMSvalue)→F32mul(rawSiLU gate)→bf順序は保持、4lane SIMDでowned RMS Vec再利用。checker6734 terminal13条件52query全scalar/SIMD/withtable/withouttableと独立NumPy orderedF32/BF digest一致。1/3/4/5/7/8/4096/9216/442368/516096/525312要素、nonBF rounding境界/符号zero/underflow、saved算術operand617（actual RMS intermediateではない）。large body約45.4%減、no table約23.4%減。report raw52Candid再decode/hash/ZIP complete、small停止。通常全体gated-norm-finalize builder開始、まだ全体paid performance/Dense correctness未検証。

通常gated-norm-finalize full builder69258 terminal成功06a1b8f8…/8,040,672bytes。gated_norm行に限り元ordered RMSを計算後、同Vecとborrowed gate sliceをhelperへ渡し、元loopをcontinueで迂回。元finite reply check/他norm分岐は保持。測定helperからgate引数のVec→sliceとzip copiedだけ変更、同FP演算順。audit_gated_norm_finalize_sources.pyで追加helper削除+1literal edit逆転後にlatest attention runtime全Rust byte一致、root20WATと22patch export/sourceSHA保持、build source/dependency/validator確認。paid build21623 terminal fbc07c63…/8,413,418bytes。scheduler update_inference/paid accounting2fileが最新paidとbyte一致、22patch sourceSHAとruntime rlib依存hash確認。初回auditはrootWATを22と誤認したassertionで停止（実際20root+22patch）、22patch一致へ修正してterminalcomplete。paid upgrade guards開始、全体性能/参照bits/最終復元未完了、最新confirmed full最大124,363,130,521維持、goal active。

Gated norm paid guards47638 terminal成功（active/refund upgrade refused、pending/done receipt preserved）、baseline restored。両full.wasm実bytes SHAがbuild wasm_sha256と最終patch output_sha256に一致、source-audit.zip/hash一覧保存。全体3input proof60052開始、snapshot saved出力。次は同live60052を待ち、3case worker instructions/referencebit全検証/restore/delete terminal確認→report_paid_gated_norm_finalize.py→独立postreport audit。60052終了前にfull4caroを別候補で変更しない。fullDense correctness captureは新norm候補について未実施。confirmed bestは旧keylane e9c84a65…1243.6313億、1000億未達、goal active。

2026-10-07: 前goal turnはprogress（新候補ビルド/component52queries/nativebits/guards/fullproof開始）。現turnは60052同livehandleを再検証して待機し、fullproof terminal成功。fbc07c63…で617=121,595,528,204 /620=104,903,053,400 /653=123,736,933,555 instructions、各4workers/heap4,224,647,168。旧attention最大から626,196,966減（約6.26197億）、最新最大1237.36933555億、1000億まで237.36933555億。snapshot000a復元削除、callers52jen/55icz停止finally完了。report7843 terminal全32hidden（30/terminal独立saved carryも含む）/32conv-KV/finalnorm/判断-logits-prob bits/payment/refund/replay/conflict/Busy/upgrade/config/receipt確認。内側generic reporterはduplicate summary ZIP warningを出したが外側最終ZIPは再生成され、独立audit_paid_gated_norm_finalize_report.pyで32rawCandid再decode・worker sum・全hash・ZIP一意性/最新summarybyte・復元flagsをterminal確認。summary normal/runtimecomparison参照を最新gatednormに正してからreportした。新best paid性能complete、最新Dense直接captureは未検証、100B goal active。
同算術Dense counterpart builder87357開始。scripts/build/prove/report/audit_delta_capture_gated_norm_finalize*.pyは既存keylane完成版から最新normal/paid/capture artifact参照のみ置換。次は87357terminal→capture source逆変換監査→新full4caroへsnapshot保護proof開始。同targetで別操作は行わない。現在fullはbaseline復元済み、small4xhad Stopped。

87357 terminal capture1b84c2f6…/8,048,323bytes。audit_delta_capture_gated_norm_finalize_sources.py terminal成功、runtime capture hooks2literalとmodule追加のみを逆転して最新gatednorm全runtime byte一致、scheduler8owned/capture edits逆転してnormal byte一致、先頭6owned変更countがpaid同一、元22patch sourceSHA/validator/source-dependency SHA一致。最新Dense correctness proof18103開始、同handleを継続して待つ。まだcapture/independentrecurrence/final復元は未完了。このsession終了前にfull4caroを変更しない。次: 18103 terminal復元削除確認→report_delta_capture_gated_norm_finalize.py→audit_delta_capture_gated_norm_finalize_reference.py→new goal-fidelity-statusでimmutable paid性能123,736,933,555と最新Dense証拠をhash付きリンク。goalactive、全3input100B未達。

2026-10-07: 前goal turnはprogress（paid best123,736,933,555/reference32hidden/境界/復元report完成、直接Dense builder/proof開始）。現turnは18103livehandleを再検証、weights ready後617各layer Dense/出力bits一致が順次出力、まだ全72/finalrestore未完了。同fullへ別候補をinstallしない。
新INT8 K2/four-column layout候補を独立作成。旧rank7 Winograd rawweight capacity保持、各I32laneが独立出力pair1つを持ち2K dot×64vectorで128項を蓄積、4lane/4outputpair=8outputrows。旧4K×2column配置よりinteger reconstruction groups半減し、partial水平加算を省く。weight layoutはoctet*2cols+block512+(k/2)*8+pair*2+k%2、plane stride(rows/4)*cols同容量。query dataは128係数を連続保存しv128.load32_splatで2Kを4laneへ複製、入力expansion Vec同容量（未使用半分zero initialized）。F32 conversion→scaleX→scaleW→originalblockadd順を保持。v1最初build store64 intrinsic引数順ミスでcompile失敗、failed-store-args1保全。修正後8ca0…build15217完了だがrawblock offset*2不足をsource確認で発見、rejected-before-installへ保全し未install。correctedv1 51829…93025terminal、v2は後続奇数tailでzero product3/4とfirstrow未参照6を省きrowpair1dispatch、初回全cacheinit保持。build55661terminal 76fb6671…/locals7677。audit_s1_k2_four_columns_layout.py raw32/128rows×256/512cols全offsetbijection、672native rank7lane checks（random/minmax/zero/alternating×n1/2/3/8×256/512）、128K exactlyonce、tail符号式・source/dependency/modulehash・Wasm validators成功。
専用small4xhad K2v2 install96060開始。次terminal→start→scripts/check_s1_k2_four_columns_probe.py --canister 4xhad-gd777-77775-aaacq-cai、既存same-module原S1controlと独立nativeexpectedで全21条件42query比較。現layoutは元同量子化INT8と同fixedraw容量、prefix/input/weightsそのものは変更しないがrawweightの格納順だけ変更する未採用component。fullへ未接続、100B goal active。

K2 four-column128 candidate check90672 terminal成功、21条件42query全nativebits一致。109token1,162,400,203対oldS1control1,173,034,376=0.90655%減、6170.9572%、n48/56/57/59/67=1.21989/1.14849/1.08684/1.07430/1.03116%減、normal132token0.54459%減。report初回はwrapper無しcheckerを使ったためfrozen-check.pyがなくartifact reporter path error、参照を実際のcheckerへ正して再実行terminal成功（保存42Candid/native/input/source/dependency/整数bounds監査、frozen-workflow.zip保存）。全体採用はpending: oldS1controlは128幅/tilecopyで、最新full direct168/zero-seedとの公平な比較ではない。small停止。
K2 adaptive168 candidateをnewartifactへ作成。128/168/32三width WAT（K2 layout等は同じ）、8192/4096 rowsに168body、remainder128/32へdispatch（すべてrows%32から安全）。Rust kernel stubは全9ABI引数をblack_box markerに含めopaquewrite保持。maxlocals9927<10000、4patch validator/source/module/dependency audit、raw bijection/672native lane algebra audit成功。build96707 terminal21b914e3…、smallinstall54480 terminal→startterminal→adaptive168 checker開始（livehandle tool state参照）。full未接続、actual n1 single_quad等の新weightlayout互換対応とlatest full comparatorは未実施。確認済み全体best fbc07c63…123,736,933,555、100B未達。
Dense18103最新確認はlive、617/620全24capture完了、653layer28まで独立Dense/state/preBF outputbit一致。final2captureとsnapshot復元削除まだ未確認。同fullへ別mutation禁止。次18103terminal→report_delta_capture_gated_norm_finalize.py→audit_delta_capture_gated_norm_finalize_reference.py→latestgoal-fidelity-status links。

Dense18103 terminal成功、latest capture1b84c2f6…全72/37,748,736 Dense values/preBF ascendingK独立再構成とall32hidden一致。snapshot000b復元削除。report80844とdirectreferenceaudit9470 terminal、全72capture.binの全bytesが旧referenceに一致（initial/final Dense/Q/K/V/G/B/header）。latestgoal-fidelity-status.json/zipはimmutable paid fbc07c63…3case121,595,528,204/104,903,053,400/123,736,933,555とlatestDense証拠をSHAリンク、fidelitycomplete/goalfalse。
K2 adaptive168 checker86349 terminal42query/nativebits一致、109token1,154,211,036対oldcontrol1,173,034,376=1.60467%減、6171.65423%、n48/56/57/59/67=1.90514/1.83404/1.77684/1.76192/1.72408%減、normal1321.09367%減。report saved42Candid/native/input/整数bounds全hash、ZIP一意性・最新summary/workflow全bytes独立監査、small停止。
現latest full comparator専用build_s1_k2_latest_control_probe.py作成。control method3は実際の最新normal rlib+PreparedPairs::project、同25patch中最初22patch全export/sourceSHAがlatestfullと完全一致。candidateWinograd/K2 Rust2filesと3WATは完成済みadaptive168とbyte保持。入力量子化は最新runtime、controlのquery-local原Strassen operand準備はproject body内に含むため性能はtotalを比較（phase splitの公平性を断定しない）。旧exactcontrolを比較から除く。初回compileはruntimeの依存検索先が旧targetでE0463、failed-deps1保全、最新fullcommandの-L/externへ変更後build90405 terminal2ed9aa8e…成功。25validator/currentfull22sourceSHA/module/source/dependencySHA/candidate同byte audit完成。smallinstall75217開始、次terminal→start→check_s1_k2_latest_control_probe.py --canister4xhad。100Bgoal active、最新full最大1237.3693億。fullbaseline復元済み、mainnet/保護対象変更なし。

75217 install terminal→startterminal。latest-control checker28047開始、同livehandleを継続。21条件42query全nativebits/性能は未完了。次28047terminal→report_s1_k2_latest_control_probe.py→saved workflow/archive/source監査→small stop。その結果でactual full direct168/zero-seed/単一tokenquadと比べて有効か判断する。新rawlayoutをfullへ採用する場合、既存single_quad/OriginalView::index/continuationを全て同layoutと係数へ対応させてからfull paid/Dense/proof必須。今はcomponentのみ、旧fullbestとcomplete arithmetic証拠は維持。goalactive。

Latest-control checker28047 terminal42query全nativebit一致。109token: current full control1,126,292,846 /K2adaptive temporarycopy1,145,636,119（1.71743%増）、617control903,699,719 /candidate918,690,245（1.6588%増）、n67+1.56883%、normal132+2.187%。saved42Candid reporter completed、source/workflow/module/dependency/input/native整数bounds/hash/ZIP全byteと一意性をpost-report-auditで独立監査、small stopterminal。旧128controlより1.6%改善はcurrent fullより速いという意味ではなく、full未採用。次の公平化候補はcandidateの一時tile Vec/copyをcurrent fullと同じdirect strided outputへ変更しblock0 positivezero seedで初期化、重複finite output走査をcurrent raw projectと同じ検査境界に揃える（full後段finite検査を保持）、そのbody-only効果をまずcomponentで測定する。n1はcurrent single_quadが既に速く、K2generic初回all7weightcache経路は不利なのでそのままfull接続しない。現時点liveprocessなし、full4caro baseline復元済み、small4xhadStopped。最良actualpaid fbc07c63…最大123,736,933,555、Dense/all32hidden/最終decisionbits証拠完全、100B未達/goalactive。

2026-10-07: 前goalturnはprogress（latestbestDense完了+K2 current-full comparator新実測棄却）。本turnはcurrent-full bytecontrolを保ったK2direct+seedを作成。28patch=現22control+新128/168/32幅direct/seed6WAT、rows出力stride/cols入力scale strideを分離、全n*rows outputspanをblock0 positivezero addで初期化しMaybeUninit→全書込み後Vec変換、一時tileVec/copy省略。入力値/積和/FP順/元postfinite check保持。最初compileは注入後p変数がwinograd.rsを指しsourcearg誤り、failed-source-arg1保全、lib.rs指定へ修正。build26349terminal d9a82d08…、audit seedWATのload禁止assertionをscaleのloadまで広く誤認、出力yp/yp1 loadだけの監査へ修正、terminal sourceaudit完成。smallinstall27484/startterminal、check8324terminal21条件42query nativebits一致。109tokencurrentfull1,126,292,846 /K2direct1,133,617,434=0.65033%増、6170.59549%増、n67 .51539%増、normal1321.11821%増。report42savedCandid/native/source/dependency/bounds complete、smallstop。まだfullへ未採用。
次compact query版e9b9b9e0…build22698terminal。各block128I16係数を隙間なく保持、元未使用半blockを除去しlen=7*pairs*(cols/2)、全7/group/blockの連続128書込み後にlen確定、Qoffset bytes=group*cols+block256。6WAT差分はこのqo式各3箇所だけ、rawweight/output/FPdot同じ。candidateだけにあったwide outputfinite scanを削除しcurrent raw projectと同検査境界へ揃えた（現在のraw controlもこのscan無し、graph後段finite checkは維持）。audit_s1_k2_compact_sources.py denseinitialization/64load32_splat spans/current22control/all28validator/source依存hash/graph finitecheckを確認。smallinstall32119開始、次terminal→start→check_s1_k2_compact_probe.py。currentfullbest123,736,933,555/100B未達、fullDense証拠完成、goalactive。

K2compact checker64606terminal全21条件42query/nativebits一致。current-full comparator againstmodelQ: 109token1,112,542,436対1,126,292,846=1.22086%減、6171.26637%、n48/56/57/59/67=1.50463/1.44326/1.38305/1.37088/1.33478%減、normal1320.99577%減。n1 genericrank7は49.46%増なのでそのままfullへ使わない。report saved42Candid/source/dependency/native整数bounds/input complete、direct/compact両workflow hashes・ZIP一意性/最新summary/workflow全bytesをpost-report-audit独立確認、small停止。
K2 single-path prototype aa38107a…build10510terminal。7係数prepをn1だけ省略、raw4plane×64K2 dotに戻して独立I32lane毎4even/4odd出力を得る。old singlequad同256項I32dot→F32 scaleX→scaleW→blockadd、4,161,536 I32bound・rows2560/4096/8192/9216×cols256/512/2560/9216 pointerbounds、全28WATbyte/source/dependencyhashがcompact同一を監査。helperのみwinograd.rsへappend、libのwa/dispatch2literalのみ変更。n1finite checkはcurrent singlequad同all_finiteを保持。smallinstall30282開始、次terminal→start→check_s1_k2_single_probe.py、21条件42query native＋actualfull comparator。fullweightpack/index/carry/continuation互換とMLPgate/down shape検証は未実施、full未採用。currentfullbest123,736,933,555/Denseall32hidden証拠完全、100B未達goalactive。

30282 installterminal→startterminal→single checker53494開始。最新live出力ではn1current-fullcontrol26,234,183 /K2single23,633,561=9.9131%減、bits一致。n48/56/57/59/67mainサイズ1.50463/1.44326/1.38305/1.37088/1.33478%減を維持（追加dispatchによる数命令差のみ）。まだ全21/42terminalは未確認、full未接続。次53494terminal→report_s1_k2_single_probe.py→workflow ZIP/source全hash独立監査→smallstop。その後K2新rawlayoutのOriginalView::index/pack/continuation/native/1token・fullschedulerの全tensor形状互換を揃え、MLPgate9216x2560・down2560x9216とcarry/rangeをactualsame全体controlで比較する。MLP形状は現在fullの160幅に対してcandidate128幅なので160direct/seed追加候補が必要。fullbest/Dense証拠は123,736,933,555維持、100Bgoalactive。

Single checker53494terminal全21条件42query/nativebits一致。report_s1_k2_single_probe.py terminal raw42Candid再decode/bounds/input/source/dependency hash/ZIP完成、postreportで全workflow hashes/ZIP一意性/最新summaryとarchive全byteを独立監査。smallstopterminal。q-projection promising component only：n1約9.9131%、main1.0〜1.5%減。現liveprocess無し、fullbaseline復元済み、smallStopped、workspace tracked diff無し。次の実作業: four output K2 single-compatible raw layoutに160幅追加しMLPgate/down同module currentfull比較、range/carryとoriginalview/index互換を実装後に全体normal/paid/Denseを測る。最新wholepaid最大1237.36933555億、目標まで237.36933555億、goalactive。

### K2 compact tile160: both MLP shapes verified (2026-10-07)

`artifacts/s1-k2-mlp160-v1` module
`6793e46ab7d8023fdfe37d982eaade886f9c2bf8e279cce2032afe1828630a83`
adds two direct/seed tile160 kernels to the previously completed K2 single candidate.
All previous 28 patched WAT source identities are unchanged; all 30 Wasm patches
validate. New tile160 uses 9477 locals, below 10000. Reversing the two runtime
dispatch edits and appended stubs recovers the previous component source bytes.

- Gate: genuine layer3 MLP gate weights, 9216x2560, 16 conditions/32 queries.
  All results match the independent ascending-K native INT8/F32 reference digest.
  At 48/56/57/59/67 tokens, total reduction is
  1.430921/1.368580/1.308571/1.296368/1.259550 percent against the actual current
  full runtime control. Single token reduces 9.934358 percent.
  Saved Q activation rows here are arithmetic operands for MLP shape testing,
  not a claim that they are real MLP gate activations.
- Down: genuine layer3 MLP down weights, 2560x9216, 12 conditions/24 queries.
  All native/control/candidate digests match. Synthetic boundary inputs at
  32/40 tokens reduce 1.296839/1.067227 percent. Single token reduces
  10.629049 percent. Largest request remains under the diagnostic 1.5MB bound;
  no ingress-limit bypass or actual full-input claim.

Independent saved Candid re-decode checks, source/dependency hashes, positive-zero
seed preserving F32 add, compact operand pointers, all output tile spans (48
conditions), and unique archived entries/latest summaries passed in
`post-report-audit.json`. Evidence is frozen in separate gate/down workflow ZIPs.
Dedicated small canister was stopped after both runs. Full canister and full best
remain unchanged; these component results are not full inference measurements.

Next: integrate the layout with `strassen_raw::pack/index`, compact full/range
operands, aligned and padded PackedView paths, K2 single-token full/continued
projection, and original F32 carry. Keep original 22 full WAT sources and all
FP32/Delta/norm/prefix behavior. Then re-run actual paid 617/620/653 and complete
hidden/state/final/Dense fidelity before adopting. The maximum confirmed full
count remains 123736933555; the 100000000000 goal remains active and unmet.

### Full K2 integration built; actual paid proof running (2026-10-07)

New `scripts/strassen_k2_runtime.rs` implements the changed raw layout and index,
compact rank7 full/range operand preparation, packed row/tail padding, original
F32 block accumulation, native reference path and K2 single-token full/carry
projection. Range preparation initializes only consumed blocks. Continued
projection copies initial F32 carry and uses unseeded kernels throughout.
Normal first blocks seed positive zero while retaining the original F32 add.
The generic unaligned single path retains ascending K256 dot and scale order.

Normal module `5e83aaa076f674fdd8f5bcae07774fe112ed9b71a1f9eb7920d730b174d4ed99`
(10,355,327 bytes) is in `artifacts/update-k2-compact-v1`.
Paid module `fa00e510c628491ff2ce3122053b4224bd0b6a4105e97968825afd6c9930fd7e`
(10,728,073 bytes) is in `artifacts/paid-k2-compact-v1`.
All30 patched bodies validate; original22 WAT source bytes are identical.
All other normal runtime RS files and all paid wrapper/scheduler/billing RS
files are byte-identical to the previously verified gated-normalization build.
Independent native harness checks300 conditions, including full projection,
unaligned row views, nonzero/signed-zero carry and recomposed split-K projection,
and bijective restoration of6,422,528 weight bytes. This native harness checks
actual new source with tiny type adapters, not full Wasm inference fidelity.
Normal source-audit ZIP independently checked unique/current entries.

First construction failed a stub-count assertion (15 required,14 assumed),
preserved in `update-k2-compact-failed-stub-count1`. Next compilation passed but
patcher rejected merged direct/seed stub aliases; preserved in
`update-k2-compact-failed-alias1`. Distinct fail-closed marker salts resolve the
aliases; all30 patched function indexes are unique. These failures were not
installed or measured as valid candidates. The normal builder's generic source
ZIP duplicated one last WAT; authoritative `source-audit.zip` deduplicates and
hash-checks all sources.

Paid upgrade guards completed all4 cases and restored/deleted their snapshot.
Actual paid proof is now executing using `prove_paid_k2_compact.py`, dedicated
full canister4caro and newly created callers5iptu/5pova, with a new protected
snapshot. It must finish/restore before any conflicting canister work. After
termination, run `report_paid_k2_compact.py`, independent raw-Candid/hash/ZIP audit,
and latest Dense capture/reference verification if adopted. No completion or
new full instruction count is claimed yet. Goal remains active and unmet.

### Actual paid K2 result completed (2026-10-07)

Paid `fa00e510c628491ff2ce3122053b4224bd0b6a4105e97968825afd6c9930fd7e`
completed `prove_paid_k2_compact.py`, `report_paid_k2_compact.py` and independent
`audit_paid_k2_compact_report.py`. All32 hidden (including independent saved-carry
hidden30), 32 conv/KV states, final norm/decision/logits/probability bits match.
31 saved Candid calls independently re-decoded. Refund/replay/conflict/Busy,
upgrade-order/completed-receipt checks and deterministic upgrade guards passed.
Snapshot000d was restored/deleted; baseline cache/pack/module verified equal.
Proof callers5iptu/5pova were stopped by the proof's finally block.
Unique frozen ZIP entries and latest summary/workflow/reference hashes verified.

| Input | Previous gated-normalization total | K2 total | Saved | Reduction |
| --- | ---: | ---: | ---: | ---: |
|617|121595528204|120776822748|818705456|0.673302%|
|620|104903053400|104126447676|776605724|0.740308%|
|653|123736933555|122975466647|761466908|0.615392%|

Workers4/3/4, maximum heap4224647168 for each; all worker handlers below40B.
All three remain above100000000000. Largest residual22975466647 (229.75466647億).
`performance-comparison.json` records genuine total counters, not component gains.
Latest full paid outputs verified, but latest direct Dense capture remains pending.
Do not mark complete or claim latest complete Dense fidelity yet.

Dense counterpart `13d17603d870a3a29d14e3b338a00a93abe00544c2db013248c1b77d8cbc2fe4`
(10,362,890 bytes), `artifacts/delta-capture-k2-compact-v1`, built successfully.
All30 WAT source identities match normal/paid. Only the same two capture hooks
and8 owned/one-stage scheduler edits are added; reversing them reproduces normal
sources. Native source audit passed. Direct proof process is confirmed running
on full4caro with protected snapshot000e and target capture module. Do not start
conflicting canister work. After completion run report_delta_capture_k2_compact.py
and audit_delta_capture_k2_compact_reference.py, verify ZIP/source/restoration,
then update full fidelity pointers. These scripts now correctly expect30 kernels.

Next arithmetic direction after latest direct proof: evaluate rank49 with K2
independent output lanes, compact operands and direct seeded output against actual
current K2 runtime. Older rank49 K4/temporary-output variants were slower; repeating
those identical candidates is not justified. Full100B objective remains active.

### Latest K2 full fidelity completed; rank49 K2 candidate rejected (2026-10-07)

Dense proof `13d17603d870a3a29d14e3b338a00a93abe00544c2db013248c1b77d8cbc2fe4`
completed all72 captures. Independent ascending-K recurrence checked37,748,736
Dense F32 values and all pre-BF outputs. Full capture bytes (initial/final Dense,
Q/K/V/G/B, header and pre-BF values) match frozen `delta-capture-v2` exactly.
All32 hidden including independently carried hidden30 verified. Snapshot000e
restored/deleted, baseline6052cc... cache/pack/module checked equal. Report,
direct-reference audit, unique latest workflow ZIP and all hashes passed.
`delta-capture-k2-compact-v1/goal-fidelity-status.json/zip` now link paidfa00...
and capture13d...: complete_fidelity_evidence=true, all_targets_met=false.
This is the latest fully verified full best:617120776822748,620104126447676,
653122975466647. Historical summaries remain unchanged. Full canisterRunning
baseline6052cc...; small canisterStopped after the following diagnostic.

New rank49 K2/four-output-quartet layout was generated in `s2-k2-kernels-v1`.
Tiles80/16 have9633/3345 locals; all4 direct/seed bodies validate. Independent
96-pattern/shape/token conditions verify raw bijection(962,560 indices), compact
operands, emitted transpose masks and rank49 integer roots. MaxI32bound84271104;
original-sized raw payload preserved. This differs from rejected older K4 rank49
and temporary-output candidates.

`artifacts/s2-k2-probe-v1` module
`6d7798c4e26e7708add4fef9fd50f637d3d3b496fa79b9df6392152ecfe9f822`
benchmarked actual current K2 full runtime control (all30 control WAT sources
identical) against rank49 with compact operands, direct/uninitialized output and
no extra generic finite scan. Single-token case intentionally uses current K2
single fallback; rank49 applies only for multiple tokens. All21 conditions/42
ordinary queries match independent native INT8 reference bits. Candidate is
slower: n48/56/57/59/67 by7.1148/5.9807/10.2760/6.8662/5.9442 percent;
617n87 by4.3628 percent, n109 by4.9504 percent, normaln132(output4096) by1.5893
percent. Small was stopped. Independent replies and frozen unique archive hashes
passed; rejected_performance=true, adopted=false. Do not integrate unchanged
candidate into paid/full inference. Failed source assertions and missing unused
kernel declaration are preserved in separate failed artifact directories.

Four synthetic same-shape multiples-of4 (n8/32/64/88) show extra instructions
40,106,740/37,446,236/33,925,490/31,265,060. Empirical linear fit suggests roughly
41M fixed startup penalty and about110K savings per additional token. This is
inference from counters, not phase attribution; zero-cross extrapolation outside
supported rows is not a performance claim. Next useful step is to isolate/fuse
first-use weight coefficient initialization (current generator separately sets
44 DAG temporaries and49 cached leaf vectors), plus tail-specific work, before
repeating a materially changed rank49 benchmark. The100B goal remains active.

### Rank49 coefficient-slot aliasing isolated and measured (2026-10-07)

`generate_s2_k2_alias_kernels.py` writes final B DAG nodes directly to persistent
weight slots; removes49 separate get/set leaf copies per K2 vector. All query,
dot, reconstruction, transpose and F32 sections remain byte-equal to preceding
rank49 kernel. Symbolic execution of emitted B instructions verifies18,816 cached
slots. All4 Wasm bodies validate; tiles80/16 use9584/3296 locals (49 fewer).
96 independent shape/pattern/token lane/layout tests remain bit-exact.

`artifacts/s2-k2-alias-probe-v1` module
`1251f2defffb2a02068838a50124ff81dbbffd7dd799f79aa6d50fd596c07336`
uses exactly the same diagnostic Rust, current full K2 runtime and30 control
WAT sources as previous rank49 probe. All21 conditions/42 ordinary queries match
native bits. Every8192-row multi-token case saves exactly16056320 instructions
relative to prior rank49; 4096-row case saves8028160, equal
49*2*32*(rows/16)*(cols/256), isolating the removed first-use copies. Single-token
fallback is unchanged. Saved Candid re-decode and unique frozen archive/hash audit
passed; small canister stopped after measurement.

Despite the reduction, rank49 remains slower than current full rank7 K2:
n48/56/57/59/67 by3.913608/3.221748/7.571503/4.250207/3.631177 percent;
617n87 by2.569078 percent; n132(4096outputs) by0.409567 percent.
Not adopted, full best and completed fidelity pointers unchanged. No full inference
performance gain is claimed from this component improvement.

Next: measure whether immutable coefficient preparation plus bounded reads can
remove the remaining startup penalty without violating heap/canister resource
bounds. This is a proposed experiment, not a claim that prepacking or stable reads
will be faster. Keep the100B/full three-input objective intact; it remains unmet.


### Immutable rank49 prepacking and stable-read cost (2026-10-07)

`artifacts/s2-k2-prepacked-stable-probe-v1` module
`99ec1fd0306651040e2e02e6896363b273675d574f33761a2fdf5567d7f2bb5a`
prepares identical I16 coefficient leaves from original weights before measurement.
Four kernels load leaves at first dot use, with9543/3255 locals for80/16 rows.
Independent21 layout conditions verify emitted offsets and integer roots; all21
native/control cases (42 ordinary queries) preserve output bits. Capacity grows
6.125 times: layer3 Q grows from20971520 to128450560 bytes. Inference still reads
heap; no stable-backed full inference is claimed.

Prepacking is slower than current full K2 control: n48/56/57/59/67 by
6.228922/5.217229/9.527583/6.142253/5.304106 percent;617 n87 by3.866446 percent;
n132 with4096 outputs by1.218747 percent. Every tested case is slower, including
single-token fallback by5 instructions due to this diagnostic variant. Not adopted.

Separately sealed coefficient bytes are copied into stable memory. An independent
NumPy reconstruction verifies SHA256 of each returned read prefix. Eight measured
sizes from0 through16777216 bytes cost exactly bytes+220 instructions, excluding
allocation, digest and Candid. A67108864-byte request exceeded the5B whole-query
limit, including SHA256 outside the read interval; preserved in
`stable-read-check-failed-64m`. This failure does not measure its read cost.
The bounded successful run, saved reply re-decode, source hashes and unique frozen
archives passed `audit_s2_k2_prepacked_stable_report.py`. Small canister stopped.

Current full best remains120776822748/104126447676/122975466647 instructions for
617/620/653 with complete fidelity evidence. The100B goal remains unmet, residual
maximum22975466647 instructions. Do not integrate this unchanged prepacked
candidate or allocate its approximately21.86GB expanded full model. Next useful
rank49 experiment is pruning algebraically zero query leaves for padded token
tails while preserving all surviving integer roots and ascending-K F32 carries;
that changes first-use work without increasing model capacity. It must be measured
against current full K2 and all native tail cases before any full adoption.


### Rank49 zero-tail pruning, per-leaf guards (2026-10-07)

`artifacts/s2-k2-tail-probe-v1`, module
`b3cd2ed72d413ea8c8319a5f7b5e71a100e5c8a03199ffce2df410c9a0d5e02c`,
guards49 leaf dots per output group and token quartet. A leaf is identically zero
if every contributing original token index is outside the remaining valid tokens;
tails1/2/3 skip24/14/4 leaves respectively.144 independent integer/layout conditions
verify emitted thresholds, zero operands and all reconstructed roots. Removing
only new guards restores the preceding alias kernels byte-for-byte. All diagnostic
Rust and30 current-full control WAT bodies remain unchanged. All21 cases/42 queries
match native output bits; saved replies and unique frozen archive/hash audit pass.

This implementation is rejected:617 component grows6.564975 percent versus current
full K2 and35768000 instructions versus alias rank49. Full quartet synthetic n8/32/
64/88 grow3512320/14049280/28098560/38635520 instructions versus alias, exactly
439040 per token. Thus repeated per-leaf guards cost more than tail pruning saves
on relevant large inputs. No full implementation change. Single-token fallback
matches preceding alias, current control differs by5 diagnostic instructions.

Next variant `s2-k2-tail-dispatch` emits specialized static paths for1/2/3/4 valid
tokens and dispatches only at quartet entry. Its full-quartet path matches alias
byte-for-byte and its144 integer/layout conditions pass. Four kernels validate,
build95dce5ea361ac826a0a8d9a1bb6a8f079604b67764fed29cd5c8daaf8998680d
is ready; performance and actual output proof remain pending at this point.


Rank49 static tail-dispatch proof completed:21 conditions/42 native/control queries
all bit-equal. Saved replies, source/dependency/input hashes and unique frozen
archive/latest-summary bytes independently verified. Component617 n87 uses
915414276 versus current895118076 (2.267433 percent slower), saving2700080 versus
preceding alias918114356. n57/59 are4.694332/3.801993 percent slower than current,
n56 is3.239261 percent slower, n109 is1.990069 percent slower, normal n132 with4096
outputs is0.427219 percent slower. All tested cases still lose to current full K2.
Not adopted; small stopped; full4caro remains original6052cc94... baseline Running.

Both tail variants provide measured evidence that tail pruning cannot close the
remaining whole-model gap. Full best and all fidelity artifacts are unchanged.
100B goal active. No live build/install/check handles remain from these candidates.
Avoid rerunning unchanged rank49 variants. Future changes need an independently
measured reduction in full-quartet projection/weight setup, rather than expanding
tail specializations whose maximum contribution is now bounded by these counters.


### Rank7 odd-row original-basis experiment (2026-10-07)

Current adopted K2 first-use weight coefficients are already fused with their
first dot; no duplicate set/get elimination remains there. Examination also
confirmed the adopted odd-tail dispatch already retains4 dots, not5. An initial
generator asserted5 and stopped; preserved at `k2-odd-four-kernels-failed-count1`,
not installed. Revised candidate replaces4 transformed query leaves0/1/2/5 with
2 original query leaves0/1; cached weights recoverB01 asw0+w4. Dot count stays4,
but first-query loads reduce from4 to2 streams. Output integer sums become
C00=A00*B00+A01*B10; C01=A00*(B00+(B01-B00))+A01*B11.192 independent original-dot
and cached-weight integer cases match, bound4161536. First/full pair instruction
prefix and loop suffix are byte-equal; original F32 conversion/scales/add/store
body remains unchanged. All8 replacement exports uniquely validate.

`artifacts/k2-odd-four-probe-v1`, module
`396adeb317a42e1ed9c8200222744875b62cf762c3944a8c86b426073f2d2a95`,
is the completed original rank49 diagnostic with only its8 current-full rank7
control WAT bodies patched. No Rust or runtime dependency change. It calls original
method3; rank49 method4 is unused. Each input/native digest and original control
reply is checked against the completed same-method/module predecessor.21 fresh
ordinary queries and21 independently redecoded saved baseline queries match all
native output bits. Even-token cases and n1/n2 have identical instruction counts;
all odd-token n>=3 cases add exactly908800 instructions.617 grows from895118076
to896026876 (0.101529 percent); n57 adds0.153078 percent, n109 adds0.081419 percent.
Repeated per-dot B01 recovery costs more than saved query loads/reconstruction.
Candidate not adopted; saved Candid/source/dependency/input hashes and unique
frozen archive/latest-summary bytes all verify. Small stopped after completion.
Full best and completed fidelity evidence unchanged;100B goal active.

A useful narrower derivation for future kernel work is retaining the current
4 cached Winograd dots while directly reconstructing only the2 valid first-row
roots (p0+p1 andp0+p5+p2), avoiding zero-product/general second-row reconstruction.
It cannot by itself close the23B gap. Full-quartet arithmetic remains the primary
optimization target; do not adopt original-basis odd-row variant unchanged.


### Rank7 odd-row valid-root specialization (2026-10-07)

`artifacts/k2-odd-roots-probe-v1`, module
`7fcdf3c96dceda4e55b8946de6efa4154aa8e7c9e61c0504772c39798ccfd834`,
retains original4 cached Winograd products0/1/2/5 and replaces general output
reconstruction with only valid roots p0+p1 andp0+p5+p2. Removes assignments to
zero products3/4/6; original dot, input-load, first/full-pair prefix, loop suffix
and F32 conversion/shuffle/scales/add/stores remain unchanged.192 independent
integer cases match original256-term dots; all8 replacement bodies validate.
The first generator had a Python parenthesis typo and failed before creating
artifacts; corrected before generation/build/install.

21 fresh same-method queries plus21 redecoded archived baseline replies all match
native bits. Every odd n>=3 Qcase saves exactly225280 instructions; even cases,
n1 andn2 are unchanged.617 Q895118076 becomes894892796 (0.025168 percent); n57
593686127 becomes593460847 (0.037946 percent). Unique frozen source/input/reply
archive and current-summary bytes verify; small stopped after completion.

Normal full module6ee013ea50d51de4905b7d464e757362b3234226846bb9cb6031398a633f8cc0
built successfully (10343743 bytes). All runtime and wrapper Rust is byte-equal
to completed K2 normal; original22 WAT unchanged; only8 K2 odd-tail bodies change.
300 completed native K2 layout/carry cases remain applicable since runtime Rust
is identical; new192 integer and21 actual component conditions are separately
recorded. Generic builder sourceZIP repeats a final WAT entry; authoritative
source-audit.zip is unique and all hashes verified. No full inference claim yet.
Paid wrapper build started; full3-input paid counters, full hidden/state/final
fidelity and upgrade/receipt checks are pending. Best full paid remains
120776822748/104126447676/122975466647;100B goal still unmet/active.


Paid odd-roots build completed: module
`ae5b713a84cd4f23c10aae84354a00699bd6b135c54a174ca6dfb9dc48fed432`,
10716489 bytes. Paid scheduler, canonical billing and wrapper sources byte-equal
to proven versions; all30 patched bodies match normal candidate; independent
source/dependency hashes verified and source-audit.zip saved. Upgrade guards
started on dedicated full4caro via snapshot-protected existing recovery workflow;
current exec session42117 remains live. Continue polling that handle, do not
restart. After successful guards/restoration, run prove_paid_k2_odd_roots.py for
3 actual inputs, then report and saved-reply audit. Full performance/fidelity
still unproven; Dense direct capture counterpart will remain required after full
paid proof if the candidate is retained. Goal remains active/unmet.


### Paid odd-root full proof completed (2026-10-07)

Upgrade guards session42117 terminal: active and refund upgrade refused; Pending/
Done receipts preserved; baseline6052cc94... cache/pack restored, snapshot000f
deleted. Actual paid proof session81993 terminal:617=120776484828 (4 workers),
620=104126447676 (3),653=122938605207 (4), maxheap4224647168 bytes and each handler
below40B. Savings versus completed K2 best:337920/0/36861440 instructions. Full
maximum residual22938605207; all_targets_met=false. Actual caller canisters
5gn64-6l777-77775-aaaha-cai /5bmyi-tt777-77775-aaahq-cai stopped in finalization.
Full baseline/cache/pack restored, snapshot0010 deleted.

All32 hidden (hidden30 independently reconstructed from saved exact carry),32
conv/KV states, final norm/decision/logits/probability bits match. Paid boundaries,
refund/duplicate/conflict/Busy/worker authority, concurrent upgrade ordering,
completed receipt and upgrade replay checks pass. Reporter59625 terminal;
independent audit redecodes31 saved raw Candid calls, verifies worker sums,
source/reference/workflow hashes, unique latest frozen-paid-proof.zip and restored
baseline. An intermediate reporter ZIP warns duplicate summary; final frozen-paid-
proof.zip is unique and latest-summary bytes verified. Dense capture build started
from this paid arithmetic; direct72 capture/state recurrence proof still pending.
100B goal active and unmet; do not substitute microcomponent success for full goal.


Dense odd-root correctness counterpart build2766 terminal:module
`e91ad7080e201fc4ca53731d0a22d941aab57fddefd9eefc3e249070418cb6d0`,
10351306 bytes. Source audit confirms all30 WAT sources equal normal/paid, only
2 capture hooks and8 scheduler edits, reversal byte-exact to normal; owned graph
changes match verified paid scheduler. Full baseline6052cc94... Running verified
before capture. Snapshot-protected capture proof started, current exec31875 live;
continue that handle without restarting. On terminal success, run
report_delta_capture_k2_odd_roots.py, audit_delta_capture_k2_odd_roots_reference.py,
then report_goal_k2_odd_roots_fidelity_status.py. Direct72 Dense recurrence values
and frozen complete capture equality remain unproven until those checks pass.
New runtime-comparison and scheduler-comparison metadata reconstructed from
verified current byte-equal runtime/wrapper sources for reporting/capture audits.


### Complete odd-root Dense fidelity; memory64 capability (2026-10-07)

Capture proof31875 terminal:all72 captures/32 heads compare37748736 final Dense
F32 state values and pre-BF outputs with independent ascending-K recurrence.
All32 hidden incl independent saved-carry hidden30 match. Snapshot0011 restored
baseline6052cc94... cache/pack equal and deleted; reporter47904 terminal. Independent
reference audit verifies whole capture.bin bytes (initial/final states, Q/K/V/G/B,
headers and pre-BF outputs) equal frozen reference for all72, plus current source/
reply hashes and archives. New goal-fidelity-status.json/zip link paid ae5b713a...
and capture e91ad708... with complete_fidelity_evidence=true,all_targets_met=false,
goal_complete=false. Full latest120776484828/104126447676/122938605207 remains
above100B; maximum residual22938605207. No live proof/build/report handles remain.

A new storage capability investigation checks the official current resource-limit
reference https://docs.internetcomputer.org/references/resource-limits/ :wasm32
heap4GiB, wasm64 heap6GiB. Existing4GiB prototype capacity findings do not establish
that every larger-memory approach is impossible. Installed rustc1.97.1 includes
wasm64-unknown-unknown target definition but only wasm32 target std is installed.
No toolchain/component installation or model migration performed.

scripts/probe_memory64.py emits a156-byte binary with memory64 initial1 page,
i64 ic0.msg_reply_data_append pointers and no model data. Dedicated small4xhad
successfully installed module1c6f2f9c8244a2ccf65148d2b45b912c4f13ade6a91a6fd77b17615bdff8a920;
query reply exactlyCandid nat64=64, module hash/status match. saved reply/source/
Wasm hashes in artifacts/memory64-capability-v1/verified.json and evidence.zip.
This proves local memory64 and i64 pointer ABI only, not above4GiB allocation or
inference performance. Small stopped; full capture/baseline unaffected. It opens
a resource option for materially changed immutable-coefficient prototypes; the
old rank49 I16 full expansion (~21.86GB) still exceeds6GiB and is not adopted.
Large-memory migration must follow component cost and exact-capacity evidence;
never omit helper/transfer/callback instructions from paid worker totals.


### Exact mixed-width coefficient capacity evidence (2026-10-07)

scripts/analyze_precomputed_coefficient_capacity.py verifies full pack SHA,
MODEL_LOCK and manifest, then reads actual layer3 Q/gate/down INT8 weights.
Four immutable B plans:rank7 Winograd,rank49 Winograd,rank343 classical andrank343
Winograd. K2/four-output-lane layout preserves original row/K positions. All12
real tensor/plan cases decode exactly using signed I8 narrow vectors, little-
endian I16 escape vectors and one-bit selectors; no coefficient saturation/rounding.
Streaming grouped computation keeps temporary allocations bounded. Same worktree
and baseline unchanged; no canister/model migration or input-dependent prework.

For Q/gate/down, mixed vector payload+flags ratios respectively:
rank7 1.920404/1.960558/1.893083;
rank49 3.927629/4.062134/3.859380;
rank343 classical7.748671/7.995808/7.428808;
rank343 Winograd7.961584/8.225425/7.843133.
Selectors/escape lists decode bit-exact; random-access indexes, allocator metadata
and runtime/working buffers omitted, so these are capacity lower estimates, not
inference memory/performance claims. Whole inference projection raw bytes3569090560;
other resident non-embedding tensor bytes496668160. Embedding is already excluded
from this heap accounting; do not infer an extra embedding-sized free region.

Even ideal all-I8 coefficients plus other resident tensors use6742576640 bytes
forrank7,11427008000 forrank49 and19624762880 forrank343; all exceed6GiB before
selectors, escapes or work buffers. This rejects whole-model resident coefficient
preparation under that storage policy, not every selective/distributed policy.
Evidence artifacts/precomputed-coefficient-capacity-v1/report.json,evidence.zip
and post-audit.json verify allsource hashes,12 roundtrips/capacity arithmetic and
unique archive/latest report bytes. Analyzer98044 terminal. No performance gain
claimed. Candidate experiments must use explicit bounded/sharded capacity and
include every helper/transfer/callback instruction in actual paid worker totals.
Current full best120776484828/104126447676/122938605207, all72 Dense/full fidelity
complete;100B goal active and still unmet.

The next substantial arithmetic candidate is rank343 with K2/four independent
output lanes, rather than the previous K4/two-output partial-lane prototype.
It can remove partial horizontal reductions and halve physical reconstruction
replication. A single32-output tile needs343*16=5488 cached weight vectors and
no query-local copies across output subtiles, keeping locals below10000. This
is a derivation to test, not a speedup claim; integer ring roots and ascending-K
F32 carry must first match an independent reference, then actual current full
K2 counters. Prepared capacity remains explicit; do not attempt unchanged old
rank343 or silently exclude immutable/helper read work from the paid objective.


### Rank343 K2 actual projection rejection (2026-10-07)

New scripts generate_s3_k2_prepared_kernels.py/audit_s3_k2_prepared_layout.py/
validate_s3_k2_prepared_kernels.py generate direct and seed 32-output K2 kernels:
5488 cached weight vectors,6192 locals, four independent output lanes, no partial
horizontal reductions. All60 host layout/register/shuffle/F32 conditions match
ascending-K original-dot oracle, including157616 actual uint32 wrap operations.
Both kernels Wasm-validate. Actual source prep preserves QuantizedRows padding8;
input pointers use343 leaves and groups*cols/8 I16 strides, weight block87808 bytes.

Initial isolated moduleff526aa1... installs, but immutable full coefficient seal
exceeds40B update limit (IC0522) before any inference query; preserved failed-seal
report and16 raw chunk replies in artifacts/s3-k2-prepared-probe-v1. Not adopted.
Changed preparation to512 immutable rows per update, explicit cursor/seal guard.
Chunked module35145b7cf64a9a2558bffb28dce585bd4eb1135caccff6b67259c3642b7cd4c5
prepares all8192 rows in16 updates; largest prep5857688192 instructions. Prepared
Q storage224788480 bytes; whole resident model expansion still fails capacity.

Checker66263 terminal:21 native/current-control pairs,42 actual projection queries
all exact output digest bits. All30 control WAT bodies equal latest normal odd-root
implementation. All21 candidate costs worsen. Saved617 Q total894872565 ->1038321108
(+16.0300526%); n132680498698 ->782289726 (+14.9582987%); size67693925027 ->864653236
(+24.6032644%). For617 quantization2737441 equal; current input prep220 and projection
892134904; candidate input prep6633544 and projection1028950123. The regression is
inside projection as well as prep, not simply input preprocessing. Fixed prep is
reported separately; no helper/input-dependent computation excluded from inference.

Independent audit redecodes all75 saved Candid replies (33 prep +42 query), counter
sums/output shapes/native digests/control sources and latest unique evidence ZIP.
artifacts/s3-k2-prepared-chunked-probe-v1/post-audit.json and frozen-evidence.zip.
Dedicated small4xhad stopped; full4caro baseline6052cc94... remains Running; saved
verified-final-state.json verifies status/archive hashes. No live process handles.
Reject this rank343 K2 candidate; do not rerun it unchanged. Best full paid remains
120776484828/104126447676/122938605207 with complete fidelity;100B goal active,unmet.


### K2 dominated pair guard folding (2026-10-07)

Previous goal turn yielded completed evidence rejecting rank343 K2, hence progress.
Current candidate removes keep guards only in the loop full-pair true branch and
odd-tail false branch of the exact identical predicate t+1<n. First pair byte-equal
(including n1 safety), full-pair integer/F32 operations byte-equal, tail second-row
loads/stores unreachable and deleted; tail yp1 address deleted. 4290 allowed row/
loop predicate conditions enumerated. Eight kernels preserve data layout/capacity.
No source/runtime dependency/model/input/prefix/quantization changes.

Component module006d7ee98f5fce473a4ff1cc15b051ae244e9f5b71a889e0f145b5aea03e4d7f
checker70365 terminal:21 fresh method3 queries match independent native and archived
latest odd-root same-method3 replies. All19 n>=3 conditions improve; n1/n2 exact same
cost. Saved617 Q894892796 ->893380246,saves1512550 (0.1690202454%); size67693945104
->692783854,saves1161250. n132680498852 ->679348352,saves1150500. These are the same
original diagnostic Rust/runtime counters, not the separately compiled S3 canister
counters (there is a small wrapper cost difference); do not compare those absolute
values across different diagnostics as if identical wrappers. Quantize/prepare/
project included, output digest/Candid excluded as in the saved same-module control.

Reporter independently redecodes21 fresh+21 saved raw Candid measurement replies,
checks native digest/counter sums/current sources/unique frozen-workflow.zip.
Entry builder/checker/reporter hashes separately verified. Artifacts in
artifacts/k2-pair-guard-fold-probe-v1/{summary.json,frozen-workflow.zip,verified-final-state.json}.
Small4xhad stopped; full4caro6052cc94... unchanged. No live handles.

Normal full candidatec499cd03b6c94b1f726929600801b128660e3a95ccdf4f65d87b32e391855c5b
10328293 bytes; builder12674 terminal. Independent source-audit verifies allruntime/
wrapper Rust byte-equal previous odd-root normal, original22WAT equal, only8guard-
fold bodies changed, latest unique source-audit.zip and21 component bits. Generic
original source.zip warns one duplicate finalWAT entry; unique source-audit.zip is
authoritative. New audit key component_native_bits_equal replaces old native_bits_equal;
paid builder/auditor must use that key when adapting. No paid full proof or Dense
capture yet, so candidate not adopted and no full-inference gain claimed. Next
required work:build canonical paid wrapper with unchanged owned scheduler; source
audit; snapshot-protected upgrade guard and full617/620/653 performance/fidelity;
Dense72 capture direct recurrence counterpart and goal status report before adoption.
Best fully verified paid remains120776484828/104126447676/122938605207 with complete
fidelity.100B goal active, unmet. This source-only branch optimization is incremental,
not a replacement for the original all-input100B objective.


### Paid guard-fold build and real upgrade guards; full proof live (2026-10-07)

Previous goal turn was progress:21 component bit checks and full normal source
build/audit completed. New workflow scripts freeze six paid derivatives and six
Dense derivatives from the completed odd-root workflow, provenance entry hashes
saved in k2-pair-guard-fold-paid-workflow-v1 and dense-workflow-v1. Normal new audit
uses component_native_bits_equal; adapted paid builder/source audit read that key.
Runtime-comparison.json built from unchanged Rust/first22WAT plus exact eightguard
fold changes; paid scheduler-comparison copied with byte-equal scheduler check and
current source-audit SHA, preserving unchanged scheduler reference metadata.

Paid builder22615 terminal module25277c772d8d32bb1ccf6c63669095d94ce4e37d35b450c9a6162f4bd49c4e3d,
10700911bytes. Source audit terminal:canonical billing and paid wrappers equal
proven canonical implementation; all30 WAT sources equal new normal, scheduler
byte-equal. Upgrade guard50609 terminal:active/refund upgrades refused, Pending/Done
failedreceipts preserved. Snapshot0012 restored6052cc94... module/cache/pack and
deleted. Allfour verified true in upgrade-guards/verified.json.

Full proof exec63642 is CONFIRMED LIVE (write_stdin polling returns samehandle),
not terminal. Snapshot saved in proof/snapshot.json; module installed. Callers
74rwa-a3777-77775-aaaia-cai and73qqu-nd777-77775-aaaiq-cai installed. Currently running
prepare_weight_cache.py under proof/preparation.log; authoritative updates.jsonl
contains actual completed preparationcalls. DO NOT restart or re-run the proof.
Continue polling session63642, with commentary <=60s. Its finally restores/deletes
snapshot and stops callers. Full canister4caro currently temporary paid candidate;
do not claim baseline already restored from this live full proof. Small4xhad is
stopped with component006d7ee...; no mainnet/otherprojects touched.

On full proof terminal success:run report_paid_k2_pair_guard_fold.py then
audit_paid_k2_pair_guard_fold_report.py. Then build_delta_capture_k2_pair_guard_fold.py,
audit_delta_capture_k2_pair_guard_fold_sources.py, prove_delta_capture_k2_pair_guard_fold.py,
report_delta_capture_k2_pair_guard_fold.py, audit_delta_capture_k2_pair_guard_fold_reference.py,
report_goal_k2_pair_guard_fold_fidelity_status.py. Full paid and Dense required
before any adoption/newbest claim. Current fully verified best still
120776484828/104126447676/122938605207;100B goal active, unmet. No new worker counts
for candidate yet. Live proof/snapshot must be carried across goal turns.


### Guard-fold paid full proof complete; Dense proof live (2026-10-07)

Previous goal turn progressed paid build/guards; this turn completed full paid
proof63642 terminal. Snapshot0013 restored baseline6052cc94... module/cache/pack and
deleted; both callers74rwa/73qqu stopped in finally. All3 paid inputs measured:
617120619170432 (4 workers),620103992353622 (3),653122775162875 (4);
maxheap4224647168bytes. Saves157314396/134094054/163442332 from oldodd-root complete
paid totals. All remain above100B; residual max22775162875 (227.75162875億).
Reporter1614 terminal:32hidden including independent exact-carry hidden30,32conv/KV
hashes,finalnorm/decision/probability bits, refund/duplicate/conflict/Busy/authority/
concurrentupgrade/completedreceipts verified. Independent post-report audit terminal:
32 raw saved Candid calls redecoded, allworker sums/workflow/reference hashes and
latest unique frozen-paid-proof.zip/summary/restoration verified. Reporter emits
an intermediate duplicate summary ZIP warning; final uniqueZIP byte-audited.

Capture builder47374 terminal,module933bbbb8cfb0207f71486611c86382518b590fbbe92e88e77506b5cfebfe2852,
10335796bytes. Source audit terminal:all30 WAT match normal/paid, runtime differs
only two capture hooks+module export; scheduler eight declared capture edits reverse
exactly to normal, ownedgraph matches paid scheduler comparison. No performance
claim for correctness-only one-stage capture.

Capture proof session57341 CONFIRMED LIVE by write_stdin returning samehandle;
continue that handle WITHOUT restart. Full4caro may now hold temporary capture
module/snapshot, check proof/snapshot.json and proof logs for exact current state.
Do not claim capture completed or baseline restored until its finally and report
prove that. Last full paid baseline restoration was before this live capture.
Required after successful terminal capture:report_delta_capture_k2_pair_guard_fold.py,
audit_delta_capture_k2_pair_guard_fold_reference.py, then
report_goal_k2_pair_guard_fold_fidelity_status.py. The 72directDense/preBF captures
for this new module remain unproven until those checks pass. Goal100B active/unmet,
do not adopt candidate or claim complete fidelity based on paid-only evidence.
No other live build/proof/report handles from this turn.


### Complete guard-fold Dense fidelity; new rank49 leaf-outer design (2026-10-07)

Capture proof57341 terminal:all72 captures/32heads compare37748736 Dense F32
state values and pre-BF outputs against independent ascending-K recurrence. All32
hidden including independent saved-carry30 match. Snapshot0014 restores/deletes
baseline6052cc94... module/cache/pack; final-status.json verifies Running baseline.
Reporter48850 terminal; directreference audit verifies all72 capture.bin wholebytes
(initial/final states,Q/K/V/G/B,headers,preBFoutputs) equal frozen originalreference.
New report_goal_k2_pair_guard_fold_fidelity_status.py terminal links paid25277...
and capture933bbbb... with complete_fidelity_evidence=true,all_targets_met=false,
goal_complete=false; latest unique statusZIP/report/source hashes checked. Newbest
fully verified paid120619170432/103992353622/122775162875; maxresidual22775162875.
No live proof/build/report handles remain from this turn.100B goal active/unmet.

New materially changed rank49 candidate:leaf-outer scheduling allows sharing only
32 query vector locals instead of49*32=1568 across all output groups. Immutable
B49 coefficients prepared/cached for all output groups before first tokenquartet;
each m leaf's full32 K2 dot sequence then runs over allj. Retain49 leafresults perj;
C scratch slots49..94 shared only after allleafdots complete, outputj emitted
before nextj scratch overwrite. No chunked integer summation or rawcapacity change.
This makes outputtile96 fit9865 locals versus previous80tile9584. Tail16 uses1760.
Prior unchanged rank49 variants still rejected; this is a new loop/lifetime design.

scripts/generate_s2_k2_leaf_outer_kernels.py emits4 direct/seed kernels in
artifacts/s2-k2-leaf-outer-kernels-v1. validate_s2_k2_leaf_outer_kernels.py confirms
all4 Wasm bodies,module5fa38da29292c87b58db51fb9fc785bf62ea6cb029be29947c5dae5934846394.
audit_s2_k2_leaf_outer_layout.py completes96 host native integer/layout conditions,
1150976 rawindices,actualtranspose masks,49 exactleafs and emitted allj leaf
assignments/Cscratch translations/querypointerloads, unchangedrawcapacity.
The audit has wasm_execution_verified=false/performance_verified=false: no actual
component query performed, no speedup claim or adoption. Do not mistake hostproof
for fullgoal. Nextsafe step build new S2 diagnostic from build_s2_k2_probe.py using
new kernels/raw layout, current normalfold c499... control, tile96 dispatcher/exports
in place of80, then fresh method3/method4 native42queries including shorttails.
If no gain reject; if useful, verify gate/down before any fulladoption. Fixed
weights and quantizer are identical; all inference quantize/prep/project work stays
in counters. No newcanisters/worktrees/mainnet/otherproject modifications.


### Rank49 leaf-outer96 actual component result (2026-10-07)

Previous goal turn was progress:completed full guard-fold Dense fidelity and
new96-wide kernel host/Wasm proofs. Current turn builds materially changed
leaf-outer96 diagnostic0b691b85ac8dfbfe2782e632ea64b96cf184a29a27c6bf9a657059b54badc2a5
with all30 current guard-fold control WAT equal verified normal c499.... Build19961
terminal, install64742 terminal. Same rank49 rawlayout/compactquery transform,
only tile96 dispatcher/exports andnew49leaf-outer WAT changed.9865 maxlocals.

Checker51591 terminal:21 fresh current/candidate pairs (42 actualqueries) match
independent native bits. Only n132/4096-output condition improves:679348352 ->
676398702,saves2949650 (0.43418814%). All20 other conditions worsen, including
n1 fallback dispatcher +5 instructions. Saved617Q893380246 ->905098714 (+1.31170%).
Actual target suffix counts from latest paid quotes:61756,62048,65357. Corresponding
Q shape probes n56581015379 ->592420429 (+1.96295%), n48500764471 ->514090329 (+2.66110%),
n57592475247 ->629759245 (+6.29292%). Those shapeprobe inputs are savedactivation
subsets, not a new fullpaid run. One unrelated n132 microcondition gain does not
establish movement to target; candidate not adopted. No gate/down/fullpaid run.

Reporter independently redecodes all59 saved Candid replies (17 fixedpreparation+
42 measurements), checks quantize/prep/project counter sums/native digests/shapes,
all30 currentcontrol WAT, four newcandidate WAT, code/dependency/entrygenerator
hashes and latest unique frozen-workflow.zip. Artifacts/s2-k2-leaf-outer-probe-v1/
summary.json andverified-final-state.json. Small4xhad stopped; full4caro Running
baseline6052cc94... unchanged. All processes/handles from this turn terminal.
Reject adopting this unchanged rank49leaf-outer96 for goal. It eliminates a query
local capacity limitation, but the observed short/midtoken cost still fails.
Future variant needs a materially different cost mechanism and fresh comparison,
not rerunning unchanged layout. Full best remains120619170432/103992353622/
122775162875 with complete guard-fold paid/Dense fidelity, goal100B active/unmet.

### Rank7 quad-query-load actual component result (2026-10-07)

Current goal turn progresses through a new exact-byte query load substitution:
four load32_splat operations become one v128.load plus four byte shuffles.
Eight method3 WAT bodies change; all Rust/runtime and other kernel bodies remain
identical to completed guard-fold diagnostic006d7ee98f5fce473a4ff1cc15b051ae244e9f5b71a889e0f145b5aea03e4d7f.
Candidate d7d2db3893ed16b9da34dcff640ee0314efe4b9ee6e5974b31c62d9e90b0310f,
maxlocals9930. Build27365, install56562, checker2881 and reporter all terminal.
512 full-range I16 host byte cases pass; 21 fresh method3 queries and21 archived
same-method control replies independently redecoded match native bits exactly.

All20 nonfallback conditions worsen; n1 remains equal. Saved617Q893380246 ->
926857046 (+3.747206%). Target suffix shape probes n48:500764471 ->519204151,
n56:581015379 ->602528339,n57:592475247 ->614427247. These are component input
subsets, not full paid inference measurements. Candidate rejected, not adopted.

audit_k2_query_quad_load_cost.py checks all source/entry/workflow hashes and
unique frozen-workflow.zip, then reconciles every observed delta exactly:
14 * row_tiles *10 Kblocks * (7*floor(n/2)+4*(n%2)) *16 packets,
with n1 fallback zero, row_tiles49 for8192 rows and26 for4096 rows.
Old four splats cost12, new load/shuffles cost26 perpacket. Official upstream
IC instrumentation source supports the operation costs used here:
https://github.com/dfinity/ic/blob/master/rs/embedders/src/wasm_utils/instrumentation.rs
Upstream master is not pinned to installed runtime; this is an observed exact
delta audit only, not a general offline instruction interpreter. No such broader
interpreter implemented. All current normal30 WAT scanned: no shuffle whose
inputs are identical and no shuffle using only one source, so no trivial such
replacement selected. Do not repeat unchanged quad-load design for this goal.

Artifacts/k2-query-quad-load-probe-v1/{summary,cost-audit,verified-final-state}.json.
Small4xhad verified Stopped with rejected candidate; full4caro verified Running
baseline6052cc94... unchanged. Best fully verified paid/Dense totals remain
120619170432/103992353622/122775162875, maximum residual22775162875.
All3 targets still above100B. Goal active/unmet; no pause/completion/blocked claim.

### Rank7 pure address arithmetic fold (2026-10-07)

Previous goal turn progressed through quad-query-load rejection and exact cost
delta audit. Current turn creates a new pure i32 address candidate. No unused
local-get variables found in the current eight kernels; arithmetic folding is
limited to local reads multiplied by0/1/powers of2 and add-zero. All replacements
preserve modular2**32 behavior, cannot trap, and contain no memory/F32/SIMD ops.
4389 random and boundary modular cases checked. Eight kernels192 replacements;
all SIMD arithmetic, loads, stores and F32 lines remain byte equal.
Build95666 terminal validates eight unique patches. Module
072d5e2467fedf0d3162c49c86220bc7744dbb106453ae81c8e04e2da287ec8a.

Two reinstall attempts78133/54145 terminal failed because init arguments lacked
principal, then rows/cols. Both were diagnostic-only initializer traps, not
measurement failures. Correct owner+8192nat32+2560nat32 reinstall99803 succeeds.
Small canister started, checker66757 terminal:21 fresh method3/native conditions
match saved same-method guard-fold baseline bits. Reporter87901 terminal
independently redecodes42 current/saved Candid replies, all counter sums, source/
dependency/entry hashes and unique frozen-workflow.zip verified.

Every19 multi-token8192-row condition saves exactly7840 instructions (49tiles*160).
n132/4096 saves4160 (26tiles*160); n1 fallback equal. Saved617Q893380246 ->
893372406, only0.0008775659 percent. Target suffix-shape n48/56/57 save7840 each.
This is a small constant initialization saving, not a token-dependent reduction
and not a full paid result. Candidate retained as component evidence only; not
adopted/fullpaid-tested. Do not use it to claim movement of22775162875 residual
without full evidence or repeat unchanged as a substantial optimization.

Current verified local states:small4xhad Stopped on072d5e...;full4caro Running
baseline6052cc94... untouched. All current handles terminal. Results at
artifacts/k2-address-fold-probe-v1/{summary,verified-final-state}.json.
Fully verified goalbest remains120619170432/103992353622/122775162875.
Earlier eleven-phase diagnostics identify base projection as83609740295 to
98725977617 instructions; these older profiled modules are not current totals.
Remaining target requires a major base-projection change, not just the new
constant address saving. Goal remains active and all three targets unmet.

### Rank343 mixed scheme and batch48/tile128 prototype (2026-10-07)

Previous goal turn was progress: address-fold component21 conditions measured,
but the constant saving is too small for100B. Current turn changes the next
kernel architecture rather than repeating an unchanged measured rejection.
screen_rank343_mixed_schemes.py freezes all8 per-recursion-level classical(C)/
Winograd(W) combinations. Each passes64 coefficient identities and symbolic
execution of the emitted I32 reconstruction register program. Existing current
WWW already inlines single-use C nodes. WWC lowers reconstruction1978 ->1880
instruction lines, while input transform nodes372 ->421. These are structural
counts, not IC speed claims. Input/weight I16 bounds4064/4096; all final dots
retain signed bound4161536. Other C orders tradeoffs recorded, not tested onIC.

New generate_s3_k2_batch48_kernels.py uses WWC and retains343*6*4=8232 leaf
products while caching only16 current query vectors and64 current B vectors.
Each leaf serves6 token octets and4 output quartets; outputtile128, tokenbatch48.
8642 total locals, below10000. This replaces previous5488 full-B locals/tile32
with shared leaf operands and broader output/query sharing. F32 scale and K256
block-add sequence unchanged. WABI requires four tile32 weight base pointers,
not the former single pointer; no existing runtime installed with this ABI.

validate_s3_k2_batch48_kernels.py session14740 terminal validates two Wasm bodies;
stub module d3f702d02f240dde27eb235dc3ef5e2596dcd61d6fd58f32eeb53a50f7d04870.
Host auditor first8408 terminal failed because expected stores assumed token-
major order rather than actual independent output-group order. Corrected
expected index order;51331 terminal succeeds48 conditions:cols256/512,n0/1/7/8/
9/47/48/49/56/57/96/97,seed/nonseed, raw scalar ascendingK dots, modulo32 emitted
reconstruction and F32 block scale/add bits exact. All24 product/reconstruction
maps and every B/store offset checked. Actual Wasm shuffle output execution,
IC counters, production resident capacity and fullpaid/Dense fidelity remain
unverified. This is a new prototype, not an adopted or achieved target.

Next concrete work: build a dedicated diagnostic from the chunked prepared
builder with WWC coefficients, current guard-fold control, tile128 dispatcher,
four weight pointers and matching current ABI. Q rows8192/4096 are divisible128.
Any residual tile32 path needs matching WWC kernel or explicit current fallback,
not pointer reinterpretation. Actual quantize/prep/project counters must include
all input-dependent work; immutable-only preparation is distinct. Preserve
the existing full model capacity constraints: batch kernel does not establish
whole-model prepared residency. Do not infer whole-model adoption from Q tests.

Artifacts/rank343-mixed-scheme-screen-v1 ands3-k2-batch48-kernels-v1. All handles
terminal, no canister mutation this turn. Full4caro freshly verified Running
baseline6052cc94...; tracked diff empty. Disk16GiB free observed, no files deleted.
Fully verified best120619170432/103992353622/122775162875 unchanged, goal active.

### Batch48/tile128 actual diagnostic installation and live comparison (2026-10-07)

Previous goal turn progressed:8 mixed schemes, new8642-local batch kernel,
48 host conditions and two Wasm bodies validated. Current turn builds dedicated
WWC coefficient diagnostic against all30 current guard-fold control bodies.
build69957 terminal60c12752557910596ac856d653315ebf948787c346aec0f651a4fdd1fd02f0c3,
32patches. Four tile32 base pointers and tile128 shape check independently audited.
Additional audit_s3_k2_batch48_stores.py verifies3072 emitted stores' exactroot
references, actual three byte shuffle masks, row/output scales and outputoffsets.
One initial auditor line-index assumption failed; robust first-store offset
parse corrected and complete current store-audit.json recorded. No timing claim.

Original install2662 terminal IC0505:code section13134079 exceeds12582912 bytes.
Failure saved atartifacts/s3-k2-batch48-probe-v1/failed-install.json; no measurements.
Do not retry the unchanged oversized module. Shared variant eliminates duplicated
seed body, initializes entire output to positive0 inside measured project, and
uses the normal body for every K block. First-block floating add has identical
positive-zero carry. Input-dependent zero allocation/fill and outputload costs
stay inside project counters; no computation externalized.

Shared builder65268 terminal,31 patches (same30 control+one normalbody),8642locals.
Module7ebe33e41d1969eb1ecc6c6c5a6d5a60f47ffb1e9f9e27e08c12da707d40792b;
code-section10711546 <=12582912 directly parsed and recorded incode-size-audit.json.
install62706 terminal success;own small4xhad started. Own full4caro status50147
terminal freshly confirms Running baseline6052cc94..., unchangedcache/pack status
not freshly read this turn. No full mutation, snapshots, mainnet or otherproject
changes. Tracked diff empty before new untracked scripts.

LIVE checker77692:check_s3_k2_batch48_shared_probe.py. Lastwrite_stdin confirms
session still running.33 preparation replies saved through0033-seal.hex;16 fixed
raw chunks+16 bounded coefficient updates+seal complete. At6 decoded coefficient
replies maximum5111806592 instructions, all <40000000000. Final all16 must be
independently decoded after completion; do not infer final max solely from sample.
 First boundary8 pair now passes native digest equality, but worsens99509601 ->
 198289727 (+99.26693 percent). Short-token overhead is a real adverse result,
 not a speedup. size67 also passes native equality but692763777 ->867447473
 (+25.21548 percent);prefix45:472048105 ->584314697 (+23.78287 percent).
 Remaining shapes including target48/56/57 remain to be measured. Freshcurrent
 control numbers differ from old diagnostic wrapper, so use actual paired replies.
 Nextturn
poll SAME77692, never rerun checker while live. If complete, run
audit_s3_k2_batch48_shared_report.py (expects31 patches/75 redecoded replies),
inspect all21 current/native/candidate results and current hashes/unique archive,
then stop ownsmall and verify fullbaseline. If failure preserveexactraw failure,
repair based on evidence. Earlier unshared checker script never executed.

Actual computation counters, production prepared-residency capacity and full
paid/Dense fidelity remain unproven. Current install is not adoption. Do not
mark goal achieved/blocked/paused. Full beststill120619170432/103992353622/
122775162875; active goal alltargets100B unmet.

### Batch48/tile128 rejection and local meter calibration complete (2026-10-07)

Previous turn progressed through repaired IC installation and live measured
comparisons. SAME checker77692 now terminal:21 current/candidate/native pairs,
42 actual projection queries, all bits/digests equal. Every condition worsens.
Fresh617Q893360015 ->1025126785 (+14.749571 percent). Target suffix shape48:
500744394 ->586197100 (+17.065135%);56:580995302 ->715415013 (+23.136110%);
57:592455170 ->788617463 (+33.110065%). n1:23130413 ->193910576 (+738.335986%).
n132:679348198 ->777415035 (+14.435431%). These are component probes with saved
activation subsets; not full paid handler measurements. Reject unchanged shared
batch48/tile128 prepared WWC forgoal; never adopt based on dot-rank alone.

audit_s3_k2_batch48_shared_report.py session3276 terminal independently redecodes
all75 raw Candid replies, all counter sums/source/dependency/entry hashes,
same30 guard-fold control WAT bodies, and native digests. Unique current
frozen-evidence.zip reverified. All16 fixed-coefficient prep updates <40B;
actual maximum5111806592, preparedQ224788480 bytes. None of the regression is
hidden immutable preparation: target56 projection itself579226149 ->709421392,
inputprepare220 ->4224688, quantization1768933 identical. Saved Q firstpair/
short-token fixed projection overhead is large, motivating calibration.

New build_simd_meter_calibration.py v1 built20 methods, notinstalled; v2 adds
empty-callee local-count tests and frozen producer provenance. Installed only
own small4xhad:module8b18644a88e5d553c5a57967a55522e0df0a782d6a718c80c9624bcf60a91917,
install70465 terminal,start95140 terminal. Rejected prepared model diagnostic
state may be replaced; no full/mainnet/otherproject mutations. v2checker85639
terminal29 raw nat64 replies;26 methods plus repeated empty/dot/8192-local
callee results deterministic. Independently decoded all29 and checked each
exact per-iteration fraction/current source/entry hashes/unique frozen-evidence.
Artifacts/simd-meter-calibration-v2/{report.json,frozen-evidence.zip}.

Measured loopbody deltas periteration against same-count emptyloop:
const+drop2;localget+drop2;const+localset2;localget+localtee+drop3;
two vectorgets+dot+set4;two gets+I32x4/I16x8 add/sub+set4;
two gets+shuffle+set6;pointerget+V128load/load32_splat/load8x8_s+set5;
two gets+F32x4mul/add+set5;get+F32convert+set3;i32get+const+mul/shl+drop4.
Emptycallee call delta6 vs emptyloop. Adding64/5488/8192 UNUSED V128 locals
or8192 unused I32 locals yields ZERO extra compared toemptycallee. This rejects
unused declaration count alone as a measured extra fee; it does not prove
anything about used-variable lifetimes, native spills, or full handler speed.

These are compound loopbody query measurements on the tiny module, not uniquely
identified absolute opcode costs. Do not blindly apply them to full kernels.
In particular old quad-load component exact14/packet evidence is unchanged;
simple per-load extrapolation needs matching four-load versus packet+shuffle
body tests with same instrumentation/context before a general cost estimator.
Query-vs-update calibration also unperformed. Neither pricing rules nor runtime
were changed. No inference work was moved outside measurement.

Own full4caro freshly verified Running baseline6052cc94... by39053 and48121;
no fresh cache/pack read. Stop79929 terminal success for ownsmall; later67574
status pending at write time (must read before claiming Stopped; earlier19756
status raced stop and reported Running). No other active measurement jobs.
Later67574 terminal confirms Stopped on8b18644...; current verified state saved
atartifacts/simd-meter-calibration-v2/verified-final-state.json. All handles now
terminal. No pending callee/projection measurement to restart or poll.
All3 paid besttotals120619170432/103992353622/122775162875 unchanged, complete
guard-fold fidelity stillprior evidence, goal100B active/unmet. This turn progress
is completed rejection/provenance and fresh metering evidence, not goalcompletion.

### Matched update calibration and streaming reconstruction (2026-10-07)

Calibration v3 installation failed terminally with IC0505: duplicate logical
query/update export names. Saved failed-install.json; no v3 measurements.
Corrected v4 uses distinct update names with identical function bodies. Own small
module3e517267b5f4d8ad92c4cd8a8a07750e28da3951825938fdeaed7b2092eb6031;
install60815/start41795/check80146 all terminal success. All28 raw nat64 Candid
replies independently decoded, source/entry hashes checked, unique frozen evidence
verified. Query/update counters are identical in every paired condition; repeated
quad old/new calls deterministic. Empty10000-loop60222; one/two/four V128-load
body deltas per iteration5/8/14. These compound-body deltas are not additive
absolute opcode prices. Quad splat body26 versus packet body40 confirms exactly
14 extra per packet in both call modes, matching all21 prior component deltas.
No runtime/pricing changes and no inference performance claim.

New screen_rank343_streaming_reconstruction.py preserves leaf order and operand
sign/order while computing ready C nodes and recycling last-use slots. Independent
symbolic replay checks all700 nodes and all64 final outputs, plus the existing
bilinear identities and final signed bound4161536. Bank capacity/peak live101
versus prior583; retained-node C instruction lines2800 versus1880. Initial
Counter.update(mapping) sign-as-count mistake corrected to update(mapping.keys())
before the completed proof. Current source hashes and evidence.zip verified.
Host capacity candidate256outputs/88tokens uses9198 estimated locals; counts
exclude actual helper ABI, code size, globals and runtime working storage.
Artifacts/rank343-streaming-reconstruction-v1/report.json explicitly records
wasm_execution_verified=false and performance_verified=false. No adopted kernel.

Stop82440 terminal success; fresh status75554 confirms own small Stopped onv4.
Own full freshly verified Running baseline6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931.
No fresh cache/pack read, no protected-project mutations. All handles terminal;
verified-final-state.json saved under simd-meter-calibration-v4. Full paid best
totals120619170432/103992353622/122775162875 and full fidelity evidence unchanged.
Goal100B remains active/unmet; streaming Wasm code sharing/performance and
production prepared-memory capacity require actual validation.

### Multi-value vector-return ABI measured (2026-10-07)

Previous turn classified progress: matched calibration and streaming symbolic
proof changed the next design choice. Fresh small status confirmed Stopped v4
before replacement. New probe_multivalue_vector_returns.py builds only a tiny
local ABI/counter probe, modulef55dd8b965282a558a0e86a8b46d7364b5651c0fa9a0f79d809ef7d1fdc2b932.
Own small install29382/start20591/check97442 terminal success. The checker had
one slow inline64_update call; the same live process was observed and awaited,
never restarted. Installed runtime accepts helpers returning16/64/128/256 V128
values with9 I32 parameters. Each helper/inline pair performs identical vector
constant pushes and receiver localsets1000 times. Additional compound cost is
exactly15000, or15 per call, for every return count in both query/update modes.
Query/update counters all equal: inline16/64/128/256=38289/134481/262737/519249;
helper=53289/149481/277737/534249. This includes nine constant arguments and
the call; it is not an isolated opcode price nor a computation-heavy benchmark.

All16 raw nat64/nat64 Candid replies redecoded independently; counter and lane0
sum checksums136/2080/8256/32896 checked, all source/entry/report hashes and
unique frozen evidence.zip reverified. Sum checksums do not prove individual
return ordering or every lane. Actual kernel integration needs full bit checks.
Evidence under artifacts/multivalue-vector-returns-v1. This removes the assumed
multi-value ABI feasibility barrier for code sharing; no streaming inference
implementation or speedup claimed. Stop terminal success. Paid best totals and
fidelity unchanged, all100B targets still unmet, goal active.

### Streaming shared kernel built and executed on Node (2026-10-07)

Previous turn progress: actual IC multi-value ABI evidence permits helper design.
New generate_s3_streaming_shared_kernel.py emits WWC batch88/tile256,9044 main
locals, one9-I32/88-V128 helper,343 calls and1043 streaming events. Same compact
prepared weight ABI, sixteen ascending pair-dots per leaf, integer wrapping
reconstruction, original root shuffles and ordered F32 scale/block additions.
Tail guards cover each token octet and each output row; no input mutation.
Canonical standalone module84ab500e701d4de4b28459b7a3b98404ddf7c7088ff92cf23e25cf9eab415d74.

execute_s3_streaming_shared_kernel.py v1 failed on512cols/n1 after13 completed
conditions: test driver did not advance sx pointer for second K block. Production
wrapper already passes q.scales().as_ptr().add(block), independently inspected.
Preserve execution/failed-run.json and original inputs. Corrected execution-v2
completed24 conditions (cols256/512, n0/1/7/8/9/48/56/57/87/88/89/132), comparing
ENTIRE memory with an independent original scalar integer dot and ordered F32
oracle. Nonzero carry/signed-zero, scale values, strided output padding and
all immutable inputs preserved exactly. Test-only memory export; same bodies.
Frozen execution-v2-report.json/evidence.zip source hashes and ZIP reverified.
Node v24.19.0 execution only; not IC fidelity or instruction performance.

New offline-built scripts/wasm_patch_shared appends one helper type/function/body
and relocates only343 direct helper calls while replacing one unique9-I32 export.
Reject aliases/imports/unhandled indirect/reference calls; wasmparser validates.
validate_s3_streaming_shared_patch.py independently compares every other original
section/body byte-for-byte and executes patched Rust Wasm on the same24 full-memory
conditions. All pass; module28446f242ce1c1cd13a644378287cb0577b6c825f2015297a02401f7a0f613de.
Patch proof source hashes/unique ZIP reverified. No relaxed SIMD, FMA, changed
quantization or runtime metering introduced.

Full component diagnostic build66876 terminal success:
module53a6c645cfd65c453438ad2341a99c0cd2a96021744b615a9c84b1369562d9ca,
10361283 bytes/code section9802159 <12582912. All31 patches validated, last
appends helper; all30 current guard-fold control body/source hashes independently
equal. Dependency/source/builder entry/module hashes checked. Same512-row immutable
weight prep chunks and same measured input preparation; wrapper uses256 outputs
and8 prepared-group pointers. Fresh ownsmall status Stopped onf55dd8b... before
installation. Install80629 currently live; do not restart solely for slow response.
check_s3_streaming_shared_probe.py prepared for21 fresh native/current/candidate
pairs after successful install/start. IC output equality/performance not yet proven.
Own full baseline not mutated. Paid best totals120619170432/103992353622/
122775162875 unchanged; goal100B active/unmet. Tracked diff empty.

### Streaming diagnostic installed; live component checker (2026-10-07)

Prior turn classified progress (implementation and actual Node bit proof), followed
by verified wait on live install80629. This same handle returned terminal failure
IC0503: omitted diagnostic init arguments. Preserve failed-install.json. Inspected
init(owner:Principal,rows:u32,cols:u32), then retry83905 with verified own controller
cibxp-okw3l-gfzvi-p6ltu-23ss3-7tlfz-nk65x-tvhvb-mg56y-b5hkf-eqe and8192/2560.
Retry terminal success; original failed attempt left small onf55dd8b... .
Start36279 terminal success. Status44937 sampled before start completion and
reported Stopped but verified new53a6c645... module. Do not treat this raced
status as later failure. install-result.json records exact arguments/outcomes.
Fresh full status34946 terminal Running6052cc94... baseline, no cache/pack read.

Checker9513 live, PID39678; first8 prepare_chunk raw replies saved at inspection,
then live child uploading weights4096. Same checker includes16 raw-weight updates,
16 fixed coefficient updates/seal and21 native/current/candidate pairs42 queries.
No IC output/performance claim until completed replies audited. New
audit_s3_streaming_shared_report.py prepared to redecode all75 replies, verify
all30 control hashes and sums, and freeze unique evidence. Never restart live
checker based solely on delay or missing completed report. All source/fidelity
and paid-goal requirements unchanged; full100B best totals still above target.
Later same checker9513 still live;14 raw prepare_chunk replies saved. No completed
query comparison/report yet; continue same handle on next goal turn.

### Streaming first IC comparisons and selective inline proof (2026-10-07)

Previous goal turn progress: corrected install/init/start and14 saved uploads.
Continued SAME checker9513/PID39678 without restart; all33 prep/seal replies now
saved and fresh paired comparisons begun. Actual counters/native equality so far:
boundary8 current99509601 ->244359743 (+145.563986%);
size67 692763777 ->838817009 (+21.082689%);
prefix45 472048105 ->615954633 (+30.485564%);
617 activation87 893360015 ->990612801 (+10.886181%);
insufficient80 821748208 ->916523074 (+11.533322%).
These are Q projection component counters including quantization/input prepare,
not full paid totals. Checker still live; full21-pair report/audit pending.
Observed regressions reject adopting current streaming configuration; completing
remaining conditions before freezing evidence. No changed inputs/weights/prefix.

While waiting, screen_rank343_streaming_inline_reconstruction.py completed a
host-only single-use-inline schedule:143 register slots,1880 instruction lines
versus101/2800. Independent symbolic slot replay verifies all kept nodes/all64
roots and same leaf dot order; original inline expression op sequence preserved.
New screen_rank343_selective_streaming_inline.py retains single-use nodes with
long readiness-to-consumer delay.10 complete symbolically audited schedules:
gap0:101/2800;1:101/2448;2:101/2252;4/8/16:100/2056;
32/64/128:108/1944;344:143/1880 (slots/instruction lines).
All64 outputs/intermediate coefficients exact; signed operand order maintained,
same modulo2**32 integer semantics. No Wasm/performance claim for these variants.
First selective screen failed from frozen-file ROOT path resolving into artifacts;
failed-gap-0.py preserved, corrected absolute ROOT and all10 reports completed.
All current source/variant hashes and unique frozen ZIPs independently reverified.
Evidence artifacts/rank343-{streaming-inline-reconstruction,selective-streaming-inline}-v1.
The100/2056 candidate reduces old stream reconstruction lines by744 without
raising register count, but must be actually executed/measured before adoption.
Full paid best totals unchanged; goal100B active/unmet.

### Shared stream rejection complete; selective stream executed/built (2026-10-07)

Previous goal turn progress: first real comparisons and10 selective host schedules.
SAME checker9513 now terminal success:21 paired conditions42queries, all native
digests/current control bits equal. Every condition regresses. Target48:
500744394 ->617772716 (+23.370870%);56:580995302 ->692475109 (+19.187730%);
57:592455170 ->762928759 (+28.774091%).617savedQ87:+10.886181%;
n89:+27.240433%;n132:+14.226755%;n1:+938.159898%.
Auditor46152 terminal independently redecoded all75 raw replies, checked sums,
input/build/entry/dependency/module/current30 body hashes/native digests. All16
coefficient updates <40B, max5111806592; preparedQ224788480bytes. Unique current
frozen ZIP and all source hashes independently reverified. Reject unchanged v1
streaming shared configuration; not adopted/full paid inference.

New selective kernel scripts generate/execute/validate_s3_selective_streaming*
use gap4 proof,100-register bank,2056 C instruction lines,8956 main locals,
671 events versus1043 old. Helper has88 results and343 calls unchanged. Canonical
standalone modulef8ec3070f51e4b3622c079f6151791a315bcb8f4e513a862d9432320f4a851db;
patched Rust module02856c12cd82478f097191bcc79ed8076ca37706110861deadabba39f5aeb0e5.
Both actual Node executions completed24 full-memory scalar/F32 oracle conditions
including multi-block512cols, n87/88/89/132 and padding/immutable memory. This
proves actual local Wasm outputs, not IC performance/full model fidelity.

Paired diagnostic build60292 terminal32 validated patches: same30 guard-fold
controls plus old stream and selective stream helpers. Full sources/dependencies/
entry/module hashes independently checked, all30 control body hashes equal.
Modulebc0e53c46dc438e268bf808e476c1d5bb1d662c0caefd42e37529a9139e401b1;
11615897bytes/code section11056462 <12582912. Methods3/4/5 share immutable
prepared data/input quantization/wrapper; new checker performs21 triples63queries,
auditor prepared for96 raw replies. This avoids cross-module counter comparison
when deciding the selective variant effect. Old small stop terminal success.
Install42861 live with correct owner/8192/2560 init arguments; do not restart
for a delayed response. No full canister mutation. Tracked diff empty.
Full best paid totals120619170432/103992353622/122775162875 still unchanged;
100B goal active/unmet. Selective IC equality/performance still unproven.
Later install42861 terminal success, ownsmall start terminal success. Same module
paired checker launched; follow its returned live handle, no repeated install.
Paired checker handle99653 confirmed live by wait; no completed report yet.

### Paired selective stream rejected with complete IC evidence (2026-10-07)

SAME checker99653 terminal:21 triples63queries, all native bits/current digests
equal. Selective stream improves every original-stream pair but remains worse
than current in ALL21 conditions. Same-module617Q87 current893360015,
original990613810, selective960496690 (+7.515075% current, saving30117120
from original). Target48 current500744394/selective597179805 (+19.258411%);
56:580995302/669977558 (+15.315486%);57:592455170/738526568 (+24.655266%).
n1 +890.302080%;n88 +6.551049%;n132 +10.494500%. Do not adopt unchanged
selective rank343/batch88/tile256. These remain component measurements, not paid.
Same-module original counters differ prior single-stream wrapper by1009 for
8192-row cases/513 for4096-row case; comparisons use the fresh common wrapper.

Auditor40886 terminal independently redecoded96 raw Candid replies, native bits,
all preparation/query counter sums, source/dependency/entry/module/control hashes.
Unique frozen-evidence.zip and current source/auditor entry hashes reverified.
Max fixed prep5111806592 remains<40B; preparedQ224788480bytes unchanged.
audit_s3_selective_streaming_delta.py independently fits all21 fresh actual deltas:
(rows/256)*(cols/256)*(28644*ceil(tokens/88)+5952*ceil(tokens/8)). Removed372
always-tested reconstruction-node guards across11octets and744 active instruction
lines across8output groups account for these exact differences. Quantization and
input preparation counters identical old/selective; all savings inprojection.
This is a checked fit to these kernels, not a universal opcode price estimator.
Delta-audit source hashes/unique ZIP independently reverified.

Own small stop terminal success; fresh small Stopped onbc0e53c4... and fresh full
Running baseline6052cc94... both terminal statuses. No cache/pack read or full
mutation. verified-final-state.json saved. All handles terminal; do not poll/restart
old checker/install. Full paid best120619170432/103992353622/122775162875 unchanged;
goal100B active/unmet. Host/Node/IC rejection evidence narrows future work away
from unchanged higher-rank prepared batch88/tile256 configurations.

### Raw rank49 copy-removal candidate screened (2026-10-07)

Previous goal turn progress: selective stream complete IC rejection plus exact
same-module delta audit. Current tracked diff remains empty; no active previous
measurement handles. Inspected current rank7 guard-fold WAT: all7 query forms
already cached/reused, not merely form0; proposed wider query cache is redundant.
Current9929 locals on168 tile includes these existing caches. Raw rank49 also
already inlines single-use C expressions: independent reconstruction generated
exact same246 instruction lines/register capacity95. A/B each have44 nodes and
zero nonleaf single-use or unused nodes, so trivial DAG inlining/dead-node
removal is unavailable. These observations prevent duplicate optimization claims.

New generate_s2_inline_roots_kernels.py emits same raw leaf-outer96/16 normal/
seed kernels removing only four root-to-c local copies per valid output row.
4 kernels retain original9865/1760 locals, cache/raw capacity/dots/F32 operations/
store order and same C reconstruction. audit_s2_inline_roots_candidate.py proves
448 aliases not overwritten before last read; candidate byte-equals the original
after ONLY these substitutions. Source/unique evidence ZIP verified. No actual
new Wasm execution or IC instruction performance claim; no canister mutation.

First estimator counts padded quartets; superseded v2 counts only valid guarded
output rows. Old report/proof preserved. v2 estimate assumes measured unit
local.get/set costs and predicts copy saving3563520 for617Q87,1966080 for48,
2293760 for56,2334720 for57. Against saved raw-rank49/current paired IC evidence,
predicted totals901535194/512124249/590126669/627424525 remain above current
893380246/500764471/581015379/592475247. n132 alone would improve, which does not
satisfy the actual617/620/653 targets. Estimates are not new measured counters;
do not adopt or claim actual regression proof from an estimator. Copy-only
candidate alone does not justify target-goal adoption; larger structural change
is required. audit-v2/source/entry/unique ZIPs independently reverified.
All tools terminal, no jobs launched. Full paid best120619170432/103992353622/
122775162875 unchanged and full100B objective active/unmet.

### Raw contiguous rank49 roots: real component gains (2026-10-07)

Previous turn progress: copy-only alias proof/estimator excludes that change alone.
New generate_s2_contiguous_roots_kernels.py permutes fixed raw output dimension:
logicalNcolumn*4+lane replaces lane*4+logicalNcolumn. Raw INT8 weight values and
payload capacity unchanged;49-leaf DAG/dot ordering unchanged. Remove all output
shuffle/root-copy stages and directly convert/store four-root vectors. Four96/16
normal/seed WAT,9865/1760 locals. Weight producer changes only corresponding
output-row source mapping; native scalar index mirrors that permutation.

execute_s2_contiguous_roots_kernels.py actually executes SIMD Wasm on Node,
104 full-memory cases: rows16/96;cols256/512/2560/9216; n0/1/3/4/5/48/56/57/
87/89/132 on short widths andn1/57 on large widths; seed/nonseed carry. Raw index
bijections, original scalar dots, ordered per-K256 F32 scale/add, stride padding
and immutable inputs all bits equal. Test-only imported memory; original bodies.
Four unique-stub Wasm patches validate, module0652b1cf... . Frozen host/execution
source hashes and unique evidence ZIP independently reverified.

Full diagnostic build67375 terminal34 validated patches: current30 guard-fold
control body/source hashes byte equal plus4 candidate kernels. Source/dependency/
entry/module hashes checked. module94ac97935a8d340039a61c0dbc82a47fe738fc262e9a53a542e7718133584072,
9699606bytes/code section9172419<12582912. Fresh smallStopped onbc0e53c4... before
install78123 terminal success (correct owner/8192/2560 args), start terminal.
Checker19913 terminal21 pairs42queries, native bits all equal. Reporter57994
terminal redecoded59 raw Candid replies and all sums/model/input/source/control
hashes, unique frozen-workflow.zip and reporter-entry hashes reverified.

Actual same-module Q results:617activation87 current893380246/candidate883717594
(1.081583% gain);insufficient80 +1.708288% gain;n132 +2.821794% gain;
size56 581015379/578657869 (0.405757% gain);boundary64 +0.944349% gain;
n88 +1.987215% gain;n109 +0.181235% gain. Size48 still0.305409% worse,
size57 3.928549% worse;prefix45 6.163508% worse;n89 .843839% worse;
n1 uses existing control fallback,5 extra dispatch instructions. No blanket
rank49 adoption; not full paid counters or full model/state fidelity proof.

New audit_contiguous_common_raw_layout.py host proof makes a single four-plane
raw INT8 payload addressable by both rank7 and rank49 (no duplicate weight bank).
16 shape cases rows16/32/96/256 ×cols256/512/2560/9216 prove complete raw index
bijections/all16 virtual rank49 aliases. An independent rank7 Winograd integer
oracle matches original scalar2x8 dots with contiguous lane roots. Rank49 aliases
require block byte offset512/group stride4*cols; rank7 raw halfK/Nquartet mapping
is described by the source. Shared-layout Wasm generators/production OriginalView/
single-token fallback/cache/continuation and full paid bits remain unimplemented.
Host-only proof, not additional IC gains. Source/unique ZIP independently checked.

Own small stop terminal success and fresh status Stopped94ac9793...; full fresh
Running baseline6052cc94...; no cache/pack read/full mutation. All handles terminal,
verified-final-state.json saved. Tracked diff empty. Full paid best remains
120619170432/103992353622/122775162875, all above100B; goal active/unmet.

### Rank7 contiguous roots: host/Wasm validation (2026-10-07)

Created `scripts/generate_k2_contiguous_roots_kernels.py` and `execute_k2_contiguous_roots_kernels.py`. All eight canonical current guard-fold kernels remove only output two-get/shuffle sequences: 1,220 static shuffle sites total. The raw four-plane common layout has SIMD lanes covering consecutive output quartets. Its plane order is K-major/N-minor, so rank7 pointer order MUST be `[0,2,1,3]`; the initial identity-pointer test correctly failed and was repaired. A subsequent harness-only issue hardcoded the tile128 export name; exports are now taken directly from the immutable kernel manifest.

Actual Node SIMD execution passed 208 full-memory bit comparisons: tile32/128/160/168, cols256/512/2560/9216, short widths n0/1/3/4/5/48/56/57/87/89/132 and wide n1/57, seed/carry and ordered F32 K256 accumulation. Input, scales, raw bank, padding and output bits all compared against independent scalar oracle. `artifacts/k2-contiguous-roots-kernels-v1/execution-report.json` and its ZIP freeze source, modules and inputs. No IC performance/full-paid claim from these tests.

Built same-module diagnostic `3b411aa7eea8f82dbd765df676e3e9b86459363eb5a9d9bf630b98e8d0df0f1c` (9,681,313 bytes): 30 unchanged current control WAT sources plus four new tile168/32 normal/seed bodies, explicit original current one-token fallback. Rank7 compact input preparation is copied from the current runtime. Candidate raw producer and pointer order are local to the diagnostic. Installation/measurements are pending; goal remains active and full paid best remains 120,619,170,432 / 103,992,353,622 / 122,775,162,875 for 617/620/653.

The new rank7 diagnostic installed successfully and was started on owned small canister4xhad. Checker `scripts/check_k2_contiguous_roots_probe.py` is running21 pairs; first boundary8 pair is99,509,601→99,499,268 (0.0103839% improvement), native bits equal. This is interim component evidence only. The frozen checker requires execution through its script wrapper (its ROOT uses script-directory ancestry); directly executing artifact copy failed before any network work and was corrected.

Also created `scripts/generate_s2_common_raw_kernels.py` / `execute_s2_common_raw_kernels.py`: four contiguous rank49 kernels now use doubled W K-block offset and quadrupled16-output-group stride. Caller uses16aliases into SAME four-plane raw bank, not16 disjoint planes. Actual Node SIMD execution passes104 full-memory scalar-oracle cases across tile16/96, cols256/512/2560/9216, seed/carry, odd/padded tokens and ascending-K F32 order. Source/report hashes verified. No IC measurement or shared production adoption yet.

Rank7 contiguous-root component checker43203 finished all21/42query pairs; independent reporter redecoded all59saved Candid replies and verified30unchanged control bodies, sources/dependencies and sum counters. Module3b411a..., source-evidence ZIP and summary saved. 617n87 gains0.0239890%; target-shaped48/56/57 gain0.0293697/0.0300200/0.0188629%; n132/4096 outputs worsens0.4427568%. Boundary2/3/5/7/9 regress, one-token fallback+5instructions. No adoption/full-paid claim. Candidate uses168/32only and differs from complete production tiling (128tail missing); the report preserves actual behavior.

Shared raw rank49 diagnostic built and validated: module2d4ead5c84d8b292c265bd62ec8ee7896c869026582563ef03be8f94ca91bc17,9,698,611bytes,34patches. All30 current controls byte-source unchanged. The Rust producer builds common four-plane rawbank;16alias pointers feed newWAT.104Node full-memory proofs and frozen source/dependency/entry hashes precede installation. Initial builder residual loop assertion was updated from1to0 after replacing the rawpack loop, before successful build. No runtime/test-source hashes from completed rank7 or prior probes were changed.

Shared raw rank49 checker85710 terminal:21pairs/42queries all native bits equal. Independent reporter38847 terminal redecoded59rawCandid replies, rechecked sources/dependencies/current30controls and counter sums, frozen unique ZIP verified. Module2d4ead5c..., raw bank capacity unchanged. 617n87 gains1.06854098%; n132 gains2.84805755%; n56 gains0.38570270%, n48 regresses0.32867667%, n57 regresses3.94821490%. This supports the shared format but not general/full-paid adoption.

Prepared separate bounded staged-input MLP diagnostic: module0ac25dde52309967a9a51d705d83175f7113ec1bd244e8859862460c27ff058f (9,702,841bytes,34validatedpatches), same canonical current control and rank49 sharedbank. Staged input is owner-checked, bounded by132*cols*4 bytes and chunks<=1,500,000bytes. An empty project argument selects stored identical F32 bytes; both paths still quantize and prepare inside measured projection. Input staging is reported separately and is not hidden as immutable-weight work. Checker scripts frozen for gate9216×2560 and down2560×9216 with synthetic n1/8/48/56/57/87, native-bit comparison, all saved Candid. No measurement of this staged MLP module yet. Pre-execution checker syntax/column-size issues were fixed and all entry hashes refreshed before any staged MLP call.

Owned small4xhad stopped on verified sharedraw2d4ead5c... after its completed21caseproof. Staged MLP module0ac25dde... installation started with correct owner and actual down2560×9216 shape; installhandle49962 pending. `scripts/prepare_s2_common_raw_mlp_checker.py` freezes bothshape checkers; `scripts/report_s2_common_raw_mlp_probe.py` will independently redecode all projection and staging replies, verify payload chunk concatenation, original30controlbody source identities, native digests, interval sums, and show staging-added totals with explicit Candid/digest/callback exclusion. These are component scopes, not complete handler totals. All new build/dependency/entry hashes reverified. Full paid best, model/prefix and protected canisters remain unchanged; goal not met.

MLP down install49962 terminal success; ownsmall started and down checker24725 launched. Pending6 pairedqueries with separately recorded staging updates. Nextpoll samehandle24725; do not replace its module or reinstall while it runs. After terminalsuccess run `python3 scripts/report_s2_common_raw_mlp_probe.py --shape down`, then verify archive/source/replies/finalstate before changing smallcanister for gate.

Staged MLP DOWN checker24725 terminal, all6nativebitpairs equal; independent reporter60778 terminal,47savedCandid replies redecoded, staging payload chunks concatenated exactly tooriginalF32 inputs, sources/dependencies/current30bodyidentities/counter sums verified. Unique archivedsource/replyhashes and final stoppedmodule0ac25dde... verified. Projection intervals n48:579,958,354→581,528,408(-0.2707184%); n56:672,992,929→670,214,221(+0.4128881%); n57:686,512,703→712,953,454(-3.8514584%); n87:+1.0595529%. Actual input-stagingintervals reported separately and added to bothpaths in summary, with explicit exclusions outside intervals. Syntheticinputs at genuine2560×9216shape; no fullDense/hiddenstate evidence/noadoption.

Next gate-shape reinstall on sameownedsmall, SAMEmodule0ac25dde..., owner/rows9216/cols2560 supplied. Gatechecker can launch only after installterminal and startsuccess; use `python3 scripts/check_s2_common_raw_mlp_probe.py --canister 4xhad-gd777-77775-aaacq-cai --shape gate`, report with matching `--shape gate`. Goalwholepaid unchanged and active.

Gate reinstall8295 terminalsuccess, ownsmall started on SAME0ac25dde module and gatechecker launched. MLPdown was audited/archived and stopped before reinstall.

`scripts/audit_s2_cached_leaf_basis.py` symbolically proves all32original basis vectors recoverable from49cached leaves with1..4 signed terms. A terms counts[1,1,1,1,2,1,2,1,2,2,1,1,4,2,2,1]; B[1,2,2,4,1,1,2,2,1,2,1,2,1,1,1,1]. In particular first logical input token's four K64 bases are directly available. This supports a possible direct cached-B one-token tail in rank49, to avoid computing the entire padded quartet for n57. Initial<=3terms bound was insufficient for one basis, extended to exact meet-in-the-middle4terms; final32 coefficient identities and source hashes verified. No tailkernel generated or measured yet; no performance inference from term counts.

MLP gate checker87561 terminal6pairs/12queries allnativebits equal; reporter45405 terminal37Candid redecode/stagingconcatenation/native/counter/current30control/source audits passed. Unique archive hashes and final stopped0ac25dde... state verified. n48:-0.01698997%; n56:+0.70020303%; n57:-3.60966880%; n87:+1.38482178%. Neither shared49MLP nor rank7partialdiagnostic is globally adopted.

Created `scripts/generate_s2_cached_tail_kernels.py` / `execute_s2_cached_tail_kernels.py`: sharedraw49 main4-token quartets unchanged except a branch; exactly one remaining token uses16raw K/N products fromcached49B basis, rawfirsttokenA bases are directleaves. No extra locals(9865/1760), noextra weightbank/payload. Integer tail dot order differs (safe original I8 basis recovered from<=4I16 leaf sums); ascendingK256 orderedF32 scales/add remains. ActualNode SIMD104fullmemorycases passed seed/carry/n1/5/57/89 inclwidecols2560/9216, preservedinput/padding. NoICspeedclaim. Staged diagnostic builder launched for this newcandidate, carrying unchanged30currentcontrols.

Cached-tail staged diagnostic build82931 terminal: moduledbd53f02b83bbb8912c2d09413fc0ef01c4b437bf8893a9bdf1e1e41a273ff92,34validatedpatches including30unchangedcanonicalcurrentcontrols. Build/source/dependency hashes reverified. Checker/reporter scripts frozen for down/gate6pairs each. Installation launched ownsmall actualdown2560×9216, correctowner, pendingterminal. Noothercanistermutated. Nextlaunch cachedtail checker only after install/start terminal; measure57case, preserveallnativebits and explicitlyaccountstaging; thenauditbeforetestinggate. Fullpaid totals unchanged, goalactive.

### Cached-tail dot reuse and exact rank48 candidates (2026-10-07)

Prior goal turn was progress; current turn finished original cached-tail down6pairs:allnativebits equal, independent47Candidredecode/payloadstaging/source/control audits and finalstoppedstate archive checked. Originaltail n57 regressed0.15220563%, greatly reduced from3.85% but stillworse.

New `generate_s2_cached_tail_dot_reuse_kernels.py` reuses sameA raw/difference dotresults acrosslogicalN instead of constructingmulti-term B vectors at eachK. All16 withinK Bbasis identities provedsymbolically, no newlocals (9865/1760),4kernels. ActualNode104full-memorycases passed and sourcehashesreverified. Diagnosticbff9c9dd7f23267456f9849b0b8f3cf6c67efb3b5e087aa1e03a33b57dfe70c6 built34validpatches preserving30canonicalcurrentcontrols. Owneddown installation91291/start58315 terminal, checker16728 terminal6pairs allbits equal, reporter90615 terminal47Candidredecode. n48:-0.28636091%,n56:+0.39716138%,n57:+0.35624235%,n87:+1.04348799%. Inputstaging intervals separatelyrecorded/added; no fullpaidclaim. Ownedgate installation90977/startterminal, checker20180running/recentlypolled. Must auditgate beforechanging smallmodule.

Primaryresearch: https://arxiv.org/abs/2506.13242 and https://fmm.univ-lille.fr/4x4x4.html linkauthorcoefficientdata https://github.com/jgdumas/plinopt . Pinned commit1af82ca4df7ed7d9687bc5f9a39bc10e724e1e0f . `fetch_rank48_primary_coefficients.py` froze9rational/ALT/CoBnumericSMS pluslicense/authors, no codeexecution/install. `audit_rank48_primary_coefficients.py` exact4096tensoridentities and47-coordinateALT/CoB compositions; uniform integer scales[2,1,8], finaldiv16, I16A/Bbounds4064/2048 and finalscaledK256dotbound66,584,576. ALT47 is redundantrectangular representation, not invertible16basis; do not assumepublishedasymptoticconstantwithoutwholecost.

Author-treeinventory additionallyfoundlatest4x4x4_48_204normal/16basis/24basis/32post-programs; `fetch_rank48_latest_programs.py` froze32filesatSAMEcommit. `audit_rank48_latest_programs.py` parsesSLPonlywithstrictlinearAST/Fraction (no downloadedprogramexecution), validates15SLPs againstSMS and2missingSMS32postSLPs viaexactpairedcomposition. Base4096tensoridentities plus16/24/32compositions allpass. NormalL55adds+2halves,R55+2halves,P90integeradds;16ALT44/44/83 andCoB36/36/27. Countsare syntacticnotICcosts. SourceZIPincludesauthorlicense.

`audit_rank2304_integer_lift.py` normalizes2025rational48leafA/B toprimitiveintegerrows, absorbsfactorsintopostmatrix, provesprimitivebase4096tensoridentity. Basepostdiv8; Kroneckerdepth2(rank2304) finaldiv64, Aexactglobalbounds[-32512,32512], B[-32672,32656], scaledoriginalK256dotbound266,338,304 <signed32. I32leaf/Cwrap is intentionalmod2^32; finalglobalarithmeticshiftby6 recoversrawdotexactlybecausefullnumeratorboundedanddivisible64. Twelveactual16x256by256x16 integerconditions(256roots each) includingextremes, coefficient-signmaximization andoverflowedleafproductsallmatchscalarrawdots. NoSIMDWasm/IC/resource/fullpaidclaim. Primary/integer/depth2uniqueZIPsourcehashes independentlyverified. New mathematicalcandidate targetsmoremajorbaseprojectionchange; must boundprecomputedweightmemory/transfercost and evaluateactualICperformancebeforeadoption. Fullpaidbest remains120619170432/103992353622/122775162875; goalactive.


### Final dot-reuse gate audit and new integer-basis screening (2026-10-07)

Gate checker20180 and independent reporter32729 terminal success: all6 native bit pairs equal;37 saved Candid replies independently redecoded, source/unchanged30control bodies/staging payload concatenation/counter sums verified. Dot-reuse gate n48:-0.03126318%,n56:+0.68585124%,n57:+0.65712833%,n87:+1.37015496%. These are component intervals with separately accounted input staging, not whole paid handler totals. Final owned small Stopped bff9c9dd7f23267456f9849b0b8f3cf6c67efb3b5e087aa1e03a33b57dfe70c6 is saved in check-gate/verified-final-state.json; full baseline unchanged. All relevant runtime handles terminal.

Latest204 primitive rank48 depth2 was rejected BEFORE creating/casting I16 coefficient payloads. Actual primitive row maximum L1 is20 for both A/B, giving activation bound50800 and weight bounds[-51018,50982]; these exceed I16. `rank48-latest-programs-v1/depth2-rejection.json` and unique evidence ZIP freeze this failure and source hashes. Earlier rational rank2304 proof remains valid, but is a different candidate. Do not attribute its admissibility to latest204.

Created `plan_rank343_integer_basis.py` and `audit_rank343_integer_basis.py`: invertible integer tensor coordinate transforms, alternate Winograd encoder and final integer inverse. All343 original leaf coefficient pairs and all64 final product expressions exactly match the prior rank343; all262144 tensor coefficients are independently verified. Every preparation intermediate fits I16 (A8128/B8192); I32 reconstruction wraps deliberately modulo2^32 and final raw dot bounds preserve exact signed lift. Ten actual scalar conditions pass. Candidate DAG nodes A/B327/327 versus prior372/372, C654 versus prior651. The prior C already deduplicates many expressions: the naive744-to654 comparison would OVERSTATE savings. No C speedup claim and no full adoption.

`execute_rank343_integer_basis.py` executed generated32-output seed/carry SIMD kernels in Node:52 full-memory bit comparisons, cols256/512/2560/9216, short widths0/1/7/8/9/48/56/57/87/89/132 and wide1/57, padded tokens, ascending-K256 F32 scale/add. All memory bytes including input/banks/padding equal independent scalar oracle. Initial frozen-plan loader ancestry failed before any case execution; corrected by using original source path for __file__ while executing frozen identical bytes. Sources and unique host/execution ZIP independently verified. Weight-prepared capacity remains10.71875x raw; this is a mathematical/SIMD screening result, not IC metering or memory architecture acceptance. Input-preparation arithmetic savings alone have not been measured and cannot establish the100B target.

Full paid best remains617:120619170432,620:103992353622,653:122775162875. Goal active/unmet; no full canister mutation, commit, PR, toolchain/runtime changes or other-project calls this screening turn.


### Rank49 integer input-basis measurement (2026-10-07)

Previous goal turn made progress: rejected unsafe latest204 I16 depth2 and executed52 exact rank343 SIMD conditions. This turn applies only the alternate input encoder to existing sharedraw49/cached-dot-tail kernels. `plan_rank49_integer_input_basis.py` preserves all49 leaf coefficient vectors and B/C leaf order/roots. Actual old rank49 DAG has44 nodes after deduplication, new41; a naive56-to49 comparison would overstate savings. All preparation intermediates globally bounded2032;20actual49x64 I16 host conditions match old direct coefficients. Host audit source hashes/evidence ZIP saved.

`scripts/build_s2_integer_input_basis_probe.py` built module2f3c7690fab28605bd882183e3a81a4dcf0f764ec876ac83fa1468a511146126 with34validpatches, same9865locals; immutable WAT source/export identities of all34patches unchanged. Projection/API/coefficients/stubs byte-identical to prior dotreuse probe; only input preparation DAG differs. Initial full patch-record equality assertion included changing intermediate module hashes and was too strict; corrected identity audit uses export/sourceSHA, captured in delta-audit.json. Installation already launched after that shell assertion failed; before any checker it was independently confirmed all kernel/source identities matched. No kernel behavior or gate requirement bypassed.

Owned small downinstall33203 terminal/start/checker64587 terminal6pairs all native bits exact; independent reporter35830 redecoded47rawCandid and verified staging concatenation/source/counter sums; uniqueZIP and finalstoppedmodule saved. Compared prior dotreuse module, n48/56/57/87 save34560/40320/43200/63360 instructions respectively, all savings exactly in input preparation; one-token fallback unchanged. Against canonical current control, n48:-0.28040186%,56:+0.40315253%,57:+0.36253503%,87:+1.04960795%. Not whole paid handler totals.

SAME module gateinstall54487/start/checker97290 terminal6pairs all native bits exact; independent reporter75734 redecoded37rawCandid/staging/source/counters; uniqueZIP and freshStoppedmodule verified. Against canonical current control gate n48:-0.02956399%,56:+0.68755979%,57:+0.65892340%,87:+1.37190101%. Incremental input-preparation savings tiny, shown in gate-delta.json. No fullcanister changes, production adoption, commit/PR or runtime/toolchain changes. All current build/install/check/reporter handles terminal. Full paid best remains120619170432/103992353622/122775162875, all >100B; goal active/unmet. Next fullhybrid adoption requires actual Q/outshape component evidence, production format/single-token/continuation adaptation and complete all32hidden/state/final paid fidelity plus whole-handler totals.


### Q/output component closure and full common-layout build (2026-10-07)

Previous turn made progress: measured rank49 input encoder41 versus old44nodes with native bit proofs. Current turn extended frozen checker/reporter toactual layer3 Q8192x2560 and output2560x4096 shapes on SAME2f3c7690... module. Q inputs are first n tokens of saved earlier real617 activation (not exact paid suffix); output inputs synthetic signed extremes/zero/tiny. All input staging updates separately counted. Full owned baseline canister untouched, fresh Running6052cc94... on4GiB; small finalStopped2f3c7690... remains3GiB.

Qinstall89287/start/checker86366 terminal6pairs allnativebits; reporter60186 independently redecoded35rawCandid/staging/source/counters and uniqueZIP verified before reinstall. Reduction relative canonical current n48:-0.34857472%,56:+0.37213396%,57:+0.34093625%,87:+1.05322724%. Fresh stopped module/state captured. Outputinstall1853/start/checker31048 terminal6pairs allnativebits; reporter37091 independently redecoded40rawCandid/staging/source/counters/uniqueZIP and finalstoppedstate verified. Output n48:-0.33153682%,56:+0.36351169%,57:+0.32923494%,87:+1.02167997%. All n8 regress and n1 fallback+5instr. Together with down/gate this supports shape-bounded56/57 dispatch, not unconditional adoption or full paid fidelity.

Created `scripts/build_update_common_raw_rank7.py`: clones complete current unprofiled full runtime/API/wrappers, changes raw pack/index to shared four-plane I8, rank7 plane pointers[0,2,1,3] fornormal/paddedtiles, aligned Rustsingle-token SIMD plane mapping and consecutive output roots. All8 complete current rank7 tile32/128/160/168 normal/seed bodies use prior208Node-verified contiguous kernels;22other current bodies remain exactlysame source/export identities. Compact Aprepare and full range/carry entry dispatch byteidentical. Source/input layout semantics still need whole runtime/Kcarry/single-token/full state proof.

Full compilation31105 completed runtime+wrapper then first patch lookup failed: baseline report's source list omitted two existing singlequadWAT files. Added explicitly known0/2sources and normal/build WAT inventories, checking SHA against eachbaselinepatch. `--resume-patches` asserts all copied runtime/wrapperfiles unchanged and regenerates/asserts exactnewstrassen source before reusing compiled rawWasm; no source changes since successful compile. Patch12288 terminal30validatedpatches. Fullcandidate e56389822773f95060da4390bc7ac67234553a70dee54cf8a67e410799369395,10,303,448bytes. Allnewsource/dependency/module hashes and uniqueZIP independently checked,8changed/22unchanged export+source identities validated. No installation/adoption; rank49 cache/dispatch integration still absent. No full baseline mutation/heap-limit/runtime/pricing/toolchain change, no commit/PR. All current handles terminal. Full paid best unchanged120619170432/103992353622/122775162875 >100B, goalactive/unmet.


### Full common-layout hybrid integration (2026-10-07)

Prior goal turn made progress:12Q/outputbitpairs, all35/40savedCandid and fullcommonrank7build. Current turn creates `scripts/build_update_common_raw_hybrid.py`, cloning verified shared-layout rank7 fullruntime/API. Cached rank49Operands contains fully written VecI16,49leaves/4token group. Four QuantizedRows constructors initialize a separate OnceCell; per-input cache reused across projections. Selector accepts onlyn56/57, startrow aligned8, measured(row,col)shapes(8192,2560),(9216,2560),(2560,4096),(2560,9216); allother shapes retain sharedrank7/single. Current48regressions excluded from rank49 selection.

Continuation builds separate Vec<MaybeUninit<I16>> and writes ONLY first..end Kblocks; consumers read only those ranges. Allpreparation charged throughcurrentinteger_k_prepare_new_columns measurement. Output copiesexisting initializedcarry then usesunseededkernels, or seedsfirstblock forfresh output; ascendingK256F32 scale/multiply/add maintained. Actualrank49inputencoder41nodes reused fromverifiedcomponent, exactsame4dotreuseWAT. Range modification changesonlyblockloopbounds. Rawfixedbank aliaseddirectly usingfullfixedrows and viewstart; no duplicate weightbank/new prephelpercanister.

Initial compiler90229 failed before linking: inserted module item beforeexisting innerdoc comment. Corrected module insertion afterdoc comment; --retry-build recopiesbaseline files/rebuilds, retainsruntime-compiler-first-failed.log. Build98484 terminalsuccess, module0aa01fdd75ccfb8583b0860ca2af39fdaacda9a5959d49dfa75d06bd0ed56484,11,419,616bytes,34validatedpatches. First30export/sourceidentities equalpriorcompletecommonrank7, added4verifiedrank49bodies. Source/dependency/module/uniqueZIP independentlychecked. Noinstallation/fullfidelity/performanceadoption claim.

`check_common_raw_hybrid_host.py` compiled actualnewrank49andparent source againstminimalnative QuantizedRows/PackedView fixture (notwholeproductioncrate):3tests terminal98105 pass. Cachepointeridentity/full49leafvectors exact againstrawcoefficients; partialrange3..5writespreserveseveryunselectedsentinel; rawviewstart8 andseed/carry0..1then1..2matchascendingKoriginalscalarbits; measuredshapeselector verifiedincludingn1/8/48/56/57/87 andexcluded4096/512views. Fixture substitutesaccessors/shapeconstructors; scalarbackend tests doNOTproveSIMD/fullgraph/paidfidelity. ActualWasm runtimecompiledallproductionQuantizedRowsconstructors; prior104SIMDkernelconditions remainunchanged. Hostreport/sourceuniqueZIPindependentlyverified.

Resourcecheck: KINGSTON initially~3.4GiBfree, systemvolume~103GiBfree. RemovedONLY62temporarypatch-stageWasm files createdbythetwonewcompletebuilds, aftercheckingfinalmodule/evidence/dependencyhashes and thatnointermediatewasfilelistedinsource/dependencyhashes. Retainedraw/finalWasm/Rustrlibs/source/validatorreports/evidenceZIP; paths/lengths/SHA recordedinintermediate-cleanup.json.396,349,843bytesreclaimed;KINGSTON~3.7GiBfree. Noothergoalhistory/model/protectedproject/cachefilesremoved. Largerrecoveryexports canusetaskownedpathonsystemvolume ifneeded, butnonecreatedthisturn. Fullownedbaselineunchanged; small remainsStopped2f3c7690...3GiB. All currenthandles terminal. Fullpaidbest120619170432/103992353622/122775162875 remains>100B; goalactive/unmet. Actualwhole-runtime/fullhidden/state/single/carry/paid totals stillrequiredbeforeadoption.


### Paid common-layout hybrid trial and local recovery (2026-10-07)

Paid builder `build_paid_common_raw_hybrid.py` completed48901, module38d8cd0f9998766b5a26d3cba2717c95e1bbd88de69a69591787a9c582a8a331 (11,792,234bytes).34kernel/source identities match normalhybrid; paid_types/paid_inference/scheduler unchanged from verified paidbest. Module/dependencies/source/uniqueZIP verified. Frozen proof adds all32hidden/state/finalhidden comparisons to independently audited paidbest.

Proof81546 failed before any inference measurement during immutable cache warmup (at least3,533,121,920bytes). Baseline4caro6052cc94... was saved inbefore.json, new snapshot15 created, candidate upgraded and2dedicated relay callers created. PocketIC abortedSIGABRT at12:26:29JST; stderr reports Unsupported overlay version45868862(maxV0) while obtaining MergeCandidateAndMetrics. Finally restoration failed because localhost8001 was unavailable. No paid totals or fidelity result from this candidate; do not infer crashcause from disk space. Crashreport/logs/descriptor and failure evidence retained under proof/recovery.

Selected local8001 restarted with sameinstalled launcher/runtime, config and rootkey. Old4caro/snapshot15 unavailable in fresh management instance. Stopped only own launcher, verified own orphan PocketIC15103 via8001 listener thenTERM, preserved entire old state as state-before-overlay-crash-20261007. Attempted state symlink to taskowned system SSD; post-start inspection shows launcher replaced it with KINGSTON directory, so relocation DID NOT take effect. Corrected relocation manifest records this limitation; protected8000/8002 untouched.

Portable baselinebackup local-baseline-recovery-v1 revalidated allSHA including module6052cc94..., stable4,702,470,144bytes and Wasm4,112,187,392bytes. Original backup remains unchanged. Taskowned CoW clone made for upload progress. Fresh sameCID4caro recreated44t localcycles,4GiBheaplimit,Stopped. Snapshot upload24297 ongoing, new snapshot0, metadata/module uploaded and Wasm offset104,000,000bytes at last inspection. BASELINE RESTORATION NOT YET VERIFIED; do not run new trial before successful upload/restore/start and exactmodule/cache/pack comparison. Full paidbest remains120619170432/103992353622/122775162875, goal100B active/unmet.

Recovery follow-up: upload24297 stilllive with104MB Wasm checkpoint; localhost8001 status HTTP200 and noCRIT/overlay/panic in currentlogs. Added read-only verify_local_baseline_recovery.py (help/import succeeds) for full module/cache/pack comparison against portable baseline and failedproof before.json; not executed against emptytarget. Corrected false SSD-relocation assertion; actual active state staysKINGSTON. Later gitdiff shows unrelated tracked PLAN/README/laya/assessment changes made externally during turn; preserved untouched. No recovery/adoption success claimed.

Next continuation: same upload24297 and processes22232/21214 authoritatively live; TCP established to8001, networkstatus local succeeds.104MB checkpoint unchanged for severalminutes. One-second process samples saved: CLI observed upload_blob_from_file/call_and_wait/read_state_raw; snapshotoffset unchanged is not terminalfailure or proof of deadlock. Prepared retry_paid_common_raw_hybrid.py and retry-plan-v1 without canistermutation: freezes originalfailedproof/source hashes, newproofdirectory, livebaselinefullstate guard beforemutation, --run requires recoveredcomplete/fullstate marker. Originalfailedproof preserved. Preparationcommand terminalsuccess; no fullproofrun while baselineunrestored. Goal remains active/unmet.


### Bounded sequential baseline snapshot transfer (2026-10-07)

Previous continuation was verifiedwait and retryentry preparation. Original24297 terminalexit1: Invalid request expiry atoffset112000000, contiguousWasm104MB. Same snapshot0 resumed53778, terminalexit1 IC0202 queuedingress timedout waiting tostart(offset2186000000), contiguousWasm206MB. No duplicateupload launched while eitherCLI live. Officialdfinity/icp-cli v1.0.2 source pinned e84048d3e92698af48929e9c49c552870a01d8d1 fetched to recovery/cli-source: upload_blob_from_file reads entireblob and launches allchunkfutures concurrently. This supports boundedtransfer change; notproof of originaloverlaycrashcause.

Created artifacts/local-serial-snapshot-upload-v1 nativehelper only: existingic-agent=0.49.2, offlinecacheddependencies, noinstalledtool/runtime/kernel changes. Usesofficialmanagementtypedargs upload_canister_snapshot_data, exactly2MBmaximum/oneinflight, fixedlocalhost8001/4caro/snapshot0, verifiesrootkeyagainstcurrentdescriptor, requiresmetadata/modulecomplete andemptywasmchunkstore. Reads only taskownedbackupclone and persistscontiguousoffset AFTER certifiedsuccess/emptyCandidreply; originalportablebackup unchanged. CLI-upload-live guardtested correctlyrefuses duplicatebefore networkcalls. Finaloffline/lockedbuild terminalsuccess andbinary/source/lock hashesrecorded. No controls/cycles/targetsettings changed.

Onlyafter53778terminal startedserial57504. Authoritativeprogress advanced206MB->276MB->564MB withoutfailure; same livehandle ongoing. BaselineNOT restored orverified, no paidinference measurement/adoption. Nextpoll57504, onlyafterterminalsuccessrestore snapshot0/start/runverify_local_baseline_recovery.py newoutputdirectory. Retainbaselineuploaded snapshotuntilverified. Goal100B full3inputs remainsactive/unmet.

Subsequent continuation verified same57504 live and persistent certifiedchunkprogress856MB->2,532MB of4,112,187,392-byte Wasm memory; stable remains0/4,702,470,144. No duplicate transfer, baseline restore, proof retry or inference result. Existing unmodifiedprepacked49 evidence reviewed: all21conditions slower, notadopted; no new performanceclaim. Official installedCLI1.0.2 network-run source audit proves it clearsstate directory on start (including nativebranch), explaining lost state-symlink/recreatedCID afterCLIrestart. Sourcepinned/hashes and network-start-storage-audit.json saved; relocation of live state/restart for persistence is NOT safe. Current transfer continues; parentstorage migration after a stop/newstart would require portablebackupuploadagain. Next poll57504 and onlyterminalsuccessrestore snapshot0/start/fullbaselineverify. Goalactive/unmet.

Next continuation: same serial57504 authoritatively live; Wasm memory now COMPLETE4,112,187,392bytes, stableoffset116,000,000/4,702,470,144. No expiry/queue failures inserialrun observed. Retrywrapper now accepts taskowned --proof-storage on systemSSD; preparedplan-v2 generated/compiled only (no --run, no canisterchanges), then added proofsymlink for futureactualrun. Earlierplan entryhash references mutablewrapper historicalversion; actualrun must create newdirectory/freshentryhashes, not consume thosepreparationmanifests as finalproof. Proofstorage change is output-only; engine state remainsKINGSTON and no live relocation/restart attempted. Native handles, actual snapshot0 retained. BaselineNOT restored/verified, no fullpaidresult; goal100B active/unmet. Nextpoll57504, terminalsuccessgate restore/start/verify.


### Baseline recovered and paid hybrid retry launched (2026-10-07)

Same serial57504 advanced stable360MB->2,538MB then terminalexit1 CertifiedReject IC0207 insufficientcycles (atleast873486538additional requested). No duplicate/resume beforeterminal. Dedicatedlocal wallet955999900000000cycles observed; toppedup ONLY4caro onlocal8001 by500T, terminalsuccess. Serial42890 resumedsameuploaded snapshot0/checkpoint; terminalsuccess allWasm4,112,187,392/stable4,702,470,144/module3,123,573 bytes. Complete marker persisted; originalportablebackup unchanged.

Restore46537 terminalsuccess snapshot0->4caro; startterminalsuccess; verification5384 terminalsuccess. Certifiedmodule6052cc94... and entirecache/fullpack EXACTLYequal both portablebaseline-state andfailedproof before.json. Verified marker saved recovery/verification/verified.json; currentlogs noCRIT/unsupportedoverlay/panic. Afterverifiedrestoration, deleted ONLY newlyuploaded snapshot0 (terminalsuccess); durableverifiedportablebackup remains. Recoverymonitor recordsall outcomes. Baseline recovery achieved; 100B inferencegoal stillunmet.

Retry-v1 entry failedimportbeforecanistermutation: actualfrozen__file__ underartifactdirectory made originalparents[1] root pointtoartifacts. Preservedfailedentry; fixedgeneratedROOT toverifiedrepoabsolute path, keepingfrozen__file__ identity forsourcehashaudit. Freshretry-v2 launched45944 withSSDproofstorage androotartifactproofsymlink, successful beforestate capture/fullbaselineguard andnewtrial snapshot creation. Candidatebuild38d8.../source/API/34WATdephashes verified bydriver. Same proof protectsrestoration/all32reference/final/API/refund/replay/auth/Busy/concurrency/upgrade checks. Current45944 LIVE, no paidresult orfullfidelityadoption claimyet. No protected/mainnet modification, no commit/PR; unrelatedtrackedchanges preserved. Nextpoll45944 (do notduplicate), inspectpreparation/progress; finalrestoreandpost-audit requiredbeforecandidateclaim.


### Paid hybrid retry path correction and immutable source audit (2026-10-07)

Previous goal turn made progress: completeportablebaseline recovery andlaunch45944. Current45944 completed721immutableweightpreparation calls252538138867instructions/4,065,416,192bytes; fixedprefix preparation andvoting/commonregistrations succeeded. Before anyinference result, configure_paid call succeeded butitsrecording raisedValueError: externalSSDlexicalD paths cannotbe relative_to(ROOT) inexisting PaidTransport. Same45944 finallyrestored6052cc94/cache/fullpack, deletedONLYtrial snapshot1 andstoppedbothownrelays; terminalexit1. Preservedfailure/rawconfigure reply; no partialinferenceperformance/fidelityclaim. Editingretrywrapper duringfailureawait changedhistoricalentryhash reference, so v2entrymanifest is not finaladoptionevidence.

Fixednewretrygeneration tokeepDlexically ROOT/artifacts/<trial>/proof throughSSDsymlink, whilecreatingtheexternalactualproofdirectory explicitly beforeuse. Preparedplan-v3 compileswithoutcanistermutation; sourcefrozenROOT stillexplicitverifiedreporoot. Onlyafter45944terminal/restoredmarker launchedfresh91773 retry-v3. Snapshotprotected, candidateinstalled andtwofreshrelays created:4xhad-gd777-77775-aaacq-cai and46el7-ql777-77775-aaada-cai. Freshnamespace means4xhad isNOW a relay, notthehistoricalsmallcomponentcanister; do notinstallprobesontoit. Current91773 LIVE inimmutableprep(index117,~2.195GB); noresult yet.

Createdandran audit_paid_common_raw_hybrid_sources.py77377 terminalsuccess. Independentlyrehashedbothnormal/paidmodule/source/dependencies,34validatedpatchidentitiesmatchnormal, paidlib/update_scheduler/paidtypes/paidinference exactbyteequalverifiedpaidbest. WritesONLYseparate source-audit-readonly.json; doesNOTmodifycandidatebuildreport/frozenproofsourcesduringliveproof. Audit explicitlyfullperformance/fullhiddenfidelity/goalfalse. Goal100B all3 remainsactive/unmet; best120619170432/103992353622/122775162875 unchanged. Nextpoll91773 (no concurrentfulltarget mutation); aftercompleteproof need independentrawCandid/totals/all32includinghidden30/report/archiveaudit andseparatedeterministicupgradeguardtests.


### Complete paid common-layout hybrid result (2026-10-07)

Previous continuation correctedrawpath recording andcompletedimmutable sourceaudit. Thisturn91773 completed721model-only prep252538138867instructions/4,065,416,192bytes, fixedprefix banks andactualthreepaidinferences. All32hidden/state hashes andfinalhidden/decisionbits matchsavedpaidbest plushistoricaldirectexports; scope includesunchangedquantization/input/prefix/weights. Paidfullhandler totals:617119862026020(4workers),620103356795366(3workers),653122016961354(4workers). Allmaxheap4,224,647,168bytes andeveryworker<40B. Againstpreviousbest saves757144412/635558256/758201521instructions(0.6277148/0.6111586/0.6175529percent). ALLstill>100B; gaps19,862,026,020/3,356,795,366/22,016,961,354.

Actualcallerbilling/refund/replay/conflict/insufficient/quote/bounds/selfauthority/statusauthority/Paused/Busy/concurrentupgradeordering/sameversionreceipts verified. 91773terminalsuccess restored6052cc94/cache/fullpack, deletedONLYtrial snapshot2, stoppedownrelays. No inferenceprocess remains. Builtreport_paid_common_raw_hybrid.py: independentrawCandidredecode, exactcurrentinputrecords, worker sums, all31historicalhidden+independentlyreconstructedhidden30, all32conv/KV, finalnorm/logits/probF32, source/dependency/referencehashes, latestuniqueZIP checks. Firstreport71224terminalsuccess. Canonicalprovenance externalrecoverymarker ishashedviaitsrepoartifactsymlinkalias, preservingarchivesandexistingdecoderpathformat; no livebuildreport mutation.

Added snapshotprotectedprove_paid_common_raw_hybrid_upgrade_guards.py; onlyafterfullproofcomplete/restored started41817. Terminalsuccess:actualpre_upgrade active/refundrefused andfailedPending/Done receipts preserved(4checks); samecandidate38d8..., noweights/inferenceexecution claimedinthese syntheticfixtures. Finallyrestored6052/cache/fullpack, removedguard snapshot3. Finalreport14099terminalsuccess incorporatesguards, independentlyredecodes35rawCandidrecords, sealsunique/latestarchive andallworkflowhashes. Additionalpost-reportaudit rehashesallreferences/workflow/archive, independentlysumsworkers andchecksall3improve; post-report-audit.json complete, goalfalse. Inlineaudit emittedlexicalSyntaxWarning foradjacentnumeric/for token; assertionsexecuted/terminalsuccess, nosemanticfailure.

Newverifiedpaidbestpointer artifacts/paid-common-raw-hybrid-v1/validated-proof-pointer.json ->paid-common-raw-hybrid-retry-v3 summary/postaudit/ZIP. Paidmodule38d8cd0f9998766b5a26d3cba2717c95e1bbd88de69a69591787a9c582a8a331; normalparent0aa01fdd75ccfb8583b0860ca2af39fdaacda9a5959d49dfa75d06bd0ed56484. Explicitgoalhidden/state/final/APIfidelity verified, butnewDenseF32directcaptureNOTperformed; flagfalse retained, notsubstitutedoldcapture. Baselinefullcanister remainsrestoredoriginal6052; candidateisverifiedartifactbest, no mainnet/protecteddeploy, no commit/PR. Allcurrentproof/report/guardhandles terminal. Goal remainsactive/unmet; nextoptimization mustuse new119862026020/103356795366/122016961354 totalsandfullcommonrawhybridparent, preserveall32checks andallinput-dependentcosts.


### Integer-basis rank343 component preparation (2026-10-07)

Current local runtime rejected relaxed SIMD dot-add capability installation (IC0505, relaxed SIMD support is not enabled). Dedicated task canister4zfnl-5t777-77775-aaadq-cai remained empty/stopped; no query or runtime change. Evidence artifacts/relaxed-dot-capability-v1/result.json.

Built s3-integer-basis-probe-v1 diagnostic7c4cb30cbfcff26c2bbf30b108933e2f3dbfdeb29cf34bc3366a9f54274574d6 with latest common-raw-hybrid control, input encoder327 nodes versus372, unchanged old651-node reconstruction kernels and36 validated patches. Independent actual Node execution of old reconstruction with new basis data passes all52 complete-memory bit comparisons (kernel-check/verified.json). Source/dependency/entry hashes reverified before installation on ONLY own empty4zfnl. Component measurement pending; no paid/full-model gain claimed. Current paidbest119862026020/103356795366/122016961354 remains above100B.

Original component checker54848 terminalexit1 atseal: IC0522 single-message40B instruction limit, sixteen raw weightchunks completed, no projection measurement. Failure retained in failed-measurement.json; target stopped. Applied existing512-row immutable coefficient preparation to fresh s3-integer-basis-chunked-probe-v1, module6012c6d30e2ce3a81fc6bd1c0cf49f7ae982a36968f2190e0aeced32372af948,36patches/6192locals. Source hashes verified; task targetreinstall92099 terminalsuccess, started, checker69968 launched. Localfake10T cycles added onlyown4zfnl. No fulltarget/relays/runtime modification.

Chunkedchecker69968 terminalsuccess: all21 native/control/candidate digests match,42 queries. Candidate slower in ALL21 conditions; size48/56/57 totals610851381/696468526/777782720 versus496812397/576901333/588396297 (22.9541/20.7258/32.1869percent increase). Independent audit37484 terminalsuccess redecodes75 replies, reruns21native outputs, verifies34 latestcontrol WAT hashes andinstruction sums. Largest immutablepreparation5,857,688,192instructions. Frozenarchive rehashed, final-audit.json records source/archive verification andStoppedown4zfnl withmodule6012... . Candidate NOT adopted; newinput327 encoding does not compensate rank343 projection/reconstruction cost. Goal remainsactive/unmet, best119862026020/103356795366/122016961354 unchanged. No live proof/check/auditprocess remains. Next pursue substantial projection/reconstruction savings against latestcommonrawhybrid, preserving full100B/all32 fidelity andinput-dependentcost scope.


### Exact signed-pair reconstruction screen (2026-10-07)

Previous goal turn wasprogress: real21condition comparison completed andrank343integerbasis candidate rejected againstlatestcontrol. Current screen_rank343_output_pair_cse.py performs signedpair commonexpression elimination overall64 rank343 leafoutputexpressions;16deterministic seeds, best645add/sub versuscanonical651. Independent topological integerexpansion inindependent-audit.json verifiesall64 expressions againstoriginalplan (no F32approximation). Same search oncurrentlyused rank49 reconstruction findsbest78versuscanonical77, so no rank49candidateadopted. Rank3436addition savings do not establishWasm/ICperformance and are not enough structural justification to adoptpreviously16–32percent slower relevantcomponent. Evidence artifacts/rank343-output-pair-cse-v1/report.json,best.json,independent-audit.json. No canister/runtime changes thisturn, no activeprocess. Fullpaidbest119862026020/103356795366/122016961354 remainsactive/unmet. Next seeklargerbaseprojection changes; historical prepackedstable/control evidence remainsnegative andmustnotbe blindlyrerun/adopted.


### Two-stage rank7 tile256 prototype (2026-10-07)

Previous goal turn completedexact signedpair screen; no adoptedgain. New two-stage rank7prototype handles256outputrows withfourleafcoefficient banks followedbythree, sharedqueryloads andonlyfourleafproductssurvive in caller-owned2048bytes/token-pair scratch. Allscratchallocation/read/write mustcount. Stages preserve integerWinograd expressions, oddrow skipsunused3/4/6, ascendingK256 F32scale/add order andexplicitseed/carry. v1fullseven-productscratch wasnotadopted;v2removeslastthree staging writes/reads. Nodefirstattempt faileddueincorrecttestweightaliasorder; preservedfailureandcorrectedcaller toK-major[0,1,2,3]sharedbankaliases. v2Node42full-memoryconditions pass including independentstagedproductsscratch andsentinels.

Currentv3 usesexistingnine-I32 ABI, scratchpointer in eighthinputtable slot,8484locals below10Klimit. FreshNode42fullmemory/scratchconditions pass (0/1/2/3/48/56/57tokens,256/512/2560cols,seed/carry includingnegativezero). Diagnosticcompileattempts1–3 failedsafelybeforecanistermutation(missingunusedmodule; mergedstubaliases; nineparameterpatcherrefusedtenparameters). Correctedattempt4build94ce1352339e2fa5167fc006ff24c8075e6cf0190cda64a75d22c82c41a70c01,36validatedpatches, first34exactlatestcommonrawhybrid; distinctseedstubsandnineparameterABI. Source/dephashesverified. Installation60610 currentlyliveonlyownstopped4zfnl; nofull4caro/relay/protected/mainnet change. ICperformancepending, fullbest119862026020/103356795366/122016961354 unchanged/goalactive.

Installation60610terminalsuccess; checker45221terminalsuccess21pairs/nativebitsallmatch, butALL21slower. Relevant48/56/57 totals533277144/615336190/627425307 versus496832474/576921410/588416374 (7.3354/6.6586/6.6295percent increase). Audit8684terminalsuccess:59rawreplies redecoded,21native outputs regenerated,34currentcontrolWATexact,42Nodefixture/sourcehashes reverifiedandarchived, finalsource/archive hasheschecked. Dedicated4zfnl stoppedwithmodule94ce..., final-audit.json saved. v4NOTadopted; originalfullpaidbest unchanged.

Createdrank7-batched256-v4kernel revision: stage2 derivesB5=B21+B3, B4=B22-B5, B6=B5-B11 fromretainedcoefficientlocals, avoidingallrawstage2reloads; only4productsscratch retained. Symbolicindependent coefficienttransitionaudit exact andI16bounds; fresh42Nodefullmemorycomparisons pass. Fresh diagnosticbuildv5module1bfed03d74aea873ed78e1f4568a04a49e29ad587b6effdf0b01cd0dd539d0e9,36validatedpatches,34actualcontrolidentities/source/dephashesverified. Installation3812currentlyliveonlyown4zfnl; nofulltarget/protected/mainnet/runtimechanges. Newv5performanceunmeasured, goalactive/unmet.

Installation3812terminalsuccess; inplacechecker33243terminalsuccess21native/control/candidatepairs allbitexact. Revision saves6,799,360instructions for8192outputrows (3,399,680 for4096) onall21conditions, independentoftokens, confirmingfixedcoefficientreloadsaving. ALLstill slower thanlatestcontrol:48/56/57 totals526477784/608536830/620625947 versus496832474/576921410/588416374 (5.9669/5.4800/5.4739percent increase). No adoption. Independent audit31725terminalsuccess redecodes59savedreplies, regenerates21nativeoutputs, verifies34latestcontrolWAT and42Nodefixturefullmemory/scratchproofhashes. Finalarchive/source rehashed, final-audit.json recordsStopped4zfnl withmodule1bfed... . Allcurrentbuild/install/check/audit handles terminal. Full4caro nevermutatedthisturn, paidbest119862026020/103356795366/122016961354 unchangedandgoalactive/unmet. Largeroutputtileswithpartialcoefficientbanks are notjustified by theseactualresults. Nextinvestigation shouldtargetlargerdot-countreduction whileincludinginputpreparation/reconstruction/storage costs; existingexactrank2304integerlift is mathematicalevidence only, notperformanceready.


### Latest rank2304 conditional scaled SLP basis (2026-10-07)

Previous goal turn wasprogress: two actualrank7batch256 componentvariants completelychecked/rejected. Currentrevalidation foundrank2304-latest-integer-lift-v1/report.json doesNOTexist; oldlatest universalI16 auditor hadnotcompleted. Runningit now terminalexit1 atfullrangebounds. Latest204coefficient basis hasactivationbound50800 andweight[-51018,50982], exceedingI16. DoNOTclaim it universallysafe ortruncate operands. Original earlier rank2304 coefficientproof isdifferentprimarydata.

Fresh audit_rank2304_latest_conditional_lift.py preservesoriginaluniversalassertion andrecordsfalse innewartifact; exact4096baseidentities plus12actual16x256/256x16 scalarI32modular dots verified withfinaldiv16/scaledbound66,584,576. Elevenconditions haveI16safeoperands, oneexplicitcoefficientmaximizing legalI8 witness doesnot andmustfall backunchanged. No SIMD/fallbackimplementation/performanceclaim.

Derivedscaledauthorbasis2L/2R withoriginalP SLP: allintermediateinputlinearforms integral; per-leafreconstructionrescalings eliminated. audit_rank2304_scaled_slp_basis.py checksSLPsymbolicoutputs againstprimarySMS, factorizedtwo-axisinputtransformagainstexpandedmatrices, two-axis90-add/suboutputprogram(withactualI32wrapping), fullscalarrawdotsall12. InitialnumericcheckerfailedbecauseNumPyscalaroperationswereunwrapped; fixedscalaraswellasvectorwrapping, freshterminalsuccess. Independentpost-audit rehashesallprimary/workflow/referencefiles andregenerates12matrix/modular/guardedrawdots; unsafeconditionusesunchangedrawmathematicalfallback. UniqueZIPsealed. Evidence artifacts/rank2304-scaled-slp-basis-v1. Newbasisneedsconditionalgate, actualSIMDexecution andICcost/fullfidelity/resourcechecks. No canistermutationthisturn; own4zfnl remainsstopped, fullpaidbest119862026020/103356795366/122016961354 unchanged; goalactive/unmet.

### Rank2304 actual modular SIMD execution and core screening (2026-10-07)

Implemented two actual SIMD Wasm exports using the same scaled author P program: universal32 uses I32 wrapping multiplication/addition, while checked16 scans all prepared operands before narrowing and falls back to raw scalar dots on unsafe inputs. I32 ring arithmetic avoids the I16 restriction for universal32; the final signed numerator remains uniquely bounded and divisible by16. This does not make unchecked I16 narrowing safe.

`validate_rank2304_modular_simd.py` executed both exports on all12 prior raw fixtures. All24 complete memory images match independently constructed matrix oracles, including products, both reconstruction stages, output, read-only operands and sentinels. The explicit I16-overflow case succeeds in universal32 and takes the checked16 fallback. Evidence: `artifacts/rank2304-modular-simd-v1/execution-report.json`.

Built and measured core-only IC update probes on ONLY own small canister4zfnl. Initial guard checker failed before installation because controllers are nested under settings; retained guard-attempt-v1 and corrected the guard. Initial v2 builder failed before mutation because a frozen script used the wrong root; retained build-attempt-v1 and corrected the explicit repository root. Both corrected probe measurements completed. The same candidate functions are embedded byteexact in both meter modules.

The looped direct SIMD control costs246649 instructions; universal32 costs230852 (6.4046% less), checked16 safe costs581068 and the unsafe scalar fallback costs2221503. After hoisting addresses and unrolling the direct control, the same raw integer dots cost74713 instructions. Universal32 is therefore208.9850% more expensive (3.08985x) than that improved screening control. All48 v2 update outputs match fresh raw integer dots. Prepared input/weight data are reset outside the measured interval; ALL input preparation, weight packing, F32 scale/add and worker orchestration remain excluded here and must be counted before any full inference claim. The direct control is not the current best rank7 full projection. Reject these implemented rank2304 prototypes as adoption candidates; do not infer a universal performance bound for every possible rank2304 implementation or a measured full inference regression.

`audit_rank2304_modular_meter.py` independently redecoded all48 raw Candid replies, regenerated raw integer references and executed all48 installed-module exports in Node. Source/entry hashes and exact candidate embedding verified; archive bytes rechecked. Frozen ZIP SHA256:7a76acdacb7b0ee228724cf04dff6942f83342967fce337253db529a568a1f7d. Own4zfnl stopped with v2 probe; full4caro remains Running with original6052cc94 baseline, own relays remain stopped. No live build/check/audit processes, runtime changes, mainnet changes, commit or PR. Best full paid totals remain119862026020/103356795366/122016961354; goal100B all3 remains active and unmet. Future work must compare against the latest common-raw hybrid and count all input-dependent preparation and reconstruction.


### Eight-lane rank7 input preparation (2026-10-07)

Previous goal turn completed actual rank2304 SIMD correctness and IC core screening, rejecting implemented prototypes. Current inspection found the latest rank7 input preparer still processes four I16 values using duplicated64-bit loads and64-bit lane stores. New eight-lane revision uses128-bit loads/stores and identical seven signed integer expressions. No quantization, operand layout, projection or F32 ordering changes. First isolated compile failed safely due reversed intrinsic store argument order; retained build-attempt-v1, corrected explicit pointer/value order.

Actual Rust-compiled old/new encoders share one probe module. All14 Node operand buffers match fresh scalar reference bytes, including read-only input/sentinels, odd n1/3/57, full/partial/empty K ranges. All14 local IC update operand buffers likewise match. Relevant48/56/57 full input-preparation counts:668893->365531,806989->453067,811513->444951, savings303362/353922/366562 (45.3528/43.8571/45.1702percent). Partial57 range3..7 saves146626; empty range unchanged200. These measure preparation only, excluding allocation/quantization/projection/orchestration; no full inference saving claimed. audit_rank7_prepare8_probe.py independently reconstructs seven coefficient-matrix outputs and redecodes everyraw Candidreply. FrozenZIP bac7853abbd32a23c755519a0e6e9c5707aa490d6c853175b6bbe847fc42f594; archive bytes verified. Ownsmall4zfnl stopped withprobe module.

Built fullnormal candidatec17b0525d53dd51b24fdd6f3641cbf34b622454d0dfed8ddb046fcc93792e2a7 (11,419,397bytes), exactlyonechangedruntimefile strassen_raw.rs andsame34validatedprojectionWATpatches. Builtpaidcandidate359bd834798c65b74741382fc5076675d050a74a7be5b3d7f775bb56b0ef0295 (11,792,143bytes). Paidlib/update scheduler/paidtypes/paidinference are byteexactlatestbest; paid-source-equivalence.json checked. No fullfidelity orwholepaidperformanceadoptionclaimyet.

Launched prove_paid_rank7_prepare8.py --run session29805, confirmedLIVE bywrite_stdin andsnapshot saved. Driver rehashescandidate/workflow, verifiesportablebaseline recovery marker, readsactualfullcache/pack/module andrequires exactoriginalbaseline equality beforemutation. Snapshotprotected proof retainsallthreeinput/all32hidden/state/final/historical/API/concurrency/refund checks, addsall32comparisonagainstlatestbest retry-v3, andfinally restoresoriginalfullcanister/deletesONLYtrial snapshot/stopsitsfreshcallers. Proof output usesROOTlexicalsymlink paid-rank7-prepare8-v1/proof toownSSDstorage, avoidingoldrelativepath failure. DoNOTduplicate/interrupt/restart this liveproof. Fullpaidbest119862026020/103356795366/122016961354 remainsunchangedand100Bgoalactive/unmet. Nextpollsame29805, monitor preparation andcompletefinalrestoration/independentreport/upgradeguards beforeadoption.


### Live prepare8 paid proof and independent reporting preparation (2026-10-07)

Previous goal turn wasprogress: measured/auditedexacteight-lanepreparation, builtfullnormal/paidcandidate, launchedsnapshotprotectedproof29805. Currentwrite_stdin confirms SAME29805LIVE; installation completed359bd834... ontoOWNfull4caro, createdOWNfreshcaller canisters5uljf-s3777-77775-aaaea-cai and5tkpr-7d777-77775-aaaeq-cai, bothcaller modules installed. Savedtrial snapshot00000000000000047fffffffffa000020101. Immutablepreparation advancedindex30/~732MB throughindex224/~2.812GB andlatest365/~3.632GB. No inference result yet. DoNOTrepeatinstallation, restorewhilelive, restartnetwork ormistakeoldstoppedrelay4xhad/small4zfnlfornewcallers.

Executed audit_paid_rank7_prepare8_sources.py terminalsuccess: bothfullmodule/source/dependencyhashesverified, all34actualprojectionWATidentities matchnormalANDlatestpaidbest, lib/paidcontract/schedulerbyteexact. Separate source-audit-readonly.json preserveslivebuild/proofmanifests andexplicitfullperformance/fidelity/goalfalse. Prepared/compiled report_paid_rank7_prepare8.py defaultplan terminalsuccess; independentreport driver reusesverifiedcompleteall32/rawCandid/report/archiveauditor, compareslatestbest andrecordspriorworker totals, retainsDenseF32directcapturefalse. Its --run isgatedoncomplete/baseline_restored/snapshot_deleted. Prepared prove_paid_rank7_prepare8_upgrade_guards.py defaultplan terminalsuccess; --run preservesfourdeterministicguardfixtures andrequiresrestoredproof BEFOREcanistermutation. No guard orfullreportexecutedyet; donotclaimplannedchecks passed.

Onlylivehandle29805. Fullpaidbest119862026020/103356795366/122016961354 unchanged,100Bgoalactive/unmet. Nextpollsame29805; uponterminalsuccessandrestoration runnewreport, thennewupgradeguards, rerunnewreport andindependentpostreport/archiveaudit beforeanybestpointer update. Keepallfullrequirements intact; prep-only44–45percent isnotfullinference reduction.


### Complete paid prepare8 result and new validated best (2026-10-07)

Previous goal turn completedsourceprovenance audit/preparedreport andguards while29805remainedlive. Current29805 advancedimmutableprep491->721/weightsready, fixedprefix andbothvoting/commonbanks ready. Actualallthreepaidinferences completed:617119784478888(4workers),620102886793233(3workers),653121936703954(4workers). Maxheap unchanged4,224,647,168bytes;everyworker<40B. All32hidden/state hashes matchlatestbest andhistoricalreferences;finalhidden/decision/logits/probabilities exact. Fullsavingsfromprevious119862026020/103356795366/122016961354 are77,547,132/470,002,133/80,257,400instructions. Allstill>100B:remaining19,784,478,888/2,886,793,233/21,936,703,954. The44–45percent preparation-only improvement mustnotbe presentedaswholeinferencereduction.

29805terminalsuccess:billing/refund/replay/conflict/insufficient/quote/bounds/auth/Paused/Busy/concurrentupgradeordering/sameversionreceipt checks passed,originalfull6052/cache/fullpack restored, ONLYtrial snapshot4 deleted andfreshcallers5uljf/5tkpr stopped. Initialindependentreport5369terminalsuccess auditedrawCandid/historicalstatesincludingindependentlyreconstructedhidden30. Thenadditionalguard10455terminalsuccess active/refundupgrade refusals andPending/Donefailedreceipts preserved;finallyoriginalbaseline restored andguardtemporarysnapshot deleted. Finalreport4052terminalsuccess incorporatesguards. audit_paid_rank7_prepare8_report.py terminalsuccess redecodes36rawCandidcalls, verifiesfullworkflow/referencehashes/latestuniquearchive, independentlysumsworkers, comparescurrentbest inputs/quotes/state/finalbits, confirmsfreshRunningoriginalbaseline. Allthreeimprove;goalfalse.

NEWvalidatedbestpointer artifacts/paid-rank7-prepare8-v1/validated-proof-pointer.json. Paidmodule359bd834798c65b74741382fc5076675d050a74a7be5b3d7f775bb56b0ef0295;normalparentc17b0525d53dd51b24fdd6f3641cbf34b622454d0dfed8ddb046fcc93792e2a7. Frozenfullproof ZIPc4a1022c413636f48e0c72229503c2424f3807b26ed75bcb02927f79f0f16890. AdditionalDenseF32directcaptureNOTperformedforthisnewbest;explicitfalse retained,oldcapturesnotreusedasproof. No livebuild/proof/report/guardhandles remain. Fulltargetremainsoriginalbaseline,notpermanentcandidate deployment;mainnet/protectedprojects/runtime untouched,no commit/PR. Goalactive/unmet. Nextoptimization mustuseNEWprepare8normalcontroland119784478888/102886793233/121936703954 totals,retainall32fidelityandallinput-dependentcostscope.


### Hoisted, unrolled rank7 prepare8 prototype (2026-10-07)

Previous goal turn completedallthreepaidprepare8proofs/guards/audits andupdatedvalidatedbest119784478888/102886793233/121936703954. Currentcandidate fixesfourinput/sevenoutputpointers perpair/Kblock andunrolls16independent eight-I16 groups. Nameprepare16 means16unrolledgroups,NOT16-lane SIMD. Currentbestprepare8Rustbody checkedbyteexactbeforecreatingprobe; pairedbinary usesidenticalfixtures/controlononenewmodule. All22Nodeand22ICoperandbuffersmatchincludinginputread-onlyandguardbytes, n0/1/3/48/56/57,cols256/512/2560/4096/9216/10240 andfull/partial/emptyranges. Nonemptycasesallimprove. Relevant48/56/57cols2560:365528->330637,453064->412365,444948->402797 (9.5454/8.9831/9.4732percent). Partial57range3..7:259280->241663;larger57cols9216:1576176->1427711. EmptyKandzero-token both200->200. Pairedcompiledcontrolcounts differfromearlierstandalonecounts because this is a new diagnosticbinary/scaffold; onlysamebinarypairedcomparison supportssavings. Allocation/quantization/projection/orchestration excluded, nofullinferencegainclaim.

Independentmatrix/Candid/Node/archiveauditterminalsuccess,ZIP60a302d5be603eae54fbe3071aafe775dbb14b1f46ae602aabaf5cf85fcbadb2. Onlyownsmall4zfnlreinstalled, measuredandstopped. Fullnormalunrolledcandidatea7f76ea328d8c1e25eb48ec90343fabcbdfb970c88f14ac2692f5ebe969a84f2 (11,425,799bytes),same34validatedprojectionWATpatchesandonlystrassen_raw.rs revision, unchangedhostscalarbody. Paidcandidate18b275b806e74a84c553bf858d5a62979a3645efae545495dca0fb2a97a126af (11,798,545bytes); lib/contract/scheduler byteexactcurrentbest. Bothbuildhandles28549/32246terminalsuccess.

Firstproofdrivergeneration failedcompileBEFOREmutation because unboundedROOTreplacement alsochangedquotedreplacementpattern; preservedattempt-v1driver/manifest andcorrectedtop-levelreplacementcount1. Fresh prove_paid_rank7_prepare8_unrolled.py --run8053LIVEconfirmedbywrite_stdin/snapshot saved. It reusesthecompleteallthree/all32/API/recoveryproofandcompareslatestprepare8savedstates, withfreshSSDproofstorageandexactlivebaselineguard. DoNOTduplicate/restart/interruptthisliveproof. Best119784478888/102886793233/121936703954 remainsunchanged,100Bgoalactive/unmet. Nextpoll8053 andverifyrestoration/finalindependentreport/guards/archivebeforebestadoption.


### Complete paid unrolled-prepare8 result (2026-10-07)

Previous goal turn measured/auditedunrolledinputpreparation, builtnormal/paidcandidateandlaunchedsnapshotprotected8053. Current8053revalidatedLIVEatindex78/~1.866GB, advanced400/~3.675GB and557/~3.867GB through721/weightsready. Model-onlypreparationunchanged252538138867instructions/4065416192bytes; noquestion-dependentcostexclusion. Bothprefixbanks ready;allthreepaidinferencescompleted:617119780973016(4workers),620102865650267(3workers),653121933075570(4workers). ALL32hidden/state andfinalhidden/decisionbits matchlatestvalidatedprepare8andhistoricalreferences. Maxheap4224647168unchanged;everyworker<40B. Saves3,505,872/21,142,966/3,628,384fromimmediatepreviousbest;ALLstill>100B,remaining19,780,973,016/2,865,650,267/21,933,075,570. Prep-only9percent mustnotbeclaimedaswholeinferencegain.

8053terminalsuccess completedbilling/refund/replay/conflict/insufficient/quote/bounds/auth/Paused/Busy/concurrency/upgradeordering/replay, restoredoriginal6052/cache/fullpack, deletedONLYtrial snapshot6, stoppedown52jen/55iczcallers. Sourceauditterminalsuccess preservesall34actualprojectionidentities andpaidcontract/schedulerbyteexact. prepare_paid_rank7_prepare8_unrolled_audits.py stagespinnedsource/report/guard/finalprogramswithrestorationgates. Firstreport39932terminalsuccess redecodesrawCandid/all32referencesincludingindependenthidden30. Guard3793terminalsuccess:active/refundupgraderefused,failedPending/Donepreserved;finallyoriginalbaseline restoredandguardtrial snapshotremoved. Finalreport3376terminalsuccess incorporatesguards. Finalauditterminalsuccess independentlyredecodes35rawcalls, verifiesallworkflow/referencehashes/latestuniquearchive, independentlysumsworkers/comparesinputs/quotes/state/finalbits, confirmsfreshRunningoriginal6052. Allthreeimprove,goalfalse.

NEWvalidatedbestpointer artifacts/paid-rank7-prepare8-unrolled-v1/validated-proof-pointer.json. Paidmodule18b275b806e74a84c553bf858d5a62979a3645efae545495dca0fb2a97a126af;normalparenta7f76ea328d8c1e25eb48ec90343fabcbdfb970c88f14ac2692f5ebe969a84f2. ZIP677b25ef36c794b3dfa8974d49f4002a51a514b7956d3a748d410fe6c8d0202c. NewDenseF32directcaptureNOTdone;explicitfalse retained. Allcurrenthandles8053/39932/3793/3376terminal;nootherprocesslaunched. Fulltargetrestoredoriginalbaseline,notpermanentcandidate deployment; protected/mainnet/runtimeunchanged,nocommit/PR. Goalactive/unmet. Nextoptimization mustuseunrolledprepare8normalparentand119780973016/102865650267/121933075570totals,preserveall32fidelity/allinputdependentcostscope.


### Historical actual quantized zero-packet screen (2026-10-07)

Previous goal turn completedunrolledprepare8paidproof/guards/independentaudits andupdatedfullbest119780973016/102865650267/121933075570. Current screen_real_quantized_zero_packets.py inspectsEXACTsavedINT16carryq_inputandproduct_q interminal-mlp-reference-v1 layers26/30forall3cases. q_input coversfull2560cols,product_q ONLYfirst1280alreadycompletedcols. Historicalprefix27 isslicedby11for617/620tocurrentprefix38;653keepsprefix27, givingactualsuffixn56/48/57. CurrentbestadditionalDensecapture remainsfalse;thesearehistoricalquantizedsamples,notfreshcurrentintermediateproof.

Transforms7/49signedinputformsandcounts zeroI16pairs. rank7n48uses16outputgroups/64packets;rank49n56/57uses6groups/32packets,completequartetsonlyandexplicitlyexcludes57finalrawtail. A49table independentlymatchescurrentrank49_raw.rsconstant. Observedtotalzeropacketfraction~0.0123–0.4percent, maximumperleaf~1.21percent. Candidate schedule needs~36.9792percent(rank7) or42.7083percent(rank49) zeropacketstobreak even:baseline4GK instructionsversusinitial2Gplus3perpacketbranchplus6Gforeachnonzeropacket(accumulatorget/setincluded). All12selectedschedulemodelsregress~55–64percent;THISpacket-gatedschedulingprototypeisnotadopted. ThisisnotactualICtiming,nofullmodelgain/regressionclaim,andnotuniversalrejectionofotherzero-skipschedules/datasets.

Audit_real_quantized_zero_packets.py independentlyindexesoriginaltoken/columnarrays(noeinsum/reshapeinputtransform), checksallleafzerocountsandcostformulas, source/carryhashes,currentruntimeA49table,uniqueZIPbytes. TerminalsuccessZIP2bf90fc3aa373078f91bb0b21b10a3d847c05c03c5d7c00c4cc0f821277d3251. NoWasm/canister/runtimechanges,noliveprocess,newfullbestunchangedand100Bgoalactive/unmet. Screenchangesnextaction:avoidper-packetzero-gatedaccumulatorsonthesesampledMLPinputs;seekstructuralreductionofdot/read/writeoverheadagainstlatestunrolledprepare8parent.


### Adjacent local set/get peephole candidate (2026-10-07)

Current validated best remains 119780973016/102865650267/121933075570 (all above 100B). generate_local_tee_projection_kernels.py replaces ONLY adjacent flat local.set/local.get of the SAME v128 local with local.tee: 524 sites across eight rank7 and four rank49 kernels. Independent audit verifies exact line replacement, v128 types and byte-identical nonlocal opcodes; both executions cover 312 complete-memory scalarordered-F32 cases including seed/carry, boundary shapes and sentinels. No arithmetic/memory access order changes. Archive f3818cb95fb02faf0e6482fd25c34495207c62acbf64616ea9330530be0570a4. Failed pre-execution generator/count and audit syntax attempts preserved/corrected; no canister mutation during those attempts.

build_local_tee_projection.py directly patches ONLY these twelve bodies into current validated unrolled-prepare8 parents, preserves other22 bodies and Rust/lib/contract/scheduler sources, independently validates all bodies and original source/dependency hashes. Normal module 5ae8f8ba47c62216a63e90f7aba5e2d41c1776a82a78b10b7db6570d56efdbcb; paid 14bdd9397f7b85e00cd55a3bc218225c68b4c78d0fb2bad8836dd1de442793c3. Runtime/compiler commands in manifests are original parent provenance ONLY; this revision performs direct body patches, no fresh Rust build claim.

Snapshot-protected full proof scripts/prove_paid_local_tee_projection.py --run launched session44797 and confirmed snapshot saved. All-three/all32/API/recovery checks reused without relaxation, additionally compare latest validated unrolled proof. Do NOT duplicate/interrupt/restart this live proof. Source audit via prepare_paid_local_tee_projection_audits.py --phase sources verifies normal/paid34 identities equal, unchanged22 against latest parent, changed12 against independently audited original/new WAT identities. Reporter/guards/final auditors staged with original restoration gates. Full IC performance/fidelity/goal success and adoption are NOT yet verified. Goal stays active; retain original best until complete restored proof, deterministic guards and independent raw-Candid/archive audit.

Session44797 freshly revalidated live: paid14bdd installed, own fresh callers5iptu-f3777-77775-aaaga-cai/5pova-id777-77775-aaagq-cai ready; preparation advanced through index46/cached1110175744. Independent-audit.json all12 line/type/nonlocal and fresh312 full-memory booleans plus every source hash freshly verified. Source audit terminal success, all four planned auditor programs prepared, no report/guards/final run before restoration. Do not launch duplicate proof; continue polling44797 and bounded preparation.log. Goal and best unchanged.


### Balanced stack-carry next candidate (2026-10-07)

Previous turn is progress (launched snapshot-protected44797). Current turn freshly polls44797 LIVE and bounded preparation log advances126 through639/~3.966GB. No duplicate/restart/interrupt. Latest proven best remains119780973016/102865650267/121933075570, no candidate full result yet.

Read-only scan of34 current WAT bodies identifies5302 separated set/get opportunities across27 kernels. Next executable prototype deliberately changes ONLY twelve rank7/rank49 kernels covered by existing312 complete-memory scalarordered-F32 fixtures:3072 sites. generate_stack_carry_projection_kernels.py replaces local.set with tee and removes a later local.get only if every intervening straight-line instruction is whitelisted, consumes ONLY intermediate stack values, leaves an empty intermediate stack, and neither accesses the held local nor crosses control/folded boundaries. Arithmetic and memory instructions remain in original order. Both total-pop and net-depth checks are required; no transform relies solely on net stack delta.

All312 actual SIMD complete-memory cases pass. Independent audit_stack_carry_projection_kernels.py verifies every3072 declared interval with an abstract HELD stack marker, explicitly proves no intervening pop touches held value, requires v128 local type, preserves every nonlocal opcode line byte-for-byte, and independently reruns312 full-memory cases. Archive49bd596628c22204af7616073c8e44b0eeeabba3190b7a2829cebcb416951daa all bytes verified. No IC/performance/full goal/adoption claim.

Next direct12-body normal/paid build via build_stack_carry_projection.py launched87586. Parent is local-tee candidate already312-case validated but its whole paid proof44797 is STILL RUNNING; do not describe it as adopted/current full best. New next candidate must wait for44797 restoration and audits before any full-target installation. No small/full canister change for stack-carry so far.

44797 freshly LIVE: all721 immutable-model preparation calls complete,252538138867 instructions and4065416192bytes; both voting/common prefix banks ready. 87586 terminalsuccess: next stack-carry normal16abb87ae3eedc89e0131c8b13f9a8869025c13d4ddf6fc313d123da57692778, paid a4b6a90a1b7593dada65c0c99c1d12a4ee57c46306440965c9d534587fd698a6. Twelve direct body patches/34total, no full Rust build claim, no installation/adoption/full result for next stack-carry. Only remaining live session44797.

44797 freshly LIVE paid617 row completed:119701499096instructions,4workers,heap4224647168,47.82sec. All32hidden/state/finalhidden/decision checks reached before printed row; saves79473920 against validated parent119780973016. Still>100B by19701499096. 620/653/API/restoration/report/guards/final notyetcomplete; no bestpointer update. Next poll44797; after terminal complete/restored/snapshot_deleted run prepare_paid_local_tee_projection_audits.py --phase report, then guards, rerun report, then final; adopt only if all audits verify. Stack-carry build87586/audit22067/generator19920 allterminalsuccess and no installation.

Current turn is progress: fresh44797 yields620102825161307(3workers),653121851280370(4workers), both heap4224647168 and all32/state/final checks passed before progress row. Savings vs previous best40488960/81795200;61779473920. All still>100B. API/concurrency/upgrade receipt/recovery are running; do not adopt before complete restored audited proof. Next stack-carry proof and staged-auditor wrappers created; sources phase completed/read-only. Next stack-carry planned sources require all3072 HELD-stack interval guards and fresh312-case booleans,22unchanged/12changed source identities against local-tee parent. No nextproof started and no nextcanister mutation.

44797 terminalsuccess: paid API/refund/replay/conflict/insufficient/version/bounds/auth/Paused/Busy/concurrency/upgradeordering/replay allpass; original6052 baseline restored with equal fullcache/fullpack; ONLY trial snapshot8 deleted; own5iptu/5pova callers stopped. First reporter33632 terminalsuccess validates raw Candid/all32 including independenthidden30 and historical decision/probabilitybits; totals119701499096/102825161307/121851280370, alltargetfalse. Deterministic upgradeguard82283 LIVE started AFTER restoration/report. No nextcandidate installation until guard restoration/final report/independent archive audit; originalbestpointer stillunchanged.


### Complete paid adjacent-tee result (2026-10-07)

Upgradeguards82283 terminalsuccess verifies active/refund refusal and Pending/Done preservation, finally restoresoriginal6052 and deletesown trialsnapshot. Finalreport81962 and final independent audit terminalsuccess:35 raw Candid calls independently redecoded, all32 hidden/state/reference finalbits, worker sums/input/quotes and unique latest archive checked; freshliveRunningoriginal6052 verified. Newbest totals617119701499096/620102825161307/653121851280370. Improvements79473920/40488960/81795200 versus previousunrolledparent. Still ALL>100B; gaps19701499096/2825161307/21851280370. Heap4224647168; everyworker<40B.

Newvalidatedbestpointer artifacts/paid-local-tee-projection-v1/validated-proof-pointer.json. Paid14bdd9397f7b85e00cd55a3bc218225c68b4c78d0fb2bad8836dd1de442793c3,normal5ae8f8ba47c62216a63e90f7aba5e2d41c1776a82a78b10b7db6570d56efdbcb. ZIPdc9650441b1545a0b74f202a8a702b3cd0dbb9727a827f9ac2a35130e5c38b20. CurrentadditionalDenseF32directcaptureNOTdone explicitfalse. Originalcanisterbaseline restored; owncallersstopped. Goalactive/unmet,no protected/mainnet/runtimechange,nocommit/PR. This is measured full improvement, not extrapolation from static524site count.

Next complete snapshot-protected stack-carry paid proof scripts/prove_paid_stack_carry_projection.py --run launched49395 AFTER priorfullproof/guardrestoration and independentfinalaudit. Latestreference now completevalidatedlocal-tee proof. Wrapper preserves all3/all32/API/recovery/livebaseline guards and source hashes. No duplicate/restart/interrupt of49395; proceedboundedpoll then staged report/guards/report/final after terminalrestore. All priorhandles44797/33632/82283/81962 terminalsuccess. Source-onlynextaudit alreadycomplete; no fullpaid/performance claim for stackcarryyet.


### Stack-carry bounded dead-store screen / full proof wait (2026-10-07)

Previous turn progress: complete adjacenttee fullproof/API/guards/archiveaudit adoptedactualbest119701499096/102825161307/121851280370; nextfullstackcarry proof49395 launched afterrestoration. Current49395 revalidatedLIVE viawrite_stdin (not inferredfromfiles). Snapshot10 saved, nextpaida4b6 installed and ownfreshcallers5gn64-6l777-77775-aaaha-cai/5bmyi-tt777-77775-aaahq-cai ready. Initialpreparation.log missing onlybecausepreparationhadnotyetstarted; psconfirmedactualpreparation child thenboundedlog progressed121/122 through386/387/~3.657GB. No duplicate/restart/interrupt.

screen_stack_carry_dead_stores.py verifiesall34 parent WAT identities and recordsread-onlyreport artifacts/stack-carry-dead-store-screen-v1/report.json. All34 havezero globallyunread local.tee variables; twelve mainrank7/rank49 kernels havezero bounded(<=500lines) flatwhitelist tee-overwritten-before-read candidates. Itstopsatfirstsame-localaccess/control/folded/unknown. ThisrulesoutONLYthese exacttwo dead-store simplifications; not a universalCFG/reorderedoptimization rejection. Arithmetic/memory/kernel/canister unchanged. Wholepaidstackcarryperformance/fidelity/goalstillpending; latestprovedbestunchanged and100Bgoalactive.

Readactualfirstdotcodeconfirmsper-productKdot chainsalready accumulateonoperandstack, so loopparameteraccumulator proposalwouldnot remove perKlocalaccumulatorgets/sets here. Historicalrank161/rank1127/prepackedstable results reread: uniformlyslower actualcontrols, rank1127corealonegreaterthanwholeS1; do notrerununchangedorclaimhypotheticalgain. Nextaction poll49395 andboundedprep, thenfullcases/API/recovery; staged prepare_paid_stack_carry_projection_audits.py phasesreport/guards/report/final AFTERterminalcomplete/restored/snapshotdeleted. No newinstalledmodule fordeadstore screen, no protected/mainnet/runtimechanges.

49395 freshly LIVE reachesweightsready:721immutablemodelprep calls/252538138867instructions/4065416192bytes,244.42sec. No paidinference row/result yet; no newbest orperformanceclaim. Only livehandle49395; do not launch anotherfullproof. Continueprefixbanks/allthree/API/restoration and stagedindependentaudits; goalactive/unmet.

49395 freshlyLIVEprefixbanksready and paid617119517490136(4workers),620102784672347(3workers),heap4224647168. Savings184008960/40488960 against latestadjacenttee best, fullall32/state/finalchecks passed before rows. 653/API/recovery/finalaudits incomplete; no bestadoptionyet.

Potential next experiment: batch independent same-length SHA256 messages in four SIMD lanes, preserving original little-endian float bytes and digest outputs. Current update_inference::digest bulk-hashes slices>=4096, so repeated per-floatupdate copying is alreadyremoved. Actual historical617 state shapes conv(3,8192), KVkeys/values(94,4,256); currentprefix38 meanssuffix56. Hidden31 large equal-size messages plus finalnorm short, conv24equal-sized andKV8equal-sized could be separate four-lane batches; mixed-size batches mustnot claim4xspeedup. Existing priorprofile hash~1.83B limitsmaximumscopebenefit; cannot solve20Bremaining alone. Any batching will require measuredcopy/transpose/padding/buffermemory and ALLinputdependentcost inclusion, actualICspeed/fidelity/API/fullworker/heap verification beforeadoption. No SHA implementation or full-runtime behaviorchange made yet; use exact currentparent and references.

Subsequent same49395 poll isTERMINALSUCCESS:653121665718770/4workers; allpaidAPI/concurrency/upgradeordering/replaychecks complete, original6052cache/fullpack restored, ONLYtrialsnapshot10deleted, own5gn64/5bmyi callersstopped. Firststackcarryreport14703 launchedAFTERterminalrestoration. Earlierincompleteentry describesbeforethisterminalpoll; do notduplicate/restart49395. GoalstillALL>100B, bestpointerunchangeduntilreport/guards/finalaudits.


### Complete paid stack-carry result (2026-10-07)

Fullproof49395 terminalsuccess, firstreport14703 terminalsuccess. Upgradeguard98532 terminalsuccess validates active/refund refusal and Pending/Done preservation; originalbaseline restored and onlyguardtrialsnapshot deleted. Finalreport7836/finalindependentaudit terminalsuccess redecodes36rawCandid calls, validatesall32 hidden/state/historical+independenthidden30/finalnorm/decision/probabilitybits, allworker sums andunique latestsummary/archive bytes andfreshRunning original6052. Newvalidatedbest119517490136/102784672347/121665718770. Saves184008960/40488960/185561600against immediateadjacenttee best. ALLstill>100B; remaining19517490136/2784672347/21665718770. Everyworker<40B/heap4224647168<4GiB.

Pointer artifacts/paid-stack-carry-projection-v1/validated-proof-pointer.json; paid a4b6a90a1b7593dada65c0c99c1d12a4ee57c46306440965c9d534587fd698a6,normal16abb87ae3eedc89e0131c8b13f9a8869025c13d4ddf6fc313d123da57692778. ZIP48c857556fc7626d72e222be0b1841d26c9b6a7f313728e51401c457e5c04919. NewadditionalDenseF32directcaptureNOTdone explicitfalse. Originalfullcanisterbaseline restored, ownnewcallersstopped; allcurrenthandles49395/14703/98532/7836terminal. Goalactive/unmet; no protected/mainnet/runtime/toolchain/pricingchange,nocommit/PR. Nextcandidate must use thiscurrentfullparent and totals, count ALLinput-dependent preparation.


### Actual same-length SHA256 batch fixtures (2026-10-07)

prepare_sha256_batch_fixtures.py terminalsuccess verifies lateststackcarrybestpointer hashes/fullproof completion/restoration, extracts36historical rawF32byte messages in9four-lane equal-length groups(all3cases×hidden/conv/KV), and requires each SHA256 to equal CURRENT verified debug hidden/state hash. Prefixslice38 for617/620,27for653; convflatten andKVkeys+values orderedconcatenation preserved. Hidden0/1/2/3,conv0/1/2/4,KV3/7/11/15 coverrealmessages. Historicalsource andrawfixture bytehashes saved in artifacts/sha256-batch-real-fixtures-v1/report.json. This is notfreshcurrentintermediatecapture ornewDensecapture.

verify_sha256_batch_fixture_bytes.cjs independentlychecksall36 rawbyteSHA256 viaNodecrypto, same-length groups andeveryfixture/sourceidentity; terminalsuccess with independent-bytes-audit.json. No SIMDcode oractualIC/performance/savingsclaim; preparation/padding/transpose/copies mustcount infuturecomponent/fullproof. No canister/runtime behavior change forfixtures. No currentliveprocess after41490terminal; originalfull6052baseline andownsmallstopped unchanged. Goalactive/unmet withlatestfullbest119517490136/102784672347/121665718770. Next concrete action implement/prove actualfour-lane SHA256prototype usingthese bytefixtures (including padding/boundaries), then compareagainstactualscalarSHA256withALLinputdependentbatchprep counted; do not blindlyadoptorextrapolate4xspeedup.


### Four-independent-message SHA256 SIMD prototype (2026-10-07)

Previous turn progress: actualallthree stackcarryfullproof/guards/independentaudit adoptedbest119517490136/102784672347/121665718770; preparedactual36matchinghistoricalhashfixtures. Current generate_sha256_four_lane.py creates ordinaryi32x4 SIMD compression ofFOURINDEPENDENTequal-length messages. K32/H256constants parsedandhashpinned frominstalledsha2-0.10.9/src/consts.rs (no downloaded/untrusted execution). Four-message rawI32loads andbyte-swap,16-wordring,64unrolledrounds/rotatedstate names andfeedforward preserveSHA256mod32arithmetic. PureWasm6-I32args compiled/typedvalidated, source andmodulehashes saved artifacts/sha256-four-lane-kernels-v1/build.json.

check_sha256_four_lane.cjs verifiesactualWasm againstindependentNodecrypto:9realmessagegroups36hashes plus19padding/boundarysizes×4lanes=76syntheticmessages, total112hashes/28batches. Lengths0/1/3/55/56/57/63/64/65/119/120/127/128/129/255/256/257/4095/4096. Entirememory includingreadonlyinput/padding/sentinels equals expectedoutput. AllPASS. Padding/setup done inNode atthisstage, NOICtiming/fullinferencesavings/adoptionclaim.

build_sha256_four_lane_probe.py buildsstandaloneRustlocalmeter usingexactcurrentpaidparent sha2/ic_cdk/candid/serde rlibhashes andcompilerflags. Nineactualfour-messagegroups embedded. scalarcontrol isCURRENTactualsha2::Sha256::digest withsamehexformatting; SIMDwindow includesallocation/zerofill/copy/padding/length,compression,laneoutput/hexformatting andbufferfree. Rawinputalreadyexists; inputgeneration/Candid/workerwrapper/possiblefull-sessionpendingcopies excluded andmustcountinanyfullproof. Firstpatch attempt failed BEFOREinstallation becauseexistingstrictpatchertargets9I32 butnewprototypehas6. Failedraw/source preserved under sha256-four-lane-meter-failed-six-parameter-v1; existingpatcherunchanged. Adapteradds3unusedI32parameters/callzeros (overheadcounted), body aritytypedvalidation retained. Correctedbuild96084 TERMINALSUCCESS.

NineparameteradapterWAT independentlyrerun throughsameactual112digest/full-memoryconditions withnonzero dummyargs23/99/123; allPASS. InitialfrozencheckerROOTpatherroroccurredBEFOREanyhash execution/canisterchange, preservedfailed-check-root-attempt.cjs, explicitROOTfixed andfreshchecksPASS. Entryhashes+nine-parameter-execution-report.json saved. MeterNOTINSTALLED; noICperformance/fullgoal/adoptionclaim. No currentliveprocess. Nextactionfreshownsmall4zfnl status/owner/cycles guard, installONLYthislocaldiagnosticmeter, actualall9×scalar/SIMD updates andrawCandiddigest/counterindependentverification; stopownsmallafterrun. Do notmodifyfull4caro/protected/mainnet/runtime/toolchain/pricing. Goalactive/unmet and fullbest unchanged.

SHA256component probe module c8096de1f53128612ca38e16465e4216405df2c61bb298bf6caba9d3e0708fc2; artifacts/sha256-four-lane-meter-v1/probe.wasm andprobe.did/build.json. Purekernel andadapter both112fullmemorytestspass. Futurechecker mustmatchall4returnhashes against groupmessagesexpected, sum/compareactualCOUNTERS(includeallSHA-dependentpadding) andfreshrestorestateproofbeforeanyfulladoption. AdditionalDensecapturefalse remainsunchanged.

### レビュー対応計画の実施状況（2026-10-07）

作業場所は `/Volumes/KINGSTON/ICP/IC-Imajev` の既存チェックアウト。今回ワークツリーは作成していない。

`inference_step` の測定開始点が Rust 関数内にあり、CDKの引数デコード等を差し引いていた点を修正した。現在のソースはメッセージ開始からの `performance_counter(0)` を記録する。ただし、その後のメトリクス更新とCandid返却処理はまだ含まれない。値は途中のチェックポイントであり、全worker費用を満たす達成判定には使えない。

`build_paid_message_checkpoint.py` で既存の最良候補を再コンパイルした。レビュー対象のカウンタ変更以外のRustソースを保持し、最良Wasmから抽出した34個の投影カーネル本体を移植して、全34個の本体ハッシュ一致とwasmparserの型検証を確認した。ビルド成果物は `artifacts/paid-message-checkpoint-v1/build/full.wasm`、SHA256は `4b420af0e94d754f8370d0da2f5e1e2fb1e99706693f2ccd9643bf7e2da06f96`。未インストールであり、全推論・API・復元の再検証は未完了。既存の検証済みポインタは更新していない。

SHA256候補は専用小型Canister `4zfnl-5t777-77775-aaadq-cai` の所有者・停止状態・旧モジュール・Cycles残高を確認して実測した。9組それぞれscalar/SIMDのupdateを実行し、SIMD側のpadding・確保・copy・圧縮・出力変換・解放を含む測定区間で50.6197〜50.9050%削減した。全72個のハッシュは実入力バイト列のSHA256と一致した。`audit_sha256_four_lane_meter.py` が独立したPythonデコーダで18件の生Candid応答を再デコードし、全命令数・ハッシュ・比較値を検証した。結果は `artifacts/sha256-four-lane-meter-v1/report.json` と `independent-ic-audit.json`。

この比較は部品の測定である。推論中の入力生成・バッチ待ちの保持やcopy・workerラッパー・返却処理は全推論への組み込み時に別途計上する必要がある。最初の2回の補助デコーダビルド失敗はpanic/ThinLTO設定によるもので、Canister変更前に発生し、失敗した作業ディレクトリを保存した。実測終了後、小型CanisterはSHA256診断モジュールのまま停止した。主Canister `4caro-hl777-77775-aaaba-cai` は前後とも元の6052cc94…モジュールでRunningだった。

外側からの費用計測も調査した。PocketICのライブtopologyにはApplication/Normal/Production設定があるが、使用中サーバーは16.0.0を表示する一方、公式16.0.0配布物のバイナリと一致しない。公式配布assetのSHA256を検証した上で解凍後のハッシュを比較し、公式 `781f643d…` に対して使用中 `7386b4cf…` だった。固定コミットの公式費用設定だけから使用中サーバーのCycles→命令数換算を証明したとは扱わない。調査資料は `artifacts/whole-message-accounting-research-v1/`。ネットワークやランタイムの再起動・置換は行っていない。

残る順序は、全workerラッパーと返却末尾を含む計測／保守的上限の検証、現在の候補で3入力の内訳を再計測、主要なINT8投影の削減、SHA256組み込みの全費用・全32層一致検証、最後に全3入力のAPI・復元・達成判定。現時点ではSHA256を全推論へ採用していない。検証済みの従来チェックポイント合計は617/620/653で119,517,490,136 / 102,784,672,347 / 121,665,718,770のまま。全範囲の1000億目標は未達成。

### CDK返却後のworkerチェックポイント（2026-10-07、次のgoalターン）

前ターンは開始点修正とSHA256の実IC実測によるprogress。現在のファイル・固定CDK 0.20.3マクロを再確認し、同期workerのCDKラッパーがCandid返却を終えて戻った後にPC0を保存する診断を実装した。

`wasm_worker_checkpoint.rs` は新しいvoidラッパーを追加し、workerのexport参照だけをラッパーへ向ける。元workerの関数本体はそのまま保持する。Rust側では成功した正規workerだけが記録をarmし、infer callbackで保存済み値をworkerメトリクスへ移す。未arm呼び出しでは前の値を保持する。型が固定されたarm/getterの2つの診断stub以外の既存関数本体を保持し、全Wasmをwasmparserで検証する。公開paid Candidのフィールド・メソッド型は変えていない。

`build_paid_wrapper_checkpoint.py` による診断候補は `artifacts/paid-wrapper-checkpoint-v1/build/full.wasm`、SHA256 `35e2a2597ea50526b94950c5e64762ba2e668352447c7f05110dcb180857640e`。34投影カーネルの本体は検証済み親と一致し、Nodeエンジンの全Wasm検証も通過した。診断候補自体は未インストールであり、paid全3入力・全32層・API・復元の検証はまだない。

`check_worker_wrapper_checkpoint.cjs` は実Wasmを実行し、返却後の保存、未arm呼び出しでの保存値維持、次workerでの更新を確認した。さらに `prove_worker_wrapper_checkpoint.py` が所有者・停止状態・旧SHA256診断モジュールをガードした専用小型Canisterだけで実ICテストを実行した。初期値0、返却後21,559、未arm update後も21,559、10,000回の追加loopを入れた次worker後81,561を確認した。全9応答の生Candidを保存し、`audit_worker_checkpoint.py` が再デコード、カウンタ一致、元workerと他の関数本体保持を独立監査した。

小型Canister `4zfnl-5t777-77775-aaadq-cai` は `423f27d23fdbede36f3d74f345b015d7172c311b60667728aa01209421f2e5a4` の返却後計測probeで停止。主Canisterは前後とも元6052cc94…モジュールでRunning。全ネットワークの再起動・ランタイム交換・mainnet変更なし。

この値は元の同期workerの引数デコード・実行・メトリクス更新・Candid返却を含むが、記録wrapper自身にはカウンタ取得後の3つのstore/constant命令と構造終端が残る。これらの費用の上限をまだ証明していない。またinfer入口・helper・callback等の費用も全範囲の判定で別途必要。全1000億目標を達成したとは扱わず、既存検証ポインタは維持する。次はこの診断候補の全3入力のpaid proofと末尾・補助費用の計上を進める。
