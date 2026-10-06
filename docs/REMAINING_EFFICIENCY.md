# 残る効率化の調査（2026-10-05）

現在の採用module bdd8ced5…・BOOM DAO617を、保存済み全50queryの要求でprofileし、全返信payloadをbit比較した。通常推論は132 token＝prefix45＋suffix87、234,521,052,335命令、149,353,220 Candid bytes、50queryのまま。今回は既存推論moduleを変更せず、投影候補だけ独立した診断canisterで測った。

## 全体の計測

profile合計234,524,124,078命令（通常実行より3,071,743多い）。inclusiveなbridge/pair/evaluateを重複加算せず、実装の計測区間を集計した。割合はprofile合計に対するもの。

|区間|命令|割合|
|---|---:|---:|
|INT8基底投影＋K継続|155,530,462,204|66.32%|
|F32 LoRA A/B|26,990,376,370|11.51%|
|wire decode/encode|4,037,810,153|1.72%|
|GQA head評価|5,146,552,648|2.19%|
|未分類（Delta、復元、norm、codec等を含む）|42,818,922,703|18.26%|

基底投影の区間はdotだけでなく、その準備・検査・scale処理も含む。重み読み出し区間は合計7,804,200命令と小さい。全部を新たなcacheへ置けば大幅削減できる状態ではない。LoRA A/Bとwireの全除去を仮定しても32queryには届かない。

32×5B＝160Bの名目予算へは74,521,052,335命令（31.77585%）の削減が必要。INT8基底区間だけで達成するなら47.91412%減が必要。今の合計を5Bで割った名目下限は47queryで、分割を詰めるだけで32にはならない。IC全体counter・Candid処理・依存関係・2MB境界を含めれば条件はさらに厳しくなる。

再集計：`scripts/analyze_remaining_efficiency.py --profile artifacts/remaining-efficiency/profile-v1/report.json --production artifacts/single-quad/tail-proof-v2/617/report.json --output <新しいJSON>`。根拠はprofile-v1/report.json、validated-source.zip、全保存要求・返信とanalysis-v1.json。計測50queryに加え、標準終端1queryを別の診断として記録している。

## 試した候補

一時変数共有：S1128行版の一時planeはquartet内で読み終わるため、出力グループ間で再利用できた。locals9,500→7,516（1,984削減）、演算token順序は不変。5実入力＋6境界の22通常query、native digest一致。命令数は従来128行版と全11条件で完全に同じ。命令削減として不採用。256行版は18,748localsで拒否された経緯があり、この一時変数共有だけで10,000以下になることはない。

奇数末尾：S1は2tokenで計算する。奇数tokenの末尾では第2tokenの整数operandがゼロになる。最初のtoken pairで重みを従来どおり準備した後、最後の1tokenだけproduct1/3/5を省く。product3は明示ゼロ、他の残存出力の整数再構成・F32変換・scale・K256加算順は元のまま。n=1は従来の初回pairを使う。

第1版は各pairで末尾を判定し、主Q0.8199%減だが80token等で約0.02%増。第2版は偶数の上限を一度計算し、完全pairと末尾を分けた。5実入力＋11境界（1/2/3/5/7/8/9/32/64/88/109）の32通常queryでnative・同module32行対照とdigest一致。

|入力|従来128行版|末尾分離版|変化|
|---|---:|---:|---:|
|prefix45|507,814,831|499,820,591|−1.5742%|
|主87|952,043,488|944,049,248|−0.8397%|
|情報不足80|868,593,200|868,598,960|＋0.000663%|
|最大変更89|973,248,348|965,254,108|−0.8214%|
|cold132、出力4096行|718,100,628|718,103,508|＋0.000401%|

測定対象はlayer3の実Q8192×2560重み、coldは4096出力行。counterは入力復元・量子化・operand準備・投影を含み、digestとCandid処理を除く。固定重み準備は各診断65 updateで、推論queryと分けた。旧128行版との比較は同じ凍結済みRust診断module・同じ32行control bodyからwide bodyだけをpatchし、元要求/hash/digestも照合した。単回時間は改善率に一般化しない。

この候補は投影単体まで。全モデルのhidden/state/readoutの一致、全体の命令削減率、query数削減は未検証。全モデルへの次の接続では、奇数かつ3token以上だけ新bodyへdispatchすれば偶数・単tokenへの増加を避けられる。採用された1token専用quadとは別の最適化である。

生成器はgenerate_s1_scratch_reuse.py / generate_s1_odd_tail.py。check_s1_wide.pyは別build-directory・追加source・境界token列を指定でき、古い証跡の上書きを拒否する。生成器、WAT、patchのhash、検証時source ZIPを保存した。artifacts/remaining-efficiency/{scratch-build,scratch-check,scratch-comparison.json,odd-build,odd-check,odd-build-v2,odd-check-v2,odd-comparison-v2.json}を参照。生成物はgitignore。

## 次の優先順位

1. 奇数末尾に完全pairの条件判定除去を加えた候補を、現在の短文構成の全体推論へ一時適用した。BOOM DAO 3入力で0.943～1.064%減少し、全117 queryの返信・最終判断と確率がbit一致。39 queryは変わらず、既定へは未採用。比較後にsnapshotで元のmoduleと重みcacheを復元した（[追加最適化の実測](INT8_PAIR_BOUNDS.md)）。旧長文の5条件全体比較とは区別する。
2. F32 LoRAの64出力共有は追加診断で、保存済み実入力のgate-Aで1.545～1.551%、down-Bで3.652～3.711%の命令削減を確認した。元の32行配置とF32 mul/addの列順を維持する。全体推論への接続は未実施で、全体の削減率やquery数の変化は未計測（[追加診断](F32_OUTPUT64.md)）。
3. 複数token INT8のさらに深い整数行列分解・tile構成を探索する。入力scaleを変えず整数block内で完結させる必要がある。以前のpair-factorは主で＋4.4%、256行版はlocal上限で拒否。乗算数の理論減少だけでは採用しない。

activation scaleをtoken単位へまとめる既試験はラベル23/23一致でも確率差最大10.9ppで、同等精度とは扱えない。LoRAのINT8化、adapterのbaseへのmerge再量子化、token/prefix削減も数値同一性を失うため別の判断精度評価が必要。今回の候補には含めない。

独立診断canisterは6y4zs（scratch）、6757g（末尾第1版）、6w6u2（末尾第2版）。全体推論6eyddはbdd8ced5…のまま読み取り確認。保護対象・Layaのソース/Git/canisterは変更しない。mainnet/push/PRは行わない。
