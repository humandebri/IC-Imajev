# Delta初期状態の転置往復を省く

2026-10-03。`experimental-delta-state-layout` を追加。通常query、クライアント保持の状態、固定INT8モデル、元F32 adapter/readout/calibrationを維持する。packetのbyte形式、token数、queryの入力・出力上限を変更しない。

`InitialState::ValueMajor` / `KeyMajor` で内部配置を区別する。通常の状態保持Deltaとprefix記録では従来のvalue優先を維持。suffix全tokenを一度に計算する経路では、ログ復元が生成したkey優先の状態をそのまま再帰演算に渡す。14 headsのログ復元後の転置と、それを戻す初期転置を省く。dense18 headsの復号は既存decoderで行い、4×4 SIMD tileのbit単位転置を同じbuffer内で一度だけ行う。Delta用のhead作業buffer確保・zero fill・コピーを省く。

public `delta_from_key_major` は `state[key * dv + value]` を受け取り、最終状態も同じ配置に更新する。各valueのkey方向の積和順、乗算と加算の分離、BF16の丸め境界を維持。`delta` は従来のvalue優先の状態更新を維持する。型付き復号結果と配置identityはqueryからDeltaまで一緒に渡し、request identity・有限値・shape・モデルの検証を維持した。

## 検証

ネイティブの候補feature構成81 unit、単独feature62 unit通過。新規4 testsは36のDelta形状で出力と最終状態を比較し、log復元9条件、typed packet復号と不正入力、任意のF32 bit列の転置・逆転置を確認する。Wasm診断24通常queryでも、初期状態の物理配置だけを変えて出力digest一致。87-token/128×128の再帰kernelは978,925命令（6.90403%）減。診断のkernel counterは初期配置の準備を含まないので、全モデルのnet削減は下記で別測定した。

専用canister `6eydd-o3777-77775-aaama-cai`、module `1f445089b66616843c84a125036f83f45a002a55c67262d037388d78d75363fe` で全5条件完走。全32層の保持hidden/state、判断4条件のvalue/abstain/logits/probabilities/unknown/最終hiddenが既存INT8版とbit一致。型付き出力検査通過、失敗/replay0。

|条件|query|合計命令|前回候補から命令減|Candid bytes|秒（単回）|最大query命令|
|---|---:|---:|---:|---:|---:|---:|
|prefix|66|136,486,268,612|2,436|71,105,691|11.810|2,399,051,830|
|baseline-617|66|262,530,038,983|957,409,314|113,697,747|52.755|4,555,649,996|
|617|64|260,701,622,342|767,186,917|124,634,637|21.978|4,555,649,996|
|insufficient|64|239,083,308,722|743,845,749|117,701,589|21.042|4,177,949,786|
|maximum|64|267,102,238,220|730,493,877|126,615,477|28.431|4,666,358,012|
|normal|291|402,505,196,433|-1,740|580,218,522|53.578|2,477,351,034|

主問題は前回のno-writeback候補から767,186,917命令減（−0.29341%）。Delta23 queryで735,220,990命令、fused prefix-startで31,966,161命令減。他の演算queryは各6命令増、合計234命令増。全体差と単独kernelの推定値を区別する。通信量とquery数、観測heap終了値は同じ。cold132の分割Deltaは状態保持が必要なため省略対象外で、計1,740命令増。prefixも微小なcompiler差がある。ピークメモリの削減とは主張しない。

固定重み準備721 update、61,042,755,368命令、271.079秒。cache payload4,065,416,192 bytes。二度目のprefix packet準備query/命令/通信0。通信はCandidのみ、handler counterはCDK encode/decodeを含まない。時間は単回で、測定の一部は独立診断と並行しているため速度向上率として扱わない。主canisterとLayaは変更しない。

既存INT8版との一致であり、公式BF16との差や既存の最大変更の見逃しは残る。50/32 query未達。主合計260,701,622,342命令は5B/queryで理論下限53 query。50へは少なくとも10,701,622,342命令の追加削減とquery配置が必要。

## 次の演算配置を実測

保存済みMLP完了入力と次Deltaを別々の通常queryで測定した。prefix packetは元log/stateへhash・bitで照合。各出力も保存済みWasm出力とbit一致。

|MLP層→次Delta|MLP完了命令|次hybrid Delta命令|個別query命令の合計|5Bまでの差|
|---|---:|---:|---:|---:|
|0→1|1,309,652,340|3,683,403,884|4,993,056,224|6,943,776|
|1→2|1,309,637,922|3,683,600,532|4,993,238,454|6,761,546|
|3→4|1,309,766,289|3,683,764,406|4,993,530,695|6,469,305|

この合計は融合後の実測ではない。残るheadroomは約647〜694万命令で小さい。融合時には中間encode/decodeを省く一方、2 MB内でMLP carryとprefixを渡すcodec費用が必要になる。MLPをprepare/finishに分けて次Deltaと結ぶだけでは、元のMLP1＋Delta1に対して2 queryのまま。query数の削減には次段の一部演算までcarryする配置が必要で、融合だけで50達成とは判断しない。

## 証拠・再現

- full build: `artifacts/prefix_codec/full-build-state-layout-v2/source.zip` / `source-hashes.json` / `features.txt` / `patch.json`。feature依存を整理したv2は最初のbuildとbyte同一で、cache再準備は不要だった。
- 全モデル: `artifacts/prefix_codec/full-state-layout-proof/report.json` / `validated-source.zip` / `comparison-no-writeback.json`。
- 固定準備: `artifacts/prefix_codec/full-state-layout-preparation/report.json`。
- 診断: `artifacts/prefix_codec/delta-state-layout-final-check/report.json` / `validated-source.zip`。
- 段間の測定: `artifacts/prefix_codec/stage-hybrid-layout-budget/report.json`。

```sh
.venv/bin/python scripts/build_full_prefix_candidate.py --directory artifacts/<new-build> --target-directory artifacts/prefix_codec/full-target --direct-input --terminal-attention --prefix-start --delta-state-layout
# raw.wasmのfail-closed stubを、既存patcherでdirect-input.watへ置換・validateする。
# 専用canisterへupgradeして固定重み準備後、通常queryで比較。
.venv/bin/python scripts/check_full_prefix_hybrid.py --canister <test-id> --codec-canister <codec-id> --wasm artifacts/<new-build>/full.wasm --directory artifacts/<new-proof> --terminal-attention --prefix-start
.venv/bin/python scripts/measure_stage_hybrid_budget.py --canister <test-id> --wasm artifacts/<new-build>/full.wasm --directory artifacts/<new-stage-report> --packets artifacts/<saved-proof>/packets --prefix-directory artifacts/<saved-proof>/prefix
```

新しいfeatureはprefix hybridとno-writebackを依存に含む。生成物はignore配下へ置く。
