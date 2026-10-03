# 使わないDelta最終状態の書き戻しを省く

`experimental-delta-no-writeback` は、全tokenを同一queryで処理し最終Delta状態を破棄する `delta_full_log_integer` の `keep=0` 経路だけに適用する。漸化式・積和順・F32/BF16の丸め・INT8重みを変更しない。通常の分割Deltaとprefix記録 `keep=1` は状態を保持する。

従来のWasm SIMD実装は、値優先の初期状態をkey優先の作業bufferへ転置し、出力計算後に最終状態を値優先へ書き戻していた。破棄する経路では最後の転置・storeを実行しない。`delta_without_final_state` のscratch最終内容はAPI上未規定。状態を次queryへ渡す呼び出しは従来の `delta` を使う。ネイティブfallbackは従来どおりscratchを更新してよい。

最終storeの省略に限定し、初期状態の転置、prefix packet内の14 heads分のlog復元、入力復号・検証は残る。主問題24 Delta層×32 heads×128×128×4 bytes = 50,331,648 bytes（48 MiB）の不要な書き戻しを除く。これは命令数やピークメモリの削減量そのものではない。

## 再現

```sh
cargo test --offline --manifest-path scripts/delta_writeback_bench/Cargo.toml --target-dir artifacts/delta-writeback-target -j1
RUSTFLAGS='-C target-feature=+simd128' cargo build --offline --release --manifest-path scripts/delta_writeback_bench/Cargo.toml --lib --target wasm32-unknown-unknown --target-dir artifacts/delta-writeback-target -j1
cargo build --offline --manifest-path scripts/delta_writeback_bench/Cargo.toml --bin writeback_args --target-dir artifacts/delta-writeback-target -j1
# 診断用canisterへWasmをinstallし、owner principalでinitした後
.venv/bin/python scripts/check_delta_writeback.py --canister <diagnostic-id> --directory artifacts/<new-diagnostic-report>
.venv/bin/python scripts/build_full_prefix_candidate.py --directory artifacts/<new-build> --target-directory artifacts/prefix_codec/full-target --direct-input --terminal-attention --prefix-start --delta-no-writeback
# raw.wasmはfail-closed stub。既存のABI検証patcherでdirect-input.watを適用したfull.wasmだけを使う。
# 専用canisterへupgradeし、固定重みを準備してから
.venv/bin/python scripts/check_full_prefix_hybrid.py --canister <full-test-id> --codec-canister <codec-id> --wasm artifacts/<new-build>/full.wasm --directory artifacts/<new-proof> --terminal-attention --prefix-start
```

診断Wasmは実際のruntime Delta SIMDソースを直接includeし、無関係なINT8 projection特殊化をコンパイルせずDeltaのみを計測する。ネイティブtestはruntime APIを使用する。小型診断の入力は決定的な合成値であり、全モデル・判断精度の証拠には使わない。Wasm同士の出力digest一致と、scratch書き戻しが省かれたことを確認する。命令counterは入力生成・digest・Candid encode/decodeを含まない。全モデルのhandler counterと通信は別に集計する。

## 検証結果

Rustネイティブの36形状・gate範囲外の拒否を検証した2 testsと、候補構成のruntime unit77件は通過。診断canisterの24通常query（12形状×従来/省略）で出力digest一致。省略経路で元のscratchが全bit維持され、従来経路では更新されることも確認した。

|tokens|従来kernel命令/head|省略kernel命令/head|削減率|
|---|---:|---:|---:|

|1|2,003,680|1,131,222|-43.5428%|
|7|2,913,994|2,041,536|-29.9403%|
|8|3,065,713|2,193,255|-28.4586%|
|32|6,706,969|5,834,511|-13.0082%|
|45|8,679,316|7,806,858|-10.0522%|
|80|13,989,481|13,117,023|-6.2365%|
|87|15,051,514|14,179,056|-5.7965%|
|89|15,354,952|14,482,494|-5.6819%|
|132|21,878,869|21,006,411|-3.9877%|

128×128条件では毎head872,458命令減。主問題24層×32 headsの単純合計670,047,744命令はkernelからの推定であり、全体実測の代わりに使わない。小型dv=4/12/16経路もbit一致。単体証拠は `artifacts/prefix_codec/delta-writeback-check/report.json` / `validated-source.zip`。全体測定は下記。

主suffix87は直前のprefix-start候補262,163,826,219→261,468,809,259命令、695,016,960命令減（−0.26511%）。Delta23 queryで666,057,920命令、fused prefix-startで28,959,040命令減。Attention7、MLP31、terminal統合1の命令は同一で、今回の差はDelta経路だけに現れた。query64、通信124,634,637 Candid bytes、最大query4,555,649,990命令は同じ。prefix準備も従来と同じ命令数。全条件比較は下記。

## 全体実測

専用canister `6eydd-o3777-77775-aaama-cai`、module `3dbdf889df5e0ecb35ffe19123caebe62ccb6f24cb35d50a37af5ae6362e8d3a` で全5条件を完走。全32層の保持hidden/state、判断4条件のvalue/abstain/logits/probabilities/unknown/最終hiddenが既存INT8版とbit一致。型付き出力検査通過、失敗/replay0。

|条件|query|合計命令|直前候補から命令減|Candid bytes|秒（単回）|最大query命令|
|---|---:|---:|---:|---:|---:|---:|
|prefix|66|136,486,271,048|0|71,105,691|20.195|2,399,051,824|
|baseline-617|66|263,487,448,297|695,016,960|113,697,747|20.963|4,555,649,990|
|617|64|261,468,809,259|695,016,960|124,634,637|20.421|4,555,649,990|
|insufficient|64|239,827,154,471|695,016,960|117,701,589|19.029|4,177,949,780|
|maximum|64|267,832,732,097|695,016,960|126,615,477|28.173|4,666,358,006|
|normal|291|402,505,194,693|0|580,218,522|53.274|2,477,351,028|

通信量・query数は直前候補と同じ。cold132は通常の分割Deltaで状態保持が必要なため今回の省略対象外。初回prefix packet準備は別集計し、二度目は準備query/命令/通信0。固定重み準備721 update、61,042,755,368命令、312.740秒。固定cache4,065,416,192 bytesで変更なし。観測heapはquery終了値でありピークではない。通信はCandid、handler命令はCDK encode/decodeを含まない。時間は単回測定で速度向上率を一般化しない。

証拠は `artifacts/prefix_codec/full-no-writeback-proof/report.json` / `validated-source.zip`、直前候補との比較は `comparison-prefix-start.json`、固定準備は `full-no-writeback-preparation/report.json`。生成物は既存ignore配下。主canister module36c04a57…の不変をcertified readで確認。Layaは操作していない。

既存INT8版に対する一致であり、公式BF16版との差や既存の最大変更見逃しを解消したという意味ではない。50/32 queryは未達。主合計261,468,809,259命令の下限は5B/queryで53 query。50へは少なくとも11,468,809,259命令の追加削減と配置最適化が必要。

次の繰り返し処理：`delta_log::restore_simd` はkey優先で復元後にvalue優先へ転置し、`delta_simd::run` が再びkey優先へ転置する。hybrid packetの未展開14 headsでこの往復がある。dense18 headsもdecode後に別bufferへ転置する。型で内部配置を区別し、復元・復号から再帰演算へ直接渡せば、この往復を省ける可能性がある。未実装・命令未測定で、固定重み演算と判断精度の変更は不要。
