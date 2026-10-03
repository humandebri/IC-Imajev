# queryごとに繰り返す処理の削減

2026-10-03追加：S1の7係数で繰り返す共通byte offsetをtoken pairごとに一度計算し再利用。全5条件の保持hidden/state/判断bit一致、graph失敗/replay0。主632,813,440命令（0.25535%）減、247,184,414,792命令。63 query・通信・観測heap終了最大は同じで50/32未達。固定pointer再利用も単体で出力一致だが1token増加により未接続。[実装と実測](S1_ADDRESS_REUSE.md)。

2026-10-03追加：S1入力係数の全領域ゼロ埋め直後の全上書きを除去し、一度だけ書き込む。全5条件の保持hidden/state/判断bit一致、graph失敗/replay0。主353,828,218命令（0.14257%）減、247,817,228,232命令。63 query・通信・観測heap終了最大は同じで50/32未達。S2共通変換再利用3候補は現行S1より増加し未採用。[実装と実測](OPERAND_WRITE_ONCE.md)。

2026-10-03追加：最終Attention/MLP/normと専用F32判断を1通常queryへ統合し、最後のhidden再送を除去。主64→63 query、495,771命令・Candid10,619 bytes減。全5条件の保持hidden/state/判断bit一致、graph失敗/replay0。同じmoduleの2→1呼び出しcontrolとqueryなしcheckpoint再開も検証。固定payload・観測heap終了最大を維持。50/32未達。[実装と実測](TERMINAL_DECISION.md)。


2026-10-03追加：INT8固定scaleのqueryごとの読み出し・byte→F32復号・正値/有限値再走査を除去。固定payload不変、全5条件の保持hidden/state/判断bit一致、失敗/replay0。主356,843,875命令（0.1436%）減、248,171,552,221命令。64 queryと通信は同じで50/32未達。観測heap終了最大は+2,424,832 bytes。第二候補32a3e1d5…を実験用基準とする。[固定scaleの実装と実測](FIXED_SCALE_REUSE.md)。


2026-10-03追加：LoRA Aのinput転置を省き、元容量の固定output32配置とK64部分和で全tokenのweight loadを共有。全5条件の保持hidden/state/判断が既存INT8版とbit一致、失敗/replay0。主さらに24.683億命令（0.9834%）減、248,528,396,096命令。64 query・通信は同じで50/32未達。主MLPを再profileし、base77.58%、A合計4.61%、B7.42%を実測。[F32_A_REUSE.md](F32_A_REUSE.md)。

2026-10-03追加：固定F32 LoRA Bを元容量のoutput32配置にし、全tokenでweight loadを共有、query input転置を除去。全5条件の全層hidden/state/判断が既存INT8版とbit一致、失敗/replay0。主さらに46.333億命令（1.8125%）減、250,996,660,791命令。cache4.065 GBを維持、64 query・通信は同じで50/32未達。LoRA AのK64診断も主18.738%減・22測定queryでdigest一致し、次の全モデル接続候補とする。[F32_WEIGHT_REUSE.md](F32_WEIGHT_REUSE.md)。

2026-10-03追加：元容量のINT8四象限配置を一度準備し、query内のS1入力変換・初回weight展開を共有。全5条件の全層hidden/state/判断が既存INT8版とbit一致、失敗/replay0。主50.717億命令（1.9454%）減、全条件0.804〜2.569%減。固定cache容量4.065 GBを維持。主64 query / Candid124,634,637 bytesで50/32未達。詳細と再現：[STRASSEN_RAW_REUSE.md](STRASSEN_RAW_REUSE.md)。

2026-10-03。固定の準備と質問ごとの演算を分け、入力・状態のコピー、prefixの再計算、演算間の往復を減らす。中間状態はクライアントが保持し、推論は通常query。モデル準備だけupdateを使う。Layaと主canisterは変更しない。

試作追加：2 symbol/4 byte復号とraw選択でdecoder単体約60%減。512行分割の87-token融合がbit一致で初完走したが、合計命令・通信は増え、既存graphには未採用。[CARRY_PAIR_DECODER.md](CARRY_PAIR_DECODER.md)。

前段の試作：毎queryのDEFLATE展開を省く可逆raw/constant/Huffman carryを実装。45-token診断で復号約12.8〜14.3%減、出力bit一致。87-token融合はまだ5B超過で未採用。[CARRY_HUFFMAN.md](CARRY_HUFFMAN.md)。

最新採用候補の削減：Delta初期状態の転置往復・head作業buffer確保を省き、全5条件で出力bit一致。主suffix64 query / 260,701,622,342命令 / 124,634,637 Candid bytes。前回no-writeback候補から767,186,917命令（0.29341%）減。[DELTA_STATE_LAYOUT.md](DELTA_STATE_LAYOUT.md)。以下は前段の削減記録。

## 削減対象

|処理|実装|検証範囲|
|---|---|---|
|固定重みの並べ替え|モデル準備updateで出力pair配置を作る|全モデルの固定cache|
|INT8入力の複製|元の量子化I16 bufferをWATから直接読む|全5条件で旧INT8版とbit一致|
|共通45-token prefixの推論|canister生成結果をクライアントに保存|全層hidden/stateのbit比較|
|prefix innovation log全体の再生|一度生成した可逆NPF1 packetを保存しDeltaへ渡す|再準備query/命令/通信は2回目0。14 heads分のlog再生は残る|
|最終Attention→MLP間の往復|最終Attention・残差・MLP・最終normを1 queryに統合|全5条件と全判断出力のbit比較|
|復号後のDelta状態コピー|所有権をqueryから再帰演算へ渡す|全5条件と全判断出力のbit比較、主問題52,675,468命令減|

## 最終層の統合

`experimental-terminal-attention` / `--fuse-terminal-attention` は最終層31に限定する。prefixの準備では使わず、最終残差だけを入力へ追加する。Attentionの新KVはクライアントへ返す。既存の積和とBF16丸めをそのまま使い、最終MLPは最後のtokenだけ計算する既存経路に接続した。decision queryは別のまま。

専用canister `6eydd-o3777-77775-aaama-cai` のmodule `0de0a40d66492d41dc95cfed0fb1aa635117e8f907c2cf623d8453b4edc902e2` で完走した。

|条件|query|合計命令|Candid送受信bytes|秒（単回）|最大query命令|
|---|---:|---:|---:|---:|---:|
|prefix45|66|136,487,322,987|71,105,691|11.567|2,399,051,818|
|主suffix87・旧log|66|264,177,813,678|113,697,747|21.296|4,555,650,095|
|主suffix87・packet再利用|66|262,243,773,917|125,973,196|8.864|4,555,650,095|
|情報不足suffix80|66|240,601,458,917|118,932,610|26.150|4,177,949,798|
|最大変更suffix89|66|268,608,217,185|127,984,762|43.089|4,666,358,111|
|cold132|291|402,500,079,294|580,218,522|66.341|2,477,344,684|

全32層hidden/stateは旧INT8版とbit一致。判断を行う4条件はvalue、abstain、raw logits、確率、unknown probability、最終hiddenも一致し、型付き出力の検査通過。失敗とreplayは0。既存の最大変更の見逃しも残り、判断精度が改善したという意味ではない。

直前の同じ直接入力kernel版と比べて主問題は67→66 query、通信11,258 bytes減。ただし命令は8,541,420増（+0.00326%）。往復削減として記録し、演算命令の改善とは扱わない。主問題の合計命令は旧採用版比約1.387%減だが、prefix packetに伴う通信増もある。時間は単回で、速度改善率として一般化しない。

固定cache準備は721 update、61,042,755,368命令、282.128秒。cache payload 4,065,416,192 bytes、観測heap終了値最大4,123,721,728 bytes。ピークheapの測定ではない。通信はCandidのみ、命令はhandler内counter。初回packet準備24 query、6,326,348,500命令、61,219,865 bytesを推論と分けて記録した。二回目の準備query/命令/通信0も検証済み。

証拠は `artifacts/prefix_codec/full-terminal-attention-proof/report.json` と `validated-source.zip`、再利用は `codec-second.json`、固定準備は `full-terminal-attention-preparation/report.json`。生成物はignore。

## Delta状態を直接使う経路

型付きoperandを消費する `evaluate_owned_decoded_with_prepared_buffer` を追加した。canisterのstep/profile_stepとnative primitiveは復号済みoperandの所有権を渡す。2 MiBのF32 Delta状態を複製せず、同じVecを可変再帰状態として使う。質問状態をcanisterに保存しない。

借用APIは残し、再利用する呼び出し側では従来どおり独立した状態コピーを作る。所有権を消費するAPIだけでコピーを省く。request identityはJSONへ2回serializeせず全fieldを比較し、scalarsはbitを比較する。有限値・shape・モデル照合の検査は維持する。

Rust native unit67件、候補の全runtime featureでunit74件、request全fieldの変更を重み読み込み前に拒否するintegration test1件通過。実入力5条件のnative Delta全出力もbit一致。Wasm buildとparser validation通過。

module `c64f1605b779d995081c55acd73b181836c14a7a445db4743ce1b247f18946c2` で全5条件を完走。全32層hidden/state、判断4条件のvalue/abstain/logits/probabilities/unknown/最終hiddenが旧INT8版とbit一致。型付き出力有効、失敗/replay0。

|条件|query|合計命令|Candid送受信bytes|秒（単回）|最大query命令|
|---|---:|---:|---:|---:|---:|
|prefix45|66|136,486,270,652|71,105,691|15.320|2,399,051,818|
|主suffix87・旧log|66|264,182,464,866|113,697,747|20.973|4,555,649,984|
|主suffix87・packet再利用|66|262,191,098,449|125,973,196|8.759|4,555,649,984|
|情報不足suffix80|66|240,548,421,313|118,932,610|19.491|4,177,949,774|
|最大変更suffix89|66|268,555,674,302|127,984,762|21.520|4,666,358,000|
|cold132|291|402,505,192,828|580,218,522|47.463|2,477,351,018|

直前の統合候補から主問題52,675,468命令減（−0.02009%）。query数と通信量は同じ。Delta24 queryの合計は58,131,256命令減、一方でAttention7 queryは5,619,509命令増など、再コンパイルによる他演算の差も含む。コピーだけの厳密な単独効果とは区別する。旧log経路は4,651,188命令増であり、効果はhybrid状態を直接使う経路のもの。

hybrid主問題の観測heap終了値は4,122,476,544→4,120,313,856 bytes。coldを含む全条件の最大値は4,123,721,728 bytesで変わらない。コピー量は24×2 MiB＝48 MiB/質問減るが、ピークheapの証明ではない。

固定cacheは同じ721 tensors/4,065,416,192 bytes、準備721 update/61,042,755,368命令/261.823秒。prefix packetの再準備query/命令/通信0も改めて検証した。証拠は `artifacts/prefix_codec/full-owned-state-proof/report.json`、`validated-source.zip`、`codec-second.json`、固定準備は `full-owned-state-preparation/report.json`。50 queryは未達。既存の最大変更の見逃しは改善していない。

候補の再ビルドは次の手順。保存先は未使用のdirectoryを指定する。検証用canisterは固定packのupload/sealと `prepare_weight_cache.py --include-f32 --require-prepared-rope --require-prepared-activation --require-all-output-pairs` による準備が別途必要。

```sh
.venv/bin/python scripts/build_full_prefix_candidate.py \
  --directory artifacts/prefix_codec/rebuild-owned \
  --target-directory artifacts/prefix_codec/rebuild-owned-target \
  --direct-input --terminal-attention
artifacts/wasm-audit-target/release/imajev-wasm-patch \
  artifacts/prefix_codec/rebuild-owned/raw.wasm \
  artifacts/prefix_codec/direct-input-build/direct-input.wat \
  artifacts/prefix_codec/rebuild-owned/full.wasm
```

raw.wasmはfail-closedのkernel stubを含む。対応する9×I32 ABIのbodyを上記で置換・validateしてからfull.wasmを使う。主canisterを変更せず、検証専用canisterで次を実行する。

```sh
.venv/bin/python scripts/check_full_prefix_hybrid.py \
  --canister 6eydd-o3777-77775-aaama-cai \
  --codec-canister 7hukf-2d777-77775-aaakq-cai \
  --wasm artifacts/prefix_codec/rebuild-owned/full.wasm \
  --directory artifacts/prefix_codec/recheck-owned --terminal-attention
```

## 次層へ途中結果を渡す容量調査

実際のMLP途中結果と次Deltaのprefixを23遷移で調査した。量子化入力だけのzlib圧縮は2 MBへ収まらない。BF16/F32をbyteごとに並べ替えて可逆圧縮すると、scale/A・conv・headerの余裕を含む要求サイズ推定は1,852,908〜1,900,402 bytes。全streamの展開byte一致を確認した。

再現は `.venv/bin/python scripts/probe_mlp_carry_capacity.py`。`artifacts/prefix_codec/mlp-carry-reproducible.json` に元request/state/report/source hashを保存する。これはホスト容量調査だけで、Wasm decoder・展開命令・query統合の予算は未検証。MLPを分けて次Deltaへ繋ぐだけではquery数は自動的に減らない。

50/32 queryは未達。主問題約262 B命令に対し5 B/queryの単純下限は53回。往復統合に加えて、演算自体の命令削減が必要。
