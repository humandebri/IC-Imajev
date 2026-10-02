# 2026-10-02 実装・測定状態

**最新の追加改善（2026-10-02）：INT8直接ロードとtile変更で、cacheなし708 queryの命令数を7393億→7110億へ3.82%削減。共通45-tokenのclient cache利用時は611 query・4926億命令・0.873 GB。ただし初回cache準備は別に578 query・2557億命令・0.464 GB。3入力で全32層hidden/state/判断が以前のINT8版とbit一致。50 queryは未達、速度向上は未確認。** [複数方向の実装・不採用候補・初回費用](DIRECTIONS.md)。

前段階：演算方式変更の許可を得て、整数base＋元F32 LoRAを全32層へ接続した。617・132 tokensは1.022兆命令・932 query・1.671 GB・単回107.274秒。最初の全層から命令数72.24%減。ホスト23件で同じpackのF32/整数ラベル一致、元順序の正解付き7問は双方6/7。ただし確率差最大約8.48ポイントで、精度・校正の一般保証はしない。50 queryは未達。[INTEGER_ARITHMETIC.md](INTEGER_ARITHMETIC.md)。

数値を変えない追加改善も全32層で検証し、1,588→1,116 query、2.264→1.762兆命令。全hidden/state/logit/確率bit一致。[追加改善](FIFTY_QUERY_ANALYSIS.md)。以下は各段階の履歴。


INT8 base＋F32 adapter/readoutで、embeddingから全32層・専用readoutまでローカルcanisterの通常queryで完走した。132 token・BOOM DAO 617・rotations=1は3,908 query、604.274秒、合計3.682兆handler命令、Candid通信5.382 GB。参照と同じlikely、候補logit最大差0.04657、校正済み確率最大差0.001207。中間状態はclient保存、量子化追加なし。公式hiddenを計算入力へ流用していない。

再現と制約は [FULL_INFERENCE.md](FULL_INFERENCE.md)、各query実測は [full-results.json](full-results.json)。以下の部分演算測定は開発時の原型記録として残す。全層のINT8差には未量子化Rust/MLXの積和差も含まれるため、完全32層BF16 A/Bは残作業。

追加の通信改善で3,108 query・2.549 GB・560.522秒へ減らした（通信52.64%減）。全32層・final hidden・判断logit・確率が改善前とビット一致。最大queryは30.59億handler命令、終端heap最大観測87.1 MB。詳細は [COMMUNICATION.md](COMMUNICATION.md)、各queryは [efficient-results.json](efficient-results.json)。

追加でactivation INT8通信を実装し、3,108 query・1.285 GB・514.639秒で完走。likelyは一致するがBF16通信版との候補確率差は最大0.120719。精度同等とは言えない。head融合と32 query目標の制約は [ACTIVATION_INT8.md](ACTIVATION_INT8.md)。

head融合版を全32層で実測：2,132 query、461.531秒、Candid通信1,283,895,563 bytes、最大query命令数3,022,101,308、終端heap最大観測86,441,984 bytes。融合前INT8版と全層・state・判断のbit一致を確認。32 query目標は未達。現在の演算量では通常query上限内に1層を丸ごと置けない。[fused-results.json](fused-results.json)。

Layaの最新レポ/整数タイルを参照し、F32射影の16行ロード共有とpointer整理を追加した。全32層の総命令数3.133兆→2.329兆（25.66%減）、2,132 query、1.284 GB、単回251.002秒。全hidden/state/logit/確率がhead融合版とbit一致。49形状のraw F32 bit比較も成功。構造・実装・LoRA再計算の分解は [LAYA_COST_ANALYSIS.md](LAYA_COST_ANALYSIS.md)。32 queryは未達、演算の整数化を全射影へ接続する作業は残る。

最新の精度維持改善は [EXACT_OPTIMIZATION.md](EXACT_OPTIMIZATION.md)。元タイルの量子化境界を保つ2タイル融合とAttention再計算削減を実装。可逆通信の全層・state・判断が以前とbit一致で命令数32.0%減/1,588 query/1.984 GB。INT8通信もbit一致で1,540 query/1.014 GB。32 queryは未達。

## モデル固定

- adapter: `mohit67890/imajev-4b@c9e5f132465da85d31735ec502d5557982671a7d`
- base: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- 公式server: `mohit67890/imajev@a0134749e0900189c129cd6bb5000969f3b64bb5`
- baseのsnapshotをadapter_configの学習パスから特定。`revision:null` をmain参照として扱っていない。
- LoRA rank64、alpha128、scale2、言語層のq/k/v/o、gate/up/down、in_proj_qkv/z、out_projが対象。未統合参照を使用。
- 実readoutはF32 `[256,2560]`。manifestのcode/token bindingを公式backendが検証する。unknownは候補数に応じた最後の行であり、常にreadoutの最終255行を使うのではない。
- tokenizer、processor、専用readout、adapter、校正を個別にSHA256固定。`calibration.json` は温度1.3051569717552742、unknown offsetなし。calibration-rot4.jsonを使っていない。
- 本revisionのprompt layoutはstandard。最終`</think>\n\n`のhiddenから専用readoutを計算。次tokenのLM語彙logitへ置換していない。

全ファイルSHA256と実tensor配置は [MODEL_LOCK.json](../MODEL_LOCK.json)、容量集計は [capacity.json](capacity.json)。公式のモデルカードmainからshapeを推測していない。

| 実ファイル/領域 | bytes | GiB |
| --- | ---: | ---: |
| base全shard（header含む） | 9,319,828,096 | 8.680 |
| 言語tensor（BF16主体、一部F32） | 8,411,510,272 | 7.834 |
| vision tensor | 667,028,480 | 0.621 |
| MTPなどその他tensor | 241,199,104 | 0.225 |
| PEFT adapter | 487,648,432 | 0.454 |
| 専用readout | 2,621,520 | 0.00244 |
| embedding単体 | 1,271,398,400 | 1.184 |

言語部分は4,205,751,296 parameters。INT8の単純なparameter bytesだけで3.917 GiBとなる（実pack容量ではない）。scale・未統合LoRA・activation・decoder・allocatorを含めて4 GiB heapへ常駐できるとは扱わない。全層のtext packからvision/MTP/tied LM headを除外した。ホストtext-only全層captureの最終hiddenは既存参照とビット一致。

32層＝24 Gated DeltaNet＋8 full attention、hidden2560、MLP9216、語彙248320。必要演算はGQA/gating、64次元partial RoPE、QK RMSNorm、causal depthwise conv(kernel4)、QK正規化、F32 recurrent state、exp/softplus/sigmoid、gated RMSNorm、SwiGLU、residual、専用readout、候補mask、校正softmax。通常のattentionだけの移植では足りない。Candle/vLLMはこの実験で動作検証していない。

## 判断の比較

入力と正解根拠は [benchmarks/cases.json](../benchmarks/cases.json)。Layaは保存済み同一質問・選択肢・構造化要約を使用し、ソース、Git、identity、network、稼働canisterを変更していない。追加問題にはLayaの結果がないため、両モデルの正解率比較はできない。

以下は公式server shared-prefix経路、1質問・rotations=1。full-forward経路も別途取得。両経路の選択結果は23/23一致するが確率には小差がある。全結果と入力tokensは [host-results.json](host-results.json)。

| 問題 | Laya保存済み | Imajev（元順序） | 観察 |
| --- | --- | --- | --- |
| 617 最低投票delay 20,000倍 | unlikely | likely (0.8140) | 重大変更の見逃しを改善する候補 |
| 620 最低投票delay 2倍 | unlikely | likely (0.5161) | 他順序ではpossible。危険度goldは未定義 |
| 653 1口座に250M mint | likely | likely (0.4410) | unknownも0.2164。既存supplyなし |
| 最低投票delay増加の事実 | 未測定 | yes | 根拠一致 |
| 最大lock duration 1,000倍増加 | 未測定 | **no** | **誤判定・変更の見逃し**、2順序とも誤り |
| stake 1,000倍増加 | 未測定 | yes | 根拠一致 |
| 新規mintの事実 | 未測定 | yes | 根拠一致 |
| 変更なし | 未測定 | no | この事例では誤警告なし |
| 旧値なし | 未測定 | abstained | unknown=0.99815 |
| maximum増加、minimum不変 | 未測定 | no | 区別はできた |

根拠を明記した7問題は元順序で6/7。これらは今回用意した小さな診断集合で、一般精度や校正性能を表すベンチではない。変更量から投票参加・集中への将来の因果効果は一意に決まらないため、歴史3件にaccuracyのgoldを付けない。並べ替えは選択肢だけをcyclicに入れ替え、unknownは最後、各回独立rotations=1で平均なし。620で順序依存を確認した。

LayaとImajevのprompt/tokenizerが異なるため入力token数・raw logits・確率を直接同じ尺度として比較しない。Imajevの歴史3件は132/124/122 tokensである。

## 演算効率化の追加測定

Layaの実コードを参照したSIMD、base/LoRA融合、候補行readout、クライアント分割を実装した。132 token・256出力行のQKVで総命令数85.2%減、通信量56.9%減。旧新の各F32積和とBF16出力はビット一致。詳細・制約・追加候補は [OPTIMIZATIONS.md](OPTIMIZATIONS.md)、各query実測は [optimizations.json](optimizations.json)。以下の原型測定は以前のWasmの記録として残す。

## Rust/Wasmの実測

測定APIは通常query。中間状態はclient保存binary（model lock hash、pack hash、input hash、version、step、op、shape、checksum、有限値検査）。query間でheapのactivationを永続化しない。重みはstable、tensorの出力行タイルを必要分だけheapへ読む。INT8のrow scaleは別rangeから読む。

- 原型はreadoutの全256行を計算。改善版`decision_fast`は候補数＋unknownの行だけ計算する。
- readout重みINT8＋activation F32、重みF32＋activation INT8を独立に比較する。
- 以下のreadout/QKV原型のactivation INT8試験はF32へ復元して送った数値誤差試験。後続の全層ではINT8 binary通信を実装・測定済み。
- 256行の実QKV base射影、rank64のF32 LoRA A/B、BF16境界丸めをcanister queryで計算。
- 実第1層DeltaNetの1 value head（128×128 state）をnativeとWasmで計算。全32 heads/全24層の検証ではない。

[canister-check.json](canister-check.json) と [projection-check.json](projection-check.json) に各queryの命令数、要求・返信bytes、時間、heap/stable、pack/Wasm hash、元ログのhashを保存する。命令数はhandler内のstate検証、stable読み出し、展開、演算、内部状態encodeを含み、CDK Candid decode/encodeを含まない。通信は成功分のCandid request＋reply、HTTP/CBOR/signatureを含まない。query cache未制御、単回測定であり時間は計算速度の証拠にしない。heapはhandler終端のpage数、瞬間ピーク測定ではない。

| 検証 | 実測 |
| --- | --- |
| readout F32 / 23入力 | 選択一致23/23、logit最大差1.63e-5、確率最大差2.41e-6 |
| readout weight INT8 / activation F32 | 選択一致23/23、logit最大差0.01836 |
| readout weight F32 / activation INT8 | 選択一致23/23、logit最大差0.14515、確率最大差0.01367 |
| 実DeltaNet 132tokens / 1 head | nativeとWasm最大差0、公式BF16との差0.000420 |
| DeltaNet 66＋66 tokens / client-held state | 一括計算との出力・state差0 |
| QKV 128 tokens・256 rows | 約35.1億命令、base+LoRAの4 query合計約3.35 MB |
| QKV 132 tokens・256 rows | 約36.2億命令、4 query合計約3.45 MB |
| 未量子化QKVの公式BF16との差 | 最大0.03125、RMSE約0.00063。累積影響未検証 |
| QKV weight INT8のみ | 最大誤差0.125 |
| QKV activation INT8のみ | 最大誤差0.09375 |

32/64/128/132はQKVの実operandを切り出したtoken prefix長であり、32/64tokensの完全な公式質問を評価した結果ではない。32～132tokens、2/3選択肢の部分検証で、4～7選択肢の全モデル精度、512tokens、Score rubricは未検証。

型付きchoiceは許可選択肢かnullのみを返し、unknown_probability、abstained、有限・正規化済み確率を出す。型安全性の検査は判断goldと別に報告。Score/ordinalのrubric出力はまだ公開APIにしていない。状態checksum/pack違い/重複選択肢の拒否、非ゼロ行タイルの部分読み出し、再送一致、upgrade後の再推論を確認した。

## 制約と次の実装

この時点で全面採用の判断は保留する。617には改善候補がある一方、maximum問題を見逃し、620で順序依存がある。代替案は既存の数値差分ルールで重大変更を必ず検出し、Imajevの判断は説明・曖昧な根拠の補助として扱う構成。原文全体/情報を増やしたpromptでmaximum誤りを調査し、未使用問題を追加して再評価する。2Bへの変更や他モデル採用は今回は行っていない。

全モデルのpack、partial RoPE、層接続、client-held分割と再開は実装済み。原型から768行tileへ拡大した第1層は239→124 query、通信331→171 MBで出力ビット一致を確認した。

残る評価・効率化は、全32層BF16 A/B、複数問題/順序のINT8完走、4〜7選択肢・512tokenの境界、query内の演算融合と入力再送削減、重みINT8と中間状態INT8の独立評価、warm/cold反復。既存のホスト診断ではmaximum変更の見逃しと620の順序依存が残る。全体採用の判断は引き続き保留。
