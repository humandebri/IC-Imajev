# 入力を共有する整数Strassenの診断

2026-10-03。[採用版](OUTPUT_PAIRS.md)は主問題67 query、265,931,336,448 handler命令で、50 queryには未達。前回の全5条件検証は完了済み。今回は積和そのものを減らす候補を専用ローカルcanisterで比較する。採用済みImajevとLayaは変更しない。

以前の[prepared Strassen](STRASSEN_PREPARED.md)はC4 leafで、87 tokenの通常INT8より14.20%多い命令を使い不採用となった。今回のC16 leafは16出力で128個の入力要素のロードを共有し、毎ロードのrow address計算を避ける。固定INT8から7種類のI16係数を準備updateで一度導出する。質問に依存する7種類の入力表現はquery内で一度用意し、出力tile間で共有する。再量子化は行わない。

各block256をK128の2部分に分け、2 token×2出力の8個の整数内積を7個の内積と加減算で再構成する。整数で元のblock256結果を復元してから、元のactivation scaleとrow scaleを元と同じ順でF32へ適用し、blockの加算順も維持する。量子化済み入力は[-127,127]、元重みは[-128,127]。導出I16の絶対値は最大256、128項の積和と4項の再構成もI32範囲内。12.5%の内積数削減だけで全体の命令削減率を断定しない。

current kernelとの比較用に、診断では元重み・採用版のK4/2出力配置・導出I16係数を同時保持する。導出係数は元weight byteの3.5倍。全モデルの4 GiB配置を証明する構成ではない。候補が計算で勝った場合も、固定係数の配置またはquery内の小さいtileへの展開を別に設計・測定する必要がある。

native testでは1/7/8/32/45/64/80/87/88/89/132 token、符号付き極値・0・複数blockのscaleで全出力bit一致。全係数strideを保ったrow prefixの87/132 tokenも一致した。V1・V2・V3は各22通常queryでnative scalarと採用版の全出力digestが一致。全モデルの判断精度とquery数の検証は未実施。

診断canisterは`7hukf-2d777-77775-aaakq-cai`。実装は`scripts/strassen16_bench`、kernel生成は`scripts/generate_strassen16_leaf.py`。生成再現とnative test2件が通過。検証script `scripts/check_strassen16.py`は採用版の保存した全5条件から同じlayer3 Q投影の実入力を取り、1/7/8/32/64/88 tokenの合成入力も比較する。基準は採用版のreal-token tileを使う`PreparedPairs`。native scalar oracle・出力SHA・module/source bookendを記録し、準備updateと入力準備の費用を隠さない。

```sh
cargo test --offline --release --manifest-path scripts/strassen16_bench/Cargo.toml \
  --lib --target-dir artifacts/strassen16/native
cargo build --offline --release --manifest-path scripts/strassen16_bench/Cargo.toml \
  --bin strassen_args --target-dir artifacts/strassen16/native
cargo build --offline --release --manifest-path scripts/strassen16_bench/Cargo.toml \
  --target wasm32-unknown-unknown --lib --target-dir target
# Copy only the diagnostic module to artifacts/strassen16/diagnostic.wasm,
# install it on its dedicated canister with owner/8192/2560 init args, then:
.venv/bin/python scripts/check_strassen16.py \
  --canister 7hukf-2d777-77775-aaakq-cai --directory artifacts/strassen16/check
```

測定counterはdigestとCandid encode/decodeを含めない。診断のrequest/replyと単回時間は全モデルの通信量・速度・判断精度を示さない。生成物はGit ignore対象。


## 反復処理を減らした3版の実測

同じlayer3 Qの固定重みと全5条件の保存入力を使用。表は採用版の`PreparedPairs`に対するhandler命令増減で、負値が削減。入力変換費用を含む。

|入力token|V1 C16|V2 C16・再構成loop除去|V3 C64・入力変換SIMD|
|---:|---:|---:|---:|
|45|+18.7150%|+6.2895%|+2.2481%|
|87|+17.0755%|+3.9230%|−0.0401%|
|80|+15.7890%|+3.0364%|−0.7166%|
|89|+16.5868%|+3.8348%|−0.1574%|
|132・出力4096行|+17.0807%|+4.1753%|−1.6012%|

V2は毎token/groupの再構成・scale/storeを定数添字に展開し、実行時loopを除去した。V3は入力ロードの共有を16→64出力へ拡張し、query内で一度だけ行う7種の入力変換をSIMD化した。87 tokenの入力準備は21,183,450→968,148命令、全費用は1,049,769,418→1,049,348,005命令。入力変換の削減率だけで全体改善を説明しない。

V3の合成1/7/8/32/64/88 tokenはそれぞれ+144.0621/+15.7069/+21.9933/+2.4167/−2.1298/+0.3062%。小入力と45 tokenでは回帰するため一律採用しない。V3でも固定I16係数は73,400,320 B、比較用に元配置と採用配置も保持した固定payloadは115,408,896 B。準備65 updateの合計は2,817,175,413命令。全モデルへの常駐配置は別途必要で、主問題の0.04%差だけでは統合費用を正当化できない。

証跡は`artifacts/strassen16/{check,v2-check,v3-check}/report.json`、各版の`vN-source.zip`・`vN-source-hashes.json`・`vN-diagnostic.wasm`。V1 module274f98c0…、V2 b244cb83…、V3 aaa9510f…。native test2件とC64生成再現を確認した。採用版55 core fileのhashも一致し、採用canisterは変更していない。

続くV4は、事前に検証したspanの内部アドレス計算に限って重複するオーバーフロー分岐を省く。公開入力検証・整数dot・F32 scale順は維持し、実queryの結果は下記。


## 毎回の処理をなくす方針

固定値は準備updateで一度計算する。採用版ではINT8配置の変換、RoPE、BF16 activation lookupがこの対象。質問に依存する量子化入力・LoRA A積は同じ入力を使う投影間で共有し、分割時にはクライアントが保持する。採用版のgate/up、Attention K/V、Delta統合とcapture/reuseはこの対象。今回の候補では7種類の整数入力表現を一度作り、全出力tileで共有する。

内積kernelでは入力ロードを複数の出力行へ共有し、tile内の繰り返す添字計算・再構成loopを除去する。V4は、公開検証で確立したspanの内部添字だけをwrapping演算に変更する。`Prepared`は元重み数30,000,000以下、`project`はtoken132以下・出力900,000以下・行数128の倍数・cols一致を検証する。導出配列の長さとtile dispatchで全pointer spanが収まるため、これらの添字は実際にはwrapしない。整数積和の演算や丸め順は変更しない。クライアント保持状態の形状・scale・有限値検証は各queryで維持する。

診断だけの固定I16係数は全モデルで3.5倍の容量になるため、計算で改善してもそのまま全モデルに常駐させない。採用版への統合には、少数tensorだけの固定配置または小さな重みtileの一度展開とtoken間共有を、メモリ上限・変換命令込みで測定する。今回の単一投影の差を全モデルの判断精度改善やquery削減として扱わない。


## V4：内部添字の重複確認を除去

公開入力検証を保ち、範囲検証済み内部添字のoverflow分岐を除去。生成WasmのR22 leafはBrIf 23→0、I64Mul 18→0、再構成tileはBrIf 200→77、I64Mul 64→0。opcode数は静的数で、効果は下記の同一入力queryで確認した。

|入力token|採用版handler命令|V4 handler命令|増減|
|---:|---:|---:|---:|
|45|545,946,169|549,927,844|+0.7293%|
|87|1,049,769,418|1,032,860,196|−1.6108%|
|80|967,158,026|943,961,528|−2.3984%|
|89|1,073,427,738|1,055,201,223|−1.6980%|
|132・出力4096行|802,588,502|777,370,322|−3.1421%|

合成1/7/8/32/64/88 tokenの増減は+117.3458/+9.3300/+15.2861/+0.3922/−3.5281/−2.1023%。全22通常queryで採用版・独立native scalarと出力digest一致。native test2件と生成再現も通過。準備65 updateの合計2,817,175,413命令、診断固定payload115,408,896 B、観測終了heap3142ページ（205,914,112 B）でpeakではない。

module `d0fe6509e5868f8ad78606ed350d077852cba9e5b3eadf49792e32149715a6c8`、Wasm7,259,113 B。証跡`artifacts/strassen16/v4-check/report.json`、`v4-source.zip`、`v4-source-hashes.json`、`v4-diagnostic.wasm`、`v4-opcodes.json`。全モデルに未統合で、判断精度・50/32 query達成を証明する測定ではない。採用canister module36c04a57…と55 core source hashは変更なし。

測定後、生成CLIの選択肢を現在のC64だけに制限した。C16/C32を生成してC64 dispatchと組み合わせる誤用を防ぐ。現行CLIで再生成したRust byteは測定版と同一で、CLI guard変更の別hashを記録する。歴史版の再現には各版source archive全体を使う。

次は、採用版の通常INT8 kernelにも残る同種の重複アドレス確認を優先する。診断WasmのR44 `output_pairs::accumulate`にはBrIf144・I64Mul37とoverflow panic call104の静的記載がある。固定係数を3.5倍保持する今回の方式をそのまま統合するより、採用配置のまま同じ除去を検証する方がメモリ面で有利。opcodeの静的数だけで削減率を予測しない。
