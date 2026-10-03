# 繰り返すbyte復号をまとめる

2026-10-03。前回のscalar Huffman carry復号を改善した。モデル演算、INT8入力、F32 scale/LoRA、BF16丸めを変更しない。通常query・クライアント保持状態のまま。

- 14-bit tableで2 symbolをまとめて取り出す。長いcode・短い末尾は従来の単一symbol経路へ戻す。
- reservoirへの入力を4 byte単位で読む。最大55 bitでu64内に収まる。末尾のbyte/zero padding検査は維持する。
- code長を昇順に並べ、2 symbolが14 bitに入らない組み合わせのtable生成を省く。
- 一時plane bufferの確保・zero初期化・最終bufferへのコピーを省き、最終planeへ直接復号する。
- 圧縮効果10%未満のplaneはrawにできる。通信と演算の交換を別測定し、2 MBを超えないことを検査する。

## 復号だけの実測

専用診断 `5tkpr-7d777-77775-aaaeq-cai` で、同一moduleの旧scalar decoderと新decoderを通常queryで比較。入力は既存canister生成済みcarryの87/45-token各3層。値推論をホストで行わない。全6件でdecoded byte countとSHA256 digest一致。counterはdecoder本体のみで、Candid・digest・model inferenceは含まない。

最初の2 symbol版は87-tokenで約11%削減。4 byte読込とtable準備打切りを加えると約34〜35%削減。さらに圧縮効果の小さいplaneをrawにすると、旧scalar/Huffman frameから約60%削減した。raw10のdecoder自身でも同じmoduleの旧decoderとdigest一致し、元Huffman frameのdigestとも照合した。

|層|tokens|旧Huffman/scalar命令|新word/pair命令|新word/pair＋raw10命令|
|---|---:|---:|---:|---:|
|0|87|189,879,944|122,583,861|74,828,370|
|1|87|190,316,975|124,720,791|75,499,139|
|3|87|190,567,849|125,595,964|76,071,814|
|0|45|127,420,969|85,621,071|47,096,212|
|1|45|127,482,543|86,372,670|47,062,158|
|3|45|127,645,211|86,794,594|47,341,545|

raw10 frameは87-tokenで1,744,495 / 1,751,273 / 1,759,669 bytes。増分は元Huffmanから約7〜11 KB。この命令削減を全モデルやquery数の達成と解釈しない。

証拠：`artifacts/prefix_codec/carry-pair-check`（2 symbol）、`carry-pair-word-check`（word）、`carry-raw10-check`（raw10）、各`report.json` / `validated-source.zip`。raw frameの生成は `carry-raw10-frames/reframe.json`。nativeで旧DEFLATEと6 groupのbyte一致を確認した記録は `carry-raw10-native-final/carry-huffman-native-parity.json` / `native-test.log`。

## 分割行数を明示する

`mlp_prepare_partial_down` のdimsは旧 `[n,2560]`＝256行を維持し、新 `[n,2560,rows]` を追加。carryのdimsは旧 `[n,p]`＝256行、新 `[n,p,rows]`。rowsは32の倍数かつ32〜2528。private prepared carryがrowsを保持し、request identityに含める。旧carryの先頭256行を512行と暗黙に読み替えない。

先頭rows行は前queryで完成BF16 hiddenにし、残りは元の残差として保持する。次queryは未計算行だけを計算し、既に完成した行へzeroを加算し直さない。末尾normは全hiddenへ一度だけ適用する。入力が通常Deltaの900K float制限を超えるshapeは、MLP演算前に拒否する。

69 unit、partial rowsの照合を含む6 integration、24-bit codeとodd tail/word boundaryのテスト通過。候補全runtime flagsでは86 unit通過。モデルmodule `full-build-carry-pair` の固定cache準備後、256行と512行を同じmoduleで比較した。50/32 queryはまだ未達。

```sh
.venv/bin/python scripts/check_carry_planes_native.py --raw-threshold .1 --directory artifacts/<fresh-native-proof>
.venv/bin/python scripts/check_carry_pair.py --canister <decoder-diagnostic-id> --directory artifacts/<fresh-decoder-proof>
.venv/bin/python scripts/check_mlp_delta_fusion.py --huffman --raw-threshold .1 --partial-rows 512 --canister <model-diagnostic-id> --wasm artifacts/<build>/full.wasm --directory artifacts/<fresh-fusion-proof>
```

## 87-token融合の完走と採否

モデル実験module `17348099e57e7f40514a6aff20081733e46acae9beface0b25ed4ba0046d196d`、canister `6eydd-o3777-77775-aaama-cai`。256行分割は新word/pair decoderでも、raw10でも3箇所すべて5B超過。512行＋raw10は3箇所すべて87-tokenで完走し、同じmoduleの分離MLP/Deltaと全出力bit一致。profile queryともbit一致。reply shape修正も今回実際の87-token返信で検証できた。

|MLP層→次Delta|prepare命令|融合命令|2 query合計|分離MLP＋旧log Delta合計|合計増分|
|---|---:|---:|---:|---:|---:|
|0→1|3,582,125,196|4,922,188,054|8,504,313,250|8,312,327,406|191,985,844|
|1→2|3,582,112,303|4,923,113,209|8,505,225,512|8,312,480,380|192,745,132|
|3→4|3,582,854,883|4,923,406,343|8,506,261,226|8,313,448,510|192,812,716|

合計命令が約2.3%増え、2 queryのCandid通信も約370万→486〜488万bytesへ増えた。比較相手は同じmoduleの旧log Deltaで、既存hybrid Deltaはさらに安い。この融合を既存graphへ採用しない。完走だけで「query削減」「性能改善」と扱わない。変えたのは分割位置・可逆codecであり、32/50 query目標の達成ではない。

最初の成功融合profile（層0）ではwire decode174,642,585、うちplane75,657,648、MLP finish1,005,542,127、次Delta3,703,357,467、wire encode29,862,372命令。nested spanはinclusive。decoder単体の高速化は確認できたが、carryの並べ直し・型検証・checksum等と、32-head prefix log復元が残る。通常MLP1＋Delta1をprepare1＋融合1へ置換しても回数は同じ。50へは総命令を少なくとも約107億削減したうえで、複数段の部分実行を詰める必要がある。

融合の証拠：`artifacts/prefix_codec/carry-pair-256-huffman` / `carry-pair-256-raw10` / `carry-pair-512-raw10` の各 `report.json` / `validated-source.zip`。512版は18通常query、失敗0。256版は各21通常query、87-token失敗3を含み、45-token診断はbit一致。本番tokenを縮めて成功扱いにしていない。

## 全条件の回帰

同じmoduleの既存graphで全5条件（主旧logも含め6経路）を完走。全32層hidden/state、判断4条件のvalue/abstain/logits/probabilities/unknown/最終hiddenは保存済みINT8版とbit一致、typed output有効、失敗/replay0。主suffix87は64 query / 260,701,702,281命令 / 124,634,637 Candid bytes。新carry融合はこのgraphで使用しない。既存の最大変更の見逃しも残る。

固定cache準備は721 owner update、61,042,755,368命令、359.580秒（単回）。元pack4,702,451,200 bytes / 固定cache4,065,416,192 bytesは同じ。推論は通常query。source bookendsを確認し、`full-carry-pair-base-proof/report.json` / `validated-source.zip` に固定した。prefix packet2回目の準備query/命令/通信は0（`codec-second.json`）。

分割行数のPython互換性3 tests、既存wire4 testsも通過（ログ参照）。生成物はignore。主canisterとLayaは変更しない。次は通信を増やす融合から切り離し、既存graphの主な演算自体、とくにF32 LoRAで繰り返す入力・重みの配置と積和を測る。bit順序を保つSIMDの配置変更をまず候補にする。

## 次のcore調査へ渡す計測

`scripts/profile_mlp_core.py` で同じmoduleの通常87-token MLPを層0/1/3の3通常queryでprofileし、返信frame全体が分離baselineとbyte一致した。層0のhandler4,553,317,587命令に対しbase project inclusive3,452,832,664（約75.8%）、LoRA B407,574,502（約9.0%）、wire decode28,286,079、wire encode28,335,814命令。gate/upの共有A積とdown A積には現状profileラベルがなく、残差約6.36億命令をA積と断定しない。shared Aとdown Aを切り分ける追加計測が必要。

証拠：`artifacts/prefix_codec/mlp-core-profile/report.json` / `validated-source.zip`。主canisterのcertified module照合は `main-unchanged-carry-pair/report.json`、従来hashと一致。

INT8本体側では、既存 `generate_wat_s1.py` が7つのStrassen積それぞれのSIMD lane水平和を先に取ってから、4出力の整数線形結合へ渡すこともコード確認した。整数線形結合と水平和は交換できるので、未還元の7積を線形結合し、最後に出力laneをまとめて還元する候補がある。元のI32上界とK-blockごとのF32 scale/加算順を維持する。既存S1は主Qで+2.56%だったため、そのまま再採用せず、この重複還元の除去を別候補として実装・実測する。固定係数3.5倍の配置制約も残り、性能や50 query達成は未証明。
