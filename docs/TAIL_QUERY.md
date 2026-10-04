# 50queryの後段接続

2026-10-04。レビュー修正は`8ebecac`にコミット済み。全体3条件をローカルcanisterで検証し、主BOOM問題と情報不足は50query、最大変更は62queryで完走した。採用済みINT8参照と返却hidden・状態・最終hidden・専用readoutの判断がビット一致した。

## レビューで修正した問題

- bridgeのcanister featureが型付きqueryデコーダを有効にしない依存欠落を修正した。
- MLP全体→Delta部分という新しいcarryを、次段が旧MLP streamの進捗だけに限定して拒否する条件を修正した。全体MLP由来はbegin=0/count=9216を明示する。
- layer30のDelta→MLP準備を`experimental-terminal-stream`のときだけ許可した。token IDsから開始する方向はlayer0に限定したまま。
- 融合した最終判断のcheckpointでも、選択肢順序・query mode・判断の欠落を検査する。従来の経路は変更しない。

## 候補のquery配分

`--roll-start --join-start --tail-start`で使う実験経路。最初の37queryを既存連結から引き継ぎ、MLP22以後を13queryへ配分する。suffixが88/89 tokenなら既存経路へ戻す。モデルの計算と専用readoutはすべて通常queryで行い、クライアントは並べ替え・型検査・可逆codec・中間状態の保持だけを行う。

| query index | 処理 |
|---|---|
| 37 | MLP22部分down完了、Attention23全体、MLP23前半3584行 |
| 38 | MLP23完了、Delta24先頭18 heads |
| 39 | Delta24残り、MLP24部分down768行 |
| 40 | MLP24完了、Delta25全体、MLP25前半1024行 |
| 41 | MLP25完了、Delta26先頭10 heads |
| 42 | Delta26残り、MLP26前半6912行 |
| 43 | MLP26完了、Attention27全体 |
| 44 | MLP27全体、Delta28先頭6 heads |
| 45 | Delta28残り、MLP28前半5888行 |
| 46 | MLP28完了、Delta29先頭24 heads |
| 47 | Delta29残り、MLP29部分down1024行 |
| 48 | MLP29完了、Delta30全体、MLP30前半512行 |
| 49 | MLP30完了、Attention31最終Q、最終MLP、専用readout |

`--tail-heads28 / --tail-front28 / --tail-heads29 / --tail-down29 / --tail-front30`はsessionに記録し、再開時に変更を拒否する。範囲の不正と、`--tail-start`なしでの非default指定も拒否する。主問題と情報不足の50queryは全体の実測件数で、専用readoutも含む。

## 通信上限と部分実測

module `28317a5b…`で最初の6連結がbit一致した。第6queryは4,783,654,331命令・Candid要求1,909,785 bytesで、hiddenの可逆圧縮により以前の2MB超過を解消した。続くMLP26→Attention27は4,899,993,064命令でbit一致。MLP27→Delta28は8 headsで5B超過し、6 headsでは4,852,458,150命令でbit一致した。

準備済みMLP30→terminal/readoutは前半2560/3072/3584行の独立controlで、最終hidden/KV/logits/確率がbit一致。2560行で3,817,426,330命令だった。このcontrolは参照入力から開始しており、全体推論の証明ではない。

module `1d9dffba…`でも最初の6連結が再びbit一致したが、第9queryの要求は2,005,659 bytesで送信前拒否となった。hidden/prefixのHuffman閾値を調整しても2,000,559 bytesで収まらない。すでにINT8の入力222,720 bytesを再量子化せずbyte列としてHuffman化すると202,679 bytesになるため、追加direction tag14を実装した。実carryの要求は1,985,622 bytesに収まった。復号はshapeから容量を決め、不正descriptor・不正整数・切断・余剰bytesを拒否する。

命令上限に近い第3queryにはこの追加圧縮を適用しない。第9queryだけがINT8 byte圧縮を使い、ほかの箇所も必要な圧縮だけを選ぶ。追加の状態量子化、近似exp、加算順序の変更は行わない。

## 検証と再現

Nativeは採用feature構成149 unit（1 ignored）・15 integration・11 doc。Python codec36件、tail graph5件、追加codec4件、terminal transport3件、journal6件、joined3件。フレームのbit復元、1/80/87 tokenの13要求の配線、最初の37要求を維持するdispatch、最終hiddenを1 tokenだけ保存する契約、layer30の古いexport除去を検証した。

全体回帰は`check_full_prefix_hybrid.py`、部分後段は`check_attention_prime_down.py`と`check_tail_stream.py`、全体候補は`check_roll_graph.py --tail-start`で測る。source/module/参照bytesを前後で固定し、検証ソースをarchiveする。失敗した要求・フレーム境界のheader/operandsも`artifacts/`へ保存する。生成物はGit対象外。

専用実験canisterだけをupgradeする。固定モデルの準備はupdate、質問状態はクライアント保持、推論は通常query。主canisterは`36c04a57e9501eb9d32163c92d7725171191fa46a45a880ad03465993fceed73`のまま。Layaは変更しない。採用INT8参照とのbit一致と、元BF16との差・判断精度の問題は別に扱う。

## 13連結の成立

最終module `ce7b024c…`の`stream-tail-v4`で、5888行/24 heads/MLP29 down1024行/MLP30 front512行という配分の13連結が、全返却hidden/carry/conv/KV/最終判断でbit一致した。最後の3queryは4,408,530,055 / 4,622,388,045 / 4,432,848,988命令。MLP29の残りdownを次queryで一度だけ計算するため、MLP29入力の再正規化・再量子化・gate/upの再実行は不要。途中carryはクライアントに保持する。

6016行は128行pendingの処理も加わり第9queryが上限超過した。5888行/24 headsでMLP29全体を同時に完了しようとしても第11queryが上限超過した。そこで残りdownをDelta30/MLP30準備へ移し、13queryのまま両側を上限内へ配分した。

## 全体の実測

最終moduleは`ce7b024cf39f5d9d20679d54745c62d7c16d5aaebd99e06c3ce875962d3a045a`。`artifacts/tail50/tail-proof-v1`の3条件はすべて通常queryで完走し、参照出力とビット一致、失敗・再送とも0。前後でmodule・ソース・参照hashを固定し、85ソースファイルを保存した。

| 条件 | query | 合計命令数 | Candid要求＋返信 bytes | 最大query命令数 | 時間（秒） |
|---|---:|---:|---:|---:|---:|
| BOOM主問題 | 50 | 234,609,007,209 | 149,353,220 | 4,918,963,331 | 36.880 |
| 情報不足 | 50 | 214,807,342,019 | 139,394,364 | 4,507,864,293 | 38.298 |
| 最大変更 | 62 | 232,868,323,713 | 125,231,193 | 4,795,107,234 | 44.376 |

主問題は132 token（固定prefix45＋suffix87）、rotations=1。省略・短縮は行っていない。最大変更はsuffix89 tokenなので既存62queryへ戻す。主の観測heap最大は4,144,037,888 bytes。50query達成は主問題と情報不足についての結果で、32queryと全入力50queryは未達。

直前の51queryと比べ、合計命令は750,494,512（0.3209%）増え、通信は957,790 bytes（0.6372%）減った。query数削減は成立したが、総計算量削減ではない。標準62queryの主問題227,795,678,436命令に比べても総命令は多い。時間は単発の観測値で、一般的な速度向上を示さない。

件数には質問推論と専用readoutを含む。固定prefix準備66query、モデル準備721update（78,739,929,194命令、dense配置4,065,416,192 bytes、290.087秒）、module検査は別集計。prefix未準備の通常経路290queryも回帰検証した。モデル全体を事前計算した質問回答や、質問状態のcanister保持は使っていない。

この一致は採用済みINT8基準との一致であり、元BF16からの量子化差と判断精度の改善を証明しない。最大変更問題で既に記録されたモデル判断の見逃しも解消していない。

再現コマンド（固定モデルとローカル環境の準備後）：

```sh
.venv/bin/python scripts/check_roll_graph.py \
  --canister 6eydd-o3777-77775-aaama-cai \
  --wasm artifacts/tail50/full-build-v6/full.wasm \
  --base-proof artifacts/tail50/full-proof-v2 \
  --directory artifacts/tail50/tail-proof-new --tail-start
.venv/bin/python scripts/snapshot_full_query_proof.py \
  --directory artifacts/tail50/tail-proof-new
```

主問題の50 checkpointを別ディレクトリへコピーして再開した結果、50件再利用・新規推論0件で、exportのbytesと最終判断が一致した（`tail-replay-v1/verification.json`）。

後段6要求の独立profileも保存返信とbit一致した（`tail-profile-v2`）。query39/43/45/47/48/49の基底行列演算inclusiveは各query命令の62.12% / 65.68% / 51.04% / 65.56% / 57.02% / 78.60%。inclusive spanは子処理を含むため合算しない。query45の追加INT8 byte復号は17,074,623命令。profile要求は検証専用で50queryの件数に含めず、profile APIは専用readoutの判断構築を含まない。次の総命令削減の主要対象は基底行列演算で、carryの再利用だけでは残る大半を消せない。

命令値はhandler counterでCDKのCandid decode/encodeを除く。通信値はCandid要求＋返信でHTTP/CBOR/signatureを除く。上限との余裕が小さいqueryがあるため、この入力範囲と検証済み配分を維持する。
