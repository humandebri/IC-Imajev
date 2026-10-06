# GQA入力共有と順序を保つ内積SIMD

2026-10-04。50query目標に向け、GQAがheadごとに繰り返すKVコピー・Request作成・検証と、score内積を削減する候補を実装した。主問題はprefix45 + suffix87 =132 token、rotations=1を維持する。固定モデルrevision/INT8重み/LoRA/校正/量子化境界は変更しない。

## 実装

`experimental-attention-views`を追加。GQAの全体shape検査後に、共有KVのsliceを各headへ直接渡す。個別headごとのinput Vec連結、Request clone、再帰`execute`と出力検査を省く。GQA全体の出力の有限値と容量は外側で検査する。score/確率のVecは再利用する。

Wasmのscore内積は4つの積だけをSIMD化し、lane0→1→2→3を順番に元のscalar F32 accumulatorへ加える。水平和・FMA・追加量子化・expの近似は使わない。stdのF32 Sumと同じ-0初期値を使い、端数は元のscalar順序で加える。BF16 score、softmaxのF32 sum/division、BF16 probability、valueの累積順序、最後のBF16境界を維持する。value側SIMDは既存実装を再使用する。

旧個別`attention_suffix_bf16`は変更せず、診断の比較経路として残す。featureを外せば従来GQAに戻る。新しいCandid method、composite query、質問状態のupdateは追加しない。

## 実Wasmの診断

`check_attention_views.py`で、共有GQAと個別headの従来queryを比較した。8条件の出力がすべてbit一致。各条件のfixtures、要求/返信、module/source bookendとsource archiveを`artifacts/attention_views/kernels-v1`に保存した。主と同じ87 token・幅256・16 heads・prefix45を含む。signed zero・奇数幅・短い入力・89 token・最終1 tokenも含む。

| suffix / heads / prefix | 従来の個別query合計 | 共有query | 差 |
|---|---:|---:|---:|
| 45 / 4 / 0 | 60,675,096 | 27,713,397 | -54.325% |
| 80 / 4 / 45 | 380,780,492 | 165,280,706 | -56.594% |
| 87 / 16 / 45 | 1,722,771,499 | 747,005,786 | -56.639% |
| 89 / 16 / 45 | 1,781,675,289 | 772,413,142 | -56.647% |
| 1 / 16 / 131 | 36,587,698 | 14,318,860 | -60.864% |

これはkernel診断であり、個別query合計にはheadごとのframe処理も含む。従来の単一GQAとの厳密な差や全体推論の削減率はこの表だけから主張しない。1 token・幅1・1headでは52,222→53,031命令と809増加した。

module `326ede749fe1119f742eff0bcd25daef4d8555c7396c7a555bb3782746976908`を専用実験canister `6eydd-o3777-77775-aaama-cai`にのみupgrade。raw2,008,646 bytes、4つのWAT置換はwasmparser検証済み、build前後source hash一致。主canister/Layaは変更しない。固定モデル準備は721 update・78,739,929,194命令・4,065,416,192 bytes、262.903秒。モデル準備の量は従来と同じ。

Nativeは採用feature構成145 unit（1 ignored）・15 integration・11 docが通過。最小feature49 unit/2 doc、GQAのcausal/suffix/個別head比較と境界3テストも通過。Wasm SIMDの一致は上記実queryで検証し、nativeだけで判断しない。

## query配分の探索

`check_attention_prime_down.py`で、MLP22の部分down→Attention23→MLP23前半→Delta24→MLP24部分downという三連結を独立したcontrolと比較する。参照bytes/NPZ/source/moduleを固定し、失敗要求を別名で保存する。

主87 tokenでは、MLP前半3328/3584行の初段は4,788,762,873 / 4,865,970,422命令で参照hidden/carry/KV bit一致。3584行の次段Delta16/18 headsは4,663,938,854 / 4,865,068,511命令でhidden/選択conv bit一致。3840/4096/4352行の初段と、一部のDelta20 headsは実5B上限を超えた。

続く残りDelta＋MLP部分down1280/1600行は5Bを超え、三連結は未成立。`prime-down-v1/v2`に成功metricと失敗requestを保存した。GQA削減だけで全体query数は減らないため、残り処理の配分とterminalで準備済みMLPを使うAPIを次に検討する。50/32到達は未証明。

さらに`prime-down-v3/v4`でMLP24部分downを512/768行へ減らすと三連結が成立した。768行の第3queryは4,912,719,438命令。返却状態をそのままMLP24 finish→Delta25 FULL→MLP25前半へ渡し、前半768/1024行の第4queryも4,809,517,604 / 4,888,189,594命令でbit一致。1280行は5B超過した。主132 tokenで4連結が成立したが、残りの層はまだ未接続である。

## 全体回帰

`artifacts/attention_views/full-proof-v1`の標準6条件と`joined-proof-v1`の連結3条件が、全export hidden・保持state・最終hidden・判断/確率で既存INT8参照とbit一致。失敗/replay0。ソース/参照/module bookendとsource archiveを保存した。

主の連結経路は51query、233,858,512,697命令、150,311,010 Candid bytes、最大4,918,963,064命令、観測heap4,144,037,888 bytes、単回39.466秒。前版比6,926,776,895命令（2.8767%）減、query数/通信/観測heapは不変。情報不足は51query・214,134,927,862命令、最大変更は62query・232,868,320,091命令。標準主62queryは227,795,678,436命令。単回時間はばらつきがあり、一般的な速度改善は主張しない。

入力を短縮せず、採用INT8参照への差はない。元BF16モデルとの差や既存の最大変更の見逃しを解消した検証ではない。50/32queryは未達。

## 50query向け後段の境界

`prime-down-v5`では前半3584/heads18/down768/次前半1024という4連結から、MLP25 complete＋Delta26の10 headsまで4,826,887,496命令で進み、全返却状態がbit一致した。12/14 headsは5B超過。10 headsの残り22 heads＋MLP26 front6912は要求が2MBを超えて送信前拒否となった。5連結は成立、6連結は未成立である。

この実carryのhidden BF16は445,440 bytes、2-plane Huffmanなら300,945 bytesで144,495 bytes減る。base F32の量子化を増やさずに要求を小さくできる候補であり、次はfollow要求のhidden可逆圧縮を実装・実測する。後段にはMLP stream完了→Attention全体、MLP全体→Delta先頭、準備済みMLP30→terminal/readoutの接続も必要。いずれも現時点で50query graphへは未接続である。
