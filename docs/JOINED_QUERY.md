# 全体51 queryの連結経路

`--roll-start --join-start`で、既存の8query連結とMLP完了→次Attentionを全体へ接続する。主入力はprefix45＋suffix87＝132 token、rotations=1。中間状態はクライアントが保持し、全て通常queryで進める。モデル固定準備だけupdateを使う。

0〜23層は8層13queryを3回、24〜28層は8query、29層とDelta30は3query、既存terminal tailは1query。最後のDelta残りを`delta_partial_finish`で完了し、不要なMLP30準備を省いてtailへ渡す。MLP完了→Attentionのcompact返信は、Attention内部で消費したMLP normを返送しない。既存APIと診断用のfull返信を維持する。

## レビューで修正した点

全体のMLP→Delta接続に可逆残差圧縮を渡していなかったため、2MB frame境界で失敗した。`joined-proof-v2`の失敗を残し、既に部分検証済みのraw閾値0.1＋辞書plane圧縮を接続した。量子化を増やす圧縮ではない。

全体比較コードのbaseline report/NPY/NPZを検証前にhash固定し、固定した同じbyteから復元する。実行後にsource・Wasm・参照を再確認する。Python -Oを拒否し、標準全体checkerは既存証跡の上書きも拒否する。新graphは88/89 tokenを既存経路へ切り替え、不正prefix/layersをquery前に拒否する。

## 検証

Rust 140 unit（1 ignored）、15 integration、11 doc通過。compact/finish-only Python境界4、既存MLP pair7、Delta start3、rolled境界3、連結codec6、参照固定5、新graph境界2テスト通過。

`compact-all-v2`は18条件中15成立・3 IC0522。87 token/1280 downのcompact queryは4,642,794,200命令。full返信版比7,126,007命令・445,440 bytes減、同moduleの分離2query比は2,770,505命令増・892,451 bytes減。89 token/768 downは依然超過、1280 downは成立する。

`joined-proof-v3`は3入力で、全保持hidden/state、最終判断・確率・final hiddenが既存INT8参照とbit一致。query失敗/replay0、source/reference bookend、83ファイルのsource ZIPを保存。標準経路の全6条件は`full-proof-v2`で一致し、参照固定を強めた`full-proof-v3`でも全6条件（prefix66、baseline64、主/情報不足/最大変更各62、prefixなし290query）が一致。329参照file・88 source fileをbookend検査し、source ZIPを保存した。

| suffix入力 | query | 合計handler命令 | Candid要求＋返信bytes | 最大query命令 | 単回時間s |
|---|---:|---:|---:|---:|---:|
| 主87 | 51 | 240,818,320,871 | 150,311,010 | 4,918,195,298 | 33.987 |
| 情報不足80 | 51 | 220,292,051,377 | 140,329,464 | 4,508,093,937 | 35.604 |
| 最大変更89・fallback | 62 | 240,031,063,972 | 125,231,193 | 4,816,569,729 | 38.221 |

前54query版比、主は798,023,131命令増（約0.33%）、16,414,018 bytes増（約12.26%）。同候補の標準62query版比でも命令・通信が増える。query数削減の実験オプションとして扱い、性能全般の改善とは断定しない。各queryと全体の通信はCandidの範囲、handler命令はCDKのCandid decode/encodeを除く。単回時間はcache/環境差を管理した速度評価ではない。

主の観測heap終了最大4,144,037,888 bytes。固定cache4,065,416,192 bytes、準備721 update・78,739,929,194命令・263.164秒を推論と別計上。module `b6dbf3a70040920791811d47d525d5f3f17d9fa738ee854f65b785f06adf208a`、`full-build-v2`、4 WAT patchをwasmparserで検証。Wasm演算sourceと実行時client sourceはそれぞれbuild/proof archiveに記録する。

コピーした`joined-resume-v3`で51/51 queryがcheckpointから再開し、判断一致を確認。元の完走証跡を上書きしない。既存主canister `4caro...`のRunning/module `36c04a57...`は不変。Layaのソース・Git・稼働canisterは変更していない。

50/32 queryは未達。採用INT8版へのbit一致は、元BF16モデルに対する判断精度とは別で、最大変更の見逃しも未解消。[残りの探索案](JOINED_QUERY_NEXT.md)では、不要carryの除去とQ/Delta/MLPの配分を個別実測する。

## クライアントprofiling

`joined-client-profile-v3`はコピーしたcheckpointを51/51 replayし、判断一致。cProfile全体9.588秒、module確認3回の応答待ち6.841秒、graph 2.130秒、可逆plane encoder累積1.583秒。時間はinclusiveで、足し合わせない。前のreplay graph runは26.311秒、このprofile runの同じreport範囲は2.895秒と大きく変動した。通信待ち/cacheの影響を分離しておらず、26秒を復号/梱包のCPUボトルネックとは断定できない。次は必要なmodule bookendを保持して重複確認の1回を省く案と、encoderの有限値二重走査を別々に検証する。
