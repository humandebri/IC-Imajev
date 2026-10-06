# 元のblock256量子化を維持する128行分割

`experimental-mlp-half` と `build_full_prefix_candidate.py --mlp-half` でMLPのquery境界を256行から128行刻みへ広げた。精度を落とす量子化変更ではなく、同じ計算をquery間へ配置するための実験機能。通常query・client-held状態を維持する。

入力のINT8化と元F32のgate/up LoRA A積は従来どおり初回だけ実行する。MLP productは完成した256要素blockだけをINT8化する。端の128要素は元のBF16で保存し、次chunkと結合して256要素単位で一度だけ量子化する。down LoRA A積は新しいchunkだけを処理し、pending分を再計算しない。F32積和の順序とBF16丸め位置を維持する。

## 状態の形式

従来のtag1（256行境界）はbyte互換。128行を持ち越すtag2は、全tokenの完成済みINT8 product、全tokenのpending BF16、完成済みblockのF32 scale、F32 down A積の順。tokenごとの完成block数は `floor(done/256)` であり、`floor(n*done/256)` ではない。

pendingは最大 `n*128` 要素。通常INT8保存との差は最大 `n*128` bytes（87 tokenで11,136 bytes）。同じproduct総数での表現差であり、前の5888行から6016行へ増やした場合のframe増加は別に測定する。下位16 bitが非zeroのpending、非有限値、tagと進捗の不整合、scale・frame上限を拒否する。featureなしのWasmは128行metadataを拒否する。

## 検証

- Native：131 unit（1 ignored）、15 integration、9 docが通過。`[128,128,8960]`、`[6016,3200]`、`[128,256,128,8704]` の分割で、token数1/7/87/89の完成product INT8値・block256 scaleが一括量子化とbit一致。各chunkでcarryをencode/decodeし元BF16 pendingと元F32状態のbitを確認した。
- F32 down Aの継続積和テストに128行を加え、従来一括計算とbit一致。featureなしのMLP境界テストも通過。
- Python：保存済み315 payloadのbyte互換、新tagの往復・不正pending/scale/tag/長さの拒否を確認。
- build source bookendが一致し、4つのWAT kernel patchをwasmparserで検証。module `96e58a669280e80be53bc65301585f15a33c2f7e4b376a353f26fa54eb68a935` を専用実験canister `6eydd-o3777-77775-aaama-cai` にupgradeした。

実測結果は後段に記録する。全体への採用前に通し回帰を行う。50/32 queryは未達で、128行単位を使えることだけでは全体query削減を意味しない。

## ローカル実測

固定モデル準備は721 update、内部重み4,065,416,192 bytes、78,739,929,194命令、258.787秒。Candid要求47,473 bytes・返信14,949,136 bytesと内部重み量は区別する。質問に依存する状態は準備updateへ保存しない。

`artifacts/mlp_half/stream-{617,insufficient,maximum}-v1` の3入力（87/80/89 token）、3層（0/1/3）、2分割（6016+3200、128+256+128+8704）の計18条件が成立。各入力27回、計81回の通常queryで、従来の準備状態・hidden/normとbit一致した。これは単体MLPの診断であり、全体query削減ではない。tag2を拒否する旧クライアント検査が残っていた初回失敗は修正し、Attention/Deltaの接続テストを追加した。

主87 tokenの五連結（layer1、down1408、前半1792、compact Q4、可逆残差dictionary）では、Attention→MLP前半6016行が **4,894,874,666命令** で成功し、carryが単体参照とbit一致した。続くMLP完了→Delta先頭24 headは **4,752,315,740命令**、Candid計 **3,591,917 bytes** で成功。従来5888行の **4,789,488,798命令／3,569,645 bytes** に比べ、37,173,058命令減・22,272 bytes増。ただし後段の残り8 head＋全MLPはIC0522、先頭26 headの第四queryもIC0522。

6144行・6272行では第三query（Attention後半＋MLP前半）がIC0522だった。6016行の結果から直線的に上限を推定しない。成功metric/profileと失敗要求のSHAを `chain-main-6016-v2`、`chain-main-6144-v1`、`chain-main-6272-v1` に保存。要求連結数・完了数・失敗数も明示し、診断exit 0を連結成功として扱わない。

この変更で `stream_product_quantize_once` のprofile区間には量子化とproduct/scale連結を含む。以前の同名区間（量子化のみ）とは範囲が違うため、当該bucketだけの増減を演算削減として比較しない。

四つの成立query（第一～第四）の合計は5888行の旧候補より **3,613,085命令増**。第三へ移した計算とpendingの扱いで全体負荷は減っていない。第四だけの命令減を全体改善とは扱わず、まだ全体graphへ接続しない。

新Wasmの全6条件（prefix、baseline-617、617、情報不足、最大変更、prefixなし132 token）の通し回帰は全保持hidden/state・最終判断/確率がbit一致。証跡は `artifacts/mlp_half/full-proof-v1` のreportと79ファイルのsource ZIP。主の標準経路は62 query／234,721,321,335命令／123,281,081 Candid bytes。こちらは現行連結経路54 queryとは別の比較である。

現行連結3条件の通し回帰も全保持hidden/state・判断/確率がbit一致し、失敗要求保存・replayとも0。`artifacts/mlp_half/rolled-proof-v1` に69ファイルのsource ZIPを保存した。

| 入力 | 全体query | 合計命令 | Candid計bytes | 最大query命令 | 単回秒 |
|---|---:|---:|---:|---:|---:|
| 主87 token | 54 | 240,544,043,820 | 133,896,992 | 4,900,654,558 | 36.414 |
| 情報不足80 token | 54 | 220,043,004,615 | 125,114,808 | 4,492,999,419 | 33.458 |
| 最大変更89 token・fallback | 62 | 240,030,892,427 | 125,231,193 | 4,816,563,937 | 36.371 |

主のquery数と通信量は従来と同じ。旧moduleの240,546,093,202命令から2,049,382命令（約0.000852%）減ったが、128行分割を全体へ接続した効果ではない。50/32 queryは未達。

## 次の繰り返し処理削減

現行の `Parts::flat` → `encode` は完成済み整数productをF32へ展開し、返信時に再びINT8へpackする。入力qも同様の往復を行う。次は検査済みのopaque reply型から整数値・BF16 pending・F32 scale/A積を直接byte列へ書く経路を探索する。未検査のクライアント入力への検証を省略せず、server内部生成の型で信頼境界を明確にする。新旧frameのbyte一致と通常queryの実命令数で採否を決める。
