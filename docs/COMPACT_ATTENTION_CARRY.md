# 重複norm/KVを除くAttention carry候補

2026-10-04継続。実験API `attention_finish_mlp_front_q4_compact` を追加した。前queryが返した量子化済みQ入力と元F32 A積を再利用し、Q射影に不要となったBF16 normを再送・展開しない。hidden、K/V、先行4 headのgate、prefix、入力scale、元F32 A積は保持する。後半queryの返却から、clientが前queryから保持している同一K/Vも除く。質問状態はclient保持、推論は通常query。

87 tokenでnorm入力445,440 bytes、KV返却356,352 bytes、合計801,792 payload bytesを除ける。これは形式上の差で、命令数や全体通信量の実測結果ではない。旧op/tagは維持し、新request tag6、新reply tag7を使う。Q+Aが両方ある内部経路だけnormなしを受け付け、Qなし又はAなしの通常経路では従来どおりnormが必要。

Deltaとの連結ではMLP carryのBF16残差だけをHuffman byte planeで可逆圧縮するrequest tag6も追加した。INT8 lanesと未丸めF32 scales/A/accumulatorは元bytesのまま。復元時の割当は検証済みtoken/行数から決め、圧縮descriptor由来の長さから割り当てない。圧縮サイズ・進捗・後半bytesを検証し、従来のMLP carry validatorへ渡す。旧未圧縮tag1を維持した。

保存済み主87 tokenのMLP3 carry残差445,440 bytesは、低byte plane219,549・高byte plane78,940 bytes、追加4-byte descriptorを含め146,947 bytes小さくなる。実測入力hashとサイズは `artifacts/prefix_codec/residual-plane-size-v1.json`。同じ残差を使う6144行・26 headの候補payloadは1,927,815 bytes（frame header/digestは別）。6144行のcanister出力や復号命令数を測った結果ではない。

追加後のnative: 127 unit、15 integration、9 doctest通過、fixture依存1件ignore。Python codec22件・journal6件・join3件通過。Qの4+12 split synthetic testでnormあり/なしの出力bitとweight readが一致、signed zero/F32 tail/direction/descriptor上限/truncationを検証した。新候補build/source/patchは `artifacts/prefix_codec/full-build-compact-q4-v2`。全体へは未接続、旧主経路54 query・50/32未達。canisterでの成立と出力一致を次に確認する。

前のレビューとコミットは [ROLL_JOIN_REVIEW_2026_10_04.md](ROLL_JOIN_REVIEW_2026_10_04.md)。Layaと主canisterは変更しない。

実験module `3cbdd7c9cbc363ab825a06f994e8d5fa37bc84851318a0690e1955b0e47d95e3` を専用6eyddへupgrade。固定準備は721 update・4,065,416,192 bytes・78,739,929,194命令・271.082秒。実ネットワークの準備通信は47,473 request＋14,949,136 reply bytesで、canister内の重み読み出し4GBと分離する。

`compact-q4-chain-v1` で80 token/6144行/26 headの五連結が全ビット一致し、22,126,192,038命令・13,924,287 Candid bytes。87 tokenではAttention後半6144行queryがIC0522、89 tokenでは最初の連結queryがIC0522。失敗requestとhash、途中成功metric/profileを保存した。

`compact-q4-chain-main-v2` で87 token/5888行のcompact Attention queryは4,853,839,732命令・2,672,702 Candid bytes、carryが独立参照と全ビット一致。旧形式の同じ5888行実測4,880,950,763命令・3,474,478 bytesから27,111,031命令（0.555%）・801,776 bytes減った。Candid headerの長さが変わるためpayload上の801,792 bytesとの差を区別する。moduleは前後で違うため総演算改善率の根拠として拡大解釈しない。

残差圧縮を伴うDelta先行24 head queryは4,862,445,223命令で成立しcarry/convが一致したが、未圧縮の前回4,815,444,802命令より増えた。可逆圧縮はframe上限を緩めるが、命令数削減としては採用しない。次のDelta残り8 head＋MLP全体は上限超過、先行26 headのquery自体も上限超過。主五連結は未成立で全体54 queryのまま。次はこの実測上の演算ボトルネックを減らす必要がある。

新moduleの既存全6条件は `full-compact-q4-proof-v1`、既存連結3条件は `rolled-compact-q4-regression-v1` で完走し、保存対象のhidden/state/final hidden/型付き判断/logits/確率が全ビット一致。いずれもsource/moduleを前後確認してsource ZIPを保存した。主連結54 query・241,643,997,846命令・133,896,992 Candid bytes・単回36.864秒、情報不足54 query、最大89 tokenは62 query fallback。失敗/replay0。compact API自体は全体未接続で、50/32 query未達。
