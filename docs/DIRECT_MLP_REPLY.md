# MLP carryのF32往復を省く直接返信

`experimental-direct-mlp-reply` とbuildの `--direct-mlp-reply` により、MLP streamとcompact Attention→MLP前半の返信をruntime内部の型から直接byte列へ書く。

従来は入力q（I16内部）、完成product（INT8内部）、BF16 pending、F32 scale/A積を一つのF32配列に展開し、encoderが整数範囲・有限値を再走査して元の形式へ戻していた。新経路はproductをINT8のまま転送し、qもF32化せずINT8へ書く。量子化block256、F32積和順・BF16丸め、返信のwire形式は変更しない。

公開の `EvaluatedReply::Payload` はopaqueな `PayloadReply` で、外部からrequest/payloadを構築・変更できない。runtime内で検査済み入力または計算から生成した状態だけを使う。返信encodeは元requestの全identityとstep+1を検査する。F32 A積、scale、BF16 residual/pendingの有限値・精度・shapeも確認する。クライアントの受信状態をこの型へ直接変換する公開経路はなく、従来のquery decode検証を維持する。

既存の数値APIはF32配列を返すまま。canister step/profileだけがtyped reply APIを使い、terminal decisionは数値出力を要求する。featureを無効にすると従来経路に戻る。version1/2/3、通常checksum・owner認証後version3の署名付き返信の条件も従来と同じ。

## 検証

Native132 unit（1 ignored）、15 integration、10 docが通過。token1/7/87/89、進捗128/4096/6016/9216、frame version1/2/3で新旧返信のbyte一致を検査した。全request fieldの変更を拒否し、opaque fieldの外部構築はcompile-failで確認。half/checksum等を有効にしない最小feature構成のMLP境界テストも通過した。

最終buildは `artifacts/direct_mlp_reply/full-build-v3`。source bookend一致、4 kernelのWAT patchをwasmparserで検証。module `500beb090e7cd61083b44138c3be85ab2c140832d7f80ca0948b58cc21b2b7bc` を専用ローカル実験canister `6eydd-o3777-77775-aaama-cai` にupgrade。保護対象/Layaは変更しない。

実命令数と全体回帰は以下に記録する。現時点で主の全体は54 query、50/32 queryは未達である。

## 実canisterの効果

`stream-{617,insufficient,maximum}-v1` の3入力×3層×2分割で、完成状態・hidden/normが固定参照とbit一致した。さらに旧moduleの同じ81個の返信frameと、新moduleの返信frameが **全バイト一致**。比較hashは `stream-byte-comparison.json` に保存。

主87 tokenの6016行Attention→MLP queryは4,894,874,666→**4,819,518,332命令**、**75,356,334命令減**。profileのwire_encodeは78,550,221→1,317,001命令。ただしpayloadを作る処理がevaluate側へ移ったので、削減はquery全体で判定する。同じ通信形式・byte列のまま軽量化した。

単体MLP分割の合計削減は、87 tokenで130,564,634〜136,068,799命令、80 tokenで118,248,417〜127,038,801、89 tokenで133,579,297〜140,525,072。これらは部分演算診断であり、全体への削減を意味しない。

## 五連結が成立した条件

前半6272行・Delta先頭26 head・down1408行・最初のMLP前半1792行で、主87 tokenの5連結が初めて成立した。

| query | 命令 | Candid要求bytes | 返信bytes |
|---|---:|---:|---:|
| MLP完了＋次Delta全体＋次MLP前半 | 4,843,286,342 | 1,744,728 | 1,392,044 |
| MLP完了＋Attention K/V＋Q4 | 4,649,608,580 | 1,081,692 | 1,674,526 |
| Attention残りQ12＋MLP前半6272 | 4,896,694,381 | 1,413,331 | 1,304,263 |
| MLP完了＋次Delta先頭26 head | 4,877,795,556 | 1,986,454 | 1,692,187 |
| Delta残り6 head＋次MLP全体 | 4,785,406,452 | 1,835,264 | 900,735 |

5本計24,052,791,311命令・15,025,224 Candid bytes。独立参照のhidden/norm・carry・convとbit一致。参照を作る別queryとprofile queryはこの5本の合計に含めず、診断call一覧には記録している。

全7箇所×3入力へ広げた `chain-all-6272-v1` は、要求21連結・完了14・失敗7。主87 tokenと情報不足80 tokenは各7箇所とも成立、最大queryはそれぞれ4,898,014,971／4,474,572,163命令。最大要求は1,986,454 bytes、観測heap最大4,144,037,888 bytes。89 tokenは7箇所とも入口でIC0522、既存の62-query fallbackを引き続き使う。

6144行は第四query、6016行は第四の26 headまたは第五queryでIC0522を保存した。幅の変更で命令数は単純な比例にならない。失敗を削除せず、source/module/referenceのbookendと要求hashを保存する。

主の全体54 queryを50へ減らすためには、この連結と層境界の他の組合せを通しgraphへ接続し、50本がすべて上限内で終わることを改めて測定する。五連結の成立だけでは50達成とは扱わない。

## 通し回帰

`full-proof-v1` の全6条件と `rolled-proof-v1` の3条件が、全保持hidden/state・最終判断/確率でbit一致。失敗/replayとも0、source/module bookendも一致。source ZIPはそれぞれ80／70ファイル。

| 現行graphの入力 | query | 合計命令 | Candid bytes | 最大query命令 |
|---|---:|---:|---:|---:|
| 主87 token | 54 | 240,543,020,619 | 133,896,992 | 4,900,659,775 |
| 情報不足80 token | 54 | 220,042,153,261 | 125,114,808 | 4,493,004,686 |
| 最大変更89 token・fallback | 62 | 240,031,043,882 | 125,231,193 | 4,816,568,929 |

現行graphは新五連結をまだ使わずquery/通信は同じ。前module比で主1,023,201命令減・情報不足851,354減・最大変更151,455増で、この微小差を直接返信の全体効果とは扱わない。部分queryの75M削減と五連結の成立が、次のgraph変更の根拠である。
