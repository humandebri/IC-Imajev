# INT8の完全pairと奇数末尾を分けた追加最適化

2026-10-05。INT8行列積の候補を実装し、現在の短文構成のBOOM DAO 3入力で推論全体の命令数が0.943～1.064%減少した。117 queryすべての状態返信、融合queryの追加hidden/conv、最終判断・logit・確率が元とbit一致した。重み、量子化、F32演算の順序は変更していない。

## 全体の実測

|BOOM DAO|suffix token|元の命令数|候補の命令数|削減率|query|
|---|---:|---:|---:|---:|---:|
|617|67|179,598,906,781|177,905,279,101|0.9430%|39|
|620|59|159,074,614,698|157,425,372,618|1.0368%|39|
|653|57|153,923,079,311|152,284,933,631|1.0643%|39|

1推論あたり約16.4～16.9億命令の削減。最大queryは順に4,887,431,937 / 4,404,555,846 / 4,313,660,361命令で、5B上限内だった。送受信Candid bytesも元と同じ89,875,492 / 81,475,306 / 79,466,460。

既存39 query経路の保存済み入力を、正しいcanister method・prefix・選択肢で再実行した。各返信のframe全byteを照合し、融合queryではprevious hiddenとconvも照合した。元の全経路と同じ出力になることを検証する比較であり、clientの分割計画を新しく探索した測定ではない。回数削減はまだ示していない。prefix計算・固定重み準備は合計から除外し、時間の改善率も主張しない。公開ベンチ100問の再評価は行っていない。

## 変更した計算

S1は2 tokenずつ処理する。偶数切り下げの上限fullnまでのループでは、第2 tokenが存在することが確定している。この区間の33箇所のkeep判定を省く。最初のpairは重み係数の準備を兼ねるので変更しない。

残る奇数末尾は、第2 tokenのゼロpaddingにより不要となるproduct1/3と、未出力の第2 tokenにしか使わないproduct5を省く。さらに、第2 tokenのscale loadと32出力グループの条件付きstoreを取り除く。第1 tokenの再構成でproduct3=0の加算も32箇所省く。整数部分の恒等式を記号展開して確認し、I32上界も確認した。F32変換・input scale・weight scale・K256 block加算順序は元のまま。

同一診断moduleに現在採用されている128出力版と候補を入れ、layer3 Q（8192×2560）の実重みで比較した。5保存済み実入力、保存活性値を切り出した57/59/67 token、符号・zero・極小値を含む11境界の計19ケース・38通常queryで、両版ともnative INT8基準とdigest一致。

保存済み実入力の投影全体はprefix45で1.874%、suffix87で1.148%、80 tokenで0.301%、89 tokenで1.130%、cold132の半分出力で0.304%減少した。57/59/67 tokenのサイズ試験は1.563 / 1.523 / 1.385%減少。1/2 tokenでは約0.014/0.013%増加している。現在の全体推論の1 tokenは既存single-quad経路を使い、この置換対象から外れる。短い境界での大きな相対改善を全体推論へ一般化しない。

## 証拠と状態

生成はscripts/build_s1_pair_bounds_probe.py、単体測定はscripts/check_s1_pair_bounds_probe.py、全体比較はscripts/prove_s1_pair_bounds_full.py、応答・hash・counter・記号式の独立再検証はscripts/report_s1_pair_bounds_probe.py。

artifacts/s1-pair-bounds-v1/buildにsource ZIP、依存rlib hash、WATとpatch検証を保存した。診断module 952cb396…と全体候補5f761276…の置換body SHA256は一致する。全体候補は現行6052cc94…の名前付きS1 wide bodyだけを置換し、他のsection/bodyを保持した。

単体証跡はcheck/report.json、全体はfull-proof-v2/report.jsonと各入力のreplay.json、再検証集計はreport.json。第1回の全体比較は融合queryを通常stepとして呼ぶ測定器の誤りで停止した。元のmoduleでも同じ誤呼出しの返信が候補と一致することを確認し、融合queryを正しく再現する第2回で全117 queryを完了した。失敗ログ・当時の測定器・復元記録はfull-proofに残し、成功結果へ混ぜていない。

専用ローカル診断2sm34を再利用した。全体比較では実験用6eyddをsnapshot保存し、一時的に候補へupgradeして721固定重みを準備した。終了後はsnapshotから元の6052cc94…、4,065,416,192 bytesの721重みcache、pack情報を復元し、一致を確認して今回のsnapshotだけを削除した。復元後のmoduleは独立のstatus読み取りでも確認済み。保護対象4caroとmainnetは変更していない。

候補Wasmはbuild/full.wasm。既定の推論への採用は行っていない。clientのpacked_planは現行module hashだけを許可しているため、採用時には検証済み候補hashの接続とprefix cache identityの扱いも必要になる。この測定は現在の39 queryのままでの命令数削減を示す。
