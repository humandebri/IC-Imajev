# 50 queryへ向けた追加削減

最新の細粒度計測と数値保存改善は[ボトルネック分析](BOTTLENECKS.md)。0.847兆命令・868 queryで、50 queryは未達。以下は前段階の比較とF32方式の条件付き予算。

2026-10-01。Layaの既存ソース・実測JSON・Git状態を読み取りだけで確認した。canister、network、ソース、Gitを変更していない。ImajevのINT8 base、F32 adapter/readout、公式promptを維持する。

## 「420Mで5 query」の比較条件

保存済みBOOM DAO 617のLaya入力は44 tokens。F32通信5 query・13,445,080,788 handler命令、INT8通信5 query・15,141,614,882命令。質問・選択肢・構造化要約はImajevと同じだが、tokenizerとモデルが必要とするpromptは異なる。Imajevは132 tokensであり、token数は3倍。Layaには後続の68-token実入力をF32 5 query／INT8 6 queryへまとめた別測定もある。44と68の値を混ぜない。

前の比較に使った128-token・17-query結果は長い境界入力の別実験だった。「最新」の呼び方を訂正し、BOOM DAOの実測をこの比較の起点にする。保存された同じ自然文を使っても両tokenizerで長さは揃わない。Imajevの公式の指示・unknown説明を削除して短くする操作は、精度維持改善には含めない。

モデル容量だけの10倍換算には、入力長の差と演算方式の差が欠けている。LayaのF32/INT8という表記は中間通信形式で、dense射影には整数dotを使う。Imajevの採用経路はINT8重みをF32へ展開し、F32乗算→F32加算の順とBF16丸め境界を保存する。readout・LoRAも別のF32処理。保存済み比較とhashは [fifty-query-budget.json](fifty-query-budget.json)。

## 命令数をさらに削る実装

4 tokens×16出力行から、16・32・64 tokens×16出力行へweight load/broadcast共有を広げた。独立したF32 accumulatorの列順を維持する。8-token候補も測定した。実入力8射影の命令数合計は、8 tokensで10.47%、16で17.84%、32で20.28%、64で20.73%減。QKVの132×1236×2560では1,781,833,854→1,356,379,851、23.88%減。

64 tokensは96-token down射影で32より不利、小さい32行gateも16より不利だった。採用経路はn>=128かつrows>=64で64 tokens、n>=32かつrows>=64で32、それ以外16。端数は既存4 laneとscalarへ戻す。各出力についてF32乗算・加算・列順・BF16境界が同じ。全候補8件の出力bit一致、160境界形状のraw F32とnative scalarの全bit一致を確認。[候補実測](multi-token-candidates.json)、[境界検証](linear-tiling-parity.json)。

追加で`lora_grouped`を最大3 virtual tiles・1,350M MACへ広げた。元の各tileを256 float境界にpaddingするため、INT8通信の各blockのmax・scale・丸めは保存される。clientの`--projection-group-cap 3`を明示した場合、n>=64で最大3 tiles、短い入力で最大2 tiles。従来設定の既定値2は維持し、sessionに新設定を記録する。不正な4 tile・work超過など6契約の拒否を実canisterで確認した。3 virtual tilesの部分blockも91 codec形状で検証。

## 50 queryの予算

この調査後に演算方式変更の許可を得て、整数baseを全32層へ接続した。1.022兆命令・932 query、ホスト23件のラベル一致を確認。現経路・判断精度差・50 queryへ残る予算は [INTEGER_ARITHMETIC.md](INTEGER_ARITHMETIC.md)。以下の95回という下限は旧F32積和を維持する経路に限った説明。

通常queryの50億命令枠を50回使うと、総予算は2,500億。Imajevの132-token層内base denseだけで約4,711億MACある。公式IC実装の[命令cost表](https://github.com/dfinity/ic/blob/master/rs/embedders/src/wasm_utils/instrumentation.rs)ではF32x4MulとF32x4Addはそれぞれcost 2、4 laneで4 MACを処理するため、乗算・加算だけでも1 MACあたりcost 1となる。

このcost表の下で全dense積和と元のF32順を実行する経路では、ロード・ループ・LoRA・codec・stateをすべて無視してもbaseだけで最低95 query相当。これは現在のlocal replicaのcounterを実測した値ではなく、明記した演算経路と公開cost表に依存する条件付きの下限。異なる数値アルゴリズムやキャッシュへ適用する普遍的下限ではない。

共通45-token prefixを正しいDeltaNet/KV状態付きで再利用できた場合も、未cacheの87 tokensのbase denseだけで約3,105億MAC、同じ条件なら最低63 query相当。初回cache作成・state通信の費用は別。したがって、今のF32積和をそのまま保ったロード共有・分割幅変更だけで50回に到達する見積もりは立たない。

50回を狙う大きい候補は、Laya同様の整数dotへ全射影を移すこと。ただし、activation量子化とF32逐次加算の変更を伴う単純W8A8は現経路とのbit一致を保証しない。採用には、少なくとも保存済みBOOM DAO・重大変更・誤警告・情報不足・選択肢順のケースで判断精度を別に検証する必要がある。bit保存と判断精度は異なる条件であり、前者を満たさないだけで後者の劣化を断定しない。一方、1件のラベル一致だけで精度維持と認定しない。

別候補は、整数候補の誤差区間が元のBF16丸め区間内に収まる出力だけを採用し、それ以外を元のF32で再計算する方式。これならbit保存を目指せるが、正しい誤差上限の証明、fallback率、LoRA境界、命令数の実測が必要。未実装なので削減率や50 query達成を約束しない。
