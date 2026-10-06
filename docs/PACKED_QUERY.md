# 短縮promptに合わせたquery境界の再配分

2026-10-05。従来の50 query経路から、クライアントによる分割選択だけでは47 queryまでしか減らなかった。既存APIではMLP完了、次層Delta、次のMLP開始の間に境界が残ったためである。`query_mlp_delta.rs` に owner専用の `mlp_delta_front` を追加し、その3処理を一つのquery内で既存kernel・演算順序のまま実行する。`PackedPrefixGraph` はtoken数に応じてMLP generation/downの境界を選ぶ。クライアント側にモデル演算は移していない。

| ケース | total token変更 | 初期query数 | client分割v2 | 結合query経路 | 命令合計・十億 |
|---|---:|---:|---:|---:|---:|
| 617 | 132→94 | 50 | 47 | 39 | 234.521→179.599 |
| insufficient | 125→87 | 50 | 47 | 39 | 214.719→160.069 |
| maximum | 134→96 | 62 | 47 | 40 | 232.780→184.775 |

短縮prompt固定50 queryとの比較でも、39/39/40回へ減少。代表入力の総命令は長いpromptから約23.4%減、query回数は22%減。Candid request+replyは代表入力89,875,492 bytes、insufficient82,598,479 bytes、maximum96,129,185 bytes。prefix/weight/packet準備・module照会は推論の回数と費用から除外し、別のreportへ保存する。

prefix27、suffix60/67/69で実推論を検証。全31個のexported hidden（compact terminalのlayer30を除く）、全32層の保持conv/KV/positions、最終norm、logits、確率、判断が、新promptの汎用query経路とbitwise一致。新しい量子化や近似は加えていない。maximumの既存誤判定noも維持し、推論精度の改善は主張しない。最終経路には失敗・fallback・replayがない。

suffix68以上では最初のMLP中間に1 queryを追加し、MLP downは2304/256行へ分ける。初期MLP front8448、次のMLP front3840→6656、Attention直前のfront2304を使う。小さい入力では256行境界で `rows * 67 / suffix_tokens` を丸める。kernelのhandlerカウンタだけでなく、Candid decode/encodeを含む実際の5B上限で成功することを確認した。各query最大handler命令は約4.94B以下。これは測定形状用の予算選択であり、未知形状の厳密なworst-case保証ではない。長いprefix/suffixや別moduleは固定経路へ戻し、上限超過時は独立した標準query journalへfallbackする。失敗queryも全体回数へ数える。

実験用ローカルcanisterは `6eydd-o3777-77775-aaama-cai`、測定moduleは `6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931`。旧885 moduleの固定runtimeと同じ4 WAT kernelを使用し、wrapperに結合queryを追加した。Rustのfeature集合はCargoで展開された依存featureも含めて一致させた。外部rlib・凍結runtime source・compiler・WAT hashは `artifacts/query-packing-v3/build/provenance.json` に保存した。現在の共有checkoutには別作業のruntime変更があるため、この計測moduleは凍結sourceからのビルドである。共有checkoutのclient/canisterについてもcargo checkは通過した。

使う場合は既存のtail/join/roll設定へ `--packed-start --bridge-binary artifacts/query-packing-v3/build/imajev-client` を追加する。canisterの結合query入口は `experimental-mlp-delta-query` featureで有効になり、MLP streamとDelta full logの依存featureも有効になる。従来の `experimental-update-inference` もこのfeatureを含む。この経路には結合queryを持つmoduleと、そのmoduleに紐づくprefix/packet準備が必要で、旧moduleへflagを付けるだけでは使えない。既存 `--adaptive-start` の47 query経路も残す。

計測・全層比較は `artifacts/query-packing-v3/run_queries.py`、結果は同directoryの `final-summary.json` と `final-{617,insufficient,maximum}/report.json`。境界探索中の失敗はtrial directoryとfallback reportに保存しており、最終の成功経路の回数には混ぜない。最終経路には初回計測の失敗費用を含めない一方、個々のfallback実験reportにはその実験の失敗・再実行費用を含む。不正なprefix長、front、alignment、payload長、NaNの5拒否試験も実験canisterで確認。planner/dispatch5件、従来adaptive4件、fallback3件が通過。

実測plannerのsnapshotは `measured-packed-source.py` に保存し、final-summaryのsource hashと照合済み。計測後、最大入力の追加MLP queryを層別帳票へ配賦する位置だけを修正した。保存された40個の要求/返信を全てidentity/hash照合してreplayし、層別query数合計も40となることを確認した。再実行した推論queryは0で、元の実測40回・命令数・通信量・出力は不変。確認帳票は `accounting-check-maximum/report.json`。
