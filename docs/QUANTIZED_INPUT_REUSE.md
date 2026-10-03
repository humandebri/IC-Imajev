# 量子化済み入力の重複検証を除去する

2026-10-02。`QuantizedRows`を外部から生成・変更できない型にし、`quantize_rows`だけが生成するようにした。有限入力、dimension、block256のshapeを生成時に確認し、量子化のclampで値域[-127,127]、scaleの生成式で有限かつ正のscaleを保証する。値・scale・dimensionは変更不能な参照/accessorで読む。

以前は投影するたびに同じ入力の値域・scale・shapeを全走査していた。gate/upなど同じ量子化済み入力を共有する投影では二重になっていた。生成後の不変条件を再利用し、この走査を除く。投影重みの長さ・scaleの有限性・正値、出力サイズと有限性の確認は継続する。外部queryから量子化済み配列を受け取る方式には変更していない。クライアントが保持する中間状態も従来どおり。

INT8量子化方式、積和、F32加算順、BF16境界、元のadapter・readout・calibration、weight packは変えていない。Rust40 testsと改変禁止を確認するcompile-fail doctestが通過した。極端な有限値・最小subnormal・token padding・不正入力・不正weight scaleを確認する。外部からのフィールド直接変更に依存するRust API利用者はimmutable accessorへ移行が必要。

生成物はgitignore対象。Laya・mainnet・remote Gitは変更していない。

専用canisterで10種類の実投影shapeを旧出力と比較して全bit一致した。87-token・1024行投影は336,955,812→331,034,398命令、87-token・8192行は1,817,733,989→1,811,592,159命令、87-token fused MLPは4,135,044,465→4,122,697,829命令。これらは当該operationのみの削減値。

新prefixから全5実行を通し、保持hidden/stateと最終hidden/logits/確率/unknown/判断は前版rope-reuse-v1とbit一致した。失敗/replayは0、前後の認証済みmodule hashを確認した。query数・通信量は同じで、命令数のみ減った。

| 実行 | query数 | handler命令数 | 前版からの削減 | Candid通信bytes | 単回時間s |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix45 | 282 | 204,541,694,216 | 0.557% | 318,918,016 | 74.222 |
| 主問題132 / suffix87 | 305 | 365,450,761,317 | 0.560% | 506,366,104 | 64.596 |
| 情報不足125 / suffix80 | 305 | 321,071,076,797 | 0.578% | 469,941,494 | 63.842 |
| 最大変更134 / suffix89 | 305 | 376,202,984,995 | 0.591% | 516,788,871 | 64.387 |
| prefixなし132 | 485 | 536,964,706,732 | 0.742% | 814,232,972 | 96.348 |

主問題は2,056,411,568命令、prefixなしは4,015,450,440命令減った。初回はprefix282＋主問題305＝587 query。最大queryは主問題4,163,845,360命令、最大変更4,300,105,752命令。終端heap最大観測は前版と同じ。handler命令数はCandid decode/encodeを除き、通信はCandid request＋replyでHTTP等を除く。時間は単回測定であり、速度向上は主張しない。

50/32 queryは未達。主問題でも5B予算で最低74 query相当、prefixなしでは108 query相当の命令が残る。既存のmaximum問題の誤判定は変わらない。

測定module `e1be06bc268f010d03385ef6138efbd9e81ab11c815e900ec9883133250de9ee`。生データ `artifacts/quantized-invariants-v2-*`、検証付き集計 `docs/quantized-invariants-v2-summary.json`、投影比較 `artifacts/quantized-invariants/probe-final/report.json`。

## 次の固定処理削減の候補

Layaの`warmup_next_inner`とpackの`Builder::push`は、準備updateでtensorを読み込み・検証してモデルを保持する。Imajevでも準備時に固定重みの仕事を済ませる候補として参照した。Layaのソース・Git・canisterは変更していない。

固定packの実容量はembedding636,692,480 bytes、INT8 dense3,575,191,552 bytes、F32 adapter487,587,840 bytes、その他2,979,328 bytes。embedding以外も約4.07GBあるため、全体をそのままheapに置けるとは断定しない。1層の重みは122,960,896〜128,310,208 bytes。通常stepのstable read計測分は主問題4,063,384,924 bytes、prefixなし4,949,614,608 bytes。decision_fastはread bytesを返さないため、この集計には含めない。これらはclient通信量とは別。

固定weightの保持と借用によるコピー削減、行分割でのLoRA A結果再利用を次の候補にする。単に重みをheapへ置いてreaderが毎回Vecへコピーする方式では、コピー費用を除けない。借用APIとメモリ上限の検証を伴う別候補として測る。この時点では未実装だった。続いて借用cacheを実装・全層検証し、[BORROWED_WEIGHTS.md](BORROWED_WEIGHTS.md)に採用結果を記録した。


行分割の同一入力をchecksum付きrequestのpayload・codec・tensor・dimensionで比較すると、準備済み主問題は重複0組、prefixなし132-tokenは重複62組（通常投影31＋gate/up31）だった。後者はLoRA A再計算93回の候補に相当する。まだcache再利用を実装しておらず、今回の削減には含めない。集計コードは`scripts/analyze_preparation_work.py`、生データは`artifacts/quantized-invariants/preparation-{main,cold}.json`。stepのread bytesと同一入力のrow分割を測る専用コードで、数値推論は行わない。

```sh
.venv/bin/python scripts/validate_terminal_readout.py --canister <専用local ID> --run-name <未使用run名> --baseline rope-reuse-v1 --fuse-mlp-norm --fuse-norm-rope --wire-codec bf16-block256-exact-v1
.venv/bin/python scripts/summarize_kernel_unroll.py --run-name <同じrun名> --baseline rope-reuse-v1
.venv/bin/python scripts/analyze_preparation_work.py --source artifacts/<同じrun名>-normal --output artifacts/<同じrun名>-preparation.json
```
