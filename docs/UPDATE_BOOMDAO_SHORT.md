# 短縮後のBoomDAO update推論実測

2026-10-05。保存済みBoomDAO提案617・620・653を短縮promptで各3回、合計9回推論した。queryと同じtoken IDs、重み、INT8基底/F32 adapter/readout、BF16境界、calibrationを使う。prefixは27token、質問依存のsuffixのみをupdate開始時に渡す。

| 提案 | 全tokens / suffix | query | update | handler命令中央値 | Candid通信中央値 | update呼び出し時間中央値 |
|---|---:|---:|---:|---:|---:|---:|
| 617 | 94 / 67 | 39 | 6 | 177.911 billion | 29,635 bytes | 35.892秒 |
| 620 | 86 / 59 | 39 | 5 | 157.380 billion | 26,790 bytes | 32.330秒 |
| 653 | 84 / 57 | 39 | 5 | 152.269 billion | 26,782 bytes | 29.504秒 |

各提案の3回でupdate回数は一致。全9回で最終norm、logits、校正確率、unknown確率、型付き判断がqueryとビット一致した。全32層のconv/KV状態hashと、queryが保存する31層のhidden hashも一致。compact queryはlayer30 hiddenを返さないため、同層hiddenだけは比較していない。回答は全3件ともlikely。推論失敗・再送なし。回答が同じであり、精度改善を示す測定ではない。

617の旧132token測定7updateに対し、今回94tokenは6update。旧handler命令230.817 billionから177.911 billionへ約22.9%減った。query39回との違いは、updateがcanister内に中間状態を保持し、一度の呼び出しで複数層を計算できること。計算自体は省略しない。

1updateの最大handler命令は35,667,741,335で、全48回の推論updateは実際に成功した。34 billionを超えた段の完了時に区切る既存schedulerを維持。handler命令はCDK Candid処理を含まず、Candid通信量はHTTP/CBOR/署名を含まない。各提案の時間中央値はquery28.686/25.542/24.702秒より長い。負荷・合意環境を統制した速度比較ではなく、少ない呼び出しが速さを保証する結果ではない。

初回準備は重み721update、固定prefix32updateで別集計。重み・prefixはupgradeで消えるheap cache。各推論にこの準備を繰り返す必要はない。検証用のprefix不正入力3updateと完了済み続行・空token拒否2updateも推論回数に含めない。準備時status/module照会も別。

update実装の固定prefix45を可変長1〜132へ変更した。各層のKV形状またはNPF1 headerから長さを取得し、32層の長さ一致を検査してsessionへ固定する。27/45tokenを混在させた登録、空prefix、二重登録を実canisterで拒否した。APIの引数・返答形式と既存のowner検査、進行中session保護は維持。

module `2436598c8fd2b900f42b94bae7f04c3e3dec24d0c5189cadcd62864e71f660e7`。直前query実測moduleのwrapperに可変prefix長だけを加え、同じ凍結runtime依存関係と4つのWAT kernelから構築。Wasm検証を通過。`artifacts/update-short-v1/build/provenance.json`にソース・compiler・依存関係、各patch JSONにkernel hashを保存。

実験canister `6eydd-o3777-77775-aaama-cai`だけを使用。採用canisterは変更していない。測定後、既存queryベンチのmodule `6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931`と重みcacheを復元する。復元は計測に含めない。

結果: `artifacts/update-short-v1/summary.json`、`proof/report.json`、各推論updateのJSON。入力・query参照hashを実行中と終了時に照合した。重み準備は`weights/report.json`、query環境復元は`query-restored-weights/report.json`。

再実行には実験moduleのupgradeと重み準備が必要。準備後、未登録prefixの場合は次を実行する（完了済み出力ディレクトリは使わない）。

```sh
.venv/bin/python scripts/measure_update_boomdao_short.py \
  --wasm artifacts/update-short-v1/build/full.wasm \
  --directory artifacts/update-short-v1/proof-new --prepare-prefix
```

ローカル実測であり、複数ノードの合意・本番時間・価格は未測定。既存の単一owner sessionの実験APIで、同時利用者対応やupgradeをまたぐ途中推論再開は含まない。
