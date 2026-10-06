# unknown説明の短縮実験

2026-10-05。固定45-token prefixと全chat marker、State、Question、選択肢順序を維持し、最終選択肢の説明だけを変更した。

`unknown — cannot be determined from the available evidence, the premise is false, or no listed option is correct` → `unknown`。全ケースで20 token削減。tokenizerによる元の入力のdecode/encode往復一致、変更後のprefix IDs一致を検証した。固定headerには情報不足ならunknownを選ぶ指示が残る。

| ケース | suffix token | update回数 | 命令数（十億） | 命令削減 | 判断 |
|---|---:|---:|---:|---:|---|
| historical-617 | 87→67 | 7→6 | 230.817→179.216 | 22.36% | likely→likely |
| minimum | 86→66 | 7→5 | 226.673→175.071 | 22.77% | yes→yes |
| maximum | 89→69 | 7→6 | 236.028→184.370 | 21.89% | no→no |
| stake | 78→58 | 6→5 | 206.012→154.500 | 25.00% | yes→yes |
| mint | 78→58 | 6→5 | 206.011→154.502 | 25.00% | yes→yes |
| unchanged | 82→62 | 7→5 | 216.357→164.770 | 23.84% | no→no |
| insufficient | 80→60 | 6→5 | 211.172→159.633 | 24.41% | __unknown__→__unknown__ |
| maximum-not-minimum | 76→56 | 6→5 | 200.876→149.352 | 25.65% | no→no |

8ケースの判断・棄権はすべて一致。正解付き7ケースは元も短縮後も6/7。maximumの既存誤判定（正解yesに対してno）は残る。insufficientのunknown確率は0.9979967→0.9985594、主617のlikely確率は0.8299531→0.8021637。入力変更なのでlogits・確率は同一ではない。

この検証は既存の少数診断ケース・選択肢順序offset0・rotations1のみ。未知入力、選択肢並べ替え、偽前提、正解選択肢欠如、画像入力への一般化や再校正の有効性は確認していない。デフォルトpromptと学習済みadapterは変更していない。

計測は準備済みローカルupdate graphをtoken IDsから実計算した。質問依存hiddenや回答を再利用していない。main/insufficient/maximumの元入力だけは同一moduleの既存proof-v1から再利用し、他5件は今回元入力・短縮入力を実行した。prefix再登録・重み準備・status検査は推論回数に含まない。全入力、script、tokenizer、bridge、moduleのhashを保存し、終了後に不変を確認した。

実行時間は負荷を統制していない単発計測で、速度削減率は主張しない。本番subnet未計測。各呼び出しの命令数は40B以内。

主ケースは全132→112 token、実計算suffix87→67。約179Bのhandler命令は5B/queryの予算約36回分に相当する。この値はupdateからの予算換算で、短縮入力のquery分割・通信を実測した回数ではない。10 queryの50B予算には依然大きく、文字の短縮だけでは届かない。

固定45-token prefixは各層のconv/KV等としてすでにキャッシュされている。そこを文字数だけ削っても、現在の毎回のsuffix演算は減らない。学習した短いベクトルprefixへの圧縮は別の学習・評価が必要で、今回未実施。vendored scoring.pyにもcompact形式のbare unknownがあるが、header/state形式も変わり、学習layoutとの対応が必要。今回の一か所短縮と区別する。

証跡：`artifacts/short-unknown/summary.json`、`pilot-v1/report.json`、`controlled-v1/report.json`、各入力・全呼び出しJSON。専用実験canister `6eydd-o3777-77775-aaama-cai`、module `88507a95ea9e6e2111843c5139025d8ce2b3b8e21d9a9957beca4f6f74e103be`。採用canisterは変更なし。

再実行（同一module、重み・prefix準備済みが必要）：

```sh
.venv/bin/python scripts/evaluate_short_unknown.py \
  --directory artifacts/short-unknown/new-run \
  --indices 0,9,11,13,15,17,19,21 --variants original,bare
```
