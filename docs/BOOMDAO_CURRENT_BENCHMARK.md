# BoomDAO 現行モデルのローカル実測

保存済み提案617・620・653の質問、選択肢、状態要約を固定し、現行の短縮promptで各3回、計9回のcanister推論を実行した。提案本文全体の評価ではない。

| 提案 | tokens / suffix | 回答 | likely確率 | query | handler命令 | ローカル時間中央値 |
|---|---:|---|---:|---:|---:|---:|
| 617 | 94 / 67 | likely | 81.0% | 39 | 179.599 billion | 28.69秒 |
| 620 | 86 / 59 | likely | 52.9% | 39 | 159.075 billion | 25.54秒 |
| 653 | 84 / 57 | likely | 58.2% | 39 | 153.923 billion | 24.70秒 |

全9回で応答のreplay、fallbackとも0。各提案の3回で回答・確率・命令数は一致した。最大の単一query handler命令数は4,936,792,017。handler命令数はCDKのCandid処理を含まないが、全呼び出しは実際に成功した。prefix準備は計測対象外。ローカル時間はプロセス起動を除き、prefix読み込み・転送を含む。

620はpossible 43.8%、likely 52.9%。二倍の最低投票ロック期間でもlikelyを選ぶため、変更倍率によるリスク段階の区別はこの3件だけでは確認できない。gold未確定のリスク質問であり、正解率は算出しない。

保存済みLaya INT8の回答は617 unlikely、620 unlikely、653 likely。現在のImajevはすべてlikely。別モデル・tokenizer・prompt・readoutの比較であり、短縮promptの精度改善を意味しない。Layaの保存測定は4〜5queryだが、現行Imajevは39query。

617の旧132token測定は50query・234.521 billion命令。現行94tokenは39query・179.599 billion命令で、queryは22%、命令は23.4%削減。

実行: `.venv/bin/python scripts/benchmark_boomdao_current.py`。完了済みディレクトリは再利用されるので、新たに実測する場合は出力ディレクトリを変更する。

結果: `artifacts/boomdao-current-v1/report.json`。各回の要求・応答とsource/module/input hashesを保存。入力は`artifacts/text-short-v2/inputs.json`先頭3件、質問・状態要約は`benchmarks/laya_baseline.json`と一致を検証。
