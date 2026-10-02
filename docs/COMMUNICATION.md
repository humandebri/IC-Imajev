# 分割queryの通信削減

改善前の1質問・132 tokenは、3,908 queryと5,382,428,062 Candid bytesを要した。重みuploadを含まず、同じ入力activationを出力行tileごとに再送する量が大きい。

今回、重みINT8と計算精度を維持したまま、可逆通信codecと大きい射影queryを実装した。中間状態のINT8量子化は追加しない。最終比較は `communication-improvement.json`、各queryは `efficient-results.json` に保存する。

## 全層の実測

| 項目 | 改善前 | 改善後 |
| --- | ---: | ---: |
| Candid要求＋返信 | 5,382,428,062 bytes | 2,549,073,399 bytes（52.64%減） |
| query数 | 3,908 | 3,108（20.47%減） |
| 合計handler命令数 | 3,682,485,075,234 | 3,330,114,701,691（9.57%減） |
| 最大query handler命令数 | 2,004,323,886 | 3,059,178,088 |
| handler終端heap最大観測 | 57,212,928 bytes | 87,097,344 bytes |
| クライアント所要時間 | 604.274秒 | 560.522秒 |

全32層の全token出力と、final hiddenが改善前とビット一致。readoutの全候補logit・確率も一致し、likelyを返した。公式非量子化参照に対する候補logit最大差0.04657は以前と同じ。通信形式と分割変更による追加の数値差はこの質問で0。

1 queryの処理量を増やし、最大命令数と一時heapは増えた。各queryの実測は [efficient-results.json](efficient-results.json)、比較とsource/Wasm hashは [communication-improvement.json](communication-improvement.json)。一時heapはhandler終端の観測値で、厳密な瞬間ピークではない。HTTP/CBOR/signatureは通信量に含まない。

時間は7.24%短かったが、単回かつquery cache未制御なので一般的な速度向上率とは扱わない。改善後も1質問2.55 GBあり、通信効率の課題は残る。次はquery内の演算融合と、LoRA Aの重複計算・入力再送削減を検討する。

## 変更

- 元F32の下位16 bitが0の値はBF16の上位16 bitを2 byteで送る。F32が必要な値はbitmapで区別して元の4 byteを送る。符号付きzero、非正規化数、F32 DeltaNet stateも復元時のbitを保つ。
- 要求/返信のencoded blobは2,000,000 bytes以下。可逆codecだけactivation上限を450,000から900,000要素へ拡張する。F32例外が多ければbyte上限で拒否する。
- 射影のwork上限を300Mから450M積和へ、行幅上限を768から1,536へ拡張する。実行幅はtoken数・列数・rank・byte/要素制限から計算する。通常queryだけで進める。
- SIMDの4 token境界へ分割幅を揃える。端数3 tokenをscalarで処理するより、96＋36 tokenに分けた方が効率がよく、第1層最大query命令数は4,435,931,081から3,058,787,891へ減った。各tokenの積和順は維持する。

公式resource limitsの現行ページはquery上限5 billion、query応答3 MiB、ingress payload2 MiBとしている。実装はencoded blob2 MBを維持し、CDK Candid処理を除くhandlerカウンタだけで上限保証しない。今回の実queryで完走を検証する。[公式resource limits](https://docs.internetcomputer.org/references/resource-limits/)

## 比較と再現

同じMODEL_LOCK・INT8 pack・token IDs・選択肢・calibrationで比較する。各層の出力・final hidden・判断logits・確率を改善前と比較する。圧縮したことによる新しい数値誤差と、公式MLXに対する既存のINT8/Rust誤差は別に扱う。

```sh
cargo test --workspace --offline
cargo build --release --target wasm32-unknown-unknown -p imajev-inference --offline
cargo build --release -p imajev-runtime -p imajev-client --offline
.venv/bin/python scripts/test_wire.py
.venv/bin/python scripts/test_journal.py
.venv/bin/python scripts/test_scheduler.py
# 自分で作成したlocal canisterをsealed pack維持でupgradeした後
.venv/bin/python scripts/run_full_canister.py --directory artifacts/efficient-new-run
.venv/bin/python scripts/compare_efficient.py --directory artifacts/efficient-new-run
.venv/bin/python scripts/check_full_native_wasm.py --directory artifacts/efficient-new-run
```

旧分割で独立再実験する場合は `--wire-codec f32 --row-cap 768 --work-cap 300000000 --directory artifacts/f32-new-run`。古いsession directoryへ新しいcodec/設定を混在させない。

`analyze_wire_payload.py`は旧3,907演算queryの保存要求/返信をofflineで再encodeし、全値のbit一致を検証した。Candid overheadを固定したcodec単独のサイズ推定は5,382,417,278→2,887,095,960 bytes（46.36%減）。これは実queryの時間/命令数を測った結果ではない。大きいtileによる再送削減と合わせた最終結果は全層の実測で評価する。

時間は単回のlocal実測で、query cache・warmup・ホスト負荷は制御していない。query数とCandid通信量を比較の中心に置く。途中のqueryをupdateへ移すことや、canister内へactivationを永続化する変更はしていない。

checksum破損、未対応codec、count超過、bitmap padding、余分なpayload、異なるpackを実canisterが拒否した。[wire-contracts.json](wire-contracts.json)。Python/Rustのcodec相互運用、全3907旧要求/返信のbit保存、状態破損拒否を検証した。命令上限エラーでは成功前にprogressを進めず行幅を半減し、失敗要求を別名で保存して縮小幅のpolicyをatomic保存する。再開時も同じ幅を選び、入力/model/pack/codecが違えば拒否する。縮小と再開はJournalを通す模擬上限エラーの回帰試験で確認した。全層実測では上限エラーは発生していない。

改善後の全3,107演算queryを再開時に再利用し、readout queryだけを再送した。15.460秒、実通信10,784 Candid bytesで判断logit・確率が初回と一致。[efficient-resume.json](efficient-resume.json)。このresumeは縮小policy対応を追加した現行clientで検証した。
