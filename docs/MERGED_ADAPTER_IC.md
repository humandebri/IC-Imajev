# 統合LoRAのローカルIC測定

追加の **32 query最適化経路の測定** は [MERGED_ADAPTER_QUERY32.md](MERGED_ADAPTER_QUERY32.md)。3入力で命令数が13.21〜13.52%減った。以下は132 tokenを共通の細分化経路で処理した当初の測定記録である。追加実験では同じ専用canisterを測定用Wasmへupgradeしたため、現在のWasmは当初のhashと異なる。旧条件を再実行する場合は保存済みWasmへのupgradeと重みcacheの再準備が必要。pack自体はupgradeで保持している。

2026-10-06。保存した統合INT8 packと、元のINT8 base＋F32 LoRAを、同じWasm・同じ入力・同じ分割設定で比較する実験。ローカルIC `http://localhost:8001/` の専用canisterを使う。

未統合 `2hlkr-gl777-77775-aaaxa-cai`、統合 `2akmf-lt777-77775-aaaxq-cai`。両方に `artifacts/prepared-f32/candidate.wasm`（SHA256 `86d93cda02bb5bb6c6e4d17bea38b3315206aace76cdc94d3a4941a6edbec0df`）を初回installし、Wasm memory limitを4 GiBにした。

比較入力は `artifacts/merged-adapter-v2/before.json` の最初の質問 `historical-617`。132 tokenをprefix cacheなしで全32層通し、通常queryだけで専用readout・校正・判断まで実行する。参照hidden/stateは入力しない。3回ずつ、before→after、after→before、before→afterの順に実行する。

`scripts/measure_merged_adapter_ic.py`が実行と集計を行う。各回のjournalは新規に作り、要求のstep番号を100000・200000・300000から開始する。同じ数値入力でも要求bytesが繰り返し間で異なるため、前回のquery応答やjournalのreplayで時間を短縮しない。ノードのquery cache設定自体は変更していない。

共通設定はINT8 block256射影、lossless BF16 wire、row cap3072、work cap3B、add/norm・norm/RoPE・Delta stageの融合、Delta 8 heads、attention 4 heads、terminal state破棄。LoRA専用のMLP/attention全体の融合やterminal MLP専用経路は両方で使わない。最新のquery32最適化経路との速度比較ではない。

アップロードしたpackはIC内でも全体SHA256を照合した。固定重み準備は未統合721 tensor・4,065,416,192 bytes、統合321 tensor・3,577,828,352 bytes。準備updateのhandler命令数は15,048,702,304と9,466,282,348。転送・hash・重み準備は推論の命令数・時間に含めない。準備を同時進行したため、準備のwall timeは単独実行の速度比較には使わない。

転送後のhashを進める補助script `finish_merged_pack_hash.py`は、既存の同期的な`hash_pack` owner updateを複数要求する。IC内ではstored cursorに対して直列に処理され、元uploaderとともにcursorを進める。追加updateの記録は `*-hash/report.json`。推論Wasmや数値演算を変更しない。

推論の命令数は既存handlerの`performance_counter(0)`の合計で、binary frame処理とdecision handlerを含み、CDKのCandid decode/encodeを除く。時間はrunnerの通信・bridge・シリアライズ・journal書き込み・初回module照合を含むローカルwall time。最終module照合と重み準備は時間の外。mainnetのレイテンシを表す測定ではない。

結果は `artifacts/merged-adapter-ic-v1/comparison.json`、各回は `{before,after}-{0,1,2}/report.json`、module/cache/sourceの前後検証は `experiment.json` に保存する。

## 全32層の結果

| 項目 | 未統合 | 統合 |
| --- | ---: | ---: |
| 1回のhandler命令数（3回すべて同値） | 602,002,923,942 | 529,050,777,852 |
| query数 | 772 | 772 |
| 通信込みwall time中央値 | 178.275秒 | 186.222秒 |
| wall timeの全3値 | 178.275 / 216.761 / 145.729秒 | 173.673 / 186.222 / 246.583秒 |
| Candid要求＋応答bytes | 1,354,732,818 | 1,354,615,582 |
| query終端heapの最大観測bytes | 4,114,939,904 | 3,596,222,464 |
| 判断（各3回） | likely | likely |
| likelyの校正済み確率 | 0.8299530745 | 0.7952746749 |

命令数は72,952,146,090（12.1182%）減った。INT8 baseの射影を残し、元のF32 LoRA A/B射影と、行分割に伴うLoRA Aの再計算を省いた。積和数の約3.3%削減とWasm命令数の削減は、数値精度・カーネル・分割処理が異なるため同じ比率にはならない。

実行時間の中央値は4.46%長く、速度改善は確認できなかった。時間の範囲は未統合145.729〜216.761秒、統合173.673〜246.583秒と大きく重なる。時間にはquery通信・bridge・シリアライズ・journal I/Oも含み、この3回から速度改善率を断定しない。

1問・132 tokenの判断は双方likelyで、各variantの3回のraw logitsも同一だった。校正済み確率の最大差は0.0346783996（3.47ポイント）。一般精度や校正精度の検証範囲はホスト23条件の記録を参照。

全6実行で全32層を通り、失敗・journal replayは0。認証済みmodule hash、pack hash、準備済みtensor名と容量、クライアント実装hashの前後一致も確認した。heapはquery終端のWasm page数の観測で、処理途中の厳密なピークではない。採用canisterには統合重みを適用していない。

```sh
# この測定専用canisterの記録を集計し直す。
.venv/bin/python scripts/measure_merged_adapter_ic.py --summary-only
```
