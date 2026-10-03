# 固定INT8 weight scaleの一度だけの準備

2026-10-02。全5実行で命令数が増えたため不採用。以下は試験した候補の説明であり、運用ソースは前版へ戻した。候補の検証済みソースは`artifacts/prepared-scales/validated-source.zip`、前版からの差分は`artifacts/prepared-scales/candidate.patch`に保存した。

[直接INT8状態展開](DIRECT_PROJECTION.md)に続き、fixed weightのrow scaleについて毎queryのbyte→F32復元、配列確保、正の有限値の再走査を準備updateへ移した。元のF32 bitsを保持し、量子化・積和・BF16境界・adapter/readout/calibrationは変えない。

## 不変の型とcache配置

`PreparedScales`は正の有限値を検証して作るprivate fieldの不変型で、Rcの部分viewを返す。外部から改変できない。INT8 cacheをweightsのBox<byte>とscaleのtyped arrayに分け、元のraw scale tailは除去する。272 tensorのscaleは1,328,640個、5,314,560 bytesで、二重保持しない。logical cache bytesとsealed packのoffset/bytesは従来のまま。

scaleのみのaligned rangeではprepared viewを借用してruntimeの`project_prepared`に渡す。同じ呼出しでbyte復元や有限性の再走査をしない。行数・weights形状・値数・work boundsは引き続き検査する。旧byte readerとstable fallbackでは従来の正の有限値検証を行い、未準備値をtrustedと扱わない。experimental token-scale経路は旧scale検査を維持する。

元packの全byte範囲を読む互換性も維持し、unaligned scale rangeはbyte view、weights/scale境界を跨ぐrangeは結合したowned byte列を返す。通常の整数projectionは2領域を別々に読むため、この結合コピーを行わない。cache clear後もRc viewの寿命を保つ。推論状態はclient-held、推論は通常query、準備だけupdate。

## 検証

nativeのfeature有効runtime47＋cache3＋compile-fail doctest3、通常feature runtime45＋cache3＋doctest2が通過。最小正subnormalとF32最大値のbit保持、非ゼロ行、invalid scale、shape不足、unaligned/cross-boundary範囲、cache clear後の寿命を検査した。prepared scaleの`as_ref`をpanicにするreaderでも実projectionが従来とbit一致し、typed経路がbyteを復元し直していないことを確認した。

専用部分canister `4qggx-l3777-77775-aaaca-cai`で、元QKV8192×2560＋rank64、23,756,800-byte partial packを用い、cold/warm/cleared×7入力長×4query=84通常queryを実行。独立旧scalarの出力とbit一致、compact frame再encodeもbyte一致、不正状態4件は全て拒否。module `b6e94a1e1a7cda27f4de7f2d4066c47132f70a59ab9c3af94277dd5f6532dca7`を前後確認した。

前のdirect decode版に対するwarm capture/reuse 2queryの差：

| tokens | 前版命令 | 本版命令 | 減った命令（負数は増加） | 削減率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 60,219,072 | 59,730,363 | +488,709 | +0.8116% |
| 7 | 249,522,755 | 249,026,644 | +496,111 | +0.1988% |
| 8 | 174,007,286 | 173,530,947 | +476,339 | +0.2737% |
| 32 | 636,296,505 | 635,417,138 | +879,367 | +0.1382% |
| 64 | 1,253,779,677 | 1,254,530,589 | -750,912 | -0.0599% |
| 87 | 1,813,064,188 | 1,812,624,283 | +439,905 | +0.0243% |
| 132 | 2,633,596,413 | 2,631,454,104 | +2,142,309 | +0.0813% |

132-tokenはさらに2,142,309命令減、64-tokenは750,912命令増。全形状で勝ったとは扱わず、全層で採用を判断する。通信bytesは全形状で前版と同じ。生結果は`artifacts/prepared-scales/probe/report.json`、前版は`artifacts/direct-projection/probe/report.json`。部分packの試験を全モデルquery数・判断精度の証拠としない。

整数dotのWasmを解析すると、7種kernelのlocals/dot/各opcode件数は前の全層moduleと一致した。この静的集計だけで動的命令数や他のF32ループが同じとは扱わない。audit生結果は`artifacts/prepared-scales/{before,after}-dot-code.json`。Wasm/log/reportはgitignore対象。

## 全層比較：不採用

全5実行で従来`direct-projection-v1`の保持hidden/state、final hidden、logits、確率、unknown、型付き判断とbit一致。prefixは全32層hidden/72 state arrays、それ以外は前31層全hiddenと最終層の必要last-token hidden/48 state arraysを比較した。失敗/replayは全て0。実行前後の認証module・model/pack・cache・検証ソースhashも一致した。元の最大変更gold=yes/出力noも残り、精度改善とは扱わない。

| 実行 | query | 候補handler命令 | 前版から増えた命令 | 増加率 | Candid bytes | 単回秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix | 282 | 193,854,145,946 | +134,326,737 | 0.0693% | 318,918,016 | 27.7771 |
| 617 | 305 | 354,841,435,804 | +136,656,854 | 0.0385% | 506,366,104 | 46.0523 |
| insufficient | 305 | 310,342,269,849 | +160,214,041 | 0.0517% | 469,941,494 | 42.3032 |
| maximum | 305 | 365,607,807,102 | +150,187,179 | 0.0411% | 516,788,871 | 50.9277 |
| normal | 485 | 522,896,887,480 | +246,022,045 | 0.0471% | 816,643,811 | 90.3625 |

主問題のMLP側は12,757,221命令減ったが、Delta stageは87,712,439、LoRA projectionは32,931,233命令増え、全体では136,656,854命令増。通信とquery数は同じ。省いたscale処理だけで全体の速度を判断できない。この比較だけでcacheの配置・allocator・非dot kernelのcodegenのどれが主因かは確定しない。整数dotの静的opcode集計が同じでも、全体の動的費用は同じではなかった。

候補の終端heap最大4,115,726,336 bytes、前版4,115,005,440 bytes。論理cache量が同じでも実際のpage観測は違う。瞬間peakは未測定。準備は721 update、15,120,013,614 handler命令、247.0849秒。前版準備15,027,997,708命令より92,015,906命令（0.6123%）増。準備Candid要求47,473／返信14,915,249 bytes。pack status1、cache status2、認証module read2は別。単回時間はquery cache未制御で速度改善の根拠としない。

生結果は`artifacts/prepared-scales-v1-*`、`docs/prepared-scales-v1-summary.json`、cache/source照合`artifacts/prepared-scales-v1-cache-checks/report.json`。候補B6 moduleとソースarchiveは生成物としてgitignoreに保存し、運用ソースからは外した。前の検証済みソース31 hashと完全一致することを確認してから、主canisterを`e961dda8d6b2f2f71448ba1b21512b72f231f440ef3b7b41951ff3ae0209c423`へ戻した。復元cacheの記録は`artifacts/prepared-scales/restored-preparation/report.json`。

Candid通信量はHTTP/CBOR/signatureを含まず、handler counterはCDK Candid decode/encodeを含まない。今回の不採用によって採用版の305 query、初回prefix込み587 query、prefixなし485 query、50/32未達は変わらない。

## F32状態を可逆圧縮する探索

実測prefix cacheの24 Delta states（各32×128×128 F32、計50,331,648 bytes）をホストだけで調べた。zlib level1をraw bytes、byte-plane、列方向bit XOR＋byte-planeの3種類に適用し、全て復元後の全bit一致を検査した。圧縮合計はraw47,406,448、byte-plane43,833,576、XOR＋byte-plane44,252,213 bytes。最良だったbyte-planeは平均1,826,399 bytes/state、範囲1,811,713〜1,834,991。

87-token×hidden2560を全BF16の最小445,440 bytesと仮定しても、stateとの合計は平均2,271,839 bytesとなり、既存2,000,000-byte frameに収まらない。header・gates・他の必要データを加える前のサイズであり、canisterで全stageを1queryへ融合できたという結果ではない。今回の3形式の測定で、他の全ての可逆圧縮を否定しない。

canister codecやクライアント本番経路には未接続。圧縮・展開のWasm命令数も未測定で、query削減や速度向上として計上しない。再現コード`scripts/explore_exact_state_compression.py`、生記録`artifacts/prepared-scales/exact-state-compression.json`は元fileとcacheのhashを保存した。新たな量子化はしない。

```sh
.venv/bin/python scripts/explore_exact_state_compression.py \
  --source artifacts/direct-projection-v1-prefix/queries \
  --output artifacts/new-exact-state-compression.json
```

復元完了後、前の全層検証記録とWasm hash・pack hash・cache4,065,416,192 bytes・全721 tensor名・運用ソース31 hashを再照合し一致した。復元記録は`artifacts/prepared-scales/restoration.json`。復元準備は721 update、15,027,997,708 handler命令、226.0021秒。候補比較の費用と別に記録する。
