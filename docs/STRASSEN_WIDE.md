# Strassenの整数補正をquery内で一度だけ復元する試験

2026-10-02。`experimental-strassen-wide` featureとして、事前変換INT8重み＋疎な±256補正をqueryの冒頭でI16へ正確に復元し、各token/output tileで補正積和を繰り返さない経路を実装した。元のINT8 pack、block256 activation scale、元のrow scale、F32加算順、BF16境界は同じ。重みの再量子化ではなく、query-localの一時表現を変えた。

固定のlayer 3 Q投影8192×2560を持つ診断canisterで、同じmoduleの通常INT8と比較した。全6形状の出力はbit一致し、独立したnative scalarにも一致した。native fixtureでは元のINT8積和とのbit一致と不正metadata/correctionの拒否を確認し、feature有効のRust36 testsが通過した。

| tokens | 通常INT8命令数 | 一度復元するStrassen命令数 |
| --- | ---: | ---: |
| 1 | 72,672,608 | 736,899,762 |
| 7 | 185,305,342 | 752,521,770 |
| 8 | 186,328,308 | 755,131,936 |
| 32 | 572,432,548 | 1,423,718,487 |
| 64 | 1,086,964,478 | 2,167,431,153 |
| 87 | 1,483,402,064 | 2,677,084,723 |

87-tokenでは通常1,483,402,064→候補2,677,084,723命令で約80.5%増となり、不採用。以前の疎な補正を繰り返す候補の3,042,874,526命令よりは小さいが、以前のmoduleとの比較だけでこの変更の単独効果を断定しない。同moduleでの通常積和に勝てていないことを判断根拠にする。

87-tokenのstable readは21,004,288→38,278,204 bytes、handler終端heapは421→1,862 pages。変換済み重み全体を一時的にI16へ広げる費用があり、1-tokenでも736,899,762命令を要する。固定重みの変換を準備時へ移す余地は残るが、事前準備済みI16 packでの実測はまだ行っていない。未実装の準備費用除去を性能改善に計上しない。

これは部分base投影の試験で、LoRA/readout・全モデル精度・query数の改善を示さない。主モデルcanisterはこの候補へ変更していない。診断canister `4fbx2-kt777-77775-aaabq-cai` は試験後に実験前のmoduleへ戻し、認証済みmodule hashで復帰を確認した。Layaは変更していない。

候補Wasm `469d224c235cbe193f51c497e093a30a91baf4a6b613bda622dc6d4c4d948b8f`、復帰先 `7c366d0b7958a3034b89dff98c0fe1486239d3419cad4882324231c7268b59b5`、部分pack `0aa00943520728ed0a610e0120402466e8d9778b4fb7f6d516aec55a1f2ab9b6`。生データ `artifacts/strassen-wide/probe/report.json`、復帰検証 `artifacts/strassen-wide/diagnostic-restored/report.json`。生成物はgitignoreする。

通常版にはこのfeatureを有効にしない。検証コードは `scripts/check_prepacked_strassen.py`。候補の比較中は同じ固定入力を使い、nativeの独立したscalarと通常INT8の二つを照合した。

後続で固定係数を準備updateへ移す候補を実装・実測したが、通常INT8より重く不採用。[STRASSEN_PREPARED.md](STRASSEN_PREPARED.md)を参照。
