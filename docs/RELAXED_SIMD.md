# Relaxed SIMD 整数dotの対応を実canisterで確認

2026-10-03。通常SIMDのI16 dot＋加算を、I8/I7 dot＋I32加算の融合命令へ替えられるか調査した。量子化済みqを `low = q & 127` と `negative = q < 0` に分ければ、`q = low - 128 * negative` が成立する。両成分は0〜127に収まる。重みは元のsigned I8のまま、整数結果を再構成してから元のblock256 F32 scaleを掛ける候補であり、量子化粒度を減らす方式ではない。

[公式Rust実装の説明](https://doc.rust-lang.org/stable/core/arch/wasm32/fn.i16x8_relaxed_dot_i8x16_i7x16.html)では、第二operandの上位bitが立っていなければsigned/unsignedの違いがなく、I16途中和に飽和/overflowの違いがあり得る。本候補のlow7の２積和は-32,512〜32,258以内、negativeは0/1なので、いずれもI16範囲内である。`scripts/probe_relaxed_simd.py` はq=-127〜127と全I8重みについて分解の整数等式を検証する。完全なモデルkernelは実装していない。

[IC公式のWasm validation](https://raw.githubusercontent.com/dfinity/ic/master/rs/embedders/src/wasm_utils/validation.rs)は `wasm_relaxed_simd(false)` を設定している。実際の選択中local networkでも確認するため、`i32x4.relaxed_dot_i8x16_i7x16_add_s` のopcodeを残した151 byteの最小Wasmを生成し、専用canister `73qqu-nd777-77775-aaaiq-cai` にinstallを試みた。

結果はIC0505、`relaxed SIMD support is not enabled (at offset 0x90)`。通常query以前のvalidationで拒否され、モデル推論・性能改善は未実行。この候補は現在のlocal環境では不採用。IC設定を変更して制約を迂回しない。推論canister、Laya、モデルと精度条件は変更していない。

probe SHA256 `b0e949bed872fad039102a1246868f27f8563cd83ebde2e46a0b61179ea6e2e0`、記録 `artifacts/relaxed-simd/{probe.json,probe.wasm,install.log}`。生成物はgitignore対象。再確認時は `scripts/probe_relaxed_simd.py` でWasmを生成し、新しい診断local canisterでinstallを試す。拒否を環境全体の停止理由とはせず、対応している通常SIMDの改善を続ける。
