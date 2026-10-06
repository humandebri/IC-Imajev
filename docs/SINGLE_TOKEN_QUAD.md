# 1 tokenのINT8基底投影を直接計算する

2026-10-05。最後の1 token用MLPは、従来2 tokenのStrassen演算で片方をゼロ埋めしていた。`experimental-single-quad`では、既存の4平面INT8配置を直接SIMDで読み、ゼロ側の演算と7種類の入力変換を省く。4出力の境界に合う1 token投影だけを切り替え、ずれた範囲と複数tokenには従来演算を使う。追加の固定重みコピーはない。

整数内積はI32で厳密に求める。256要素ごとのactivation scale、weight scaleの乗算とF32累積順序は従来と同じ。LoRA、専用readout、tokenizer、calibration、重み量子化、中間状態は変更しない。

## 実測

最初の小ループ版は終端profileで4,533,194,094命令となり、従来の4,432,537,049から2.27%増加した。採用しない。内側32反復を定数オフセットで展開し、4平面のポインタを共通化した第2版は4,344,580,462命令、1.9843%減。両版とも保存済み要求への返信payloadはbit一致した。profileは計測用queryで、通常推論のquery数には含めない。

情報不足は50query・214,719,444,971命令・139,394,364 bytes、最大入力は標準経路62query・232,780,394,973命令・125,231,193 bytesで完走した。

主問題の通常推論の終端queryは4,432,848,988→4,344,892,554命令。全体は234,609,007,209→234,521,052,335命令（87,954,874命令、0.03749%減）。132 token＝prefix45＋suffix87、rotations=1、50通常query、Candid要求＋返信149,353,220 bytes。最大queryは4,918,963,366命令、最大観測heapは4,144,037,888 bytes。時間37.221秒は単発測定で、速度の一般的改善率を表さない。

削減は最後の1 token処理に限られるため全体の効果は小さい。32queryは未達。複数tokenの基底投影が引き続き主要な計算負荷で、従来最重queryも今回の対象外。メモリ・通信量・query数の削減は主張しない。

## 検証と実行

`cargo test --offline -p imajev-runtime --features experimental-single-quad`で54単体＋既存integration1＋compile-fail2が通過。追加の`--test single_quad`も通過し、256/512/2560/9216列、INT8端値、ゼロ、複数K256 block、4行ずれ、2行ずれのfallback、最終行境界を元INT8投影とbit比較する。実Wasmは標準6条件と50query経路3条件で返却hidden/state/最終hidden/logits/校正確率/判断を採用INT8基準と比較する。compact tailが省略するlayer30 hiddenは比較対象に含めない。型安全な判断のvalidatorも通過。公式BF16との一致や一般精度改善を意味しない。既知の最大変更問題の誤判断も残る。

ビルドは`build_full_prefix_candidate.py`の従来採用引数に`--single-quad`を追加する。raw Wasmの4つの既存ABI stubは従来同一のWATでpatchしてから使う。ビルド・patch・測定の証跡は以下（生成物はgitignore）。

- `artifacts/single-quad/build-v2`: configuration、feature一覧、source ZIP/hash、raw/patched Wasm、4 patch報告、使用WAT。module SHA256 `bdd8ced5ac7afb938168c7d88bafb03a42570a12fafe9470a8723c8e17786087`。
- `artifacts/single-quad/profile-v1` / `profile-v2`: 不採用版と第2版の固定要求、bit比較、演算別命令数、source ZIP。
- `artifacts/single-quad/preparation-v2`: 固定重みのみ721 update、4,065,416,192 bytes。質問依存推論は通常queryのみ。
- `artifacts/single-quad/full-proof-v2`: 新module用prefix、packet、標準6条件、参照hashと検証済みsource ZIP。
- `artifacts/single-quad/tail-proof-v2`: 50query経路3条件の完全実行、queryごとの要求・返信・命令数、参照hashとsource ZIP。

専用実験canister `6eydd-o3777-77775-aaama-cai`に第2版を適用する。保護対象`4caro-hl777-77775-aaaba-cai`のmoduleは`36c04a57e9501eb9d32163c92d7725171191fa46a45a880ad03465993fceed73`のまま確認済み。Layaのソース・Git状態・canisterは変更しない。mainnet、push、PRは行わない。
