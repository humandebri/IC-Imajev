2026-10-08更新：この文書は当時の実験・測定記録。旧column16/column32・balanced44・dot-scale・whole-token scaleの実装とfeatureは削除済みで、以下の旧featureを指定するコマンドは現行ソースでは使えない。現行実装と検証は[INDEPENDENT_RUNTIME.md](runtime/INDEPENDENT_RUNTIME.md)を参照。

# 整数dotの結果取り出しとDelta状態の書き込み削減

2026-10-02。入力tokens、W8/A8 block256、F32 LoRA、BF16丸め境界、readout、calibrationを維持する候補。モデルの公表値から改善を推定せず、専用local canisterで比較する。

`int8_kernel.rs` のdot結果取り出しを定数添字で展開した。最大64×8タイルのコンパイル後Wasmでは、dot命令数16,384は同じだが、`memory.fill` が64→0、`v128.load` が2,568→2,048、`v128.store` が642→254となった。静的命令数の減少だけを実行命令削減とは扱わない。実データの10射影shapeでは全出力bitが一致し、全shapeの命令数が減少した。87-token gate/upは4,728,057,501→4,180,627,101、downは2,312,099,961→2,038,384,761命令。

`delta_simd.rs` はdecayを適用した中間stateのstoreを省き、更新passで同じF32積を再計算する。積・加算の順序は維持する。87-tokenの14-head実入力は1,143,780,860→1,133,805,440命令でbit一致。削減は約0.87%であり、整数射影ほど大きくない。

全層検証は新しいprefixを準備し、主問題・情報不足・最大lock変更・cacheなしの通常queryを実行する。中間hidden、保持state、raw logits、確率、unknown、型安全出力を比較する。既存最大lock問題の誤判定を改善したとは主張しない。回数を下げるscheduler変更はこの候補には含まれない。

再現手順:

```sh
cargo build --offline --release --locked --target wasm32-unknown-unknown -p imajev-inference
icp canister install 4caro-hl777-77775-aaaba-cai --mode upgrade --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm --network local --identity imajev-local
.venv/bin/python scripts/probe_projection_shapes.py --canister 4caro-hl777-77775-aaaba-cai --source-directory artifacts/terminal-v3-617 --output artifacts/reduction-unroll/reprobe
.venv/bin/python scripts/validate_terminal_readout.py --canister 4caro-hl777-77775-aaaba-cai --run-name kernel-unroll-v1 --baseline terminal-v3
.venv/bin/python scripts/summarize_kernel_unroll.py
```

新しい検証には未使用の`--run-name`を指定する。同名directoryの再実行を新規実測と数えない。個別演算の生測定は `artifacts/reduction-unroll/probe.log`、`artifacts/delta-store-{before,after}/report.json`。コンパイル後Wasmの演算調査器は `scripts/wasm_audit/`、生成物は `artifacts/wasm-audit-target/`。測定JSON・Wasm・重み・実行ログはGit管理から除外する。

## 全層実測結果

5実行とも失敗query 0、replay 0。prefixは32層hiddenと72 state、判断実行は保持31層hidden・最終行・48 state・最終hidden・raw logits・確率・unknown・ラベルが前版とbit一致。

| 条件 | query | 命令（前→後） | 削減率 | Candid bytes | 単回秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix 45 tokens準備 | 330 | 248,234,620,521→231,258,059,241 | 6.84% | 360,695,462 | 42.178 |
| 主問題132 tokens / suffix87 | 352 | 450,660,351,980→409,789,765,100 | 9.07% | 579,259,841 | 49.071 |
| 情報不足125 / suffix80 | 352 | 400,002,358,771→362,649,426,931 | 9.34% | 537,098,941 | 66.748 |
| 最大変更134 / suffix89 | 383 | 471,751,558,567→429,404,119,847 | 8.98% | 606,349,640 | 61.894 |
| cacheなし132 | 501 | 665,196,829,946→600,673,550,586 | 9.70% | 885,497,502 | 86.693 |

時間は単回で、前版より増えたケースもある。再現性ある時間短縮は未証明。命令はhandler counterでCandid decode/encodeを含まず、通信はCandid request+replyでHTTP等を含まない。主問題の最大queryは4,221,755,478命令、handler末尾heap最大観測43,450,368 bytes。

Wasm SHA256 `a1d90e53a42336741615e9e804df5a57c9d20d71f8f3734407d3fd07a55d81d3`。prefix系4実行は実行前後に認証済みmodule hashを確認。cacheなし実行は記録器がまだ前後確認に対応しておらず、全5実行後に同hashを確認した。今後の通常実行記録器にも前後確認を追加した。生測定 `artifacts/kernel-unroll-v1-*/report.json`、比較 `docs/kernel-unroll-v1-*-results.json`、検証付き集計 `scripts/summarize_kernel_unroll.py`。

主問題の命令の76.68%は整数射影・MLP群、15.18%はDeltaStage。4098億命令のまま詰めても5B/queryで理想的には最低82回必要。50回の2500億予算へさらに38.99%以上、32回の1600億予算へさらに60.96%以上削減が必要。query依存関係・2MB通信上限による制約はこれに加わる。目標回数は未達で、探索を継続する。

## 16列幅の再試行と89-token統合

定数添字化後に64×16タイルも再試行した。実shape10種類すべてbit一致したが、全shapeで命令数が増えた。87-token gate/upは4,180,627,101→4,192,094,109、1-token終端MLPは723,054,940→793,425,820。8列版に復帰した。生測定は `artifacts/tile16-unroll/probe/`。

8列改良版の89-token gate/upを全9216列で実行すると、従来2 query /4,522,485,155→1 query /4,346,355,691命令、全出力bit一致。`scripts/check_wide_mlp.py` が保存済み2分割出力を再結合して比較する。`--wide-mlp` の4.5G MAC予算の適用を88→89 tokensへ延長し、90以上は従来4G予算の分割を維持する。命令上限エラー時の幅縮小は維持する。新graph hashに対応したprefix準備から検証し、旧cacheのhashを改変しない。

疎な活性値のスキップも調査した。主問題の31 MLP down入力で、INT8値がゼロの比率は中央値5.88%。8要素groupが全てゼロになる比率の中央値は0、最大約0.008%。このgroupを飛ばす枝分かれは削減余地が乏しく、カーネルには追加していない。生統計は `artifacts/reduction-unroll/activation-sparsity.json`。

89-token統合後の全5実行 (`kernel-wide89-v2`) も、失敗query/replay 0、保持hidden/state/判断がbit一致。前後認証済みmodule hashを全実行で確認した。最大変更は383→352 query、471,751,558,567→423,754,131,594命令（10.17%減）、Candid通信606,349,640→591,305,752 bytes（約2.48%減）、最大query4,388,628,713命令。単回66.086秒。主問題・情報不足・cacheなし・prefixの回数/命令/通信は上表と同じ。最新graphとWasmのcacheを使う実行directoryは `artifacts/kernel-wide89-v2-*`、集計は `docs/kernel-wide89-v2-summary.json`。最大変更の既存誤判定は残る。生成物はGit管理しない。

## token全体のactivation scale候補

`host_integer_reference.py --activation-scale token` を追加した。デフォルトのblock256は維持する。候補は元のINT8 packとF32 adapter/readoutを使い、各tokenの全列から共通scaleを求める。各256列の正確な整数dotをI32で合計してから一度だけF32 scaleを適用する。中間stateの通信量子化は追加しない。

ホスト23条件（選択肢順を含む）で現行block256版とラベル23/23一致。確率差は最大0.108998（約10.9ポイント）で、確率・校正の同等性はない。最大lock変更の見逃しと620の順序依存は残る。これは公式nonlinear graphを使うホスト評価であり、Wasm全層精度や一般精度の証明ではない。`artifacts/host-token-scale-orders.json` と `artifacts/direct-dot/host-token-scale-comparison.json` に記録。別Rustカーネルを実装し、性能とnative/Wasm差を測定する段階。既定graphの演算方式はblock256のまま。

## 直後の結果取り出しと短いtokenタイル

87-tokenの射影では結果を計算直後に取り出す候補がさらに約0.8〜1.3%改善。1-tokenでは小幅に増えたため、R<=8は前の取り出し方式を使う。1〜7 tokensを8行までpaddingして計算する経路は、4/2/1行タイルを使う。quantization bufferのpaddingと検証は維持し、余った演算だけを省く。1-tokenの終端MLPは723,054,940→408,890,024命令（約43.45%減）、全出力bit一致。1〜134 tokens・非ゼロ出力row・raw整数積和/LoRAを含む126条件でnative scalarと実Wasmがbit一致した。

`kernel-direct-v3` は新prefixを含む全5実行で失敗/replay 0、保持hidden/state/最終logits/確率/unknown/ラベルbit一致、前後module hash一致。

| 条件 | query | handler命令 | terminal-v3比削減 | Candid bytes | 単回秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix準備45 tokens | 330 | 229,449,072,201 | 7.57% | 360,695,462 | 83.178 |
| 主問題132 / suffix87 | 352 | 405,337,693,132 | 10.06% | 579,259,841 | 79.458 |
| 情報不足125 / suffix80 | 352 | 358,176,808,659 | 10.46% | 537,098,941 | 51.322 |
| 最大変更134 / suffix89 | 352 | 418,467,276,522 | 11.29% | 591,305,752 | 57.685 |
| cacheなし132 | 501 | 593,351,422,154 | 10.80% | 885,497,502 | 80.619 |

Wasm `0cb945804dcfaebc4da7b73376b1e517fa066fbb6c31ffd1f0175e1238fe1366`。prefix準備と主問題の単回時間は同時にホスト候補評価を実行した影響も受けるため、前版との速度比較には用いない。命令数と出力の比較を採用根拠とする。目標回数は未達、最大変更の誤判定も残る。個別入力・候補・実測は `artifacts/direct-dot/`、全層は `artifacts/kernel-direct-v3-*`、検証集計は `docs/kernel-direct-v3-summary.json`。

token-scaleカーネルの最初の32列unroll候補は8実shapeすべてnative scalarとbit一致したが、7/8 shapeで命令増加。87-token gate/upは4,126,220,453→4,167,890,706命令。downは2,011,159,805→2,009,982,568命令と約0.06%だけ改善。scaleの回数を減らしても内側の整数accumulator更新が増えるため、理論上のscale削減だけで採用しない。256列unrollの候補を別途測定する。実験用opは`*_token`の別名で、既定graphのblock256と混在させない。

256列unrollにしたtoken-scale候補は87-tokenの6形状で約3.4〜6.0%の命令削減。gate/upは4,126,220,453→3,928,961,298、downは2,011,159,805→1,890,328,168。1-tokenの2形状は小幅に増加。8形状すべて同じtoken-scale方式のnative scalarとbit一致し、認証済みmodule hashを前後に確認した。Wasm全層token-scale推論はまだ接続していない。ホストのラベル一致だけでcanisterの精度維持を確定しない。生測定 `artifacts/token-scale/probe256/`。

実験APIはRust feature `experimental-token-scale` を明示した場合にだけ有効にする。既定ビルドはblock256。実験用ビルドは次の通り（変更したWasmにはprefixを新規準備する）。

```sh
cargo build --offline --release --locked -p imajev-runtime --bin primitive --features experimental-token-scale
cargo build --offline --release --locked --target wasm32-unknown-unknown -p imajev-inference --features imajev-runtime/experimental-token-scale
# 選択した専用local canisterへupgrade後、実験用の出力directoryで実行する。
.venv/bin/python scripts/probe_projection_shapes.py --canister 4caro-hl777-77775-aaaba-cai --source-directory artifacts/kernel-direct-v3-617 --output artifacts/token-scale/new-probe --token-scale
```

重みと中間stateの量子化はこの実験で変更していない。activation scaleの変更は追加演算誤差として評価する。既定graphへの全面採用は保留。

## 次の候補：変換を事前準備するStrassen

従来の不採用Strassenは重み変換とI16展開をquery内で行っていた。`prepare_strassen_pack.py` は元のINT8重みから7種類のStrassen operandを事前生成する。加減算結果をINT8にwrapし、はみ出した±256を疎な符号付きINT8補正とUInt8列indexで保持する。元の量子化は変更せず、整数積を正確に再構成する候補。

実Attention Q行列（8192×2560）では、加減算のoverflow率は各式で0.66〜0.94%。部分試験packは59,249,724 bytes、変換INT8重み36,700,160 bytes、CSR補正1,545,276 bytes。全モデルpackではない。固定した元pack・tensor・準備ソースhashをmanifestへ記録し、元packは変更しない。生成物は `artifacts/strassen-prepacked/`。元のfull-model canisterのsealed packを書き換えず、別の診断canisterで試すための準備物。

`test_strassen_pack.py` は±overflow、CSR不正形式と、符号付き整数行列のStrassen再結合を検証する。現段階では事前準備と整数同値性のみを検証し、Wasm queryへの接続・命令数削減は未測定。理論上の積の削減率を性能向上と扱わない。

## 現在の既定ビルドと検証済み結果

実験featureを無効にした既定ビルドで、新しいprefixを含む全5実行を再検証した（`kernel-default-v4`）。保持hidden/state・最終hidden/logits/確率/unknown/ラベルはterminal-v3とbit一致、失敗query/replay 0、実行前後に認証済みmodule hashを確認した。

| 入力 | query | handler命令 | Candid bytes | 単回秒 | 最大query命令 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix準備45 tokens | 330 | 229,437,549,313 | 360,695,462 | 34.493 | 2,311,826,661 |
| 主問題132 / suffix87 | 352 | 405,322,437,010 | 579,259,841 | 44.284 | 4,167,348,841 |
| 情報不足125 / suffix80 | 352 | 358,162,787,577 | 537,098,941 | 43.012 | 3,642,405,503 |
| 最大変更134 / suffix89 | 352 | 418,451,724,030 | 591,305,752 | 55.549 | 4,322,854,140 |
| cacheなし132 | 501 | 593,293,161,174 | 885,497,502 | 78.558 | 3,783,030,089 |

主問題はterminal-v3比10.06%命令減、最大変更は383→352 query・11.30%命令減。初回prefix準備は別費用330 queryで、主問題との合計は682 query。50/32 queryは未達。主問題の整数射影/MLP群は約76.5%、DeltaStageは約15.3%。50回の理想予算2500億へ、さらに約38.3%の命令削減に加えてquery統合が必要。最大lock問題の既存誤判定を改善したとは扱わない。

現在の既定Wasm SHA256は `8a84a27ea5885949e3d9e18316a80567364102f0529377be7289b7e55ff0a3f2`。全5実行のsource hashにはruntimeの全Rust moduleを記録し、実験カーネルも含むソース構成を残す。Rustは既定/実験feature両方で30 tests、Python 40 tests。整数経路の126条件native/Wasm照合も通過。生成物をgitignoreし、Laya・mainnet・Git remoteは変更していない。

再検証には `validate_terminal_readout.py --run-name kernel-default-v4 --baseline terminal-v3` と `summarize_kernel_unroll.py --run-name kernel-default-v4` を使った。新規実測には未使用のrun名を指定する。生測定 `artifacts/kernel-default-v4-*`、検証付き集計 `docs/kernel-default-v4-summary.json`。
