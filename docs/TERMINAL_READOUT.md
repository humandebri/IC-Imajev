# 判断に必要なtokenだけを計算する

2026-10-02。`--terminal-readout` は通常queryのINT8判断用の追加最適化。入力tokens・重み・activation量子化・calibration・選択肢は変更しない。

## 採用版の全層実測

| 入力 | query 前→後 | handler命令 | Candid bytes | 単回秒 | 最大query命令 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 617（132 tokens、45 prefix準備済み） | 379→**352** | 450,660,351,980 | 579,259,841 | 51.174 | 4,769,185,878 |
| 情報不足（125 tokens、同prefix） | 379→**352** | 400,002,358,771 | 537,098,941 | 52.461 | 4,197,250,156 |
| 最大lock変更（134 tokens、同prefix） | 411→**383** | 471,751,558,567 | 606,349,640 | 64.936 | 4,582,503,238 |
| 617 cacheなし（132 tokens） | 508→**501** | 665,196,829,946 | 885,497,502 | 75.797 | 4,395,987,614 |
| 45-token prefix準備 | 354→**330** | 248,234,620,521 | 360,695,462 | 26.524 | 2,556,437,714 |

617ではquery 7.12%減、命令 3.19%減、通信 5.80%減。prefix準備は別費用で、初回は330+352=682 query。cacheなしは501 query。handler終端heap最大観測は43,450,368 bytesで前版と同じ。命令はCDKのCandid decode/encodeを含まず、通信はCandid request+replyでHTTP/CBOR/署名を含まない。単回時間から速度向上の再現性は断定しない。

3入力とcacheなし実行で、前31層の全hidden、最終層の最後のhidden、保持48 state配列、最終正規化hidden、raw logits・確率・unknown・ラベルが従来INT8版とbit一致。prefix準備も全32層hiddenと72 state配列が一致。最終版の全5実行は失敗query 0、replay 0。型安全出力は全判断で検証済み。最大lock変更の既存誤判定（gold=yes、回答no）は今回も残る。

現行617の射影・MLP query群は命令の78.66%。現在の4507億命令を詰め直すだけなら5B/queryで最低91回が必要で、32回の1600億予算へはさらに64.50%以上の削減が必要。今回32回は未達。これは現行演算量に対する下限であり、別の演算方式でも不可能と示したものではない。

Rust 28 tests、Python 37 tests、実Wasmのsuffix/終端MLP比較、および全層5実行を通過。Wasm SHA256は `6e4b6cd5639e7fa9ab7d05fc7ad940be8b2b539cf550a33a6a6fd06c303674f0`。5実行の記録済みsource/Wasm hashは作業中のファイルと一致する。

生測定は `artifacts/terminal-v3-{prefix,617,insufficient,maximum,normal}/report.json` と各queryのrequest/reply/metric。比較は `docs/terminal-readout-*-results.json`、失敗・replayを除外していないことを検証する集計は `scripts/summarize_terminal_readout.py`。中間のv2測定を最終版へ混在させない。

## 省略する演算
最終31層目（0始まり）の出力は、最後のtokenだけが専用readoutで使われる。この層のQ射影、Q norm、Q RoPE、Attention出力、O射影、MLP、残差・最終normを1 tokenに限定する。K/V射影と履歴は全tokenを保持する。前の31層は全tokenを処理する。RoPEは元の絶対位置を使う。

さらに、prefix cacheを使う各Attention層では、使われないprefix側のQを0で埋めて計算する処理を省く。`gqa_suffix_bf16` は全K/Vとsuffix Qを入力し、元のcausal位置・内積順序・softmax・BF16丸めを保つ。

最後の1-token MLPは `terminal_mlp_integer` でpost-attention residual/norm、gate/up/SwiGLU、down射影、final residual/normをまとめる。元のF32 LoRAと各BF16境界を維持し、専用decision readoutは別queryで実行する。単一query内で重みを順番に読み出し、サーバー側に推論セッションを保存しない。

`--fuse-delta-input` では、96 tokens以下でQKV全体が既存のtoken/row/work上限に収まる場合、QKV射影とA/Bゲート射影を `delta_input_integer` にまとめる。Z射影は別query。元の計算結果を連結するだけなので、再量子化・adapter統合・再帰stateの圧縮は行わない。132-tokenのcacheなし実行ではこの条件を満たさず、従来の分割を使う。

## 保存形式と適用範囲

最終hiddenと `layer-31.npy` は1行だけになる。比較対象は前31層の全hiddenと、最後の層の最後の1行。省略したhiddenが一致したとは表現しない。保持したconv/KV状態は全て比較する。終端Delta状態は以前と同様に省略する。

このflagは `--compact-heads` と終端推論を必要とする。prefix準備では無効化し、全hiddenと継続stateを保存する。cacheのgraph/Wasm hashが変わるため新しいdirectoryへ再準備する。途中層までの実行では最終tokenへの縮小は発生しない。

## 不採用の整数行列積候補

one-level Strassenで2×2ブロックの8本の積を7本へ置き換える候補を実装した。各block256のscale適用前に整数で再結合するため、成功した実shapeは全出力bitが一致した。しかし、事前変換・拡張重み・中間配列・再結合の費用を含めたWasm命令は増加した。

87-token実入力の代表例は、QKV射影2,122,518,670→4,347,355,773命令。7 shapes中5つはbit一致、2つは5B命令上限超過。不採用として元のinteger kernelへ戻した。乗算数の理論上の削減を性能向上と扱わない。

候補生成コードは `scripts/generate_strassen_candidate.py`。`artifacts/strassen/baseline.rs` を必要とし、実行するとkernelソースを書き換える実験用scriptである。生入力比較・候補ソース・Wasm・測定は `artifacts/strassen/`。この候補の自動deployは行わない。

## 検証

初期suffix実装はWasm SIMD出力sliceに絶対位置を使う不具合を検出し、相対位置へ修正した。失敗した実行は `artifacts/terminal-readout-617/` に残し、採用版の測定から区別する。`scripts/check_suffix_attention.py` は実Wasmでprefix 0・45・最後の1 tokenを旧出力とbit比較し、この経路を回帰検証する。

```sh
cargo test --offline -p imajev-runtime
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
cargo build --offline --release --target wasm32-unknown-unknown -p imajev-inference
icp canister install 4caro-hl777-77775-aaaba-cai --mode upgrade --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm --network local --identity imajev-local
.venv/bin/python scripts/check_suffix_attention.py --canister 4caro-hl777-77775-aaaba-cai
.venv/bin/python scripts/check_terminal_mlp.py --canister 4caro-hl777-77775-aaaba-cai
.venv/bin/python scripts/validate_terminal_readout.py --canister 4caro-hl777-77775-aaaba-cai
```

検証はImajev専用local canisterで行う。Laya、mainnet、Git remoteは変更しない。生成物はgitignore対象。測定directoryは新規にし、replayを新しい実測と数えない。

個別実Wasm試験：132-token・16 headsのAttentionはprefix 0、45、131で全出力bit一致。handler命令はそれぞれ2,229,957,874、1,918,027,898、87,740,271。終端MLPは実保存activationから最終hiddenとnorm出力がbit一致し、793,971,284命令。これらの個別試験だけを全体の高速化とは数えない。
