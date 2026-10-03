# 残差・正規化をMLPへ統合

2026-10-02。post-attentionの残差加算・正規化をgate/up/SwiGLUの通常queryに統合した。正規化済み配列の往復を省くため、clientは加算前のhiddenとattention outputを保持する。MLP後にはその2配列とMLP outputを送り、`BF16(BF16(hidden+attention)+MLP)`を再現して次の層の正規化を行う。二つの加算の間のBF16丸めを省略しない。client側で数値計算は行わない。

新演算は `mlp_add_norm_integer` と `add_norm_chain_bf16`。Candidとowner制限は同じで、推論は通常query、中間状態はclient保持のまま。base W8/A8 block256、元F32 LoRA、readout、tokenizer、calibrationは変更していない。`--fuse-mlp-norm --fuse-mlp --fuse-add-norm`で有効にする。3配列が900,000-float上限に収まらない132-token cacheなし入力は従来経路へ戻す。部分グラフの最後と、既存のterminal layer 31も元の経路を保存する。

45/87/89 tokenの実layer 0で、統合MLPと後続chainを前版のquery出力に照合し、全bit一致した。chainはnative scalarにも一致した。Rustのfixtureは統合MLPを元のnorm＋MLPと比較し、別テストは途中の丸めを省略すると値が変わる入力で両方の丸めを検証する。Rust34 tests、Python40 tests通過。

固定4B packを保持した `4caro-hl777-77775-aaaba-cai` で新prefixから全5実行を再検証。保持hidden/state、最終hidden/logits/確率/unknown/判断は前版 `packed-reduction-v1` と全bit一致。prefixの72状態配列／各問題の48状態配列、終端層の実際に使う最後のhiddenを比較した。失敗・replayは0、前後の認証済みmodule hashが全実行で一致した。

| 実行 | query：前→後 | 命令数 | 通信bytes | 時間s | 最大query命令数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix45 | 330→298 | 212,408,296,161 | 345,001,126 | 43.158 | 2,291,174,835 |
| 主問題132 / suffix87 | 352→321 | 377,303,015,436 | 549,890,193 | 59.356 | 4,189,039,621 |
| 情報不足125 / suffix80 | 352→321 | 332,031,946,117 | 510,090,253 | 61.289 | 3,657,474,583 |
| 最大変更134 / suffix89 | 352→321 | 388,433,506,539 | 561,261,544 | 65.672 | 4,326,621,824 |
| cacheなし132 | 501→501 | 554,462,878,785 | 885,497,502 | 87.444 | 3,732,834,908 |

主問題はquery8.81%減、命令0.576%減、通信5.07%減。初回主問題はprefix準備298＋推論321＝619 query（前版682）。cacheなしは同じ演算経路で、追加dispatch等により63,083命令だけ増えた。主問題の観測heapは47,054,848 bytes（前43,450,368）。handler命令数はCandid decode/encodeを含まない。通信はCandid request＋replyでHTTP等を含まない。時間は単発実測で、速度改善とは報告しない。

50/32 queryは未達。準備済みの主問題でも命令数だけで最低76回の5B予算に相当し、50回へさらに約33.7%の削減とquery統合が必要。初回準備費用を省いて達成扱いにしない。既存の最大lock問題の誤判定は残り、判断精度や校正の改善を示す結果ではない。

既定Wasm SHA256: `821f969e04cf6902d8bbe6c0203fc10c0c0f26e9370e325b7b6adfd6e23dc295`。生測定 `artifacts/mlp-norm-v1-*`、検証付き集計 `docs/mlp-norm-v1-summary.json`、単体照合 `scripts/check_mlp_norm.py` と `artifacts/mlp-norm/probe{45,87,89}`。生成物はgitignoreし、Laya・mainnet・Git remoteは変更していない。

再検証は未使用run名で実行する。

```sh
.venv/bin/python scripts/validate_terminal_readout.py --canister <専用local ID> --run-name <新しいrun名> --baseline packed-reduction-v1 --fuse-mlp-norm
.venv/bin/python scripts/summarize_kernel_unroll.py --run-name <同じrun名> --baseline packed-reduction-v1
```

## 当時の通信試作と後続実装

後続でblock256 codecのRust/Wasm実装と全層検証を完了した。[BLOCK_CODEC.md](BLOCK_CODEC.md)。以下は当時のpacket試作の記録であり、QKV/gates/hiddenの融合演算と940,992-float対応は依然未実装。

`scripts/explore_block_codec.py`はcanister未実装のprototypeとして、256要素ごとに全BF16かF32 blockかを示す形式を調べた。各要素のbitmapを減らし、F32例外を含むblockは全値をF32で保存する。量子化や丸めは追加せず、代表的な全operationのrequest/replyで全bitのroundtripを確認した。

87-token実QKV・gatesにhiddenを併せる試作payloadは940,992 floatsで、現形式の推定2,005,620 bytesがprototypeでは1,888,548 bytesになった。ただし当時の900,000-float上限を超え、この時点のcanisterはcodecも融合演算も受け付けなかった。拡張された値数の明示的な検証、2 MBの厳密な上限、実canisterの命令費用、全層の照合が必要。これは次の統合の可能性を示すpacket試算であり、現在の通信量やquery数の改善には計上していない。データは `artifacts/mlp-norm/block-codec-screen.json`。
