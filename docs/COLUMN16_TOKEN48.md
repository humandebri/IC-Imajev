この文書の測定は48-token INT8採用時点。現在の採用版は[元F32 LoRAの端数処理](MATRIX_TAIL.md)。

# 48 tokenで固定重みの展開を共有する

2026-10-02。前の改善はDelta後半と出力投影を統合し、主問題を193→169 queryへ減らした。この候補では各投影の中で繰り返す重みロード・INT8→I16展開を減らす。

採用済み16出力行kernelのtoken tileは最大32。候補は48 tokenをまとめ、45 tokenのprefixはpaddingを含む48 tokenを一度に処理する。87 tokenは48+32+8、89 tokenは48+48となる。block256の整数dot、F32 scalingの順序、padding、出力の有限値検査、LoRAとモデル構成を維持する。追加の量子化はない。

`experimental-column16-token48`は直接診断APIを有効にする。単独では通常の`project`と`project_column16`は従来経路を維持する。追加の`experimental-adopt-column16-token48`で通常の`project`を48-token版へ接続できる。canisterの`experimental-column16-token48`がこれをforwardする。全層検証後、下記の新Wasmを採用した。診断canisterの`column16-token48` featureは同じWasm内に32-token版と48-token版を組み込み、同じ固定重みと同じ入力で測定する。返信の互換名`pair`は今回48-token版を意味する。

native scalar oracleは1/7/8/31/32/33/45/47/48/49/64/87/89/95/96/97/132 token、256/512列、符号端値の全出力bit一致を確認した。不完全な16行tile、不正scaleを両方のAPIで拒否した。記録は`artifacts/column16-token48/tests.log`。

専用local診断canisterは`5iptu-f3777-77775-aaaga-cai`。診断測定後、下記の全層検証を完了した。50 queryは未達。

## 同じWasm内での通常query測定

| 条件 | token数 | 32-token版命令 | 48-token版命令 | 削減率 |
| --- | ---: | ---: | ---: | ---: |
| prefix | 45 | 637,250,238 | 630,803,673 | 1.0116% |
| 617 | 87 | 1,196,202,809 | 1,189,761,364 | 0.5385% |
| insufficient | 80 | 1,058,785,102 | 1,052,339,561 | 0.6088% |
| maximum | 89 | 1,263,694,439 | 1,254,422,146 | 0.7337% |
| normal | 132 | 918,849,634 | 914,214,013 | 0.5045% |

全5条件で両Wasm出力digestが独立native全出力digestと一致した。nativeでは従来8行、32-token×16行、48-token×16行の全値bit比較を行った。fixed weight8192×2560、132-tokenのみ出力4096行。65準備update・10通常query・module status read2 updateを分けて記録した。counterは入力復元・量子化・base投影を含み、LoRA・digest・Candid encoding・全層推論を含まない。命令削減を全層結果やquery数へ外挿しない。

診断module `2a583bda911721a2e6bb637884c1678a3494fec454e331a2f22b44e37f6cf4fa`、記録`artifacts/column16-token48/check/report.json`、測定時29ソースhashとarchiveは`artifacts/column16-token48/source-hashes.json`と`source.zip`。全層用feature追加前の診断ソースである。生成物はgitignore対象。

固定packのLoRA A200行列のSHA256も調べたが、200個すべて異なり、同じ入力に対するbit同一のA投影を省く候補はなかった。記録`artifacts/column16-token48/lora-a-duplicates.json`。近似的な同一視は行わない。

## 次のquery統合に向けた実packetの確認

`scripts/size_attention_full.py`は採用済みDelta-finish v2の40 Attention層packetから、正規化入力・prefix KVだけを一度送る完全Attention候補を組み立て、投影結果と現在tokenのKVを返信する容量を算出する。全40候補はrequest/replyとも2MB以内。prefixの既存handler合計最大1,885,999,356、主問題4,907,793,402、情報不足4,206,864,104、最大変更4,983,653,561命令。後者は5Bに近いため、単純な連結では上限遵守を証明できない。prefixなし132 tokenは7層が5Bを超え、最大6,580,940,971命令。最終層だけは5B未満。

この計算はhostでの容量算出と既存の別query counterの合計であり、統合Wasmの実測ではない。予測を採用結果と扱わず、部分統合・量子化済み入力の共用・重複検査削除と実測を次に行う。記録`artifacts/column16-token48/attention-full-sizing.json`。

## 全層検証と採用

| 条件 | query | handler命令 | 削減命令 | 削減率 | Candid bytes | 実測秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix | 170 | 166,526,069,137 | 1,102,783,992 | 0.6579% | 237,812,108 | 25.2926 |
| 617 | 169 | 315,497,848,425 | 1,055,170,104 | 0.3333% | 354,192,982 | 42.8215 |
| insufficient | 169 | 270,053,571,031 | 1,061,152,920 | 0.3914% | 330,090,604 | 38.8137 |
| maximum | 169 | 319,408,607,270 | 1,524,540,792 | 0.4750% | 361,100,413 | 53.7416 |
| normal | 294 | 457,626,940,696 | 872,590,160 | 0.1903% | 580,794,410 | 73.9408 |

全5条件の保持hidden・状態・判断・確率が直前Delta-finish v2とbit一致し、失敗/replay0。prefixは全32層の全token hiddenと72状態配列、質問は31層の全token hidden＋最終層の最後のtokenと48保持状態配列を比較した。全32層・全tokenの演算を維持し、終端の不要状態を省く従来仕様も維持する。公式BF16参照への誤差や、最大変更gold=yesをnoとする既存誤判定は残る。型安全な返信と判断精度を区別する。

主問題は1,055,170,104命令（0.3333%）減。prefixは1,102,783,992命令（0.6579%）減。主169、初回prefix込み339、prefixなし294 queryと通信量は同じ。50/32未達。今回の主handler合計でも5B/queryだけから少なくとも64 queryが必要で、通信・CDK・依存関係を含む実際の下限は別途必要になる。

最大queryは全条件3,839,353,404命令。最大観測heap4,119,986,176 bytes。query5B/heap4GiB/input logical900K/frame2MBの上限を維持する。counterはCDK Candid encode/decode、Candid通信はHTTP/CBOR/signatureを除外。heapは終端page数で瞬間ピークではない。時間はquery cache未制御の単回測定で、今回は全条件で前回より遅かった。命令削減を実時間の改善と扱わない。

721 tensor・4,065,416,192 bytesの固定cacheとRoPE131,072 bytesをmanifest順でfresh heapへ準備した。固定準備は721 update、15,041,482,761命令、241.5128秒、Candid request47,473/reply14,927,506 bytes。pack status1/cache status2 query・認証read2は準備時、cache status2 query・認証read2は全層前後に別途実行し、推論query数と分ける。

全層用構成で56 runtime tests・2 integration tests・2 compile-fail doctestsが通過。境界token数と符号端値、不正scale/不完全tileの拒否は直接APIと採用APIを検査した。記録`artifacts/column16-token48/full-tests.log`。

採用module `55e47876ea61efe590d6cb71ee53ac433e7d3b55c6e2c024932f1b38dc02a656`、Wasm `artifacts/column16-token48/full.wasm`、canister `4caro-hl777-77775-aaaba-cai`。全層前後の認証moduleとcache不変、38実装hashは`artifacts/column16-token48-v1-cache-checks/report.json`。検証時ソースは`artifacts/column16-token48/validated-source.zip`、全層記録`artifacts/column16-token48-v1-*`、比較`docs/column16-token48-v1-summary.json`。中間状態はclient-held、推論は通常query、固定準備だけupdate。Laya環境は変更していない。

```sh
cargo build --offline --release --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-projection-reuse,experimental-full-weight-cache,experimental-blake3,experimental-attention-fusion,experimental-delta-projected,experimental-prepared-rope,experimental-column16-token48,experimental-delta-finish
# 専用local canisterへのupgrade後:
.venv/bin/python scripts/prepare_weight_cache.py --canister <local-id> \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --directory artifacts/new-token48-preparation --include-f32 --require-prepared-rope
.venv/bin/python scripts/validate_prepared_weights.py --canister <local-id> \
  --run-name new-token48 --baseline delta-finish-v2 \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --preparation artifacts/new-token48-preparation --reuse-projection-inputs \
  --frame-checksum blake3 --fuse-attention --fuse-delta-projected \
  --fuse-delta-finish --require-prepared-rope
```

## 残る計算の内訳と次の探索

直前Delta-finish v2の主問題169 queryでは、`mlp_add_norm_integer`の31層分（gate/upとSwiGLU）が117,318,563,119命令、`delta_project_finish`60,922,647,035、汎用`lora_integer`59,167,773,281、`delta_project_capture`45,555,990,657、Attention Q/GQA24,797,098,984だった。MLP downは汎用投影に含まれ、別の残差・次層normも3,998,206,273命令ある。単一のquery境界削減だけで50になるとはしない。

F32 LoRAの`matrix_multi`も固定重みを繰り返しロードしている。87 tokenでは32-token group×2のあと23 tokenを4-token group×5とscalar3で処理する。残りを16/8-token groupにまとめること、末尾scalarを独立SIMD laneへ入れることは、列ごとの積和順序を維持して検証できる次の候補。現時点では未実装・未測定であり、削減として計上しない。
