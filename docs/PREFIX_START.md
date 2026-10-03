# 冒頭の埋め込み・norm・Deltaを通常query内で接続する

2026-10-03。毎回クライアントへ返して再送していた埋め込みとnorm0の出力をquery内に留める。実験feature `experimental-prefix-start` と `run_prefix_canister.py --fuse-prefix-start`。主canister・Layaは変更しない。

## 経路と境界

- クライアントはsuffix token ID、既存conv、canisterが生成したNPF1 prefix packetを送る。ホストでembedding/norm/Deltaを計算しない。
- 型付き `PreparedPrefixStart` がop・tensor・shape・方向・token ID整数・conv BF16/有限値・packet prefix長を検証する。全request fieldのidentity照合を保つ。
- 既存embed、rms_bf16、delta_full_logの演算を同じ順序で使う。prefix stateの所有権をそのままDelta再帰へ移す。
- 埋め込み、Delta出力、convをクライアントに返す。埋め込みはlayer0のMLP残差に必要。suffix状態をcanisterへ保存せず、推論updateを使わない。
- replyは2つの既存可逆blockで構成する専用encoding `prefix-start-exact-v1`。それぞれのcountを独立に検証し、結合countも `2*n*2560+3*8192` と一致させる。汎用float上限を変更しない。
- 1〜89 suffix tokens、1〜132 prefix tokensの実験範囲。hybrid packet、full Delta、suffix state破棄を必須とする。prefix準備・cold推論・旧log経路は従来どおり。上限外の入力を黙って短縮しない。

## 検証

Rust unit70件、候補の全runtime featureで77件、prefix系Python16件通過。方向、誤ったcount、非BF16 conv、非整数/範囲外/非有限token ID、packet長/identity、変更requestを拒否する。主87・情報不足80・最大変更89-tokenのnative統合出力は同じビルドでembed→norm→Deltaを別々に実行した結果とbit一致した。

nativeと保存済みWasmのDelta境界には主5・情報不足14・最大変更7要素の差、最大6.103515625e-5がある。統合/別演算のnative同士には差がない。これをWasm全推論の一致と代用しない。`artifacts/prefix_codec/native-prefix-start-check/report.json` / `validated-source.zip` に比較の範囲と差を保存する。

候補module `89ee343165e149e867d4de9847180e1c0dff7b7735bf21c6f4eb7863539b3667` を専用canister `6eydd-o3777-77775-aaama-cai` へ導入。固定cache 721 tensors/4,065,416,192 bytes、721準備update/61,042,755,368命令/277.441秒。固定packは既存のものを保持し、再uploadしない。

全推論5条件の全32層hidden/stateが旧INT8版とbit一致。判断を行う4条件のvalue/abstain/raw logits/probabilities/unknown probability/最終hiddenも一致、型付き出力有効、失敗/replay0。Wasmの統合replyを直前のWasmのembedとDelta0 replyへ分解して比較しても、主・情報不足・最大変更の3条件で全bit一致。native/Wasm間の前述の境界差と区別する。

|条件|query|合計handler命令|Candid送受信bytes|秒（単回）|最大query命令|
|---|---:|---:|---:|---:|---:|
|prefix45|66|136,486,271,048|71,105,691|12.924|2,399,051,824|
|主suffix87・旧log|66|264,182,465,257|113,697,747|21.382|4,555,649,990|
|主suffix87・冒頭統合|64|262,163,826,219|124,634,637|21.798|4,555,649,990|
|情報不足suffix80・冒頭統合|64|240,522,171,431|117,701,589|36.545|4,177,949,780|
|最大変更suffix89・冒頭統合|64|268,527,749,057|126,615,477|29.797|4,666,358,006|
|cold132|291|402,505,194,693|580,218,522|44.230|2,477,351,028|

直前の所有権移譲版から主問題66→64 query、27,272,230命令減（−0.01040%）、1,338,559 Candid bytes減（−1.06257%）。冒頭query自身は3,808,275,603命令、request1,490,234 bytes、reply940,798 bytes。各queryは5 B以内。再コンパイル後の非統合旧log経路は391命令増など、小さな他箇所の差も含む。

観測heap終了値の全条件最大は4,123,721,728 bytes。主suffix自身は4,120,313,856→4,123,328,512 bytesと増えており、メモリ削減の主張はしない。瞬間ピークの測定ではない。秒数は単回で他処理の影響を含むため速度改善率としない。Candid通信はHTTP/CBOR/signatureを除き、命令counterはCDK Candid処理を除く。

prefix packetの初回準備24 query/6,326,348,500命令/61,219,865 bytesとprefix推論は、上記suffixから分けて計上。二度目のpacket準備query/命令/通信は0を再確認した。証拠は `artifacts/prefix_codec/full-prefix-start-proof/report.json`、`validated-source.zip`、`raw-start-parity.json`、`codec-second.json`。最大変更の既存の見逃しは改善していない。50/32 queryは未達。主問題262 Bの単純な5 B/query下限は53回で、さらに演算命令の削減と分割配置の改善が必要。

## MLP後半を次Deltaへ渡す案の予算

保存済み実operandを専用canisterの通常queryで再実行し、各replyの全bit一致を確認した。MLP後半と次Deltaを別々に測り、合計は以下となった。

|MLP層→次Delta|合計handler命令|5 Bとの差|
|---|---:|---:|
|0→1|5,137,534,927|137,534,927超過|
|1→2|5,137,683,643|137,683,643超過|
|3→4|5,138,033,244|138,033,244超過|

これは実際に融合したqueryの値ではなく、個別queryの合計。MLP途中状態とprefixの可逆圧縮で2 MBに収まる容量証拠はあるが、展開命令も必要になる。単純な融合だけで5 B以内やquery削減を主張できないため、先に予算内の冒頭を統合した。

再現は `scripts/measure_stage_pair_budget.py --canister <診断canister> --wasm <対応Wasm> --directory <新規保存先>`。記録 `artifacts/prefix_codec/stage-pair-budget/report.json` にsource/report/request/expected response/module hashと各query命令・Candid bytesを保存する。

## 再ビルドと実行

`build_full_prefix_candidate.py` の `--direct-input --terminal-attention --prefix-start` で未使用directoryへビルドし、[対応WATの置換・validate手順](REPEATED_WORK_REMOVAL.md)でfull.wasmを生成する。raw stubをinstallしない。指定した診断canisterだけを準備後、次を実行する。

```sh
.venv/bin/python scripts/check_full_prefix_hybrid.py \
  --canister 6eydd-o3777-77775-aaama-cai \
  --codec-canister 7hukf-2d777-77775-aaakq-cai \
  --wasm artifacts/prefix_codec/full-build-prefix-start/full.wasm \
  --directory artifacts/prefix_codec/full-prefix-start-proof \
  --terminal-attention --prefix-start
```

新module/graphに結び付けたprefix cacheとpacketを新たに生成する。既存cacheの検査を緩めて流用しない。生成物はartifacts配下でignore。50/32 queryは未達。
