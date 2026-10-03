この文書の測定は16行tile採用時点。現在の採用版は[Delta後半と出力投影の統合](DELTA_FINISH.md)。

# 16出力行の共有ロードと重み展開の明示化

2026-10-02。INT8投影の出力tileを8→16行にし、同じ量子化済み入力のSIMDロードを多くの出力行で共有する。token tileは最大32へ抑え、入力padding・整数積和・block256のF32 scale/加算順序を維持する。16行で割り切れない投影は従来8行版を使う。base/activationの精度とLoRAは変えない。

## 生成Wasmから修正した点

最初の16行版は全5条件でbit一致したが、命令数が0.22〜3.04%増えた。`array::from_fn`で16行の重み配列を組み立てる処理がループとして残り、INT8からI16へ展開した重みを一時メモリへ書いて読み戻していた。

列と32個の8-byteロードをmacroで明示し、compilerが各値をlocalとして扱えるようにした。元の8行版も同じWasm内に残して比較した。512出力値・block256の代表kernelの静的命令は以下。ループ内命令の静的個数をそのまま動的命令数へ加算していない。

| kernel | I32 dot | V128Load | INT8拡張load | V128Store | LocalGet | Loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 token×8行 | 16,384 | 2,048 | 256 | 128 | 35,864 | 1 |
| 初期32 token×16行 | 16,384 | 1,536 | 32（重み展開loop内） | 160 | 35,299 | 2 |
| 明示展開32 token×16行 | 16,384 | 1,024 | 512 | 128 | 35,136 | 1 |

dotと整数加算の個数を減らす方式ではない。入力ロードを共有する範囲を増やし、重み配列の準備ループ・一時store/loadを除いた効果を測る。

## 専用診断canisterの実測

第3層Qの固定INT8重み8192×2560、既存の全層実行から保存した5条件の正規化済み入力を使用した。132 tokenだけは900K出力値上限に従って4096出力行にする。同一Wasm・同じ重みcacheに対する8行/16行の通常queryを各1回実行し、独立native計算の全出力SHA256 digestと両方が一致した。nativeでは全値bit比較を行う。

| 条件 | token数 | 8行版命令 | 明示16行版命令 | 削減率 | 初期16行版の増加率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix | 45 | 671,564,048 | 636,214,542 | 5.2638% | 1.4484% |
| 617 | 87 | 1,224,979,573 | 1,194,557,555 | 2.4835% | 3.0379% |
| insufficient | 80 | 1,108,121,013 | 1,057,004,979 | 4.6129% | 1.4907% |
| maximum | 89 | 1,326,249,893 | 1,261,493,155 | 4.8827% | 0.2201% |
| normal | 132 | 947,346,049 | 917,426,303 | 3.1583% | 1.6042% |

counterはF32入力復元・量子化・base投影を含むhandler区間。LoRA・全層推論・digest・Candid encode/decodeは含まない。単回時間やquery数の改善をこの表から主張しない。

診断用実行器の返信フィールド`pair`は従来pair-factor診断との互換名であり、今回の`candidate_kind=column16-explicit`では16行kernelを表す。重みseal時に旧pair-factor用データも保持するが、16行queryでは使わず、8/16行版の両方で同じ固定cacheを使う。

初回diagnostic canister `52jen-jl777-77775-aaafa-cai`、module `50bc71d65e5b43308b057bcd164072959fde3d8b0b609b3beda0c1be8976a8e1`、記録`artifacts/column16/check/report.json`、ソースarchive`artifacts/column16/source.zip`。明示版canister `55icz-et777-77775-aaafq-cai`、module `731f5062d1c74d0fc0aa649bf62ae939de2203f7e9fdac3f2890858490586aff`、記録`artifacts/column16-explicit/check/report.json`、ソースarchive`artifacts/column16-explicit/source.zip`。各65準備update、10通常query、module status読取り2 updateを分けて記録する。

## 型・境界と全層確認

`experimental-column16`で16行の直接診断APIを公開する。`experimental-explicit-weight-loads`は重みロードの明示展開を選ぶ。runtimeの`experimental-adopt-column16`は両方を有効にし、既存`project`を16行版へ接続する。canisterの`experimental-column16`がこれをforwardする。外部のCandid/wire、query上限、heap上限、client-held状態は変えない。

native scalar oracleは符号端値、1/7/8/31/32/33/64/87/89/132 token、256/512列、32出力行でbit一致。不完全な16行tileと不正scaleを拒否する。通常運用feature構成の55 runtime tests、2 integration tests、2 compile-fail doctestsが通った。opaqueな量子化済み入力を維持し、同じ境界検査の後にC=8/16のprivate helperへ渡す。重みmacroの各loadは検査済み行のblock256内に収まる。

## 全層実測と採用

721 tensorをfresh heap・元のmanifest順で準備し、全5条件を完走した。保持hidden・状態・判断・確率は直前の採用INT8版とbit一致し、失敗/replayは0。全32層・全tokenを処理し、終端層は従来どおりreadoutに必要な最後のtokenだけ保持する。query数とCandid通信量は変わらない。

| 条件 | query | handler命令 | 削減命令 | 削減率 | Candid bytes | 実測秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix | 194 | 167,732,915,946 | 5,979,045,056 | 3.4419% | 246,693,116 | 25.0887 |
| 617 | 193 | 316,924,219,278 | 4,853,778,624 | 1.5084% | 371,329,222 | 41.0281 |
| insufficient | 193 | 271,460,082,252 | 8,282,609,536 | 2.9608% | 345,843,820 | 36.9228 |
| maximum | 193 | 321,311,408,003 | 10,529,779,392 | 3.1731% | 378,624,781 | 44.6806 |
| normal | 294 | 458,499,530,856 | 5,819,328,980 | 1.2533% | 580,794,410 | 65.5248 |

全5条件で命令を減らしたため採用する。主問題は48.54億命令（1.5084%）減、prefixは59.79億（3.4419%）減。主193・初回prefix込み387・prefixなし294 queryで50/32未達。主問題のhandler命令を5Bで割るだけでも64 query、初回prefix込み97 queryが下限で、CDK・通信・状態の依存を除いた楽観値である。

最大queryは主問題3,804,503,338、全条件3,860,151,590命令。最大観測heap4,119,986,176 bytes。query limit5B/heap4GiB、2MB frame、900K公開logical値上限は維持した。時間はquery cache未制御の単回実測で速度改善の保証ではない。命令counterはCDK Candid encode/decodeを、通信はHTTP/CBOR/signatureを除外する。heapはhandler終端page数で瞬間ピークではない。

追加の精度劣化がないことを既存INT8実装との比較で確認した。公式BF16との差や最大変更gold=yesに対するnoの誤りは残る。型安全性と判断精度を同じ評価にはしない。

採用module `d49fd4e1a943c43270826c465979a6869d1e35c3d5cba575ec2b2ce87531aa52`、専用local canister `4caro-hl777-77775-aaaba-cai`。`artifacts/column16-v1-cache-checks/report.json`で37実装hash、認証module bookends、721 tensor・4,065,416,192 bytesのcacheとRoPE表131,072 bytesの不変を確認した。ソースarchive`artifacts/column16-explicit/inference-validated-source.zip`、全層記録`artifacts/column16-v1-*`、比較`docs/column16-v1-summary.json`。

固定準備は別途721 update、15,041,482,761命令、228.1266秒、Candid request47,473/reply14,927,506 bytes。準備時pack status1/cache status2 query・認証read2、全層前後cache status2 query・認証read2を推論数から分ける。質問依存の中間状態はclientが保持し、推論は通常queryのみ。

```sh
cargo build --offline --release --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-projection-reuse,experimental-full-weight-cache,experimental-blake3,experimental-attention-fusion,experimental-delta-projected,experimental-prepared-rope,experimental-column16
# install/upgrade後の固定重み準備・全層確認はACTIVATION_BUFFERS.mdの手順に従う。
```

## 次のquery統合の容量確認

`size_delta_finish.py`でDelta後半とその後の出力投影を1 queryにまとめる仮のframeを組み立てた。先のqueryの前半16-head gated出力をclientから渡し、後半のgated出力は投影へ直接接続する想定。prefix・主問題・情報不足・最大変更の全24 Delta層で、入力997,832〜1,961,376 bytes、返信も2MB以内に収まった。別々に実行したhandler命令の合計最大は2,681,906,593である。

prefixなし132 tokenは909,352 logical値となり、現行900K上限で拒否された。既存の分割をfallbackとして維持する必要がある。容量と別実行counterの合計だけを確認した段階であり、統合kernel・precision・合計命令・query削減は未実測。生記録`artifacts/column16-explicit/delta-finish-sizing.json`。次はこの境界を実装・測定する。


## 可逆F32状態圧縮の追加探索

`explore_state_codecs.py`で実prefixの第0/10/30層の全32-head F32状態を調べた。raw、byte-plane、列方向XOR、行方向XOR、bit-planeの5変換とzlib1/zlib9/lzma3の3codecを組み合わせ、45件すべてbit可逆を検証した。最良はbyte-plane/lzma3で、それぞれ1,757,008 / 1,802,136 / 1,789,024 bytes。87-token hiddenとconv履歴をBF16で加えた楽観的なframe見積りも2,252,624 / 2,297,752 / 2,284,640 bytesとなり2MBを超えた。

これは3層・15方式のホスト容量調査であり、あらゆる可逆圧縮が不可能という証明ではない。Wasm decoder・命令数・capture返信/reuse入力の実packetは未実装。追加量子化は行わない。記録`artifacts/column16-explicit/state-codecs.json`。生成Wasm・pack・全測定・中間状態はgitignore対象。
