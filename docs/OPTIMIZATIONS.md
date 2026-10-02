# Imajevの演算・query分割の効率化（2026-10-01）

この文書の前半は部分演算の開発履歴。最新の全32層測定とLaya比較は [LAYA_COST_ANALYSIS.md](LAYA_COST_ANALYSIS.md) を参照。

Layaの実コードを参照し、Imajevの実重み・operandで改善を実装、ローカル通常queryでA/B測定した。**対象は第1層QKVの256出力行と専用readout。全4Bのcanister推論は未完了。** 全32層の速度・精度改善には読み替えない。

実装・測定・参照元のSHA256、各queryの命令数・Candid通信量・時間・heap/stableページは [optimizations.json](optimizations.json)。Layaのソース/Git/network/canisterは変更していない。Laya Gitは読み取り確認時にclean。

## 実装した改善

1. **tokenを4個ずつSIMD並列化**。入力を演算内部で列×4 tokenに詰め直し、同じ重みを4 laneで使う。各tokenの積和は元の列順にF32加算する。横方向のreduce/FMAへ変更せず、数値誤差を増やさない。4未満の端数はscalar。`matmul_reference`を比較用に残す。
2. **base＋LoRA A/B＋BF16境界丸めを融合**。`lora_project`が3 tensorを必要な出力行だけstableから読む。LoRA rank64/scale2、base BF16丸め→LoRA scale丸め→加算丸めの境界を保持。元の4 queryを1 queryにし、入力の再送や中間配列の往復を削る。
3. **readoutの不要行を省略**。`decision_fast`は選択肢数＋unknownの3〜8行だけ計算する。全256行の旧`decision`も比較用に残す。入力stateの重複decode、logitsのencode/decode往復も除く。専用readout、unknown位置、温度、型付き出力を保持する。
4. **クライアントでtoken/出力行を分割**。`Transport.project`は入力・出力float上限、演算量上限、設定したtoken/行上限に収める。命令上限エラー時は行幅を半減し、同じ入力/progressから再試行する。成功したタイルのみ結果に採用する。未知のmodel mismatchなどは再試行せず終了。通信一時障害の再送は429/502/503/504/connection/timeoutに限定した。

token分割が使えるのは各tokenが独立した**線形射影**。attention、conv、DeltaNetに同じ分割方法を流用しない。recurrent stateの継続は既存のDeltaNet経路を使う。中間状態はクライアントが保持し、計算中にupdateを呼ばない。

## 実測

入力は公式ホスト実装から取得した132 token、hidden2560。base実BF16をF32に展開した重みと未統合F32 LoRAを使用。新たな重み/activation量子化は行っていない。

| QKV 256出力行・132 token | query数 | 総命令数 | 最大query命令数 | Candid要求＋返信 |
| --- | ---: | ---: | ---: | ---: |
| scalar / 旧4演算 | 4 | 4,751,616,092 | 3,615,596,409 | 3,450,144 bytes |
| SIMD / 4演算 | 4 | 878,564,740 | 573,612,349 | 3,450,084 bytes |
| SIMD＋融合 | 1 | 704,673,861 | 704,673,861 | 1,487,774 bytes |
| 融合を127/129行へ分割 | 2 | 947,034,373 | 475,328,112 | 2,840,392 bytes |
| 融合をtoken64/64/4、行129/127へ分割 | 6 | 1,017,682,158 | 239,525,433 | 2,844,090 bytes |

融合版はscalarに対して総命令数**85.2%減**、通信量**56.9%減**。必要以上に分割すると入力再送とLoRA Aの再計算が増える。分割の目的は各queryの上限に収めることで、総費用を必ず減らすものではない。132 tokenの融合版handler終端heapは12,648,448 bytes。瞬間ピークではない。

単回のクライアント時間はscalar 0.604秒、融合0.100秒だった。query cache未制御、HTTP/署名などの影響も含むため、これだけで速度倍率を断定しない。命令カウンタはhandler内部（検証、読み出し、展開、演算、内部state serialization）を測り、CDKのCandid decode/encodeを含まない。通信はCandidのみでHTTP/CBOR/signatureは含まない。今回の実測queryはすべて成功し、命令上限による分割再試行は故障注入テストで検証した。

専用readoutの23実入力では総命令数が1,162,932,079→36,992,978（**96.8%減**）。旧decisionのcounterは外側decode/calibrationを除き、新decision_fastはそれらも含むので測定区間は完全に同じではない。23/23でlogits・確率・選択が旧経路と一致。候補数2〜7×F32/INT8重みの12組合せでも旧新一致し、重複選択肢の拒否を確認した。この12組合せは型・演算契約試験で、判断精度ベンチではない。

4/5/32/64/128/132 tokenで、base・LoRA A・LoRA Bの**丸め前F32出力すべてが旧新ビット一致**。融合BF16出力とtoken/行分割の結果もビット一致。公式MLX BF16出力との差は最大0.03125で、既存の積和/丸め差は解消していない。改善によりその差を増やしていない。既存の最大lock変更の誤判定や選択肢順序依存も、この演算効率化で改善したとは主張しない。

Rust 9テスト、schedulerの故障注入2テスト、実canister 6分割の一致、pack違い/checksum/重複候補拒否/query replayを確認済み。

## 探索した次の候補と制約

| 候補 | 根拠・見込み | 状態 / 制約 |
| --- | --- | --- |
| byte shuffle＋可逆zlib圧縮 | 融合132 tokenの入力1,352,093→396,720、返信135,581→34,738 bytes（16-byte envelope込み見積もり） | ホスト上で完全復元を検証。canister codecは未実装、追加命令数/時間未測定。量子化とは別の検証が必要 |
| baseをBF16で保存 | baseは既にBF16学習重み。検証packはF32展開しており、元BF16保存なら該当重みのstable読み出しbytesを半減できる | runtimeはBF16 unpack対応。新packの実測と出力一致は未実施。LoRA/readoutのF32精度は保持する |
| 同じtoken群のLoRA Aを出力行分割間で再利用 | 現在の分割では各queryでAを再計算し再読する。132 tokenの2分割で余分なA読み出し655,360 bytes | クライアント保持Aを受け取る専用経路が候補。入力増加・境界検証と合わせて測定が必要 |
| DeltaNetをvalue head/行方向でSIMD化 | 各value行の更新は同一q/k/g/betaを使用。token依存を守りながら独立行を並列化できる | 未実装。F32 recurrent stateと列順積和、split継続の一致が必要 |
| shared-prefix再利用 | 公式serving経路は共通prefixを再利用している | hybrid DeltaNet state＋attention KVをクライアント保持する設計が必要。独立問題/選択肢順序の評価とキャッシュを混同しない |
| 重みINT8の整数dot | Layaはactivation量子化、integer dot、scale/bias融合を実装 | 今のImajev INT8経路はF32展開後のdot。activation量子化の精度検証が別途必要なので今回の数値一致改善には混ぜない |
| LoRAをbaseへ統合 | 射影数を減らす候補 | 未統合参照のBF16丸め順序が変わり得る。重み量子化/丸めと判断精度の再検証なしに採用しない |

32 value headsのDeltaNet stateは1層524,288 F32（2,097,152 bytes）。現在の450,000 float上限を超えるため、全headを1 state返信へまとめられない。24 DeltaNet層のstateだけで50,331,648 bytesになる。head単位またはhead群をクライアントに保持する分割が必要。これはqueryをさらに細かくする理由となるが、その転送費用は未測定。

Layaのencoder固有phase数・幅テーブルはImajevへコピーしていない。Imajevは24 DeltaNet＋8 full attentionのdecoderなので、全層schedulerには演算依存とKV/recurrent state管理が必要。現段階の分割schedulerは線形射影に限定する。

## 参照箇所と再実行

Layaの`crates/laya-candle/src/int8.rs`（token/出力行tile、scale/writeback融合）、`epilogue.rs`、`tools/client_held_query.py`（失敗時分割幅縮小とclient state）、`tools/query_transport.py`およびcanisterの`query_transport.rs`（可逆圧縮）を読み、ImajevのF32/BF16/LoRA境界に合わせて実装した。固定公式MLXのQwen3.5/LoRALinearを数値境界の基準にした。Layaの実行環境は使用していない。

今回のImajev専用networkとsealed packを使用する。各スクリプトのURL/IDは今回の環境に固定されている。

```sh
cargo test --workspace --offline
.venv/bin/python scripts/test_scheduler.py
.venv/bin/python scripts/benchmark_optimizations.py
.venv/bin/python scripts/check_optimized_scheduler.py
.venv/bin/python scripts/check_fast_readout.py
.venv/bin/python scripts/explore_transport.py
.venv/bin/python scripts/summarize_optimizations.py
```

生ログは`artifacts/optimization-check/`、`optimized-scheduler/`、`fast-readout-contracts/`。これらの検証スクリプトは再実行で自分のログを置き換える。以前の原型測定`artifacts/projection-check/`、`canister-check/`はbenchmarkでは上書きしない。

全層のINT8 pack・768行tile・通常query完走の追加測定は [FULL_INFERENCE.md](FULL_INFERENCE.md) と [full-results.json](full-results.json)。以下の旧Wasm測定とは分けて扱う。
