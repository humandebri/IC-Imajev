この文書の測定は入力バッファ削減時点。最新の採用版は[16行tileと重みロード明示化](COLUMN16.md)。

# 毎queryの入力バッファの二重書き込みを除く

2026-10-02。INT8量子化・client-held準備状態の復元・その返信への追加で、全領域をゼロ初期化した直後に同じ領域を実値で上書きしていた。実入力の領域は一度だけ書き、token paddingだけをゼロにする。量子化式、block256 scale、INT8範囲、LoRA、BF16境界、F32再帰状態は変更しない。

`int8_kernel.rs` の `quantize_rows`、`QuantizedRows::from_bytes` / `from_wire`、`append_wire_values`が対象。確保したVecのlengthは初期化が完了するまで増やさず、spare capacityへ書く。復元SIMDの宛先は`MaybeUninit`であり、末尾の不正値を検出して戻る場合も未初期化要素を参照・公開しない。整数型にはdrop処理がない。量子化は全block256を書いた後にpaddingを初期化する。返信追加は元のprefixを保持し、新しく追加した全要素を初期化してからlengthを更新する。

スケールの正の有限値検査と整数範囲・整数性検査は残す。opaqueな`QuantizedRows`のコンストラクタ以外で初期化を確定させない。queryをまたいだactivationのheap保存や、状態の追加量子化は行わない。

## 安全性と部分検証

native feature有効55 tests＋compile-fail doctest2、default44 tests＋doctest1が通過した。1/7/8/31/87/132 rows、512列、符号端値、padding、既存返信prefixの負ゼロ・subnormal、2回の追加を検査した。byte128やfractional/範囲外/非有限値を先頭と末尾に置いた拒否経路も検査した。

`check_delta_projected.py`で5条件・第0/30層・capture/reuse合計20通常queryが保存された従来Wasmとbit一致。不正入力7 queryを拒否し、末尾のfractional/-128も含む。`check_activation_restore.py`は実132-token MLP reuse frameのINT8 byte列の先頭と末尾を128へ変更し、checksumを正しく更新して送った。両方を拒否し、その後の正常queryは保存された従来Wasmとbit一致した。clientだけで弾いた検証ではない。

部分記録は`artifacts/activation-init/partial/report.json`と`artifacts/activation-init/restore-check/report.json`、検証時ソースは`artifacts/activation-init/build-source.zip`。生成物はgitignore対象。

## 固定RoPE準備の切り分け

同じモデルの固定theta/rotary幅、最大512位置のsin/cos表131,072 bytesをownerの準備updateで一度生成する候補も実装した。通常queryは表を読むだけで、範囲外や異なるtheta/rotary幅は従来演算へ戻す。表は質問依存の状態ではない。

13件の部分比較はbit一致し、固定条件のRoPE handler合計は1.4662%減った。ただし全層で準備順序とfresh heapをそろえた候補`prepared-rope-v3`は、主問題321,911,152,912→321,913,095,110命令と0.0006033%増えた。prefixも0.0011259%増え、他3条件は0.0051〜0.0088%減った。全5条件の保持hidden/state/判断/確率はbit一致するが、部分演算の削減を全層の改善とは扱わない。RoPE単独の全面採用は保留する。

全層バッファ最適化候補はこの表も含む。`prepared-rope-v3`との比較でバッファ変更だけを、採用済み`delta-projected-v1`との比較で候補全体を区別する。全層測定の前に同一Wasmへupgradeし、heapを空にして721 tensorを元のmanifest順に準備する。部分診断によるallocatorへの影響を混ぜない。

## 全層実測と採用

全5条件で、以前のINT8 Wasmに対する保持hidden・状態・型付き判断・確率がbit一致した。失敗・replayは0。表は採用済みDelta投影統合版との比較で、固定RoPE表も含む候補全体の差である。全32層・全tokenを計算するが、質問の終端層は候補readoutに必要な最後のtokenだけ保持する従来仕様を維持する。

| 条件 | query | handler命令 | 削減命令 | 削減率 | Candid bytes | 実測秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix | 194 | 173,711,961,002 | 69,562,508 | 0.04003% | 246,693,116 | 28.3076 |
| 主問題suffix | 193 | 321,777,997,902 | 133,155,010 | 0.04136% | 371,329,222 | 49.6271 |
| 情報不足suffix | 193 | 279,742,691,788 | 138,526,391 | 0.04949% | 345,843,820 | 37.1899 |
| 最大変更suffix | 193 | 331,841,187,395 | 167,492,495 | 0.05045% | 378,624,781 | 70.1162 |
| prefixなし | 294 | 464,318,859,836 | 304,742,867 | 0.06559% | 580,794,410 | 79.7182 |

バッファ変更だけを切り分ける固定RoPE表版との比較では、prefix71,519,096、主問題135,097,208、情報不足124,248,616、最大変更138,201,976、prefixなし272,546,613命令を減らした。同じ準備順序・fresh heapで、コアソースの変更は`int8_kernel.rs`のみ。主問題のop別差は`lora_integer`71,812,868、Delta capture32,075,184、reuse10,690,440命令などである。projection出力バッファやwire codecにも二重のゼロ書き込みが残っており、別候補として検証できる。

候補全体は全5条件で命令を減らしたため採用する。query数・通信量は同じ。主問題193、初回prefix込み387、prefixなし294 queryで50/32未達。主問題のhandler総命令を5Bで割っても65 queryが下限であり、整数dotの削減が引き続き必要である。

主問題の最大query3,872,962,026、全条件の最大4,005,858,790命令。最大観測heap4,119,986,176 bytes（handler終端page数、瞬間ピークではない）。selected query limit5B/heap4GiBは維持する。通信はCandid request＋replyでHTTP等を除外、命令はCDK Candid encode/decodeを除外。時間はquery cache未制御の単回測定で、増えた条件もあるため速度改善は主張しない。

既存INT8版から追加劣化がないことを検証した。公式BF16参照との差、最大変更gold=yesに対するnoの誤り、選択肢順序依存は解消していない。型安全性と判断精度は別に評価する。

採用Wasm SHA256 `7fc0e373b1f13c05d35db7410e591b47447c188910db2c9905ce9ced68701b83`、専用local canister `4caro-hl777-77775-aaaba-cai`。`artifacts/activation-init-v1-cache-checks/report.json`で37実装hash、721 tensor・4,065,416,192 bytesのcache、RoPE表131,072 bytes、認証module hashの前後不変を確認した。検証ソースarchiveは`artifacts/activation-init/validated-source.zip`、全層記録は`artifacts/activation-init-v1-*`、比較は`docs/activation-init-v1-summary.json`。

全層比較に使った固定準備は別途721 update、15,041,482,761命令、226.0872秒、Candid request47,473/reply14,927,506 bytes。準備時pack status1/cache status2 query・認証read2、全層前後cache status2 query・認証read2は推論数から分ける。質問の推論と中間状態受け渡しは通常queryを維持する。

```sh
cargo build --offline --release --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-projection-reuse,experimental-full-weight-cache,experimental-blake3,experimental-attention-fusion,experimental-delta-projected,experimental-prepared-rope
# 専用local canisterへのinstall/upgrade後に固定重みを準備する:
.venv/bin/python scripts/prepare_weight_cache.py --canister <local-id> \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --directory artifacts/new-preparation --include-f32 --require-prepared-rope
.venv/bin/python scripts/validate_prepared_weights.py --canister <local-id> \
  --run-name new-buffer-validation --baseline delta-projected-v1 \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --preparation artifacts/new-preparation --reuse-projection-inputs \
  --frame-checksum blake3 --fuse-attention --fuse-delta-projected --require-prepared-rope
```

