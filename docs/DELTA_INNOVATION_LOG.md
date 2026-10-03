2026-10-03後続：全32 headのDelta統合を全5条件で検証・採用し、主91→67 query。現在の採用版は[DELTA_FULL_LOG.md](DELTA_FULL_LOG.md)。以下は探索・採用時の履歴。

# Delta状態を元のF32更新量で保持する候補

2026-10-03。採用済み推論は[COLUMN32.md](COLUMN32.md)の91 query。ここではDeltaの2分割を減らすための可逆状態表現を実装・検証する。推論全体へは未接続で、query削減や一般精度の改善を示すものではない。

32 headの128×128 F32状態は2,097,152 byte。87 tokenの入力やconv履歴もあるため、通常queryの2 MB frameに収まらない。以前の可逆圧縮も入力との合算では不足した。

## 保存する値と演算

元の更新は各トークン、各value laneで次の順に行う。

```text
S_i *= g
mem = 元の順序で sum(S_i * K_i)
innovation = (V - mem) * beta
S_i += K_i * innovation
```

prefix処理の際にK、g、innovationを記録する。Kは既存のBF16丸め済み値をそのまま保持し、隣接するvalue headの組で共有する。gとinnovationは元のF32 bitのまま保持する。復元はゼロ状態から各トークン順に`S_i *= g; S_i += K_i * innovation`を実行する。prefixの`mem`内積、betaによる更新量計算、Qによる過去出力計算を再実行しない。K・g・innovationのF32演算順は変えない。

これは近似的なlow-rank分解や状態のINT8量子化ではない。小数の再結合はしていない。一方、状態を直接受信する場合より復元演算が追加されるため、通信削減だけを見て採用できない。

## 実装と段階検証

`scripts/delta_log_bench/src/replay.rs`にcapture/reference/restoreとSIMD復元を実装。専用canisterはquery内で入力から復元し、SHA256 digestと命令数だけを返す。質問状態はcanisterに保持しない。initで固定ownerを設定する。

`check_delta_log.py`は保存済みWasmのprefix stage入力からconvとK正規化をRust nativeで再現する。保存済みの元F32 gateを使い、全24層・前後16 headずつの48条件について、capture後の状態、logから復元した状態、従来recurrence、保存済みWasm状態の全F32 bitを比較する。

native captureは検証用の値を抽出するだけであり、モデル推論をクライアントへ移す実装ではない。採用する場合はcaptureをprefixの通常query内へ接続する必要がある。

nativeでは48条件すべてビット一致。元のgateが0/1の場合、token数1/7/45、shape/finite/decay boundsのunit testも通過。

## 実Wasm復元の測定

専用ローカルcanister `7st3i-3l777-77775-aaaja-cai`で144通常queryを完走。前後16 headの48条件×2方式と、32 headの24条件×2方式を比較した。従来recurrenceと新しいSIMD復元のdigestが独立nativeとすべて一致。さらに48個の復元stateを現在採用中のColumn32版prefixの保存済みstateへ直接比較し、全bit一致を確認した。

| 45 tokenの状態復元 | 従来recurrence handler命令 | log復元 handler命令 | 従来kernel命令 | log復元 kernel命令 |
| --- | ---: | ---: | ---: | ---: |
| 16 head | 158,528,826 | 87,515,764 | 156,761,621 | 85,736,839 |
| 32 head | 316,981,969 | 174,969,404 | 313,428,444 | 171,432,439 |

32 headのlog復元は従来recurrence再実行に比べhandler命令44.80%減。ただし実運用の旧経路はF32状態を直接受信しており、毎回prefixのrecurrenceを再実行していたわけではない。この測定はlog表現を採用する場合に必要な追加復元コストを示す。投影、現トークンの再帰、出力投影、frame checksum/codec、digest、Candid encode/decodeはこのkernel比較へ含めない。handler counterもCDKのCandid decode/encodeを含めず、digest前で停止する。実queryが上限内で完走したことは復元診断のみを証明する。

診断module SHA256: `ca836dad2fff35217a699583ffb3046877ea4cd344dfb7cca67164d1ef5dcd1b`。bookendのmodule status read 2回は管理updateであり、144推論診断queryとは別。固定モデル重みのuploadや準備updateはこの診断では不要。

raw証拠は`artifacts/delta-log/check/report.json`、現在のprefixとの直接比較は`artifacts/delta-log/current-prefix-audit.json`。36 source hash、manifest、Wasm/native helper hashを固定し、`artifacts/delta-log/validated-source.zip`へ保存した。採用済みmain moduleと44 source hashが変わっていないことも確認した。生成物はGit ignore対象、実装と手順は管理対象。

旧mainのDelta前後2 queryのhandler合算最大4,052,681,200命令に32 head復元を単純加算すると4,227,650,604命令。ただし分割方式と新しいbuffer/codec/投影形状の違いがあるため、これは統合queryが5Bへ収まる証拠ではない。実装後に実queryで確認する。

## 32 head統合案の実packetサイズ

全5条件・24層の実入力・出力・履歴を用いる。prefix以外の継続入力には45 token分のlogを加え、質問の返信には次回不要なDelta状態を含めない。prefix返信には新しいlogを含める。encodeには既存の可逆BF16/F32 block256 codecを使用する。

| 条件 | 最大入力frame byte | 最大返信frame byte |
| --- | ---: | ---: |
| prefix 45 token | 280,098 | 1,207,593 |
| 主 suffix87 token | 1,422,687 | 495,191 |
| 情報不足 suffix80 token | 1,386,838 | 459,342 |
| 重大変更 suffix89 token | 1,432,930 | 505,435 |
| prefixなし132 token | 725,648 | 725,648 |

120 packetすべて2 MB以内。最大logical入力530,336 floatsで既存900,000上限以内。これらは仮の統合op名を使った実サイズであり、32 headの投影・再帰・出力投影を1 queryで実行した証拠ではない。

## 再現と次の判断

```sh
cargo test --offline --manifest-path scripts/delta_log_bench/Cargo.toml \
  --target-dir artifacts/delta-log/target --lib
cargo build --offline --release --manifest-path scripts/delta_log_bench/Cargo.toml \
  --target-dir artifacts/delta-log/native --bin log_args
cargo build --offline --release --manifest-path scripts/delta_log_bench/Cargo.toml \
  --target wasm32-unknown-unknown --target-dir artifacts/delta-log/wasm --lib
.venv/bin/python scripts/check_delta_log.py \
  --canister 7st3i-3l777-77775-aaaja-cai --directory artifacts/delta-log/check
```

次は復元命令数と32 head分の投影・再帰の合計を測り、通常queryの実5B上限を満たすか確認する。prefixのlog生成もquery内へ実装し、全モデル5条件で従来hidden/state/判断/確率のビット一致を再検証してから採用する。モデルの重みやcalibrationは変更しない。
