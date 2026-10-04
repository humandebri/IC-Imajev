# 最終2層の接続で繰り返す処理と不要な中間出力の除去

2026-10-03。layer30 MLPとlayer31の最終Attention・MLP・norm・専用F32判断を、通常query一回へ接続した。主問題は63→62 query、247,184,414,792→247,133,233,383 handler命令（51,181,409減、0.02071%）、Candid124,624,018→123,281,081 bytes（1,342,937減）。50/32 queryは未達。

## 除去した処理

`terminal_tail_integer` はlayer30の残差・Attention出力とlayer31のprefix KVを受け取る。復号と有限値検査を一度行った値を、private fieldの `PreparedTail` に保持する。request全体の同一性も検査し、別requestで検査済み値を流用できない。MLP出力とprefixを借用sliceで最終Attentionへ接続し、連結してから再び切り分けるコピーと接続時の全走査を省く。checksum・shape・有限値・モデル/pack照合を維持する。内部演算の既存検査は残る。

dims `[n,p,0]` は通常推論用で、最終hidden・normと新KVだけを返す。layer30の全token hiddenは内部で消費した後、返却配列へのコピー・BF16通信符号化・Candid返送・client保存を省く。dims `[n,p]` は全layer30 hiddenを返す比較用の経路。両者は演算順序とBF16境界が同じ。INT8 base、元F32 adapter/readout/calibration、入力token数を変更していない。

clientはsuffix 1〜87 tokenのみ融合し、89 tokenの最大変更とprefixなし132 tokenは従来の分割を使う。状態はclient保持、質問に関するupdateや隠れたqueryはない。layer31 KVとlayer30 conv/Delta stateを保持し、layer30のhidden非返却を `layers.json` の `hidden_exported:false` で明示する。

## 同じmoduleでの切り分け

`artifacts/prefix_codec/terminal-tail-compact-control/report.json` は同じmoduleで従来2query・全hidden返却融合・非返却融合を比較した。旧経路の全配列と判断を保存済みINT8版に照合し、成功した融合出力も全要素と判断・確率をbit比較した。

|suffix|従来2query命令|全hidden返却融合|非返却融合命令|非返却の通信削減|
|---|---:|---:|---:|---:|
|87|5,017,847,266|5B上限超過|4,966,663,248|1,342,935 B|
|80|4,609,997,481|4,576,763,980|4,563,085,208|1,235,389 B|
|89|5,124,060,546|5B上限超過|5B上限超過|対象外|

成功した87/80では最終hidden・norm・KV・typed判断・logits・確率がbit一致。80の全hidden返却経路ではlayer30 hiddenも全要素一致。87の全hidden返却は失敗なのでlayer30 hiddenを直接取り出して比較したとは扱わない。87の成功したqueryは実際の5B上限を通過したが、handler counterはCDKの後段Candid処理を含まない。33Mのcounter差を実行全体の余裕と見なさない。

前段の二候補も保存した。最初の連結・再走査版は80で14,498,385命令減、87/89は上限超過 (`terminal-tail-control`)。検査済み型と借用slice版は80で33,233,528命令減、87/89は上限超過 (`terminal-tail-typed-control`)。不要hiddenの返送除去によって初めて87が実行上限へ収まった。

## 全32層の実測

`artifacts/prefix_codec/full-terminal-tail-compact-proof/report.json` と `before-after.json`。直前のS1 address reuse版との比較。

|条件|query|handler命令|Candid bytes|最大query命令|秒（単回）|
|---|---:|---:|---:|---:|---:|
|prefix45準備|66|129,971,420,715|71,105,691|2,276,638,235|11.090|
|旧log・suffix87|64|248,966,180,469|112,344,191|4,966,663,279|20.238|
|主・suffix87|62|247,133,233,383|123,281,081|4,966,663,279|19.755|
|情報不足suffix80|62|225,962,542,748|116,455,579|4,563,085,247|23.563|
|最大変更suffix89|63|252,749,478,524|126,604,858|4,384,713,594|28.994|
|prefixなし132|290|384,398,247,493|580,207,902|2,379,472,033|67.742|

全条件の返却hiddenと全保持state、最終hidden・typed判断・logits・確率が既存INT8版とbit一致。融合した主/情報不足/旧logはlayer30 hidden非返却なので、全32層のhiddenを全て取り出した比較とは区別する。比較driverは欠落をlayer30に限定し、非返却記録とlayer31の統合記録を検査する。query失敗/replay0。最大変更の既存見逃しは改善していない。時間は制御していない単回値で、情報不足/coldは前回より長い。

観測heap終了値の全条件最大は4,131,258,368 bytesで前回と同じ。固定cache721 tensors /4,065,416,192 bytes、準備721 update /78,739,929,194命令 /261.750秒を別計上。prefix packet二度目の準備はcache hitで準備query・命令・通信0。

module `3449f313159240ea70f338e6d508e4a37be982313c57e75c73945694546a0b4b` は検証専用 `6eydd-o3777-77775-aaama-cai`。既存main module `36c04a57…` は不変、Layaは変更していない。検証ソースZIP・kernel hash/ZIP・patch log・準備reportを保存した。Rust97 unit・9 integration・4 compile-fail doctest、client7 tests通過。生成物はignored。

## 再現

新規directoryを指定する。固定packのupload/seal済みの検証専用canisterが必要。

```sh
.venv/bin/python scripts/build_full_prefix_candidate.py --directory artifacts/prefix_codec/rebuild-tail --target-directory artifacts/prefix_codec/rebuild-tail-target --direct-input --terminal-attention --terminal-tail --prefix-start --delta-state-layout --strassen-raw --f32-output-generic --instruction-profile
artifacts/wasm-audit-target/release/imajev-wasm-patch artifacts/prefix_codec/rebuild-tail/raw.wasm artifacts/prefix_codec/direct-input-build/direct-input.wat artifacts/prefix_codec/rebuild-tail/pair-patched.wasm
artifacts/wasm-audit-target/release/imajev-wasm-patch artifacts/prefix_codec/rebuild-tail/pair-patched.wasm artifacts/s1_address_reuse/build/kernel.wat artifacts/prefix_codec/rebuild-tail/raw-patched.wasm __imajev_s1_raw_accumulate
artifacts/wasm-audit-target/release/imajev-wasm-patch artifacts/prefix_codec/rebuild-tail/raw-patched.wasm artifacts/f32_block/build/kernel.wat artifacts/prefix_codec/rebuild-tail/full.wasm __imajev_f32_output64
```

rawのkernel stubはfail closed。3bodyを置換・validate後にupgradeし、`prepare_weight_cache.py --include-f32 --require-prepared-rope --require-prepared-activation --require-all-output-pairs` で固定重みを準備する。`check_terminal_tail.py` で同一moduleの切り分け、`check_full_prefix_hybrid.py --terminal-attention --terminal-decision --terminal-tail --prefix-start` で全層検証する。canister/wasm/directoryは各scriptの必須引数。

次は他の層間接続でも同じ復号・入力検査・一時連結を繰り返す箇所を調べる。演算が必要な値の検査を根拠なく削除せず、固定値は準備update、検査済み動的値は変更不能な型で再利用する。主MLPではbase整数演算が支配的なので、処理の重複除去だけで50/32回へ到達したとは見積もらない。
