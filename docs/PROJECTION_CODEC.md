# 量子化済み入力の可逆INT8通信と分割間の再利用

2026-10-02。前のF32配列経由の[投影再利用試験](PROJECTION_REUSE.md)で、受渡し費用が再計算の節約を上回ったため、既に量子化済みの整数を1byteで送る`projection-block256-exact-v1`をRust/Wasm・Pythonに実装した。新たな量子化は行わない。base/activation量子化とF32 LoRA/readout/calibrationは固定したまま、scale・LoRA A結果のF32全bitを保存する。

## Frameと安全条件

専用codecは`lora_integer_capture`と`lora_integer_reuse`に限り、dimsを`[tokens,rows,cols,row_start,rank]`とする。rankはsealed A/B tensorと照合する。capture入力とreuse出力はtag0＋従来block256の可逆payload。capture出力はtag1＋BF16投影結果＋INT8入力＋raw F32 scale/A結果。reuse入力はtag2＋同じINT8/raw F32 state。model/pack/input識別・tensor/aux/scalars・progressを含む既存headerとSHA256で全frameを包む。checksumは認証ではなく、owner-onlyアクセスを維持する。

値数900,000、frame2,000,000 bytes、header16,384 bytes、token/column/rank/work boundsを維持する。rank0/巨大dims、方向tag/長さ、reserved整数-128、小数・範囲外・整数表現のnegative zeroを拒否する。F32 A結果のnegative zeroとsubnormalは元bitで保存する。scaleは正の有限値、Aと入出力は有限値でなければ拒否する。plain出力数とstate数が同じになる曖昧なshapeも拒否する。INT8 pack/unpackはSIMD化したが、未検証状態をtrustedと扱わない。

## 実際のQKV投影の比較

専用canister `4qggx-l3777-77775-aaaca-cai`で、元layer0 QKV8192×2560＋rank64の23,756,800-byte部分packを用い、4096+4096行の2queryを比較した。全7入力長×cold/warm/clear×4query=84通常queryで独立旧nativeとbit一致した。Rust/Pythonのcompact response再エンコードもbyte一致。別の4queryでreserved整数、zero scale、不正tag、truncated payloadを送り、全て拒否した。

| tokens | 通常2query命令 | INT8 reuse 2query命令 | 削減率 | 通常Candid bytes | INT8 reuse Candid bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 66,726,193 | 60,348,266 | 9.558% | 29,278 | 29,926 |
| 7 | 271,466,007 | 250,532,305 | 7.711% | 189,064 | 193,245 |
| 8 | 177,453,783 | 175,159,983 | 1.293% | 215,694 | 220,464 |
| 32 | 646,291,715 | 640,690,443 | 0.867% | 854,830 | 873,730 |
| 64 | 1,273,268,541 | 1,263,731,165 | 0.749% | 1,707,006 | 1,744,746 |
| 87 | 1,859,367,491 | 1,825,789,398 | 1.806% | 2,319,508 | 2,370,789 |
| 132 | 2,672,025,428 | 2,651,654,611 | 0.762% | 3,517,884 | 3,595,659 |

132-token warmは2,672,025,428→2,651,654,611命令（0.762%減）。coldは2,739,590,410→2,715,157,056命令で、この部分投影の値は全層値と分けて評価する。通信は3,517,884→3,595,659 bytes（2.211%増）。前のF32配列経由の21.44%増から縮まったが、通信が通常版より小さくなったわけではない。短い入力だけの勝利ではなく、全7形状で同moduleの通常2queryより命令を削減した。codec module `f563fc4251b3e56c7fde8d5775c93e12bacbae8995df12e829d43ccceeb1f044`を試験前後に認証済みhashで確認。partial試験は全モデル精度/query数の証明ではない。

Rust45＋cache3＋compile-fail doctest1、通常feature runtime43＋cache3＋doctest1が通過。Python既存wire7・block2・journal4とscheduler10が通過。生reportは`artifacts/projection-codec/probe/report.json`、native/canonical比較・module・ログも同directoryに保存しgitignore。再現は`check_projection_reuse.py --compact-state --verify-malformed --native artifacts/projection-codec/primitive --canister <専用ID> --wasm <candidate> --directory <fresh>`。

## 全層への接続

`--reuse-projection-inputs`をfull/prefix CLIへ追加した。最初の成功tileがstateを返し、以後のtileで再利用する。返送stateのために出力tileを縮めてもquery数が増えない場合だけ有効にし、入力ごとにstateを破棄する。instruction failureでは幅を縮め、未完了tileの進捗やstateを確定しない。sessionとjournalはreuse方式を識別し、新旧checkpointの混在を拒否する。codecはこの投影だけで切り替え、終了・例外時に通常codecを戻す。再開時も最初の保存responseからstateを再取得する。

全層用module `0829445c9641f274162b0eec5036ab265b596a5bf419cebf29ed5e4835d41b50`は`experimental-projection-reuse,experimental-full-weight-cache`でbuildした。selected local主canisterをこの版へupgradeし、sealed full packを保持して固定721 tensorを準備し直した。全5実行の検証を完了し、この実測済みmoduleを保持する。新しい投影再利用はCLIの明示オプションで有効にし、通常codecや従来経路も維持する。Laya/mainnet/Git remoteは変更していない。

## 全層の実測と採用範囲

新しい45-token prefixから、主問題・情報不足・最大変更と、prefixなし132-tokenを全32層で実行した。全5実行で、prefixの全32 hidden/72 state arrays、問題の前31層全hiddenと最終層の必要なlast-token hidden/48 state arrays、final hidden/logits/probabilities/unknown/判断が旧`prepared-f32-v1`とbit一致。terminalで省いている最終層の不要なtokenを全32層hidden検証と呼ばない。失敗/replayは0。各実行の前後認証module hash、721 tensor・4,065,416,192-byte cacheとソースhashを照合した。元モデルの最大変更gold=yes/出力noという既存誤判定も残り、一般精度や校正の改善とは扱わない。

| 実行 | query | handler命令数 | Candid bytes | 単回秒 | 最大query命令 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix | 282 | 193,719,417,681 | 318,918,016 | 28.6905 | 2,134,114,814 |
| 617 | 305 | 354,704,392,013 | 506,366,104 | 46.5944 | 4,021,004,508 |
| insufficient | 305 | 310,181,669,347 | 469,941,494 | 42.1287 | 3,489,465,359 |
| maximum | 305 | 365,457,232,850 | 516,788,871 | 50.7746 | 4,157,302,094 |
| normal | 485 | 523,244,923,486 | 816,643,811 | 73.8889 | 3,612,832,517 |

prefixなしでは31組のcapture/reuseを利用し、従来523,750,449,327→523,244,923,486命令（505,525,841命令、0.09652%減）。queryは485のまま、通信は814,232,972→816,643,811 bytes（2,410,839 bytes、0.2961%増）。row境界を変えた全モデルでは、同じ4096+4096行での部分投影の0.762%をそのまま適用できない。prefix準備済み経路は再利用対象が0で、主問題の305 query/506,366,104 bytesは同じ。handler命令は354,704,377,597→354,704,392,013と14,416増え、意味のある改善はない。codec対応moduleのわずかなdispatch差を隠さない。

準備は別途721 update、15,027,997,708 handler命令、Candid request47,473/reply14,915,249 bytes、221.231秒。pack status1/cache status2/認証module読取2も別に記録。heapの最大終端観測は最大変更で4,115,005,440 bytes、主問題/prefixなしで4,114,612,224 bytes。4GiBの残量は約180 MBだが、割当peakの保証ではない。時間は単回であり、cache/負荷を制御した速度改善の証明ではない。handler counterはCDK Candid decode/encodeを、通信はHTTP/CBOR/署名を含まない。

初回prefix込み282+305=587 query、prefixなし485 queryで50/32は未達。主問題のhandler命令だけでも5Bを最大利用する仮定で最低71query、初回prefix込みでは110、prefixなしでは105が必要。query統合だけで目標を達成できるとは扱わない。生データは`artifacts/projection-codec-v1-*`、`docs/projection-codec-v1-summary.json`（gitignore）。この節の数値はcompact codec初版の記録。後続の[直接INT8展開](DIRECT_PROJECTION.md)でF32経由の再変換を除去し、全5実行を比較して追加削減を実測した。現在の採用moduleと数値は同文書を参照。MLP分割間の2つのLoRA A結果の再利用と整数dotの削減は残る対象。

再現手順（既存のselected local full pack/4GiB設定を使用し、Layaには実行しない）：

```sh
cargo build --release --offline --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-projection-reuse,experimental-full-weight-cache
icp canister install 4caro-hl777-77775-aaaba-cai --network local --identity imajev-local \
  --mode upgrade --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm --yes
.venv/bin/python scripts/prepare_weight_cache.py --canister 4caro-hl777-77775-aaaba-cai \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm --include-f32 \
  --directory artifacts/new-reuse-preparation
.venv/bin/python scripts/validate_prepared_weights.py --canister 4caro-hl777-77775-aaaba-cai \
  --run-name new-reuse --baseline prepared-f32-v1 \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --preparation artifacts/new-reuse-preparation --reuse-projection-inputs
```

既存の検証fixtureと比較元artifactが必要。新規環境でarchiveしたartifactを使う場合はmodel/pack/sourceを同じhashで確認する。
