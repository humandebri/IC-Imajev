# 数値を変えない命令・query削減

2026-10-01。採用条件は既存の同じ重み・入力・通信精度に対するbit一致。新しいactivation/weight量子化、近似exp/SiLU、LoRA merge、集計順の変更は採用していない。重みは固定したINT8 base＋F32 LoRA/readout、stateはclient保持、推論は通常query。Layaは読み取りのみ。

## 精度優先の結果

追加で16/32/64-token SIMD共有・3タイル融合・可逆通信の分割幅統合を実装し、全層bit一致で1.762兆命令・1,116 queryへ減らした。[最新の全queryと比較](multi-token-compact-lossless-results.json)、[候補・比較条件](FIFTY_QUERY_ANALYSIS.md)。以下の2タイル結果はその前の履歴。許可された演算変更の別経路は [INTEGER_ARITHMETIC.md](INTEGER_ARITHMETIC.md)。

BOOM DAO 617、132 tokens、1質問、rotations=1で、**追加のactivation量子化がない可逆BF16通信版**を完走した。

| 指標 | 以前の可逆版 | 今回 |
| --- | ---: | ---: |
| 総handler命令数 | 3,330,114,701,691 | 2,263,644,653,152 |
| 通常query | 3,108 | 1,588 |
| Candid request＋reply | 2,549,073,399 bytes | 1,984,111,747 bytes |
| 単回local時間 | 560.522秒 | 217.409秒 |

命令数32.03%減、query数48.91%減、通信22.16%減。全32層hidden、conv/DeltaNet/KV state、final logits・候補確率・unknownが以前の可逆版とbit一致。typed choiceは有効でlikely。[全queryと比較](exact-grouped-lossless-results.json)。

これは最適化による数値劣化を検出しなかった結果で、一般正答率の保証ではない。全層の比較は1質問。元の重みINT8/公式MLXとの差は残り、公式参照とのlogit最大差0.04657、hidden最大差0.12109。ホスト比較で見つかったmaximum変更の見逃しや選択肢順依存を、この改善で解消したとは扱わない。

## INT8通信版も別に比較

既存INT8通信版に対してもbit一致で改善した。2.329兆→2.197兆命令（5.68%減）、2,132→1,540 query、1.284→1.014 GB（通信21.03%減）、単回209.638秒。[各query](exact-grouped-int8-results.json)。前のLaya参照で採用した16行SIMDを含め、head融合時点3.133兆からは約29.88%減。

こちらは既に導入していたINT8通信による、可逆BF16版との候補確率差最大約0.121を残す。最適化前後で差を増やしていないが、精度優先では上の可逆版を選ぶ。

## 実装

### 元の2タイルを1 queryへまとめる

`lora_grouped` dims=`[tokens, original_tile_width, input_cols, row_start, total_rows]`。aux=`[LoRA A, LoRA B]`、scalars=`[scale]`。最大2 virtual tiles、900M MAC。元の`lora_project`450M契約は変更しない。

baseの連続行、A全rank、Bの対応行を読み、Aの積を1回だけ計算する。baseのBF16丸め→LoRA scaleのBF16丸め→加算のBF16丸めは維持する。出力はvirtual tileごとのtoken-major順に並べ、各tileを256 float境界まで零paddingする。INT8 encoderのblock境界・max scale・RNEは元の各queryと同じになり、入力も同じINT8 blockを使う。clientはpaddingを外して元の列順へ戻す。

単純に出力行幅を倍にするとINT8のblock境界が変わるため、その方法は使わない。LoRA Aを別queryでclientへ返す案は、queryとF32転送が増えるため、この内部共有を優先した。

実入力9組の2 query→1 queryで元のINT8出力とbit一致。[grouped-projection-check.json](grouped-projection-check.json)。token/行幅/部分blockの91組合せで旧新codecの全bit保存を検証。zero width、2タイル超過、cols違い、row範囲、work超過、zero totalの6条件を実canisterが計算前に拒否。[grouped-contracts.json](grouped-contracts.json)。

### Attentionの再計算をなくす

softmax後の確率の除算・BF16丸めはchannelごとに同じ結果を作っていた。各確率について1回へまとめ、4独立value channelをSIMD laneで処理する。key順の積和、scale/mask/exp、F32の乗算→加算、BF16境界は維持。既存16 Attention queryの合計38.223→18.653十億命令、約51.2%減で全出力bit一致。[attention-exact-check.json](attention-exact-check.json)。

Rust浮動小数点Iterator.sumの初期値は負のゼロ。境界検査でSIMDもこれに揃え、全負ゼロの値でも符号を保存した。ホストとWasmには既存のexp実装差があるため、raw F32の精度検査は同じWasmの旧scalar参照を使う。token1/3/4/7/16、width3/4/5/8/16、負ゼロを含む50条件でbit一致。[attention-boundaries.json](attention-boundaries.json)。参照opは`attention_reference`/`attention_bf16_reference`で、既存owner制限のstep内にある。

全層実測後に負ゼロ修正と参照診断を加えた最終Wasmをinstallし、両完走の変更対象Attention全32 queryを再送して元の全出力とbit一致を確認。[最終Wasm/hashと再検証](exact-final-wasm-replay.json)。全層レポートのcounter/時間/Wasm hashは実測時のmoduleを指し、最終moduleと混同しない。

## 実行予算と限界

最大query handler counterはINT8通信44.740億、可逆通信45.115億。Candid decode/encodeを含まないcounterのため、任意の形状での上限内成功を保証しない。終端heap最大観測は約170 MBへ増えた。sealed重み・owner・Candidのmethod signatureは維持し、準備以外のupdateやcanisterへのactivation永続化は加えていない。

32 query目標は未達。精度を変えないことを優先し、追加量子化が必要な単純W8A8置換は未採用。さらなる候補は、複数token群でのweightロード共有、正しいKV/DeltaNet状態を伴うclient shared-prefix、最終層の最後のtokenのみ計算。いずれもbit一致、準備費用、通信、命令数を検証してから採用する。

## 再現

```sh
cargo test --workspace --offline
cargo build --release --target wasm32-unknown-unknown -p imajev-inference --offline
cargo build --release -p imajev-runtime --offline
.venv/bin/python scripts/test_grouped_codec.py
# 自分のImajev専用canisterへsealed pack維持でupgradeした後
.venv/bin/python scripts/check_grouped_projection.py
.venv/bin/python scripts/check_grouped_contracts.py
.venv/bin/python scripts/check_attention_boundaries.py
.venv/bin/python scripts/run_full_canister.py --wire-codec bf16-exact \
  --delta-head-cap 8 --attention-head-cap 8 --group-projections \
  --directory artifacts/new-exact-lossless-run
```

INT8通信を選ぶ場合は `--wire-codec int8-block256-v1 --delta-head-cap 16`。歴史的sessionへ設定を混在させず新しいdirectoryを使う。時間はcache/ホスト負荷未制御の単回local測定。通信はCandidのみ、HTTP/CBOR/署名を除く。heapは瞬間peakではなくhandler終端の観測。
