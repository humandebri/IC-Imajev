# MLP完了と次Attentionの連結

`experimental-mlp-attention-finish`で、client保持のINT8 partial-down carryとprefix KVを受け、MLP完了と次層のfull Attentionを通常query一回で実行する。量子化・F32 LoRA積和順・BF16境界は変更しない。opaque入力を全request fieldへ拘束し、方向・長さ・scope・有限値を検査する。

## レビューと検証

Rust 138 unit（1 ignored）、15 integration、11 doc、Python境界2テスト通過。1/7/87/89 tokenのcarry/reply、整数・scale・BF16 precision、NaN/Inf、request identity変更を確認した。専用実験canister `6eydd-o3777-77775-aaama-cai`だけをupgradeし、固定参照のsource/reference bookendとZIPを保存。生成物はartifacts以下でgitignore対象。

`artifacts/mlp_attention_finish/all-v1`は3入力×3層×2 down幅の18条件。15条件成功・3条件IC0522。成功条件のMLP hidden/norm、Attention output/KVは全bit一致。89 tokenの768行は失敗し1280行は成功。query一回への融合は全体推論への接続前で、全体54 query、50/32未達。

| suffix tokens | down rows | 層6の命令 | 2query controlとの差 | Candid削減bytes |
|---|---:|---:|---:|---:|
| 87 | 1280 | 4,649,920,207 | 9,896,512増 | 447,027 |
| 80 | 1280 | 4,200,055,008 | 約9.03M増 | 411,146 |
| 89 | 1280 | 4,774,399,992 | 10,105,701増 | 457,282 |

命令削減とは評価しない。v1は診断比較用にMLP normも返すため、encoderと返送に余分な処理が残る。次は後続で不要なnormの返送を除去し、8query連結と全体graphへ接続して実測する。初期prefix準備、partial carryを取得するcontrol、profile再実行は診断用で、実経路の呼び出し数とは分けて記録する。

最終module `1fe829f353f7e7ddd45db7559cecc1b3ee7f61a62b9300dc3bf961c5ff974989`、build `artifacts/mlp_attention_finish/full-build-v1`、4WAT patch wasmparser検証済み。準備721 update、cache4,065,416,192 bytes、78,739,929,194命令。元BF16モデルとの精度評価は別で、最大変更の見逃しは未解消。

## 後続のcompact返信と全体接続

不要なnorm返送を省くAPIと、終端のDelta完了専用APIを追加し、主54→51queryの全体経路を実測した。命令・通信の増加も含めて[全体結果](JOINED_QUERY.md)を参照する。v1証跡は保存し、後続結果と混同しない。
