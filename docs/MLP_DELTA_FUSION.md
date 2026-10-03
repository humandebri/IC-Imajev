# MLP完了と次Deltaを融合する試作

2026-10-03。前段の個別query合計が5Bに入ったことを受け、`experimental-mlp-delta-fusion` を実装した。先頭256行のdown投影を準備queryで先に計算し、残り2304行と次Deltaを1 queryへ結ぶ。元のINT8値・F32 scale/A積・BF16残差・conv履歴・F32 prefix logを可逆に持ち運ぶ。モデル重み、量子化、積和順、BF16丸めを変更しない。質問状態はクライアント保持、準備後の推論は通常query。

この版は**未採用**。主87-token融合は3箇所すべてIC0522（5B命令上限超過）で完走しなかった。実装した経路と失敗を保存し、codecを改善して再測定する。入力が2 MB以内であることと、命令上限内で完走することを分けて検証する。

## 実装

- `mlp_prepare_partial_down` は既存の正規化・gate/up・INT8入力準備・元F32 LoRA A積を実行した後、down投影の先頭256行だけを計算する。残差の先頭256行を完成済みBF16 hiddenへ置き換え、残りは元の残差を保持する。
- `mlp_finish_partial_integer` は残り2304行を計算し、完成hidden全体を次層のnormへ渡す。
- `mlp_finish_delta_log_integer` はそのnormを次Deltaへ直接渡す。返信はMLP hidden、Delta出力、conv履歴。間にモデル推論をホストで実行しない。
- `mlp-delta-log-carry-exact-v1` は6つのtyped streamをbyte planeへ並べ、zlibで圧縮する。`miniz_oxide=0.8.9` を固定。展開先はshapeから決め、完了状態・消費入力・出力長・Adler checksum・末尾を検証する。通常codecの900K float上限を変更しない。
- requestは全field・scalar bitで復号時のidentityへ照合する。整数−128、非有限値、非正のscale、prefix gate範囲外、次層がAttentionの融合は拒否する。

## 検証と実測

Rust候補83 unit、単独feature66 unit通過。専用integration2件はrequest全field変更がidentityエラーとなること、不正lane/gate/metadataの拒否を確認した。zlib末尾・不足・過展開・checksum検査も通過。

専用canister `6eydd-o3777-77775-aaama-cai`、module `1b9e20b8126405974a5b0db57a3e28b9f51166c7cc499bc1274b80f66b623d05`。各層で同じmoduleの通常MLPと、partial prepare→finishの全出力がbit一致。準備済みq/scales/Aも保存済みcanister出力とbit一致。融合は87-tokenで上限超過。45-tokenの診断では、canister生成済みcarryの先頭45行をコピーして融合し、同じmoduleで分離計算したhidden/Delta/convとbit一致。45-token診断は87-token本番の代用にしない。

|MLP層→次Delta|87-token prepare命令|87-token finish命令|87-token次Delta命令|融合frame bytes|87-token融合|
|---|---:|---:|---:|---:|---|
|0→1|3,460,143,735|1,363,666,568|3,759,033,623|1,821,966|IC0522|
|1→2|3,460,150,842|1,362,831,937|3,759,196,757|1,831,419|IC0522|
|3→4|3,460,893,438|1,362,365,190|3,759,417,899|1,835,210|IC0522|

21通常queryの診断（失敗3を含む）。さらに最初の失敗確認では5通常queryを送った。以下の45-token counterはprofile queryで、nested spanはinclusiveのため足し合わせない。

|層|融合handler命令|wire decode|うちinflate|MLP finish|次Delta|wire encode|
|---|---:|---:|---:|---:|---:|---:|
|0|2,881,246,950|242,337,698|162,212,787|589,212,916|2,028,187,975|16,265,240|
|1|2,876,746,181|238,766,060|158,560,743|589,238,288|2,028,275,828|16,225,271|
|3|2,876,756,937|238,439,487|158,199,654|589,406,512|2,028,464,892|16,225,312|

全体復号2.38〜2.42億命令のうちDEFLATE展開1.58〜1.62億命令が大きい。これにbyte planeの並べ直し、型検証、frame hashが加わる。先の個別query命令の合算には新しいcarry decoder費用が無く、融合できるという証明にはならなかった。

hybrid packetをそのままMLP carryへ加えると、zlib圧縮後でも概算2.45〜2.51 MBで上限を超える。元のprefix logなら87-token融合frameは実際1.82〜1.84 MBに入るが、32-head log復元も必要で、14-head復元のhybridより演算が増える。

この融合だけでは、元のMLP1 query＋Delta1 queryをprepare1＋fused1へ置き換えるため回数は同じ。50へは複数段を跨ぐ部分実行と追加命令削減が必要。今回の試作を「50達成」「通信・総命令改善」と扱わない。

## 既存経路の回帰確認

同じ実験moduleで既存の全5条件を完走。保持hidden/state、判断4条件のvalue/abstain/logits/probabilities/unknown/最終hiddenは既存INT8版とbit一致。失敗/replay0、typed output有効。融合opcodeは既存graphで使用しない。主87-token suffixは64 query /260,701,700,326命令 /124,634,637 Candid bytes。直前のstate-layout候補より77,984命令増。profile featureと新dispatchを含む診断buildであり、速度向上とは扱わない。50/32未達。

固定準備721 update、61,042,755,368命令、272.475秒、cache4,065,416,192 bytes。主canister・Layaは変更していない。元INT8版と公式BF16版の差、最大変更の既存見逃しは残る。

## 次のcodec候補

canister生成済みcarryを15のbyte planeへ分け、raw / constant / canonical Huffmanを独立選択した場合のサイズを計算した。bounded header16,384 bytesを含む保守的なframe見積りは3箇所で1,749,842〜1,768,729 bytes。Wasm decoderと命令数は未測定。DEFLATEのLZ処理を省き、高entropy planeはrawコピーで復元できる可能性がある。速度を推定値だけで断定せず、実装後に同じ87-token入力で測る。

## 証拠・再現

- ビルド: `artifacts/prefix_codec/full-build-mlp-delta-fusion/source.zip` / `features.txt` / `patch.json`。
- 融合の診断: `artifacts/prefix_codec/mlp-delta-fusion-check-v2/report.json` / `validated-source.zip`。integrationは後にidentityエラー文字列まで確認するよう強化し、`artifacts/mlp-delta-carry-identity-tests-final.log`に再実行を保存した。
- 既存経路: `artifacts/prefix_codec/full-fusion-base-proof/report.json` / `validated-source.zip`。
- サイズ診断: `artifacts/prefix_codec/hybrid-mlp-carry-capacity.json` / `carry-huffman-capacity.json`。Huffman見積りは `scripts/probe_carry_huffman_capacity.py` で再現。

```sh
.venv/bin/python scripts/build_full_prefix_candidate.py --directory artifacts/<new-build> --target-directory artifacts/prefix_codec/full-target --direct-input --terminal-attention --prefix-start --mlp-delta-fusion --instruction-profile
# rawのfail-closed stubへ検証済みdirect-input.watをpatchし、専用canisterへupgrade。
# 固定重み準備後に通常queryで測定する。
.venv/bin/python scripts/check_mlp_delta_fusion.py --canister <test-id> --wasm artifacts/<new-build>/full.wasm --directory artifacts/<new-fusion-report>
.venv/bin/python scripts/probe_carry_huffman_capacity.py
```

圧縮依存を追加したoffline resolverが既存combine4.6.8をcached 4.6.7へ解決したため、Cargo.lockにも記録した。変更後の全回帰とCandid送受信は上記で検証。モデルのロック・packは不変。生成物はignore配下へ置く。
