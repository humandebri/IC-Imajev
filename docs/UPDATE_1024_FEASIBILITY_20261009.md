# 1024-token update推論の実現性調査

後続作業で実装し、ローカル実モデルの1024-token updateが完走した。測定結果は[ローカル検証記録](UPDATE_1024_LOCAL_VALIDATION_20261009.md)を参照。以下は実装前の調査記録。

2026-10-09の実装前調査。以下は調査時点のソースと保存済みの測定記録に基づく。1024-tokenモデル推論は実行していない。ここでの1024は、質問・選択肢・chat記号・共通prefixを含む入力全体のtoken数。

結論：現行実装では拒否される。内部workerへ分割するupdate方式を拡張すれば対応できる見込みはあるが、受付上限の変更だけでは動かない。完走・数値一致・費用・時間は未検証。

## 実装前の制約

- `canisters/inference/src/paid_inference.rs` の `INPUT_LIMIT` と `REPLAY_INPUT_LIMIT` は512。現在の共通prefixは5 tokensで、1024入力ならsuffixは1019 tokens。
- `crates/imajev-runtime/src/delta_hybrid.rs` の `ServerDeltaStream` はprefix込みの総位置を512までに制限する。
- `attention_token_chunks.rs`、`attention_fusion.rs`、`attention_full.rs`、`lib.rs` のGQA処理に履歴長・総位置512の検査がある。token tile自体の行数制限とは区別して変更する必要がある。
- 現行Attention分割は16 query headsを8＋8で処理する。内部配列上限は900,000 floats、GQA演算量上限は75,000,000。
- 有料APIのworker回数上限は長い入力でも64回。stage数はtoken planから生成するため拡張可能だが、worker回数は別途設計・実測が必要。
- RoPEの事前計算表は512位置。ただし範囲外では同じ位置計算関数へフォールバックするため、表の長さ自体は1024位置の数学的な障害ではない。呼び出し側の位置上限解除と数値検証は必要。

## Attentionの具体的な障害と変更候補

最終位置1024、通常Attention tile 57 tokens、offset 967で式を評価した。終端層だけの1-query処理ではなく、途中のAttention層に必要な57-query処理を対象とする。

| groupのquery heads | KV配列のfloats | RoPE後Q＋KVのfloats | GQA演算量 | 上限内か |
| --- | ---: | ---: | ---: | --- |
| 8（現行） | 1,048,576 | 1,165,312 | 116,269,056 | 配列・演算量とも超過 |
| 4（候補） | 524,288 | 582,656 | 58,134,528 | この2制約は通過 |

4＋4＋4＋4への分割は、既存GQAのhead group単位である4 headsとも整合する。Qの量子化とLoRA A積は既存の共有方式を維持できる。現在の固定2-groupループとgroup_historyの幅を変更し、token・head順序と結果一致を確認する必要がある。上表は全処理のメモリ・命令数を保証するものではない。

## workerとメモリ

現在の89-token Delta/MLP・57-token Attention計画の式を、5-token prefixで評価すると、512入力は403 stages、1024入力は805 stagesとなる。stageは演算の区切りであり、worker自己呼び出し回数ではない。

全suffixのhidden・norm・attentionを各F32で持つ3配列の論理サイズは、512入力で約14.85 MiB、1024入力で約29.85 MiB。suffix K/Vは約3.96→7.96 MiB。合計増分は19 MiBだが、Vecのcapacity、複製、kernel一時配列、allocatorの高水位は含まない。従って実heapはこの値から確定できない。

旧27-token prefix版の保存済み512実測は内部worker34回、257.52秒、最大heap 4,203,151,360 bytes（4 GiBまで87.5625 MiB）。現在の5-token prefix版や1024版の性能証拠には使えない。時間・workerを単純に2倍した値も保証できない。Attentionの履歴計算量は入力長に対して二次的に増える。

ICの公式資料によれば、updateの命令上限は1メッセージ40B、queryは5B、wasm32 heapは4 GiB。自己呼び出しごとの新しいupdateへ計算を分ける現在の構成は、総計40B以上の推論を実行できる。ただし各workerが40B以内であることと、外部呼び出しの応答待ち時間の確認が必要。[公式資料](https://internetcomputer.org/docs/current/references/execution-errors/)

## 実装・検証に必要な作業

1. 受付・receipt replay・Delta継続・Attention/GQA位置上限を整合させ、builderと長さfixtureにも反映する。凍結runtimeを再利用する最適化builderの生成ソースも確認する。
2. 長い履歴で4 headsへ分割し、短い入力の既存経路を維持する。worker上限とworker開始前後の命令予算を検証する。
3. 513・768・1024をローカル実モデルで完走させ、1025の受領前拒否を確認する。境界付近と短い入力の回帰も確認する。
4. 全32層の状態と最終判定を独立参照計算と照合し、heap・各worker命令・時間・cyclesを実測する。途中失敗時の返金とreceipt再取得も確認する。

保存済み512実測の出典は `docs/ADAPTIVE_TOKEN_SCHEDULER.md`。現在のprefix契約・検証手順は `docs/PAID_PREFIX5_VERIFICATION.md` と `docs/PREFIX5_MIGRATION.md`。今回の調査では実装、ローカルcanister、公開環境を変更していない。
