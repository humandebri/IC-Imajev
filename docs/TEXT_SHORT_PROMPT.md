# 標準text promptの短縮（v2）

2026-10-05。利用者の指定に従ってcanister向け入力生成 `scripts/prepare_text.py` のデフォルトを `text-only-short-v2` に変更した。固定headerから画像・stateの非命令性の一文と情報不足ならunknownを選ぶ一文を削除し、最終unknown選択肢の長い説明を `unknown` のみにした。vendorの固定公式参照・MODEL_LOCK・既存計測fixtureは維持し、新しいfixtureを別に作成。

残る指示：

```text
Inspect the available evidence and answer the question using the stated criteria. Return only the single option code.
```

主入力は元132→94 token（前回unknownだけ短縮した112からさらに18減）、固定prefix45→27 token、毎回のsuffix87→67 token。2文の削除そのものではsuffix67は減らない。attentionが読むprefix履歴とprefix準備は短くなる。

23入力/選択肢順序で削除対象だけを編集した保存promptとtoken IDsが一致。文字列/辞書/空stateでも、State・Question・ユーザー選択肢説明に含まれる同じ文言を削らないことを検証した。chat template・選択肢code/token binding・512-token上限・input hash・prefix一致検査は維持。

旧45-token cacheを新入力に使うと既存のtoken IDs一致検査で拒否される。今回は新しい27-token prefixをcanister queryで準備して3ケースを実計算した。prefix/sessionは既存のupdate実験moduleでは45-token固定のため、この変更をそのまま既存update APIへ渡せない。今回そのmoduleはupgradeせず、一般prefix query経路で確認した。

| ケース | 全token / suffix | 判断 | 命令数（十億） |
|---|---:|---|---:|
| 617 | 94 / 67 | likely | 175.988 |
| insufficient | 87 / 60 | unknown（棄権） | 156.765 |
| maximum | 96 / 69 | no | 181.046 |

主ケース・情報不足・最大値ケースの判断は元の入力と同じ。情報不足で棄権を維持し、最大値ケースの既存誤判定noも残る。2文削除後の実推論評価はこの3ケースだけで、前回unknownだけを短縮した8ケースの検証と混同しない。確率・hiddenの同一性や未知入力での同等精度は主張しない。

今回は新prefixを汎用innovation logで引き継ぎ、専用hybrid packetをまだ作っていない経路のため、各ケース65 query。以前の50 queryや5〜6 updateとの呼び出し効率比較には使わない。prefix準備は別途66 query・75,772,749,319 handler命令。全queryは5B以内、replayed_queries=0。準備費用は各推論の命令数に含めない。単発・ローカル・負荷非統制で時間改善率は主張しない。

計測artifact：`artifacts/text-short-v2/summary.json`、入力、全query要求/返信、各report。計測module `88507a95ea9e6e2111843c5139025d8ce2b3b8e21d9a9957beca4f6f74e103be`。設定不足の初回準備は推論開始前に失敗し、そのjournalを残して新しいprefix-v2 directoryで完了した。

## 同じ結合経路でのquery回数確認

27-token prefixにも元と同じ9共有Kペア・18 dense headの可逆NPF1圧縮を適用。codec入力長の45固定を1..132 tokenの検査へ変更し、専用の新しいローカルcodec canisterで24層分を準備した。旧45-token packetとcodeccanisterは変更しない。rolled graph側のprefix45固定も同じbounded範囲へ変更し、元のrolled/joined/tailのchunk設定で3ケースを実行した。

| ケース | 元の長いpromptのquery | 短縮後query | handler命令（十億） | 最大query命令（十億） |
|---|---:|---:|---:|---:|
| 617 | 50 | 50 | 234.521→180.363 | 3.814 |
| insufficient | 50 | 50 | 214.719→160.710 | 3.400 |
| maximum | 62 | 50 | 232.780→185.515 | 3.923 |

主と情報不足は50→50で、prompt短縮だけでは結合経路のquery回数は減らなかった。50-query経路は固定の段・chunk分割を使っており、計算量が減っても自動で分割数を減らさない。最大値ケースは旧suffix89の標準62-query境界から新suffix69の結合経路へ入り、62→50になった。新入力の汎用経路65 queryからは3件とも50 queryへ減少。主の命令削減23.09%、Candid要求＋返信149,353,220→111,619,406 bytes（25.26%減）。

短縮promptの汎用経路と、各件の全31 exported hidden（layer30はcompact tailで省略）、全32層の保持conv/KV/position、最終norm、logits・確率・判断がbitwise一致した。最大queryは3.923B以下、replayed_queries=0、fallbackなし。実測50回はprefix準備・codec準備・module照会を除く。主50→50なので、命令数をquery予算で割った約36回という値を実測回数と扱わない。回数をさらに減らすには短いsuffixに合わせたquery分割の変更・再検証が必要。

codec準備は別途24 query・4,850,076,243命令・46,652,459 Candid bytes。次回以降は同じpacketを再利用可能。6 cache-policy検証、既存roll3件とtail5件の検証が通過。実queryからも27-token prefixを含む数値整合性を確認した。

旧長promptの比較値は保存済みsingle-quad/tail-proof-v2（module bdd8...）から取得し、新計測moduleは前節のupdate-inference build-v2（8850...）。同じkernel/同じ結合設定を使うが別moduleであり、厳密な単一要因の同一Wasm A/Bではない。新promptでの汎用/結合経路の数値比較は同じ8850... module。速度は負荷非統制・単発で保証しない。

新しい計測記録：`artifacts/text-short-v2/optimized-summary.json`、`optimized-{617,insufficient,maximum}/report.json` と全要求/返信。再実行用 `run_optimized_queries.py` は先に新prefix/packetを準備した環境で使う。codec module `e9644266f9bea8efdff5c1d5017b7c82beb67c3030279e6f27248b90fa51ef6e`、canister `2vn5i-k3777-77775-aaaua-cai`。wrapperと旧compiled依存関係はcodec-buildに記録。
