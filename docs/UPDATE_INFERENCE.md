# 中間状態をcanisterに保持する連結update推論

最新の最適化後は各3回の実測で **5 / 4 / 5 update**。詳細は [UPDATE_TEMPLATES_MEASURED.md](UPDATE_TEMPLATES_MEASURED.md)。以前の短縮版は **6 / 5 / 5 update**（[UPDATE_BOOMDAO_SHORT.md](UPDATE_BOOMDAO_SHORT.md)）。

2026-10-05。50 queryを50 updateとして再送する比較から進め、update専用入口で通常のtext graphを内部実行した。固定45-token prefixを一度登録し、token IDsから新たに推論する。保存した質問依存hiddenや返信を入力に使わない。

## 実測結果

| 条件 | 推論update数 | handler命令合計 | 最大update handler命令 | Candid要求＋返信bytes | 呼び出し時間合計秒 | 観測cycles残高減少 |
|---|---:|---:|---:|---:|---:|---:|
| 617（suffix 87） | 7 | 230,817,253,418 | 37,195,224,905 | 31,878 | 51.015 | 230,979,392,403 |
| insufficient（suffix 80） | 6 | 211,172,250,990 | 37,768,813,878 | 28,354 | 45.848 | 211,224,314,432 |
| maximum（suffix 89） | 7 | 236,028,015,932 | 38,035,146,912 | 31,856 | 45.370 | 236,087,970,233 |

全3条件の32層suffix hidden、保持conv/KV、最終norm、readout logits、校正確率、判断が採用済みINT8参照と一致した。配列はSHA256で全bytesを比較し、最終normと判断も検証する。参照にはprefix行を含む通常層と最後の1 tokenだけの最終層があり、それぞれ同じ範囲へ合わせた。既存の最大変更問題でのモデル誤判断も同じで、精度改善ではない。

主問題は132 token＝prefix45＋suffix87、rotations=1。前回の50 update再送に対して呼び出し数50→7（86%減）、通信149,353,220→31,878 bytes（99.9787%減）。handler命令234,521,052,335→230,817,253,418（1.5793%減）。観測cycles残高差401,269,034,430→230,979,392,403（約42.44%減）。計測moduleとgraphが異なるため、通信往復除去だけの寄与を分離した削減率ではない。

呼び出し時間は前回53.302秒、今回主51.015秒。今回も単発で、主の第5呼び出しだけ14.405秒、他の大きい呼び出しは約7秒と変動した。負荷を統制しておらず、速度改善率は主張しない。本番subnetの時間・料金も未測定。

## 実行方式

feature `experimental-update-inference` が次のowner-only updateを追加する。

- `update_prefix(layer, values, packet)`：各層の固定prefixを登録。Deltaはconv履歴＋可逆NPF1 packet、Attentionはhead-major K/V。形状・有限値・BF16境界・packet復号を検査し、一度登録した層は変更できない。推論開始後はprefix全体を固定する。
- `update_infer_start(ids, options)`：1〜89 suffix tokenと2〜7選択肢を検査し、embedding、初期norm、複数段のAttention/Delta/MLPを実計算する。
- `update_infer_continue(id, stage)`：canister内のhidden/norm/attentionを引き継ぎ、指定進捗と一致する場合だけ続行する。クライアントは中間配列を入力できない。

既存のfull Delta、full Attention、full MLPと同じINT8基底/F32 LoRA演算を使い、正式なBF16境界・加算順序・calibrationを維持する。layer31では最後のQと最後のMLPだけを計算する。各updateは34B命令を超えた段の完了時に区切り、次の段は後続updateへ送る。検査・hash・prefix復号もhandler計測に含め、全実測呼び出しが40Bの上限内に収まった。6回という算術上の見積もりに対し、主と89 tokenは実装の段境界と余裕により7回となった。

返答は進捗、各段の命令、比較用のhidden/state hash、完了時だけ最終normと判断。途中の大きいhiddenを返信せず、続行要求はIDと進捗のみ。全50 requestをまとめて持ち込む方式ではない。セッションは一つで、進行中の別startは拒否し、完了後は次の入力を開始できる。計算エラーはtrapし、そのupdateの途中変更をrollbackする。今回の失敗・再送は0。

## 準備・検証の別集計

upgrade後の固定重み準備は721 owner update・78,739,929,194 handler命令。heap cache4,065,416,192 bytes。準備scriptの最初の確認は512行以上だけを期待する条件で失敗したため、paired-only moduleに合う `--require-all-output-pairs` で721 tensorの既存cacheを再確認した。重みの再準備は0で、配置・容量・module確認を通過。

固定prefix登録は32 update・Candid 39,846,913 bytes・呼び出し合計8.615秒。これはmodel/module準備時の費用で、上表の推論呼び出しに含めない。prefixは旧canister計算から得た固定検証済みstateで、質問依存の回答やhiddenを事前登録していない。

検証専用update2件で、完了したstage64からの続行と空token IDsを拒否した。status検査・module検査・この2件は推論回数と時間から除外。各条件のcycles残高差には経過時間に伴う維持費とstatus照会費用が混ざるため、純粋な実行料金や本番価格として扱わない。最大観測heap終了値は4,165,140,480 bytesで4GiB内。ただし一時peakを保証する値ではない。

実NPF1 packet、非BF16/非有限入力、不一致prefix長、切断packet、入力長不足のruntimeテスト1件が通過。update featureのcanister native検証は既存7 unit通過。計測scriptのPython構文確認と差分空白検査も通過。ローカル実行であり複数ノードの合意検証ではない。

## 証跡・制約

専用実験canister `6eydd-o3777-77775-aaama-cai`、module `88507a95ea9e6e2111843c5139025d8ce2b3b8e21d9a9957beca4f6f74e103be`。採用canisterをupgradeしていない。`artifacts/update-inference/proof-v1/report.json` と各条件の呼び出しJSONに全測定・比較結果・前後module hash・固定参照hashを保存し、終了時もmoduleと参照の不変を確認した。

共有checkoutで別の変更が進んでいたため、`source-v2.zip` にコードと依存関係を固定し、コピーしたsourceからWasmと計測bridgeをビルドした。Wasmの4 ABI stubを従来の同一WATでpatchし、wasmparser検証を通過。build-v2/raw.wasm と full.wasm が計測対象。後から追加したnative用heap_pages cfgはeditable sourceとportable-update-inference.rsに保存し、計測moduleのsource archiveと区別する。この変更の再ビルドは実験canisterに未適用。

今回の対応範囲は固定45-token prefix・1〜89 suffix token・一つのownerセッション。prefix/sessionはheap保持で、upgrade後は再登録・新規推論が必要。upgradeをまたいだセッション再開、複数利用者の同時セッション、timerによる自動続行は未実装。実験結果をそのままpublicな本番APIとして扱わない。

準備済みmodule/cache/prefixがある場合：

```sh
.venv/bin/python scripts/measure_update_graph.py \
  --wasm artifacts/update-inference/build-v2/full.wasm \
  --directory artifacts/update-inference/proof-new
```

prefix未登録の新しいupgrade後は `--prepare-prefix` を付ける。同一moduleにprefixを二重登録しない。モデルの準備・読み出し・module検査は別手順。元の50/50/62 queryの計測は [SINGLE_TOKEN_QUAD.md](SINGLE_TOKEN_QUAD.md)、前回50 update再送は [REPLICATED_INFERENCE.md](REPLICATED_INFERENCE.md)。
