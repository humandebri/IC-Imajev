# クライアントによる入力長別query分割

後続の結合query対応では39/39/40回まで減少した。詳細は [PACKED_QUERY.md](PACKED_QUERY.md) を参照。この文書は既存moduleの47 query経路を記録する。

2026-10-05。`client/adaptive_inference.py` の `AdaptivePrefixGraph` を追加し、既存の `run_prefix_canister.py` に `--adaptive-start` を追加した。Wasm・重み・prefixは前回と同じで、canister upgradeなし。クライアントはtoken数からMLP generation行数、Delta head数、kernelの組合せを選ぶ。モデル演算・norm・状態更新・readoutは既存canister query内で行い、クライアントは可逆framing・選択・保存だけを行う。

`short-query-v2` は検証済みmodule `88507a95ea9e6e2111843c5139025d8ce2b3b8e21d9a9957beca4f6f74e103be`、prefix1..27、suffix1..69で短い入力用経路を選ぶ。実推論検証はprefix27とsuffix60/67/69。その他のmodule・長いprefix・長いsuffixでは従来のTailPrefixGraphへdispatchする。予算を任意の未知moduleに転用しない。未知入力での厳密なworst-case命令保証ではなく、上限/frame超過時は既存の標準query fallbackを使う。

分割は初期MLP frontを256行境界、Delta head数を共有Kの2-head境界へ合わせてtoken数から選ぶ。full MLPと次のDeltaを結合するqueryは別の小さいhead数を使う。MLPのgeneration carry/down carry/Attention出力の状態に応じて、全Attentionを完了する経路や次層のgenerationまで進める経路を選び、不要なquery境界を取り除く。整数・BF16境界と演算順序は既存kernelを維持する。汎用DAGの自動探索やオンラインの命令予測学習ではない。

| ケース | suffix | 短縮promptの固定経路 | 動的経路 | 最大query命令（十億） | 命令合計（十億） |
|---|---:|---:|---:|---:|---:|
| 617 | 67 | 50 | 47 | 4.543 | 180.363→181.281 |
| insufficient | 60 | 50 | 47 | 4.047 | 160.710→161.613 |
| maximum | 69 | 50 | 47 | 4.515 | 185.515→186.512 |

3ケースとも固定50→47 query（6%減）。元の長いpromptと比べると主/情報不足50→47、最大入力62→47だが、prompt変更と分割変更の寄与を混同しない。主の総命令は短縮prompt固定経路比で約0.51%増、Candid通信111,619,406→113,340,950 bytes（約1.54%増）。呼び出し減が計算量/通信量/速度の最小化を同時に意味するわけではない。単発時間は約27〜31秒で負荷を統制していないため速度改善率は主張しない。

全31 exported hidden（compact terminalではlayer30は未export）・全32層の保持conv/KV/positions・最終norm・logits・確率・判断が、新promptの汎用経路とbitwise一致。固定50回経路も同じ汎用結果と一致している。replayed_queries=0、最終v2のfallback/失敗query=0、各query5B以内。情報不足の棄権と既存maximumの誤判定noも維持。精度改善ではない。

初期v1ではfull-MLP結合時に20 headsを使い、suffix69で5Bを超えた。その失敗はadaptive-maximum/fallback.jsonに保存され、5成功+1失敗+62標準queryの合計68回として記録された。成功queryだけを数えて47回と扱っていない。v2ではfull-MLP結合を67 tokenで16 heads・69 tokenで14 headsへ減らして再測定。3ケースすべて新しいjournalで完走した。

4 planner/dispatch検証は未測定module・大きい入力を固定経路へ送ること、prefix不一致をquery前に拒否すること、分割のalignment、adaptive flagを標準fallbackから除外することを確認。既存のfallback3件も通過し、失敗queryの集計とinput/module/option identityを維持した。Python構文・差分空白検査も通過。

利用は既存のtail/rolled/joined設定へ `--adaptive-start` を追加する。デフォルトの固定経路は維持。利用者が分割数を手動指定する必要はない。計測設定・実行例は `artifacts/text-short-v2/run_adaptive_v2_queries.py`、最終結果は `adaptive-v2-summary.json` と `adaptive-v2-{617,insufficient,maximum}/report.json`。各要求/返信も保存。prefix準備・packet準備・module照会は推論回数に含めない。

この対応は既存kernelで表現できるcarry状態間の結合に限定される。queryを任意にまとめたり、canisterの5B命令上限を変更したりするものではない。未測定形状ではまず固定経路へ戻る。
