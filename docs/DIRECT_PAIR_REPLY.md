# Delta/MLP入口の直接返信

`experimental-direct-mlp-reply`をDelta/MLP completion側にも適用。従来の`Preparation::flat`による量子化入力のF32展開、全出力有限値走査、encoderでの整数再検査・INT8復元を省く。checked `QuantizedRows`をそのままINT8へ書き、生成されたhidden/historyのBF16・有限値、F32 A/baseの有限値、gate範囲とshapeを検査する。

通常の数値APIとfollow queryは従来経路。opaque返信は内部だけで構築され、元request全fieldとstep+1に拘束される。既存tag0・frame version・checksumと返却byte列を維持し、量子化block256・F32積和順・BF16境界を変更しない。

## Native検証

最小featureの関連9テストが通過。全featureは134 unit（1 ignored）、15 integration、10 docが通過。直接返信は1/7/80/87/89 token、18/20/26 head、frame1/2/3・通常/owner署名checksumで従来encoderとbyte一致。不正shape・非BF16・非有限値・gate・directionを拒否する。

最終build `artifacts/direct_pair_reply/full-build-v1` はsource bookend一致、4つのWAT patchはwasmparser検証済み。module `35633dff491ebc102115a7f70294516fa01b05c2c28840619c37fc07b28db8af`。専用ローカル実験canisterだけへupgradeし、固定重みcacheをupdateで準備する。

## 部分queryの実測

87 tokenのMLP完了＋Delta26headは旧4,877,795,556→新4,875,523,546命令で、2,272,010命令減。wire_encodeだけなら45,755,343→1,727,859だが、payload構築がevaluate側へ移るため、44Mの全体削減とは報告しない。shape・演算出力・通信形式は同じ。

同じ入口front4096/20headは直接返信でもIC0522。front4352へ前倒しすると最初は4,834,719,202命令で通るが、次のdown1408は超過した。down1280へ移すと入口2query＋五連結5queryが主87 token/layer1で成立し、hidden/norm/carry/convが固定参照とbit一致。入口2queryは4,834,719,202／4,893,679,130命令、五連結最大4,899,014,496。これは配分変更を伴う診断で、直接返信だけの効果とは扱わない。

冒頭Deltaを含めた`entry-start-all-v1`は要求21条件、成功13・失敗8。主87はlayer5/9/13/17/21/25の6箇所、情報不足80は全7箇所で8queryが成立。成功条件の最大queryは4,922,437,243命令。主87の最初はIDs/embedを含むstartがIC0522、最大変更89は7箇所ともstartがIC0522。すべての成功出力は参照bit一致で、失敗要求も保存した。全体graphへの接続とquery削減は未検証。

`--entry-start`はlayer0だけIDsを渡しcanisterがembed/normする。他の層は参照の直前MLP hidden/normをそのままframe化する。クライアントで追加の推論演算をしない。独立準備のqueryと実際の入口3query、五連結5query、profile queryを分けて記録する。

追加レビューでfront128境界を過剰に許していたCLIガードを修正。最初のMLP-front連結とDelta startは現在256単位のため、attention_frontとstartを含まないentry_frontの128単位と区別する。対応する境界テストが通過。

## 通し回帰

既存prefix cacheはWasm hashに拘束され、旧moduleのcache再利用をquery発行前に拒否した。その失敗記録を`rolled-proof-v1`に残し、新moduleでprefixを作り直した。

`full-proof-v1`標準6条件と`rolled-proof-v2`連結3条件が全保持hidden/state・判断・確率で既存INT8参照とbit一致。失敗/replayは0、source bookendとZIP保存は成功。さらに旧候補の同じ170通常queryについて要求・返信の340 frameが全byte一致し、`rolled-byte-comparison.json`に各hashを保存した。

| 連結graph | query | 合計命令 | Candid bytes | 最大query命令 | 単回時間s |
|---|---:|---:|---:|---:|---:|
| 主87 token | 54 | 240,513,749,852 | 133,896,992 | 4,900,656,355 | 37.534 |
| 情報不足80 token | 54 | 220,021,215,405 | 125,114,808 | 4,493,001,266 | 33.552 |
| 最大変更89 token・fallback | 62 | 240,031,063,831 | 125,231,193 | 4,816,569,724 | 35.228 |

旧module比で主29,270,767命令減、情報不足20,937,856減、最大変更は19,949増。最大変更は直接返信を使わず分岐追加後の微小増として記録する。query/通信は同じ。主の観測heap終了最大は4,144,037,888 bytesで同じ。単回時間から速度改善は断定しない。

主入力は共通prefix45＋suffix87＝132 token、rotations=1。新しい8query連結は全体へ未接続、主は54query、50/32未達。次に調べる箇所はIDs/embedを含む最初のDelta/MLP start返信のF32展開と、後続MLP/Attention境界の配分。元BF16モデルに対する判断精度とは別の回帰であり、最大変更の見逃しを解消したものではない。

保護対象`4caro-hl777-77775-aaaba-cai`はRunning、module `36c04a57e9501eb9d32163c92d7725171191fa46a45a880ad03465993fceed73`のまま。Layaは参照のみ。生成物は既存gitignoreの対象で、mainnet/push/PRは行っていない。
