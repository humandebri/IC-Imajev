# 連結queryの実装レビューと境界測定

2026-10-04。固定INT8 base、元F32 adapter/readout/scales、既存BF16境界と積和順序を維持した。中間状態はclientが保持し、推論は通常queryだけで行う。

## 修正

分割Attentionのdecodeでprefixをsplitして再結合する重複コピーを除き、最終配置へ直接BF16を展開した。旧形式とQ4形式の配置を維持する。

五連結の検証driverは後段で失敗した場合、前段の成功queryの命令数・profileをreportへ残していなかった。各成功queryのmetricとprofileを個別保存し、reportにも全測定を残すように修正した。独立参照生成queryも記録するため、report全測定を五連結本体の費用として合算しない。失敗queryは上限超過requestを保存し、成功命令数を捏造しない。

## 実験API

`mlp_finish_delta_log_mlp_front` は現MLP完了・次Delta全体・次MLP前半を連結する。残差hidden、MLPの未丸めF32 carry、Delta convを返す。`delta_partial_mlp_full` は残りDeltaと次MLP全体を連結しhidden/normと残りconvを返す。既存APIの形式を変更せず、実験APIとして追加した。現行全体graphへはまだ接続しない。

## 実測と限界

実験module SHA256 `df16f9e4c134b3e889e1bfce139c13241a13debf44856b8200468456db56710e`。情報不足80 tokenのlayer1起点、down1792行・MLP前半1792行・Attention後MLP5120行・Delta先行26 headで五連結が全ビット一致した。先行24 headも一致した。layer1だけの検証で、全層・全判断の改善とは扱わない。

主87 tokenでdown1792・前半1792・Attention後5888・先行24 headを再測定した。前半4 queryは参照hidden/norm、MLP carry、convとビット一致。命令数は順に4,722,219,281、4,649,850,246、4,880,950,763、4,815,444,802。第五のDelta残り8 head＋MLP全体はIC0522で失敗した。前処理を別queryで用意した部分測定なので、前段全体のスケジュール成立も証明していない。

先行head増加は後段を軽くするが、MLP carry＋prefixの2MB frameと前段5B命令上限が制約になる。89 tokenでは5120行・26 headのframeが上限を超えた。単純なquery連結だけで50/32 query達成を宣言しない。

証拠: `artifacts/prefix_codec/roll-join-chain-{v1,down-v2,main-wide-v3,review-v4}`。review-v4では途中成功metric/profileを個別保存した。生成物はignore。

## 検証

追加Rust境界テスト2件を含む実験featureセット125 unit、15 integration、9 doctest通過（fixture依存1件ignore）。Python codec21件、journal6件、追加join境界2件通過。既存全6条件の全推論と連結3条件の回帰は別証拠として追記する。

Laya、保護対象の主canister、mainnet、remoteを変更していない。既存INT8とのbit一致はBF16参照との判断精度同等性ではなく、重大変更の見逃しも解消していない。

全6条件を `artifacts/prefix_codec/full-roll-joins-review-v2`、現行連結3条件を `artifacts/prefix_codec/rolled-joins-review-v1` で再実行し、保存対象hidden/state/final hidden/型付き判断/logits/確率が全ビット一致した。いずれもmodule/sourceを前後確認しsource ZIPを保存した。主連結54 query・241,643,997,534命令・133,896,992 Candid bytes、情報不足54 query・221,143,011,943命令、最大変更fallback62 query・240,511,224,211命令。失敗/replayは0。並行実行したためwall timeは速度比較に使わない。cold132 tokenも290 queryで完走した。追加APIはこの全体graphでは未使用なので、全体回帰だけで追加APIの全層対応を証明したとは扱わない。

保護対象主canisterはRunning、module `36c04a57e9501eb9d32163c92d7725171191fa46a45a880ad03465993fceed73` のまま。
