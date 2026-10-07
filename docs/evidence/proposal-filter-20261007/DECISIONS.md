# BOOM DAO proposal 500〜660：Imajev検査

2026-10-07新規推論。2026-10-06取得済みの161件のsnapshotをhash照合し、ローカルに取り込んだ二択ハーネスで全件の処理経路を再計算。既存proposal_assessmentの重要項目選別・数値抽出・二択参加方針・128-token上限・未校正スコア閾値0.6・承認証拠検査を適用。資金等の証拠不足・対応範囲外はTool hold、Motion/表示変更等は除外。投票は実行していない。

集計: {"complete": true, "total_proposals": 161, "selected": 73, "skipped": 88, "counts": {"hold": 61, "reject": 10, "approve": 2}, "routes": {"skipped": 88, "outside_direct_eligibility_scope": 17, "participation_model": 50, "evidence_gate": 5, "budget_overflow": 1}, "model_distinct": 18, "evaluated_distinct": 18, "reused_distinct": 0, "new_distinct": 18, "accuracy_measured": false}

既存の予測結果とprefix状態は再利用していない。モデル対象の同一token列のみ今回の実行内でまとめて新規推論した。全161件が独立した新規モデル推論という意味ではない。過去GPTラベルや採決結果は入力しない。正解ラベルがないため正答率は測定しない。

旧値はproposal rendering由来でchain検証なし。判定対象は数値上の投票参加条件であり、proposalの安全性全般の承認ではない。128 tokens以内でも32 query保証はない。

|ID|処理|tokens|二択|スコア|最終判定|理由|
|---|---|---:|---|---:|---|---|
|500|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|501|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|502|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|503|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|504|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|505|participation_model|128|reject|0.8699|reject||
|506|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|507|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|508|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|509|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|510|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|511|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|512|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|513|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|514|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|515|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|516|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|517|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|518|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|519|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|520|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|521|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|522|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|523|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|524|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|525|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|526|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|527|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|528|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|529|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|530|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|531|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|532|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|533|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|534|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|535|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|536|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|537|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|538|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|539|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|540|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|541|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|542|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|543|outside_direct_eligibility_scope|0|—|—|hold|Outside numerical stake/voting-lock participation scope.|
|544|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|545|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|546|skipped|0|—|—|除外|only unchanged parameters in projection|
|547|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|548|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|549|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|550|skipped|0|—|—|除外|only unchanged parameters in projection|
|551|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|552|participation_model|112|reject|0.8145|reject||
|553|participation_model|67|reject|0.6996|reject||
|554|participation_model|68|reject|0.7780|reject||
|555|participation_model|68|reject|0.7103|reject||
|556|participation_model|68|reject|0.7103|reject||
|557|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|558|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|559|participation_model|88|reject|0.7529|reject||
|560|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|561|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|562|participation_model|123|approve|0.5354|hold||
|563|participation_model|88|reject|0.7529|reject||
|564|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|565|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|566|participation_model|123|approve|0.5354|hold||
|567|participation_model|88|reject|0.7529|reject||
|568|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|569|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|570|participation_model|110|reject|0.5883|hold||
|571|participation_model|110|reject|0.5883|hold||
|572|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|573|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|574|participation_model|110|reject|0.5883|hold||
|575|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|576|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|577|participation_model|110|reject|0.5883|hold||
|578|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|579|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|580|participation_model|110|reject|0.5883|hold||
|581|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|582|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|583|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|584|evidence_gate|0|—|—|hold|treasury capacity/control/recipient relationship evidence missing|
|585|budget_overflow|0|—|—|hold|All changed values preserved; input exceeds 128 tokens.|
|586|participation_model|126|reject|0.5344|hold||
|587|participation_model|126|approve|0.5165|hold||
|588|participation_model|126|approve|0.5036|hold||
|589|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|590|participation_model|126|reject|0.5498|hold||
|591|participation_model|126|reject|0.5498|hold||
|592|participation_model|126|reject|0.5498|hold||
|593|participation_model|126|reject|0.5498|hold||
|594|participation_model|124|reject|0.5176|hold||
|595|participation_model|124|reject|0.5176|hold||
|596|participation_model|124|reject|0.5176|hold||
|597|participation_model|124|reject|0.5176|hold||
|598|participation_model|126|reject|0.5207|hold||
|599|participation_model|126|reject|0.5207|hold||
|600|participation_model|126|reject|0.5207|hold||
|601|participation_model|124|reject|0.5176|hold||
|602|participation_model|67|reject|0.5385|hold||
|603|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|604|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|605|participation_model|67|reject|0.5385|hold||
|606|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|607|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|608|participation_model|67|reject|0.5385|hold||
|609|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|610|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|611|participation_model|67|reject|0.5385|hold||
|612|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|613|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|614|participation_model|67|reject|0.5385|hold||
|615|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|616|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|617|participation_model|97|reject|0.6865|reject||
|618|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|619|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|620|participation_model|67|reject|0.5385|hold||
|621|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|622|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|623|participation_model|67|reject|0.5385|hold||
|624|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|625|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|626|participation_model|67|reject|0.5385|hold||
|627|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|628|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|629|participation_model|67|reject|0.5385|hold||
|630|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|631|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|632|participation_model|67|reject|0.5385|hold||
|633|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|634|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|635|participation_model|67|reject|0.5385|hold||
|636|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|637|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|638|participation_model|67|reject|0.5385|hold||
|639|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|640|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|641|participation_model|67|reject|0.5385|hold||
|642|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|643|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|644|participation_model|67|reject|0.5385|hold||
|645|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|646|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|647|participation_model|67|reject|0.5385|hold||
|648|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|649|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|650|participation_model|67|reject|0.5385|hold||
|651|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|652|skipped|0|—|—|除外|display/branding-only fields excluded by local policy|
|653|evidence_gate|0|—|—|hold|mint concentration/control evidence missing|
|654|evidence_gate|0|—|—|hold|treasury capacity/control/recipient relationship evidence missing|
|655|evidence_gate|0|—|—|hold|treasury capacity/control/recipient relationship evidence missing|
|656|participation_model|78|approve|0.8763|approve||
|657|skipped|0|—|—|除外|non-executing motion excluded by local policy|
|658|evidence_gate|0|—|—|hold|executable content missing or payload sources conflict|
|659|participation_model|101|approve|0.8625|approve||
|660|skipped|0|—|—|除外|non-executing motion excluded by local policy|


新規実行の集計: {"fresh_inferences": 18, "reused_predictions": 0, "completed_prefix_banks": 11, "inference_queries": 774, "preparation_queries": 990, "prior_logits_bit_equal": 18, "changes": []}
