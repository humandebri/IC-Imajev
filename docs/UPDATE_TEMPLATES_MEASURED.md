# 最適化後のupdate推論実測

2026-10-06。最新query32候補の演算最適化と投票用38-token prefixをupdate推論へ適用し、BOOM DAO 617/620/653を各3回、全9推論で計測した。推論のupdate回数は **5 / 4 / 5**。以前の実測 **6 / 5 / 5** に対して投票2件は各1回減り、653は回数を維持した。

|提案|prefix / suffix|以前→今回update|handler命令中央値|以前からの命令削減|Candid要求＋返信中央値|呼び出し時間中央値|
|---|---:|---:|---:|---:|---:|---:|
|617|38 / 56|6→5|145,508,909,711|18.213%|27,428 bytes|28.676秒|
|620|38 / 48|5→4|125,615,520,758|20.183%|24,063 bytes|23.797秒|
|653|27 / 57|5→5|146,527,267,281|3.771%|27,432 bytes|27.996秒|

以前の比較対象は [UPDATE_BOOMDAO_SHORT.md](UPDATE_BOOMDAO_SHORT.md) と `artifacts/update-short-v1/summary.json`。新しい全9推論で回数が一致し、再送・失敗は0。新規測定の推論updateは合計42回で、以前の9推論48回から12.5%減った。初回準備・境界確認・中断した初版の測定はこの42回へ含めない。

モデルの重み・入力本文・質問・選択肢・calibrationは変更していない。投票2件では固定文の「Minimum voting dissolve delay changes from 1 day to 」までを再利用し、新しい日数・倍率・質問を毎回推論する。mintの653には共通27-token prefixを使う。全9推論で、元queryが保存した31層のsuffix hidden、全32層のconv/suffix KVのhash、最終normalized hidden、判断・校正確率・unknown確率・logitsがbit一致した。layer30 hiddenと未exportのDelta密状態は直接比較していない。全3入力の回答はlikely。

updateは既存のserver-held graphを使用する。開始要求にsuffix token IDsと選択肢を渡し、続行要求はsession IDとstageのみ。canister内の中間配列を保持し、約34 billion命令を超えた処理段の終了時に区切るschedulerは以前の実装と同一ソースを再利用した。queryを32回へ分割するフロント側の融合は使っていない。

最大update handler命令は **35,865,130,885**、最大観測heapは **4,204,265,472 bytes**。全42回が実際に成功し、handler counterは40B未満、要求と返信はそれぞれ2MB未満、heapは4GiB未満。handler命令はCDKのdecode/encodeを含まず、通信量はCandidのみでHTTP/CBOR/署名を含まない。時間はローカル実測の中央値で、負荷を統制した本番速度比較ではない。

今回のupdate実装は一つの固定prefix bankを登録し、推論開始後は変更を拒否する。投票2件と653は同じ候補Wasmを2回upgradeして別々に準備・計測した。各bankの準備は重み721 update、共通27-prefixの密状態cache 24 update、graph prefix登録32 update。prefixの事前計算自体は既存の検証済みquery成果物を再利用し、各質問の推論回数には含めない。各bankの不正prefix 3 update・不正開始/続行2 updateも別集計。任意の入力や本番で5/4/5になる保証ではない。

初版 `proof-v1` は投票4推論の完了後、3回目の617でローカルcycles凍結閾値に達して中断し、snapshot復元も一度拒否された。ローカル実行用cyclesを10兆補充し、元module・重みcache・packをsnapshotで復元・照合してsnapshotを削除した。記録は `proof-v1/voting/measurement/failure.json` と `proof-v1/recovered.json`。採用結果は全9推論を最初からやり直した `proof-v2` のみ。補充はlocalhostの開発ネットワークで、本番cyclesの支出ではない。

測定後に元module `6052cc94…` と721項目の重みcache、packをsnapshotで復元・照合し、今回のsnapshotを削除した。比較canisterは元moduleでRunning。既定・本番へは導入していない。

候補module: `de7f5f07a045caaee173ab560777ab7f7c2c91fb4a57b672700ca3d08abf169a`。検証済みquery32 runtimeのrlibと6つのWAT kernelを再利用し、可変prefix対応の旧update wrapperを組み合わせた。wrapperのoverflow検査と外部入力検査を維持している。

- ビルド: `scripts/build_update_templates_full.py`、`artifacts/update-templates-v1/build/report.json`
- snapshot保護・実行: `scripts/prove_update_templates.py`
- 各呼び出しの計測: `scripts/measure_update_templates.py`、`proof-v2/{voting,common}/measurement`
- 保存返信からの独立検証: `scripts/report_update_templates.py`、`proof-v2/{verified.json,verification.log}`
- 最終ローカルstatus: `artifacts/update-templates-v1/final-local-status.txt`
