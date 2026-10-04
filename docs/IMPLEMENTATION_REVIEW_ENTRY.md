# 入口を含む連結検証のレビュー

2026-10-04。直近のtyped reply、half128と連結検証をレビューした。演算経路の公開数値API・内部opaque型・request全fieldの拘束・block256量子化・F32積和順を確認。型を外部から構築できず、通常queryの入力検証を通ることを維持している。

## 修正した問題

- 同じbinary/NPZ参照の再読込がhashを上書きし、途中変更を見逃す場合があった。`read_bytes`が最初のhashを固定し、その同じbyte列をdecode/NPZへ渡す。再読込の変更と終端の変更は拒否する。
- 新しい入口を使う試験で、入口head/front・成功query・profileが成功caseへ記録されなかった。入口2queryを別項目で保存し、五連結5queryの集計と区別する。全CLI設定もreportへ保存する。
- 数値範囲の誤りを診断query発行後に検出していた。head、MLP前半、down行、圧縮設定を発行前に検証する。half128を許可しblock256量子化は維持する。
- source snapshotにcanister source/Cargo manifest/lock/build scriptが欠けていたため追加。Wasmファイルの終端hashも検査する。Python `-O`は出力一致のassertを無効化するため、checker起動時に拒否する。

## 検証と制約

参照変更・設定境界の5テスト、Python compile、diff checkが通過。実ローカルmodule `500beb090e7cd61083b44138c3be85ab2c140832d7f80ca0948b58cc21b2b7bc` に対し、80 token/layer1、入口20head/front4096/down1408、五連結front1792/6272・26headを検証した。入口2query＋五連結5queryのhidden/norm/carry/convが固定参照とbit一致、上限超過0。入口は4,500,759,289／4,535,049,343命令。`artifacts/review_entry/entry20-info-v2` に設定・全metric・profile・source/reference hashを保存。

最終checkerを固定した再実行でも一致。`-O`での起動拒否を別途確認した。

主87 tokenの同じ入口20/22headは直前の実験でIC0522、証跡は `artifacts/roll_entry/entry{20,22}-down1408-main-v1`。80 tokenの成功を主入力の成功や全体query削減に換算しない。通し主入力は54query、50/32未達。次はDelta/MLP pair返信に残るINT8→F32→INT8展開・再走査を削減する。
