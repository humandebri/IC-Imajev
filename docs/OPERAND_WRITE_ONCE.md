# S1入力作業領域の二重書き込みを除く

2026-10-03。通常queryで毎回作るS1入力係数bufferのゼロ埋めを除いた。固定重み・入力量子化・演算順序・中間状態形式は変更しない。

## 実装と安全性

`strassen_raw::Operands::new`は`vec![0;len]`をやめ、capacityだけ確保したVecへ全係数を書き込む。書き込み完了後にだけlengthを確定する。nativeはspare capacityのMaybeUninitへwriteし、Wasmは書き込みだけのSIMD helperを使う。未初期化要素を参照しない。

opaqueなQuantizedRowsのconstructorはcolsの256整列、rows上限512、cols上限262144、8tokenへの初期化済みpaddingを保証する。各係数・token pair・blockに256要素の独立範囲があり、k=0,4,…124の32回×8要素のstoreで全範囲を一度だけ初期化する。奇数token末尾もpaddingから安全に読める。検証testは1/7/8/45/87/132/512token、256/512colsで全係数、書き込み範囲の完全被覆、同じQuantizedRowsの係数再利用を確認する。

## 独立診断

同じmodule内で旧column32、S1ゼロ埋めあり、S1ゼロ埋めなしを比較した。元のINT8 layer3 Q、8192×2560。5実入力＋6境界入力、33通常queryで全出力digestがnative参照と一致。主入力87tokenは準備3,439,580→1,862,501命令、投影全体1,005,678,014→1,004,100,935命令（1,577,079 /0.15682%減）。これはQ単体の実測であり、全モデルの削減率ではない。

証拠 `artifacts/wat_s1_nozero/check-fixed-source/report.json` / `validated-source.zip`。最初の試行は測定中のソース変更をbookend checkが拒否したため根拠に使わず、同じmoduleを再初期化してソース固定の再測定を完了した。kernel・依存rlib・モデル・入力・moduleのhashを保存した。module54e70d31…。

再現は`build_wat_s1_nozero_cached.py`→既存raw S1 WATのbody patch→独立したローカルcanisterのreinstall→`check_wat_s1_nozero.py --canister <scratch ID> --directory <fresh path>`。generated artifactsとbench targetはignore。

## 全モデル

検証用候補module5ff4e16d…をビルドした。runtime86 unit、9 integration、3 compile-fail doctest、canister8 unit通過。入力準備buffer全体の初期化を確認するtestを追加した。以下に全モデルの測定結果を記す。


全5条件＋旧log対照を完走し、全保持hidden/state/最終hidden、判断value/abstain/logits/確率/unknownが旧INT8版と一致した。型付き出力の検査通過、graph失敗/replay0。主87tokenは45tokenのprefixを再利用する。重大変更を見逃す既存の判断は残り、判断精度の改善を意味しない。

|条件|query|handler命令|前候補から削減|Candid bytes|最大query命令|秒・単回|
|---|---:|---:|---:|---:|---:|---:|
|prefix45|66|130,311,746,653|189,938,944 (0.1455%)|71,105,691|2,283,394,708|17.154|
|主87・旧log対照|65|249,650,175,306|353,828,218 (0.1415%)|113,687,128|4,301,761,566|25.603|
|主87|63|247,817,228,232|353,828,218 (0.1426%)|124,624,018|4,301,761,566|20.599|
|情報不足80|63|226,584,777,243|321,686,906 (0.1418%)|117,690,970|3,931,672,916|26.233|
|最大変更89|63|253,396,661,543|361,863,546 (0.1426%)|126,604,858|4,397,932,787|29.145|
|cold132|290|385,348,681,343|693,832,994 (0.1797%)|580,207,902|2,386,652,826|52.294|

主は353,828,218命令（0.14257%）減、247,817,228,232命令。全条件で減り、query数と通信量は変わらない。63 queryで50/32目標は未達。handler counterはCDKのCandid処理を含まない。時間は単回で、実行速度の改善率には一般化しない。

固定cache721 tensors /4,065,416,192 bytes、観測heap終了値最大4,131,258,368 bytesを維持した。瞬間peakの測定ではない。準備721 update /78,739,929,194命令 /275.024秒は推論と分離する。prefix packet二回目の準備query/命令/通信0、bridge binary hashの不変も確認。通常queryに質問状態を永続保持させない。

証拠 `artifacts/prefix_codec/full-nozero-proof/report.json` / `before-after.json` / `validated-source.zip` / `codec-second.json` / `client-binary-check.json`、固定準備`full-nozero-preparation/report.json`、build/patch/source hashと`change.patch`は`full-build-nozero/`。主canister36c04a57…のmodule不変を`main-unchanged.json`で確認。Layaは変更していない。

全モデル再現は[前候補](TERMINAL_DECISION.md)と同じbuild flags、3body patch、固定cache準備を使い、保存先を新規directoryにする。

```sh
.venv/bin/python scripts/check_full_prefix_hybrid.py \
  --canister 6eydd-o3777-77775-aaama-cai \
  --codec-canister 7hukf-2d777-77775-aaakq-cai \
  --wasm artifacts/prefix_codec/full-build-nozero/full.wasm \
  --directory artifacts/prefix_codec/recheck-nozero \
  --terminal-attention --terminal-decision --prefix-start
```

さらに大きな削減を探る対象は、固定重みの初回展開とkernel内部の共通アドレス計算、query間で持ち越せる準備済みoperandの復元である。[S2共通変換再利用](S2_SHARED_TRANSFORMS.md)は現行S1を上回らず未採用。queryをまとめるには別途stage/packet境界の検証が必要で、今回の合計命令÷5Bを実query数とは扱わない。
