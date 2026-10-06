# 各queryのframeハッシュをクライアントへ移管

2026-10-03。推論は通常query、中間状態はクライアント保持のまま。`experimental-host-checksum` と `--frame-checksum host` はframe version3を使う。主245,111,663,149→243,089,913,177 handler命令、2,021,749,972減（0.824828%）。62 query・Candid123,281,081 bytesは同じ。50/32 queryは未達。

## 省いた繰り返し処理

従来は保存frameのchecksumをクライアントが検証し、署名したIC requestを送り、canisterが全payloadをもう一度ハッシュしていた。返却時にもcanisterでchecksumを計算してから、IC nodeがquery response全体へ署名する。version3ではcanisterの入力・出力のframeハッシュを省く。

クライアントは保存requestのSHA256を検証してから署名・送信する。canisterはownerを認可してからversion3を復号する。返却frameは32 byteのゼロfooterとし、Rust bridgeがIC query response署名を検証し、request identity・step+1・ゼロfooterを照合した後、SHA256 footerを埋めて保存する。保存形式はchecksum付きなので従来のcheckpoint破損検出を維持する。version1/2のcanister内SHA256/BLAKE3検証は維持する。

署名検証を明示的に有効にした`ic-agent 0.49.2`を使う。ローカルroot keyを取得する既存bridgeはlocalhost専用であり、mainnetへの変更ではない。通常query署名は返答の真正性を検証するが、replicated executionの証明としては扱わない。

`HostBoundRequest`はchecksum確認済みのversion3 headerをprivate fieldで保持する。sealed replyのmodel・pack・input・op・tensor・dims・aux・encoding・version・scalars bitとprogressを照合する。payloadの長さ・codec canonicality・有限性・モデル照合はcanister内で維持する。質問状態をcanisterに保存する機能は追加しない。

## 検証

実ローカルqueryの応答CBORに含まれるCandid bytesをプロキシで一byte書き換えると、bridgeは`Query signature verification failed.`で拒否し、出力ファイルを作らない。保存requestのpayload/footer破損は送信前にchecksumエラーとなり、プロキシが観測するqueryは0。正しいversion3応答とversion2の数値出力はbit一致。匿名step/decisionはpayload復号より前にowner onlyで拒否した。

正しいchecksumを付けた不正block payloadもversion1/2/3の48通常queryですべて拒否。checksum移管によってNaN/Inf・bitmap padding・不正長・非canonical判定を省いていない。証拠は`artifacts/host_checksum/transport-matched/report.json`、`frame-rejections-matched/report.json`、`anonymous*.log`。

最終候補と同じfeature構成のnativeでruntime93＋canister9 unit、integration9、private operandのcompile-fail doc test5が通過。実装後にdecision自身にもowner guardを置き、呼び出し先stepの認可に依存しない入口にした。

## 再現

`scripts/build_full_prefix_candidate.py --host-checksum`で専用候補をビルドし、既存pair・S1・F32 WAT bodyを検証付きでpatchする。固定cache準備後、`scripts/check_full_prefix_hybrid.py --host-checksum`で全6条件を比較する。実測profileは`scripts/profile_host_checksum.py`、署名改ざん試験は`scripts/check_host_checksum.py`、不正payloadは`scripts/check_codec_frame_rejections.py --host-checksum`。生成物・モデル・ログはGit ignore対象。

## 全32層の実測

専用canister `6eydd-o3777-77775-aaama-cai`、module `1e85f150083d173fb4245800f143e53ffef93b708851dba3ada0acab15a0333f`。比較は直前のcodec streaming候補。

|条件|query|handler命令|削減命令|Candid bytes|最大query命令|秒（単回）|
|---|---:|---:|---:|---:|---:|---:|
|prefix45|66|127,492,304,185|1,166,986,781|71,105,691|2,238,635,064|11.555|
|旧log主87|64|244,771,859,503|1,842,736,515|112,344,191|4,895,382,546|20.146|
|主87|62|243,089,913,177|2,021,749,972|123,281,081|4,895,382,546|20.221|
|情報不足80|62|222,192,595,274|1,908,731,658|116,455,579|4,497,066,083|18.686|
|最大変更89|63|248,592,116,619|2,076,122,453|126,604,858|4,309,603,689|20.900|
|cold132|290|367,976,478,140|9,509,973,171|580,207,902|2,353,243,151|43.797|

全6条件の返却hidden・全保持state・最終hidden・typed判断・logits・probabilities/unknownは既存INT8版とbit一致、失敗/replay0。主/情報不足/旧logでは従来同様layer30 hidden非返却であり、未返却hiddenを直接比較したとは扱わない。元の最大変更の見逃しも残り、判断精度改善の主張はしない。通信・観測heap終了値最大は変わらない。時間は単回で、主20.245→20.221秒、cold42.361→43.797秒であり、一律の高速化は確認していない。

同一moduleの実MLP layer0/1/3 profileは1queryあたり29,257,685命令減、出力bit一致。layer0は4,242,428,165→4,213,170,480。wireのdecode/encodeは17,624,124/19,324,116→2,896,893/4,696,318。SHA/BLAKE3 spanはversion3にない。base演算3,319,242,044は変わらず、新handlerの約78.78%。数値演算を高速化した結果とは扱わない。

最後の2層のcompact統合は87/80-tokenでbit一致、4,895,382,517 / 4,497,066,046命令。89-tokenの2 query合計は5,014,885,217で、compact統合も実5B上限を超過した。clientの87-token capを維持する。失敗した診断queryは通常推論の失敗数・時間に含めず、`artifacts/host_checksum/terminal-tail-matched/report.json`に分けて記録。

固定cache準備は721 update、78,739,929,194命令、281.725秒、payload4,065,416,192 bytes。推論とは別計上。prefix packetの2回目準備はquery/命令/通信0。主canister moduleとcodec module不変、Layaは未変更。

証拠は`artifacts/prefix_codec/full-host-checksum-proof/report.json`、`before-after.json`、`validated-source.zip`、`codec-second.json`、`main-unchanged.json`。buildは`full-build-host-checksum-v3/source.zip`、feature/configuration、Wasm body patchログと追加kernel hash/ZIP。bridge変更はソース・binary hashを保存。設定差のあったv2候補は性能比較から除外し、`artifacts/host_checksum/unmatched-build-not-adopted.json`に記録した。
