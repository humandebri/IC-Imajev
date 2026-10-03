# 可逆prefixを本体Deltaへ接続する実装

追加候補：最終Attention/MLPを統合し、復号済みDelta状態の再コピーも省いた。全推論5条件でbit一致、主suffix66 query/262,191,098,449命令。最新の実測と再現手順は [REPEATED_WORK_REMOVAL.md](REPEATED_WORK_REMOVAL.md)。以下は各段階の検証記録として保持する。

2026-10-03。既存の共通prefix計算の保存に加え、各suffixで繰り返していたinnovation logの再構成を減らす。実験feature `experimental-prefix-hybrid` と明示CLI `--hybrid-cache` を追加した。既存のデフォルト・主canisterを置き換えていない。

## 経路

1. canisterが生成した45-token prefix logを、独立codecの通常queryで一度だけNPF1 packetへ準備する。全状態はクライアントに返す。
2. model/pack/input/source module/各層の元state hashにpacketを結び付け、クライアントで保存する。再実行時は準備queryを呼ばない。
3. `run_prefix_canister.py --hybrid-cache <directory>` が通常のprefix cacheと照合する。suffix・compact heads・full Deltaを必須とする。
4. `PrefixTextGraph`がBF16 suffix入力/convとopaque packetをframeへ入れる。ホストがF32状態を再計算しない。毎層のlogコピー/サイズ走査も省く。
5. `decode_query`が型付き`PreparedDeltaHybrid`としてpacketを検査・復号する。既存Deltaの射影/再帰/出力経路へ直接stateを渡し、元の24層×45-token log全体の再生を避ける。NPF1の14 heads分の残るlog再生は必要。状態のcanister永続化と推論updateを追加していない。

## 検証済み

- Rust native 58 unit tests、Python prefix系13 tests、wasm32 canisterのcargo check通過。生成物はartifacts配下でignore。
- 実際の主入力87-token suffixのlayer 0/22/30、情報不足80-tokenと最大変更89-tokenのlayer 0、計5条件で完全なnative Deltaの全出力・convが旧経路とbit一致。source/moduleが生成した既存logとpacketの元logもbyte一致を確認。
- 実測packetを元のprimary prefix cache各層のstate file hashへ結び付ける検査通過。誤ったprefix長、入力precision、suffix state保持、reply方向、変更requestの拒否を検査。
- 90-tokenまで実packetサイズ相当を含めたrequest frameは2 MB以内。replyも可逆block codecで元のrequest identityを保持する。

raw `artifacts/prefix_codec/native-delta-check/report.json`、完全な検証時source `native-delta-check/validated-source.zip`。native helperは `cargo build --release -p imajev-runtime --bin primitive --features experimental-prefix-hybrid,experimental-blake3 --target-dir artifacts/prefix_codec/native-target`。チェックは `check_prefix_hybrid_native.py`。ここでの時間はホストnative単回測定で、IC命令数を示さない。

## 未検証・進行中

全推論候補Wasmは元採用feature群にprefix-hybrid・LoRA input sharing・MLP89を加えてビルドしたが、通常release(opt3/thinLTO/cgu1)、cgu8、非generic共通body、WAT用stub、同stubのsandbox外実行、opt2の6条件でruntimeのrustcがsignal 9で終了した。原因は未確定で、実行時性能やIC上の制約を示す結果ではない。source/feature/compiler/logは `artifacts/prefix_codec/full-build*` に保存し、主Wasmを上書きしていない。

検証専用の新規ローカルcanister `6eydd-o3777-77775-aaama-cai` には既存検証済みmoduleを入れ、固定INT8 pack 4,702,451,200 bytesのupload・hash・sealまで完了した。3201 update、435.93秒。候補moduleへのupgradeとweight cache準備はまだ行っていない。旧主canisterを変更せず、候補のビルドと出力検証を続ける。

その後、固定pairを必須とする実験featureで未使用fallback特殊化を除き、全tokenで重みを共有するWAT kernelを接続したfull候補が通常releaseでビルド成功した。初回準備と推論開始で、32-row gate48個が旧byte cacheに残るため明示errorを返すことを確認。これらもpair化する修正を行い、canister unit5件通過・再ビルド成功。module `44640eb4c3f3cfc513002281eef618da2e428fc9852dd342425b735cdecbef80` を専用canisterへupgradeし、全248 INT8 tensorsの再準備を進めている。元の200 pair tensorsに3,932,160 weight bytes分のgateを加えるが、元byte配置を置き換えるので固定payloadは増やさない。

残る条件は、完成Wasmのsource/hash整合、固定モデルuploadとweight準備、token IDからの新prefix生成、packet準備を初回費用へ計上、全5条件のhidden/state/decision/probabilities比較、各query命令/通信量/時間の実測。新しいcacheは本体module hashとgraph hashへ結び付け直す必要があり、旧cacheの検査を緩めて流用しない。

## 全推論の実測

修正候補`44640eb4…`の全32層とdecisionを専用canisterで実行し、prefix・主suffix・情報不足・最大変更・cold入力の全hidden/stateが旧INT8版とbit一致した。判断を行う4条件ではvalue/abstain/logits/probabilities/unknown probabilityも一致し、型付き出力も有効。失敗・replayは0、各queryは5 B命令以内。判断精度を新たに改善したという意味ではなく、最大変更のgold=yesに対する既存noの見逃しも維持されている。

|条件|query|合計命令|Candid送受信bytes|実行秒（単回）|最大query命令|
|---|---:|---:|---:|---:|---:|
|共通prefix45 tokens|66|136,581,517,179|71,105,691|12.733|2,400,908,664|
|同候補・旧log経路の主suffix87|67|264,360,817,275|113,709,005|21.253|4,559,237,137|
|主suffix87・packet再利用|67|262,456,310,514|125,984,454|8.676|4,559,237,137|
|情報不足suffix80|67|240,767,196,100|118,943,867|19.733|4,181,248,637|
|最大変更suffix89|67|268,826,136,490|127,996,020|46.027|4,670,027,599|
|cold132 tokens|292|402,920,704,287|580,229,781|66.089|2,478,947,790|

旧採用版との主問題比較は命令−1.3067%、Candid通信+10.7955%、query数67で同じ。kernel等だけの旧log経路は命令−0.5906%、通信は同じ。同module内でpacketの有無を比べると復元削減は1,904,506,761命令（−0.7204%）、追加通信12,275,449 bytes。kernelやLoRA input sharingの効果と区別する。

packet初回準備は24通常query、6,326,348,500命令、61,219,865 Candid bytes、3.237秒。二回目はcache hitで準備query/命令/通信が0（file/hash/status検査0.290秒）。packet準備だけの命令費用は同じ主suffixで4回の利用後に回収する。共通prefix生成と固定モデル準備は別費用。

固定cacheの修正後準備は721 update、61,042,755,368命令、235.239秒、固定payload4,065,416,192 bytes。uploadと修正前の準備・失敗試行は、この費用に混ぜず別ログとして保存。推論の観測heap終了値の最大は4,125,949,952 bytes。peakメモリの証明ではない。命令counterはhandler内、通信はCandidのみ、時間は単回で他プロセスの影響を含むため時間の改善率を一般化しない。

検証記録 `artifacts/prefix_codec/full-paired-all-proof/report.json`、source `validated-source.zip`、codec再利用 `codec-second.json`。主canisterは従来moduleのまま。50 queryは未達で、通信増を伴うpacket経路は明示opt-inの候補として保持する。

## 入力複製も省いた最終候補

`experimental-pair-direct-input` では元のquantized I16 bufferを直接読み、初回load64_splatで同じlane配置を作る。各queryのduplicated input Vec生成を省く。専用canisterのmodule `52a7e3f4a1c7552b4985fba56c0e26e74bdf9f98ae670c9aa20c2e1d30b0a2f0` で再び全5条件のhidden/stateと全判断条件のlogits/probabilitiesをbit比較し、一致・型付き出力有効・失敗/replay0を確認した。

|条件|query|合計命令|Candid送受信bytes|実行秒（単回）|最大query命令|
|---|---:|---:|---:|---:|---:|
|共通prefix45|66|136,487,322,987|71,105,691|11.367|2,399,051,818|
|同候補・旧log経路の主suffix87|67|264,169,272,258|113,709,005|20.717|4,555,650,095|
|主suffix87・packet再利用|67|262,235,232,497|125,984,454|13.736|4,555,650,095|
|情報不足suffix80|67|240,593,369,461|118,943,867|46.359|4,177,949,798|
|最大変更suffix89|67|268,599,624,338|127,996,020|23.288|4,666,358,111|
|cold132|292|402,490,273,853|580,229,781|52.701|2,477,344,684|

主問題は旧採用版より命令−1.3899%。packet無しでは−0.6626%で通信は同じ。入力複製を省く前のfull候補より主問題で221,078,017命令減。最終候補の観測heap終了値は最大4,123,721,728 bytesで4 GiB設定内。固定cacheは同じ4,065,416,192 bytes、721準備update/61,042,755,368命令/275.581秒。時間のばらつきがあるため単回の秒数から速度改善を断定しない。

`artifacts/prefix_codec/full-direct-input-proof/report.json` と `validated-source.zip` が最終候補の証拠。各queryの命令/通信量は各条件のreportのqueriesに保存。50 queryは未達、最大変更の既存の見逃しも残る。主canister・Layaは変更していない。

再ビルドは `docs/PAIR_OPERAND_REUSE.md` の手順。候補のsource/feature/kernel hashを保存してから、matched ABIのbodyを置換・validateする。raw stubをinstallしない。
