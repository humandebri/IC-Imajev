# 32 queryへの改善結果

## 最新結果（2026-10-06）：3入力すべて32 query

同じ最適化をupdate推論でも再計測し、617/620/653は各3回すべて **5 / 4 / 5 update**。以前の6/5/5から投票2件は各1回減った。詳細は [UPDATE_TEMPLATES_MEASURED.md](UPDATE_TEMPLATES_MEASURED.md)。

固定prefixを準備した状態で、現行短文BOOM DAO 617/620/653をすべて通常query32回で実行した。元INT8モデルから重み・入力本文・質問・選択肢・calibrationを変えておらず、全exported hidden、畳み込み/KV状態、最終normalized hidden、判定・確率・logitsがbit一致した。

|入力|元→改善後query|改善後命令|元からの削減|prefix/suffix|最大query命令|Candid通信|
|---|---:|---:|---:|---:|---:|---:|
|617|39→32|144,992,136,801|19.269%|38/56|4,707,303,488|79,101,667 bytes|
|620|39→32|125,259,273,904|21.258%|38/48|4,072,571,844|71,008,643 bytes|
|653|39→32|145,133,372,393|5.710%|27/57|4,700,116,107|74,312,335 bytes|

投票テンプレートだけは、数値の直前まで11 tokenの固定文を追加で事前計算して再利用する。新しい日数・倍率・質問は毎回推論する。文字列一致から最長prefixを選び、mintなど異なる文には元の共通27 tokenを使用する。**共通27 tokenのままで617が32回になったという結果ではない**。実機測定は上記3入力の範囲で、任意の長文に対する32回保証はしない。

追加の固定38-token prefixは初回だけ66 query＋packet化24 queryで準備する。モデルの固定重み721 update・固定27-prefix状態24 updateも別に準備し、1問の推論updateは0。各queryはICの通常query上限で成功し、handler counter<5B、要求/返信それぞれ2MB未満、heap4GiB未満（最大4,176,150,528 bytes）。Candid通信量とcounterはHTTP/CBOR/certified metadata、CDKのdecode/encodeを除く。

ローカル比較後に元のmodule/cacheへ復元し、snapshotを削除した。新moduleは`artifacts/voting-template-prefix-v1/full-build/full.wasm`（2cfe5d3a…）、全実測と独立検証は`proof-v1/{report.json,verified.json,verification.log}`。`cli-check-v1`と`cli-check-v2`は自動選択/実行wrapperを実機の96 frameへoffline照合した証跡で、追加の実機queryには数えない。

候補module・固定重みを導入済みのローカルcanisterでは、次のコマンドで実行できる。元の検証用canisterは測定後に6052…へ復元してあるため、実行には候補の導入と準備が必要。

```sh
.venv/bin/python scripts/run_query32_templates.py \
  --canister <prepared-local-canister-id> \
  --record 0 \
  --directory artifacts/query32-template-run
```

`--record 0/1/2`が617/620/653。wrapperはcore CLIの`--query32-start`を使い、フロントで固定38-token templateか共通27-token prefixを選ぶ。module/bridge/cacheのidentityが異なる場合や対象範囲外は停止し、自動fallback・重複実行をしない。

以下は改善途中の測定履歴。


目標は現在の短文BOOM DAO推論を通常query 32回以下にすること。モデル・prefix・判断基準は維持し、中間状態と最終判断・確率の完全一致を確認する。653/620では32回を実測達成。617は未達成。

直前のpair境界最適化では3入力とも39回のまま、全体命令数を0.943～1.064%削減できた。最も重い617は177,905,279,101命令。32×5Bの名目予算へはさらに17,905,279,101命令（10.064%）以上の削減が必要で、通信処理や分割の余白も必要となる。詳細は[INT8_PAIR_BOUNDS.md](INT8_PAIR_BOUNDS.md)。

## 深い整数行列分解：不採用

3段Strassenで8 token・32出力を処理する候補を作成した。整数乗算の組合せを512から343に減らす。K256内で整数計算を完結し、F32への変換・scale・block加算順を維持する。入力変換をK256ごとに作るため、全列の追加展開cacheは不要。Wasm localsは9,195。

layer3の実Q重み8192×2560を使った19条件・38通常queryで、元のINT8 native計算と全出力digestが一致した。実入力、57/59/67 tokenの長さ比較、整数の符号・ゼロ・微小値を含む境界条件を測定した。

しかし命令数は増加した。67 tokenでは46.84%増、59では50.05%増、57では54.77%増。以前の実入力87 tokenでも36.32%増。乗算減少より変換・復元等の追加処理が大きかったため、全モデルには接続しない。全モデルのquery数や精度をこの診断から推定しない。

証跡：`artifacts/s3-stream-v1/{build,check}`、保存したCandid返信を再decodeして検証した`summary.json`。生成・build・check・集計は`generate_s3_stream_probe.py`、`build_s3_stream_probe.py`、`check_s3_stream_probe.py`、`report_s3_stream_probe.py`。

## Deltaの状態保持：全モデルで効果を確認

Deltaの128×128状態をWasmの一時変数に保持し、状態の再読み込み・書き込みと重複する減衰計算を減らした。各値のF32 mul/addとkeyの加算順序を維持し、最後の状態も元のkey-major配置へ保存する。単体7条件・14通常queryで、出力と最終状態が元のWasm実装およびnative F32計算と完全一致。57～87 tokenでは26.87～26.98%のカーネル命令削減。

凍結したruntimeへこの処理を追加し、前のpair境界最適化と組み合わせた全体候補を測定した。全117 queryの返信・中間状態・判断・確率は元の6052…と完全一致。

|BOOM DAO|pair境界最適化後|Delta追加後|追加の削減|query数|
|---|---:|---:|---:|---:|
|617|177,905,279,101|176,343,988,861|0.878%|39|
|620|157,425,372,618|156,052,758,474|0.872%|39|
|653|152,284,933,631|150,959,488,511|0.870%|39|

元の6052…との比較では合計1.812～1.925%減。最大queryは617で4,887,431,937命令、Candid通信量は元のまま。617を32回へ収めるにはこの候補からさらに16,343,988,861命令（9.268%）以上の削減が必要で、分割と通信の余白も必要となる。全体候補349b2c09…は比較後にsnapshotで元のmodule・重みcacheを復元し、snapshotも削除した。既定にはまだ接続していない。

証跡：`artifacts/delta-register-v3/{build,check,full-build,full-proof-v1}`、返信と保存した全117要求・返信を独立に照合した`report.json`。独立診断の初版には置換前stubの制御フローの不備があり、v1は推論に使用していない。v2は出力だけの比較、v3は最終状態も含めた比較である。

## 4 token・64出力の整数行列分解：不採用

2段Strassenのrank49版も、固定重みの追加展開を避け、64出力を共有する構成で測定した。19条件・38通常queryで全出力が元INT8 native計算と完全一致したが、67 tokenで12.76%増となったため不採用。初期版のWAT生成・一時変数再利用の不備は修正し、v3で全条件を再測定した。証跡は`artifacts/s2-wide-stream-v3/{build,check,summary.json}`。

## 固定prefix状態の再利用：全モデルで一致

27 tokenの固定prefixについて、生のlogと圧縮packetを復元し、全F32状態がbit一致する場合だけ登録する。24層で48MiBの追加メモリと準備update 24回を使用する。質問ごとの状態は保存しない。

全117 queryの返信frameと中間状態が完全一致し、判断・確率・logitsも完全一致した。decisionの命令counterはコンパイル後に変化するため、数値判断との比較から分離して記録している。

|入力|元の全体命令|改善後|削減率|query数|
|---|---:|---:|---:|---:|
|617|179,598,906,781|175,287,408,275|2.401%|39|
|620|159,074,614,698|154,988,497,025|2.569%|39|
|653|153,923,079,311|149,903,713,164|2.611%|39|

保存返信・source hash・24準備updateのCandid返信を独立検証済み。各queryのheapは4GiB未満。証跡は`artifacts/prefix-state-cache-v1/full-proof-v2`、独立集計は`report.json`と`verification.log`。比較後に元のmoduleと重みcacheへ復元し、snapshotを削除した。

## 32回への分割変更：653/620で達成

4層ごとに5回だった処理を、MLPの完了・次Attention・次MLPの先頭を融合して4回へ組み直した。最初の3層と終端を含め通常query 32回。数値計算は通常query内で実行し、クライアントは状態転送と分割を担う。推論updateは0回で、固定重み・prefixの準備updateは含めない。

|入力|元query数|改善後|全体命令|元からの命令削減|最大query命令|Candid通信|
|---|---:|---:|---:|---:|---:|---:|
|653|39|32|147,275,250,000|4.319%|4,756,527,651|74,312,335 bytes|
|620|39|32|152,277,053,079|4.273%|4,916,154,338|76,335,591 bytes|

新鮮なjournalで64 queryを実行し、replay・自動fallbackは0。保存した31層のhidden、32層の畳み込み/KV状態、最後のnormalized hidden、判断・確率・logitsが元のINT8モデルと全bit一致した。最終層のhiddenは終端readoutが必要な最後のtokenを比較している。全返信を再decodeし、各命令counter・通信サイズ・heap<4GiBを確認した。異なる分割のtransient carry同士を同じframeとして比較する主張はしていない。

module f6b399b6…、`client/query32_balanced.py`と専用bridgeを使用。各ブロックのMLP先行行数を5120→4864→4608→4352と減らし、Attention融合で5120へ戻して命令を分配する。

初版のfront固定4096では653だけ32回成功（147,259,960,067命令・71,970,775 Candid bytes）、620の2回目がICの5B上限を超えた。この失敗は`proof-v1/620/queries/failures.jsonl`に記録し、成功32回には含めていない。分配修正後は653/620を両方新規実行した。

証跡は`artifacts/query32-v1/proof-v2/{report,verified}.json`、独立検証は`scripts/report_query32.py`。いずれの実験も比較後にsnapshotで6052…と重みcacheを復元し、snapshotも削除した。

## 残る対象

最も重い617（suffix67）は元の39回構成で、改善後も175,287,408,275命令。32×5Bへはさらに8.72%以上の削減が必要で、個々のqueryの通信処理と分割の余裕も必要。32回候補の実測範囲はsuffix57/59・prefix27であり、617や長文へ一般化しない。

rank49 Winogradも追加重み展開cacheを作らず、Wasm一時変数へ重みを保持して比較した。19条件・38通常queryで全出力digestが元INT8 nativeと一致したが、suffix67で10.72%増、57で15.36%増、59で11.94%増となり不採用。証跡は`artifacts/winograd2-register-v2/{build,check,summary.json,verification.log,proof}`。旧Strassen rank49や6.125倍の重み展開cacheを使った旧Winograd2とは構成が異なる。

初回の再インストールは状態消去への承認が不明として自動レビューで拒否され、snapshotを保存するupgradeへ変更した。診断初版ではpost_upgrade初期化がなく準備updateが失敗したため、hookを追加したv2で全測定をやり直した。v1/v2とも元の診断module・状態へ復元し、今回作ったsnapshotだけを削除した。memory_sizeはsnapshot自体の保存量も含むため、v2では削除後に元のサイズを確認している。推論canisterの復元・64 queryの検証とは独立した診断である。

実行スクリプトの`--query32-start`に検証済み分配を接続した。候補module・専用bridgeのhash、prefix元module、入力範囲を確認し、失敗時に別グラフへ自動再実行しない。CLIの経路選択と64要求frame・付随prefix/expectedを、実機で保存したframeへoffline照合した（`artifacts/query32-v1/cli-check-v1/check.json`）。このoffline確認は追加の実機測定として数えない。全モデルの既定moduleは6052…のまま。

32回ルートは`--tail-start --join-start --roll-start --query32-start`と、`artifacts/query32-v1/full-build/full.wasm`・`artifacts/query32-v1/client-build/imajev-client`を指定する。prefixは検証済み`artifacts/query-packing-v3/prefix-v2/queries`、packetは同`packets-v2`を使用し、固定重み721項目と固定prefix状態24項目の準備が必要。完全な実行例は`proof-v2/operations.json`と`scripts/prove_query32_balanced.py`に保存している。

出力幅160のS1も測定した。first-pairのscratchを再利用して9,317 localsへ収め、19条件・38 queryで元のnativeと全bit一致した。ただし8192行の末尾32行を128行のゼロpaddingで処理したため、suffix67で0.876%増、57/59で約0.887%増。全体へは接続しない。証跡は`artifacts/s1-output160-v1/{build,check,summary.json,proof}`。初回のビルドには変数名の不備があり、コンパイルエラーを保存し、修正後に測定した。診断canisterは元のmodule・状態へ復元済み。

次は凍結ソースを用い、releaseの整数overflow検査による命令差を比較する。依存ライブラリや浮動小数の演算順序は維持し、runtimeとwrapperのcompiler flagを変更する。ローカル比較候補であり、正常入力117 queryのstate一致と、外部入力の整数境界の扱いを確認するまで既定には採用しない。


## Releaseの整数検査を一括で外す比較：不採用

凍結したruntime/wrapperだけを`overflow-checks=no`で比較し、依存ライブラリと5本のWAT bodyは維持した。準備は完了したが、正常617の最初のqueryが`paired projection output`エラーとなり、全体比較は完了しなかった。全体命令の削減率や状態一致は報告しない。

別途、不正なnormalization shape `[262144,16384]`（Wasm32の要素数2^32）を1 queryずつ比較した。元のmoduleはoverflow trapで拒否したが、候補は空の結果を返した。同じ要求hashで検証済み。一括設定変更は採用せず、元のmodule・重みcacheへsnapshotで復元し、snapshotも削除した。証跡は`artifacts/release-overflow-v1/full-proof-v1/{failure-summary,restored}.json`と`boundary/{candidate,baseline}`。失敗queryのraw Candid errorはこのreplay harnessでは保存されず、bridgeのRuntimeErrorとして記録している。

次の検討では外部入力のchecked演算を維持し、サイズ上限を確定できる内部ループのアドレス計算に範囲を絞る。617の32回は未達で、653/620の検証済み32回化とは分けて扱う。

## 外部境界を維持したrelease候補

一括flag変更の正常入力エラーを調査したところ、compilerが3つの同一stubを1つの関数に統合し、旧patcherが異なるkernelを同じindexへ上書きしていた。各stubを区別し、export aliasがある関数へのpatchを拒否する検証付きpatcherを別途作成した。旧patcherと元moduleは上書きしていない。

wrapperのoverflow検査と外部shapeのchecked積を維持し、凍結runtimeだけを変更した候補b23e6288…で、117 queryの全response frame・付随状態・最終判定と確率が元INT8と一致した。15件の入力境界を候補と元で比較し、不正入力12件は双方拒否、合法入力3件のresponse frameも一致した。境界確認は一般の全入力に対する証明ではなく、今回の条件を対象とする。

|入力|元からの全体命令削減|改善後命令|測定分割|
|---|---:|---:|---:|
|617|3.677%|172,995,547,107|39 query|
|620|3.808%|153,017,571,746|39 query|
|653|3.840%|148,012,081,990|39 query|

この測定は固定39回の比較。別に最初のqueryのsmokeを1回、境界queryを30回実行した。固定重みはsmoke準備23 updateと残り698 update、固定prefix状態は24 update。元moduleと重みcacheへ復元しsnapshotを削除済み。証跡は`artifacts/guarded-release-v2/{report.json,verification.log,full-proof-v1}`。32回構成への適用は別測定中。617は合計160B以下へなお7.512%以上と、分割・通信の余裕が必要。

Hopcroft/Kerrの矩形2×4×4、rank26も検証した。Lille公開の評価式を読み、全8出力の多項式恒等式とi16/i32中間値の上限を検査し、128行の重みをregister保持した。同じraw INT8容量で19条件・38通常queryを測定し、nativeと全出力bit一致したが、suffix67で14.229%増。乗算削減を前後処理の増加が上回ったため不採用。証跡は`artifacts/hopcroft-kerr26-v1/{reference,build,check,summary.json,verification.log,proof}`。generator初版のPython名解決と二版のlocal参照のビルド不備は測定前に修正し、失敗生成物も別フォルダへ残した。診断module・状態は復元しsnapshotも削除済み。

## Release候補を32回構成でも検証

同じb23e6288…と分配で、653/620の64通常queryを新規実行した。replay/fallbackは0で、各queryは5B命令・要求/返信それぞれ2MB・heap4GiB未満。31層のexported hidden、32層の畳み込み/KV状態、最終hidden・判定・確率・logitsを元モデルとbit照合した。

|入力|通常query|命令|元39回からの削減|以前の32回からの削減|最大query命令|
|---|---:|---:|---:|---:|---:|
|653|32|145,509,107,822|5.466%|1.199%|4,712,040,712|
|620|32|150,436,477,875|5.430%|1.209%|4,870,112,397|

通信量は以前の32回構成と同じ。固定重み721 update・固定prefix24 updateは推論32回から別計上。比較後に元module/cacheへ復元しsnapshotを削除した。独立検証は`artifacts/guarded-release-v2/query32-proof-v1/{verified.json,verification.log}`。凍結した旧graphのplanningメタデータは元f6b…名を保持するが、実際のmoduleは実行前後のhashと各runのwasm_sha256でb23e…を検証している。CLIには候補hashを明示許可し、plan名も選択moduleへ合わせた。新しい64 frameのoffline CLI照合は`query32-cli-check-v1/check.json`で、追加実機queryには数えない。

利用時は先の32回CLI引数を維持し、`--wasm artifacts/guarded-release-v2/full-build/full.wasm`へ変更する。専用bridgeと固定prefix/packetの入力は同一。617への32回化は未達のまま。次の候補ではF32 LoRAの出力tileを64へ拡大し、列ごとの加算順序・元の重みstrideを保持して全モデルを比較する。

## LoRA出力幅64の全モデル検証

元のoutput32重みstrideを維持し、隣り合う2 tileを一緒に処理する。各F32出力の列ごとの乗算・加算順序は維持し、64行境界に合わない投影や列継続は元の32行経路へ戻す。候補9bdae0d8…で117通常queryの全frame・付随状態・判定・確率が元とbit一致した。

|入力|全体命令|元からの削減|b23e…からの追加削減|測定回数|
|---|---:|---:|---:|---:|
|617|172,537,849,634|3.932%|0.265%|39|
|620|152,612,755,077|4.062%|0.265%|39|
|653|147,620,047,987|4.095%|0.265%|39|

固定重みはsmoke準備23 update＋残り698 update、prefix24 update。最初のqueryのsmokeは別途1回。元module/cacheへ復元しsnapshotを削除済み。独立検証は`artifacts/guarded-f32-output64-v1/{report.json,verification.log,full-proof-v1}`。この候補では32回構成はまだ測定しておらず、CLIで許可する32回moduleは検証済みf6b…/b23e…の2本に限定した。

出力幅128もdown-Bの実LoRA重みで15条件・30 queryを比較し、nativeと全bit一致、出力幅32より約5.4%の単体命令削減を確認した。モデルのLoRA rankは64であり、最初にrank128を仮定して準備したgate-A測定はsealで失敗したため、実際の2560×64 down-Bへ修正して最初から測り直した。失敗時も診断module/stateを復元しsnapshotを削除した。結果は`artifacts/f32-output128-v1/{report.json,verification.log,down-B,proof}`。この単体改善を全モデルや32回化の削減率としては数えない。

現時点の617は172,537,849,634命令で、合計160B以下へなお約7.27%＋分割の余裕が必要。32回達成は653/620、残る対象は617。

## 固定テンプレートの追加prefixを検証中

投票入力617/620には、全カテゴリ共通27 tokenに加え、変更後の数値の直前まで11 tokenの固定文がある。2入力の単純な最長共通prefixは39 tokenだが、その最後には「20000」と「2」の先頭digit「2」が含まれるため採用しない。固定templateは38 tokenで切り、「Minimum voting dissolve delay changes from 1 day to 」までを一度だけ計算する。新しい数値・倍率・質問・選択肢は毎回推論する。

固定38 tokenを元6052…で66 query計算し、24 queryで再利用packetを生成した。準備102,509,495,334命令・prefix計算Candid60,423,052 bytesは一度だけの費用で、1問ごとの32 queryへ隠して含めない。元推論の最初38 tokenのhiddenと全30層・投票2入力でbit一致した。

変更後の数値9種類（0/1/2/7/42/365/20000/999999/1000000）と質問2種類、合計18条件で、cache境界の手前が同じことをoffline確認した。古い値を変えた文やmintの文はこのprefixへ一致せず、共通27 tokenを使う。これは選択ロジックの検証であり、18件のモデル精度評価ではない。

候補2cfe5d3a…は検証済みLoRA64候補に、owner専用MLP/Delta bridgeのprefix上限を27→38へ広げたもの。要求全体1.99MB、prefix shape・finite/gate検査、suffix範囲は維持する。正常推論3入力を32回で比較中。prefix38を使わず共通27のままで重い617が32回になったという主張はしない。

別途、rank26の転置4×4×2（token4・出力64）も19条件・38 queryでnativeとbit一致したが、suffix67で17.323%命令増となり不採用。厳密な値は`artifacts/hopcroft-kerr26-transpose-v1/summary.json`。診断module/stateへ復元しsnapshotを削除した。
