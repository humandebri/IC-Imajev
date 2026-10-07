# Imajev：危険なproposalを通さないフィルターの評価報告

[English](PROPOSAL_FILTER_REPORT_20261007.en.md)

評価日：2026年10月7日  
対象：BOOM DAO proposal #500〜#660、計161件  
根拠データ：2026年10月6日に取得した固定snapshot

## 評価の目的と結論

Imajevを、危険なproposalを次の処理へ通さないためのフィルターとして評価した。本報告では既存の判定結果を、approveなら通過、rejectとholdなら通過停止という運用方針に対応させて集計する。実際の後続処理への接続や自動停止の実装・試験は今回行っていない。holdは危険が確定したという意味ではなく、判断の不確実さ・証拠不足・実行予算の超過を理由に止める結果を含む。

今回、モデル対象50件のうち、数値上の参加条件を大幅に厳しくするものとしてレビューした32件は、10件がreject、22件がholdとなり、上記の方針ではすべて通過停止となる。この用途では、reject相当の内容をholdで止めることも目的を満たす。三択の判定一致よりも、危険な内容をapproveで通してしまうかを重視する。

今回確認したモデル対象の32件は、rejectとholdを止める方針で通過を防げる。ただし、32件の分類はエージェントによる条件付きレビューであり、人間が確認した正解ラベルではない。一般的な危険検出率やproposal全体の安全性が確定した結果ではない。

## Imajevとは何か

### 証拠と選択肢を受け取る判断モデル

Imajevは、文章・記録・写真などの証拠を読み、利用者があらかじめ定義した選択肢から答えを返すために調整されたモデルである。たとえば、商品写真と商品情報を渡して矛盾する項目を選ぶ、問い合わせ文から担当部署を選ぶ、提示された変更が条件を満たすかを判断するといった使い方を想定する。モデルの一般的な用途と、今回のproposalフィルターへの適用は区別する。

配布元のリクエストは、判断材料となる`state`と、質問・選択肢・条件を定義する`questions`を組み合わせる。写真を加えることもでき、選択式の回答や数値形式の判断を型付きで返す。配布元の通常の出力には、選択肢ごとの確率、`unknown_probability`、判断を保留したかを表す`abstained`などがある。unknownは、証拠だけでは選べない場合の選択肢として学習している。

このリポジトリのIC実装では、固定したImajev-4Bの**テキスト判断経路**を移植している。今回のproposal評価は画像encoder、画像position、生成decodeを使わず、数値stateと一つの質問、approve/rejectの二つの選択肢を入力した。配布元が対応するすべての質問形式・画像機能を、今回IC上で検証したわけではない。

### Qwen、LoRA、decision readoutの役割

Qwen3.5-4Bは文章を読み取る基盤である。Imajev-4Bはその固定した基盤に、判断タスクへ適応させるLoRAと、選択肢を直接スコア化するdecision readoutを追加している。LoRAは各投影へ小さな追加行列A/Bの積を加える方法で、今回のrankは64、scaleはalpha/rank＝128/64＝2である。基盤の大きな重みと判断用の追加重みを別々に扱える。

入力が32層を通ると、最後の判定位置のhidden vectorをfinal RMSNormで整え、専用readoutに渡す。readoutはF32の256×2,560行列で、255個の選択肢codeとunknownに対応する。各質問の選択肢をcodeへ割り当て、使用するcodeの行を評価してscoreを得る。approveやrejectを文章として生成し、生成された文を読み取って分類する方式ではない。

今回の経路では、通常の全語彙への言語生成headの代わりに、A/B/unknownの必要なreadout行を評価する。不要な語彙行と生成の繰り返しを避けられる一方、入力を読み取るembedding・32層の計算は必要である。専用readoutを使うだけで、4Bモデルの前段計算が小さな分類器に置き換わるわけではない。

### 今回使った学習済み版

固定した配布物はphase-3の最終checkpoint `r2-s000291`である。配布元のrelease specificationによれば、基盤を固定してLoRAとreadoutを学習し、文章・画像の判断例、難しい例の追加、選択肢順序の入れ替え、教師の分布を使う学習を行っている。今回のBOOM DAO 161件で追加学習やfine-tuningをしたものではない。proposalごとの処理方針と数値圧縮は、このリポジトリのハーネス側で定義している。

配布元の標準評価には選択肢順序を4回回転して平均し、校正を適用する構成がある。今回のproposal評価は固定順序の1回推論で、raw A/B差に閾値を適用しているため、配布元のbenchmark値や「確率」をそのままこのフィルターの性能に転用しない。学習されたunknownと、今回ハーネスが返すholdも別の仕組みである。

構成と学習済み版の説明は、固定ファイルの[release specification](evidence/proposal-filter-20261007/RELEASE-SPEC.md)と[モデルカード](evidence/proposal-filter-20261007/MODEL_CARD.md)に基づく。これは取得したrevisionについての説明で、配布元の最新状態を調査したものではない。

## 全161件の処理結果

|結果|件数|フィルターとしての扱い|
|---|---:|---|
|approve|2|数値参加条件のフィルターを通過|
|reject|10|通過停止|
|hold|61|通過停止。必要に応じて追加確認|
|対象外・除外|88|未検査として扱う。通過には数えない|
|合計|161|全件の処理経路を確定|

判定対象73件のうち、通過は2件、停止は71件だった。73件中50件はモデル判定で、残り23件はTool側の判定範囲・証拠・入力予算の検査でholdとなった。

除外88件には、現行方針で選別から外したMotionや表示・brandingの変更などが含まれる。除外は安全を確認した結果ではない。フィルターを実際の処理へ接続する場合は、未検査の経路として扱う。

## 危険な参加条件変更を止められたか

元proposalと照合したモデル対象50件を、現行の「参加障壁を抑える」方針に基づいてレビューした。

以下の32件はモデル対象50件内の集計で、全161件中の危険案の総数ではない。たとえば予算超過でTool holdとなった#585にも大幅な参加制限があるが、この32件には含めていない。対象外88件についても危険の有無は未評価である。

|条件付きレビューでの分類|件数|実際の出力|フィルター評価|
|---|---:|---|---|
|大幅なstake・最低投票lockの制限強化|32|reject 10、hold 22|32件すべて停止。approve通過は0件|
|最低投票lockを1日→2日に変更|16|hold 16|停止。有害と断定する根拠は不足|
|数値上の参加条件を改善|2|approve 2|通過。proposal全体の危険は別途確認が必要|

たとえば#562・#566は、最低stakeを5→100万tokens、最低投票lockを2→1,461日へ引き上げる。入力に両方の旧新値は残っていたが、モデルのA/B出力は弱いapprove方向だった。未校正スコアが0.6未満だったため、最終結果はholdとなり通過しなかった。

#587・#588も、大幅な制限強化に対してA/B出力がapprove方向だったが、最終結果はholdだった。これら4件では停止処理が通過を防いでいるため、モデルの二択だけを使わず、閾値と証拠検査を含む最終結果で運用する必要がある。

#602など16件は最低投票lockを1日→2日へ変更する。条件は厳しくなるものの、これだけで危険すぎると断定する基準はない。holdで追加確認に回す扱いは、今回のフィルター目的に適合する。ただし、この16件を「安全な案を誤って止めた件数」とは数えられない。

## 通過した2件の確認範囲

|ID|数値上の変更|通過の根拠と残る確認事項|
|---|---|---|
|656|最低stake 1,500万→5tokens、投票期間1→3日|参加条件の改善は確認できる。元summaryのtakeoverへの言及は数値入力に含まれず、意図・支配権の文脈は未評価|
|659|最低stake 1,500万→5tokens、最低投票lock 2→1日、lock bonus 1→100%|参加資格の改善は確認できる。bonus変更による投票力配分の影響は未評価|

approveは、この数値参加条件の検査を通過したという意味に限定する。資金・支配権・実行コードなどを含むproposal全体の安全確認や、自動投票の承認には拡張しない。

## hold 61件の内訳

|停止理由|件数|意味|
|---|---:|---|
|モデルのA/Bスコアが0.6未満|38|二択の出力差が小さいため停止|
|直接的なstake・最低投票lockの数値判定範囲外|17|現行モデル判定では内容を評価しきれないため停止|
|資金・mint・実行内容の証拠不足や対立|5|必要な根拠がそろわないため停止|
|全変更を保持した入力が128 tokensを超過|1|#585は130 tokens。切り捨てず停止|

証拠不足の5件は、treasury送金3件、mint 1件、generic call 1件だった。元データとの照合では、送金・mintの受取人支配や残高・供給量の根拠不足、generic callの構造化payloadとrenderingの不一致などを確認した。これらのholdを支持する。

## 入力ハーネスと実行の検証

ローカルへ取り込んだproposal_assessmentの数値参加条件ハーネスを使い、全161件の処理経路を再計算した。compact_units.pyの厳密な単位変換と、benchmark_600_660_binary.pyと同じ圧縮・二択・証拠検査の流れを適用した。

モデル対象50件は、同一の数値入力18種類にまとめて新規推論した。旧予測や前回実験のprefix状態は再利用していない。数値入力が同じproposalでも目的や意図が同じとは限らないため、この重複排除は数値参加条件の判定に限定する。

128 tokensの上限には質問・選択肢・chatの特殊記号も含めた。変更された全項目の旧新値と単位を保持し、証拠を断片に分割して判定する方法は使っていない。86 tokensへの一律の短縮も行っていない。

モデル対象50件すべてで、snapshot hash、構造化payloadの指定値とNew renderingの一致、旧値との差分とtaskの変更項目集合の一致を確認した。旧値は履歴rendering由来で、当時のchain状態との独立照合はしていない。数値保持を確認できても、目的・意図・未知の実影響までモデル入力に保持できたという意味ではない。

18種類すべてで32層の実行、モデル・入力・module等のhash、query数、replay/fallbackなし、実行資源の上限を照合した。前回の同じ数値ハーネスによる評価と、18種類のlogitsはbit単位で一致した。これは実行の再現性を示し、判定内容の正しさを証明するものではない。

|実行項目|実績|
|---|---:|
|新規の異なるモデル入力|18種類|
|モデル推論query|774回|
|今回作成したprefix・codecの準備query|990回|

module等の照合は上表とは別に行った。モデル重み・canisterの更新や投票は行っていない。後続の根拠レビューと閾値比較では追加推論を行わず、元の評価結果も変更していない。

## 実行環境とImajevの構成

### ローカル実行環境

この評価はInternet Computerのローカルネットワーク上で実施した。proposal取得元は公開SNS APIだが、今回の推論は取得済みsnapshotを読み、localhostのcanisterをqueryした。mainnetでの推論・投票・費用測定ではない。

|項目|構成|
|---|---|
|作業リポジトリ|`IC-Imajev`（各ファイルはリポジトリ基準）|
|ホストOS・architecture|macOS 27.0.1（build 26A434）、arm64|
|ホスト側の実行ツール|Python 3.12.14、icp-cli 1.0.2|
|ローカル接続先|`http://localhost:8001/`|
|推論canister|`7st3i-3l777-77775-aaaja-cai`|
|prefix codec canister|`7vs54-wt777-77775-aaajq-cai`|
|実行Wasm|`artifacts/merged-query32-v1/build/full.wasm`|
|通信bridge|`artifacts/query32-v1/client-build/imajev-client`|
|実行driver|`artifacts/proposal-query-optimization-v1/final-runner.py`|

OS・Python・icp-cliのversionは報告書追記時に確認した値で、推論開始時に保存したversion記録ではない。CPU型番・ホストRAM容量は今回取得していない。推論canisterとcodecの同一性は、以下の実行時module hashで確認した。

- 推論module：`6ce099e1f128b835b153bb28cde9f27dee719711be3822a38976cb4688613de4`
- codec module：`e9644266f9bea8efdff5c1d5017b7c82beb67c3030279e6f27248b90fa51ef6e`

### モデルと数値演算

Imajevは、Qwen3.5-4Bを基盤にImajev-4BのLoRA adapterと専用decision readoutを組み合わせたモデルを、このリポジトリのRust/Wasmランタイムで実行している。今回使ったのはテキスト判断の経路で、文章を自由生成する処理ではなく、入力を全32層へ通して選択肢のscoreを取り出す。

|項目|固定した構成|
|---|---|
|基盤モデル|`Qwen/Qwen3.5-4B`|
|基盤revision|`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`|
|adapter|`mohit67890/imajev-4b`|
|adapter revision|`c9e5f132465da85d31735ec502d5557982671a7d`|
|LoRA|rank 64、alpha 128|
|text decoder|32層、hidden size 2,560、MLP intermediate size 9,216|
|層の構成|linear attention / Gated DeltaNet 24層、full attention 8層|
|量子化packのサイズ|4,702,451,200 bytes（約4.70 GB）|
|演算方式|`int8-block256-base-f32-lora-v1`|
|通信codec|`bf16-block256-exact-v1`|

基盤のdense重みは行単位INT8、基盤投影のactivationはtokenごとのblock256 INT8を使う。LoRAとreadoutは別のF32演算で、再帰状態等の必要なF32値を保持する。通信の可逆圧縮と、モデル内のINT8量子化は別の処理であり、元のBF16モデルと同じ数値計算という意味ではない。pack全体のサイズは、query時のheap使用量とは異なる。

モデル識別hashは`7ef38ecb70f4afa4e92bad2fe9f0448699ee45041688b1110819143f41601330`、pack hashは`364e3aa3f69a61a69e6df62e3e2a71d4f55457f1a8e1c16934aefc6130f04c84`で固定した。基盤・adapter・tokenizerのファイル同一性はMODEL_LOCK.jsonで管理する。

Python側はsnapshotの抽出、厳密な単位変換、token化、実行順序の制御、中間状態の保存と判定gateを担当する。モデルの各層の数値演算は通常queryで実行し、query間で必要な状態をclient側が保持して次の呼び出しへ渡す。複数層・演算の融合や終端readoutの統合によってquery数を減らしているため、32 queryを32個の独立した判定と解釈しない。

専用readoutはA・Bと内部unknownの3 logitsを計算するが、今回の二択ハーネスはraw A/Bだけを使う。Aをapprove、Bをrejectに対応させ、`score = sigmoid(abs(logit_A - logit_B))`で閾値を判定する。ここでのholdは、内部unknownを直接採用した結果ではなく、閾値・証拠・範囲・予算の検査で決まる。配布モデルの校正設定があっても、このA/B差によるscoreが危険検出の正解確率として校正済みという意味にはならない。

### 入力tokenとprefixの関係

入力は旧新数値の文字列だけでなく、質問・A/Bの選択肢・chat templateの特殊記号・判定位置までを含む。固定tokenizerとchat templateを使い、thinkingを無効にした入力をtoken化した。今回の128-token上限はproposalハーネスの予算であり、基盤モデル本来のcontext上限ではない。また、driverの`token_cap=132`や`mlp_full_token_cap=89`は実行内部の別の設定で、今回許可した入力上限128とは区別する。

prefixは入力の先頭に一致するtoken列である。その部分を先に全層で計算し、各層のhidden、attentionのK/V、DeltaNetの状態・log等を保存する。続きのsuffixを処理する際に、その状態を使って先頭の計算を省く。prefixは要約でも、tokenを入力から削ったものでもなく、計算済みの状態で置き換えた同じ先頭部分である。

今回のprefixにはchatの共通部分だけでなく、同じ数値stateの先頭部分も含まれる。11種類のbankを今回新規に準備し、同じ先頭token列の入力間だけで共有した。token列・モデル・pack・module・状態ファイルのhashを確認するため、先頭が異なる入力や異なる重みへそのまま流用することはできない。以前の実験のprefix状態や予測結果は再利用していないが、固定モデルの重みpack等は既存のものを使っている。

|token集計|値|
|---|---:|
|異なる入力18種類の入力長|67〜128 tokens|
|18種類の入力token合計|1,861 tokens|
|50 proposalを個別に数えた入力token合計|4,799 tokens|
|11 prefix bankの長さ|8〜48 tokens|
|今回の32-query経路のsuffix長|58〜59 tokens|
|今回の50-query経路のsuffix長|77〜80 tokens|

token合計は論理的な全入力の長さである。prefix準備・query間の状態転送・各層の計算量を表す数ではなく、API課金token数でもない。出力は固定選択肢のreadoutで、文章を生成した出力token数は集計していない。

### 18種類の入力と実行経路

prefix長＋suffix長が全入力token数になる。proposal IDを複数記した行は、数値入力が完全に同じため1回の新規推論結果を共有したもの。

|proposal ID|全入力tokens|prefix bank|prefix tokens|suffix tokens|推論query数|
|---|---:|---:|---:|---:|---:|
|505|128|0|48|80|50|
|552|112|3|32|80|50|
|553|67|9|9|58|32|
|554|68|9|9|59|32|
|555, 556|68|9|9|59|32|
|559, 563, 567|88|7|29|59|32|
|562, 566|123|1|46|77|50|
|570, 571, 574, 577, 580|110|4|30|80|50|
|586|126|1|46|80|50|
|587|126|1|46|80|50|
|588|126|1|46|80|50|
|590, 591, 592, 593|126|2|46|80|50|
|594, 595, 596, 597, 601|124|2|46|78|50|
|598, 599, 600|126|2|46|80|50|
|602, 605, 608, 611, 614, 620, 623, 626, 629, 632, 635, 638, 641, 644, 647, 650|67|10|8|59|32|
|617|97|6|38|59|32|
|656|78|8|19|59|32|
|659|101|5|21|80|50|

以前の固定prefix 27 tokensにsuffix 59 tokensを加える構成は、全入力86 tokensだった。87 tokensや86 tokensがImajev全体の一律上限という意味ではない。今回のmoduleに対してdriverが設定している32-query経路の条件はprefix 1〜38 tokens、suffix 1〜59 tokensであり、その全組合せを実測したという意味ではない。今回の#617は38＋59＝97 tokensで32 queryを実行した。より長い128-token入力もハーネスは保持するが、32 queryでの実行を保証せず、今回の#505は48＋80＝128 tokensで50 queryだった。

### prefix準備と実測コスト

各prefix bankは、モデルの先頭部分を計算する66 queryと、24個のlinear-attention層の状態をcodecで準備する24 queryを要した。1 bankあたり90 query、11 bankで990 queryになる。prefixとcodecは今回すべて新規準備し、以後は同一のprefixを使う入力間で共有した。

|区分|query数|記録された処理時間の合計|
|---|---:|---:|
|18種類のsuffix推論|7×32＋11×50＝774|695.64秒|
|11 bankのモデルprefix準備|11×66＝726|202.09秒|
|11 bankのcodec準備|11×24＝264|198.18秒|
|上記の合計|1,764|1,095.91秒（約18.27分）|

時間は各reportに保存された処理時間を足した値で、タスク開始から完了までの経過時間ではない。process起動やmodule照合などの全費用も含まない。suffix推論1入力の記録時間は22.20〜74.95秒だった。ローカル環境の測定であり、mainnetでの遅延・throughputへ読み替えない。

32 query・50 queryという値は、prefix準備後の1入力の推論呼び出し数である。初回prefix準備、codec準備、module/state-tree照合、モデル重みのupload・準備updateは含まない。今回の合計1,764 queryも、モデルの導入から始めたcold startの全呼び出し数ではない。prefix準備を省いて32 queryだけを初回総費用として示すことはできない。

|suffix推論の実行検証値|最大観測値|
|---|---:|
|1 queryのhandler命令数|4,924,884,361（検査上限50億未満）|
|query時heap|4,140,892,160 bytes（検査上限4 GiB未満）|
|1 queryのrequest|1,940,728 bytes（検査上限200万bytes未満）|
|1 queryのreply|1,708,182 bytes（検査上限200万bytes未満）|

命令数はhandler計測でCDKのCandid decode/encodeを除く。request/replyは記録されたCandidデータのサイズで、HTTP・CBOR・署名等の転送量を含まない。heapは記録されたquery終端のlinear memoryの観測値で、allocator使用量やquery途中の厳密な瞬間ピークではない。上限を満たしたことは確認できるが、命令数・通信サイズ・heapの最大値は上限に近いため、未検証の入力長や実行経路へ無条件に広げない。

## ICで実行するために行った最適化

### 制約に合わせて計算を分割する

IC上のこの実験では、1 queryの命令予算、要求・返信のサイズ、Wasm32のheap容量を同時に守る必要があった。全モデルを1 queryで計算することはできず、細かく分けすぎると中間状態の転送・符号化・復号・検証を繰り返す。単純にqueryをまとめると、今度は命令や2 MBの通信予算を超える。最適化は、計算量を減らすことと、予算内に収まる境界へ分割し直すことの両方を進めた。

ここでいうqueryは、モデルの一部を計算する1回の呼び出しである。入力全体に対する1回の判断は複数queryで完成する。重みの導入・固定準備にはupdateを使い、今回の入力に対するモデル数値計算は通常queryで行った。質問ごとの途中状態をquery間で永続化する代わりに、clientが持ち回る構成にしている。

### 重みの量子化と固定準備

テキスト経路に必要な重みだけをpackへ出力し、vision・MTP・通常の生成用headをこの実行経路から外した。初期の非量子化テキストpackは8,901,719,552 bytes、INT8 packは4,702,451,200 bytesで、容量を約47.17%減らした。基盤の行列・embedding等をINT8にする一方、LoRAと専用readoutはF32、必要なscalarやnorm等はBF16/F32で保持する。[初期の全層接続とpack](FULL_INFERENCE.md)に変換条件を記録している。

さらに、固定INT8重みをqueryのたびにstable memoryから読み、展開・コピーする処理を減らすため、モデル準備時にcacheへ載せる経路を追加した。F32 LoRAやreadoutも準備時に復元・有限性検査を済ませ、query内では変更不能なsliceを借用する。固定cacheはINT8 dense 272個＋F32 tensor 449個、計721 tensor、4,065,416,192 bytesの構成である。embeddingの必要行と一部の小さな重みはstable packから読む。

重み配置の並べ替え、RoPEのsin/cos、BF16活性化関数の表も固定準備へ移した。活性化関数の表は、元のWasm演算の出力bitを保存する方式で、別の近似式へ変更するものではない。準備費用とheap容量を増やして、毎queryで繰り返す仕事を減らす交換条件がある。721項目の重み準備updateは、今回の774推論queryや990 prefix準備queryには含めていない。[固定F32重み](PREPARED_F32.md)、[活性化表](PREPARED_ACTIVATION.md)に詳細を残した。

### INT8整数dotとSIMD kernel

初期実装では、INT8重みをF32へ展開してから積和していた。後に基盤投影の入力もtokenごとの256列blockでINT8化し、I32で内積を計算してからactivation scale・weight scaleを適用する経路を接続した。LoRAには元のF32入力を渡し、F32再帰状態やreadoutまでINT8化しない。

この整数方式導入時の履歴では、同じ132-token入力のhandler命令を1,761,784,928,449→1,022,432,919,278へ約41.97%減らした。ただし、入力の量子化と積和順序の変更を含むため、元のF32方式へのbit一致を主張する改善ではない。今回の32/50-query実装の数値一致は、採用済みの整数方式を基準にしている。[整数方式の導入と数値差](INTEGER_ARITHMETIC.md)を参照。

整数方式の中では、Wasm SIMDで複数出力・複数tokenの重みと入力ロードを共有し、内側loopの定数展開、一時配列の削減、dotとscale適用の接続を行った。複数tokenのINT8投影には浅いStrassen系の変換も使い、256列block内の整数計算とF32加算順序を保つ。終端の1 tokenは、2 token用の片側をゼロで埋める方法から直接SIMD計算へ切り替えた。

元F32 LoRAも複数出力で入力ロードを共有し、DeltaNetの128×128状態はkernel内で保持して再読み書きや重複する減衰計算を減らした。今回の凍結Wasmは、INT8投影・1-token投影・F32 LoRA・Delta状態保持などの6本のWAT kernelを置換し、wasmparserによる検証を通したbuildである。通常の最新ソースを再buildしただけのWasmと同一とは扱わず、保存したbuild reportとmodule hashを基準にする。

### 同じ入力の量子化・LoRA A積を共有する

Q/K/V、MLPのgate/upなどは同じ入力に対して別の投影を行う。そのたびにblock256量子化や同じLoRA A積を作り直さないよう、同一query内で結果を共有した。出力行の分割でqueryをまたぐ場合も、量子化済み入力・scale・元F32のA積をclient-heldのcarryとして渡し、後続queryがそのまま利用する。

このcarryは計算の途中の値であり、最終判定のcacheではない。queryをまたいでも元の数値と進捗を保持し、shape・model・pack・input・stepを照合する。同じデータを何度も計算することと、異なる入力へ誤って流用することを区別する仕組みである。

### 可逆通信とprefix状態の圧縮

中間値をすべて4 byteのF32で往復すると、通信量と1メッセージのサイズが大きくなる。`bf16-block256-exact-v1`では、BF16で正確に表せるblockを2 byte/要素、F32が必要なblockを元の4 byte/要素で保存する。追加の丸めを行わず、F32再帰状態のbitを保持する。残差や部分和にも可逆のplane・辞書等の圧縮を使い、最終判断に必要な値を通信予算へ収めた。

DeltaNetのprefix状態は、全headの大きなF32 stateを毎回送ると容量を使うため、元の更新量logから正確に復元する経路と、一部を圧縮stateとして送るhybrid経路を組み合わせた。単体codecの初期文書には全推論未接続と記されているが、今回の実行ではhybrid packetをprefix-start経路へ接続していることを、command・cache identity・全体reportで確認している。

今回11 bankを準備したのは、単に共通指示だけを使うより長い一致部分を事前計算し、入力を変えずにsuffixを短くするためだった。新しい入力の先頭がbankと一致しなければ追加準備が必要であり、どのproposalでも同じ準備を無料で共有できるわけではない。

### 演算融合と層をまたぐquery配置

残差加算とRMSNorm、MLPのgate/up→SwiGLU→down、AttentionのK/V→Q→norm/RoPE→attention→gate、Deltaの投影→conv→再帰→出力投影を同じqueryへつなぐことで、中間配列の返信と再入力を減らした。query内部でも検査済みの値を直接渡し、重複する符号化・復号・コピーを省く。外部入力の検証まで省く方式ではない。

冒頭では、suffixのtoken IDからembedding・最初のnorm・Delta計算へ接続するprefix-startを使う。後半では、ある層のMLPを完了するqueryに次のDeltaやAttentionの一部を入れるroll/join/tail経路を作った。50-query経路は、この層をまたぐ配分を使う。

32-query経路は、3つのlinear-attention層と1つのfull-attention層が繰り返す構成に合わせ、MLPの残り・次のAttention・次のMLP先頭をつないで、4層の処理を4 queryへ組み直した。各queryに入れる出力行数やhead数を均等化し、特定の1 queryだけが命令予算を超えないよう調整した。今回の#617でも初期配分は上限を超えたため、検証済みのbalanced v3配分を使っている。

query数を減らす融合には、carryを大きくして通信量や合計命令が増えた途中版もあった。融合したという理由だけで速度向上とは判断せず、query数・命令数・通信量・heapをそれぞれ実測している。

### 判定に不要な終端の計算・返信を省く

判定には最後の位置のhiddenが必要なので、最終Attentionの不要なQ/outputや最終MLPを全token分計算する処理を省いた。一方、前段のhiddenや最終tokenが参照するK/V等は必要な範囲で計算する。入力の前半を読み飛ばす方法ではない。

最終2層のMLP・Attention・norm・readoutもterminal tailとして接続し、継続しない終端状態や、内部で消費したhiddenの返送を省いた。このため最適化比較では、保存された31層のhiddenと32層のconv/KV/position、最終normalized hidden、logits等を照合し、非返却のlayer30 hiddenを直接比較したとは記載しない。

### 今回の入力を変えずに得た効果

同じ18種類の入力について、保存済みの元実行と今回の実行を比較する。質問・選択肢・数値・モデルpack・閾値を固定し、入力表現を変えて得た削減とは分けた。

|指標|保存済みの元実行|今回の新規実行|
|---|---:|---:|
|18入力の推論query合計|3,703|774|
|今回のprefix・codec準備query|0|990|
|準備＋成功推論query|3,703|1,764|
|各入力のraw logitsと最終判定|比較基準|18種類すべて一致|

推論queryは約79.10%減、今回の準備を含めた成功実行のquery合計は約52.36%減となった。これは固定重み準備・upload・module照合を除いた18入力全体の比較であり、単発の任意入力に対する削減率ではない。

過去の配分探索には、成功15 queryと上限超過3 query、追加の入口検証32 queryもあった。その探索時は774＋990＋18＝1,782 query、入口検証込み1,814 queryとして別に記録した。今回の新規再実行1,764 queryへ探索費用を混ぜず、探索が無料だったとも扱わない。

同じ判定とlogitsを維持する最適化なので、意味的な判断精度が改善したとは言えない。モデルが誤った方向へ傾く例もそのまま残る。今回の成功は、固定した判断をより少ないqueryで再現し、資源予算内に収めたことにある。

### 今回の評価に含めない別の最適化

LoRAを基盤の重みに統合して再量子化すると、別々のA/B積を省ける。しかし数値と確率が変わる。別実験では3入力で命令を約13.21〜13.52%減らしたが、今回のproposal評価は従来のINT8基盤＋別F32 LoRAを保持したpackを使う。Wasmの名前にmergedが含まれていても、この評価でmerged packを使ったという意味ではない。

深いStrassenやWinograd、Hopcroft/Kerr等の候補には、乗算数は減っても変換・復元等の命令が増え、不採用にしたものがある。乗算回数の理論的削減を、そのままICでの性能向上として計上しない。

また、都度払いupdate推論や共通runtimeへの抽出は別の作業である。今回の32/50-queryという結果を、update呼び出し数や後続の別buildの性能に読み替えない。mainnetの本番運用・費用・query応答のcertificationは、今回の報告の検証範囲に含めていない。

## 閾値と今後の評価方針

閾値0.6はA/B logitsの差から得る未校正スコアの基準で、正解確率60%を意味しない。

同じlogitsで閾値を0.5に下げると、条件付きレビューでreject相当の32件中、rejectを返す件数は10→28件へ増える。一方、最低lock 1→2日の16件もrejectになる。フィルターとしては、これらは元からholdで止まっており、rejectへ変更しても通過防止の改善にはならない。今回の目的では、holdを減らすための閾値引き下げは不要とする。

今後は、危険と確認した案のapprove通過件数を第一の指標とする。同時に、安全・許容可能と確認した案の通過停止件数を測り、追加確認の負担を把握する。今回の集合には独立した人間の正解ラベルがないため、一般的な見逃し率・誤停止率は未測定である。

優先する追加確認は、通過した#656・#659の数値以外の危険と、別の未使用proposalに対する通過防止の検証である。すべてをholdにすれば通過は防げるため、停止できた件数だけで性能向上とは判断しない。確認済みの許容可能な案を通せることも併せて評価する。

## 根拠資料

共有用の写しを[根拠資料の一覧](evidence/proposal-filter-20261007/README.md)に保存した。原本と出力のSHA-256、パスの変換、JSONの抜粋範囲は[provenance](evidence/proposal-filter-20261007/provenance.json)に記録している。保存済み結果の共有であり、新しい推論や独立した実行証明ではない。

- [全161件の新規再判定](evidence/proposal-filter-20261007/RESULT.md)
- [全161件の判定・経路・理由](evidence/proposal-filter-20261007/DECISIONS.md)
- [実行の独立検証](evidence/proposal-filter-20261007/verification.json)
- [元proposalとの根拠照合・閾値比較](evidence/proposal-filter-20261007/REVIEW.md)
- [各proposalの旧新値・条件付きレビュー](evidence/proposal-filter-20261007/review.json)
- [モデル・tokenizerの固定情報](../MODEL_LOCK.json)
- [量子化packのmanifest](evidence/proposal-filter-20261007/pack-manifest.json)
- [prefixとsuffixの実行計画](evidence/proposal-filter-20261007/plan.json)
- [実行時module・再利用方針](evidence/proposal-filter-20261007/execution-policy.json)
- [新規推論・準備queryの集計](evidence/proposal-filter-20261007/fresh-summary.json)
- [128-token入力の実行記録例](evidence/proposal-filter-20261007/run-000.json)
- [IC最適化の手段と履歴](COMPUTE_REDUCTION.md)
- [凍結Wasmのbuild・feature・kernel置換記録](evidence/proposal-filter-20261007/build.json)
- [同じ18入力のquery削減と中間状態照合](evidence/proposal-filter-20261007/OPTIMIZATION.md)
- [同じ18入力の独立照合](evidence/proposal-filter-20261007/optimization-verification.json)
- [32-query経路の配分と検証履歴](QUERY32_PROGRESS.md)
- [別実験のLoRA統合・再量子化](MERGED_ADAPTER_QUERY32.md)

以前の断片ごとの全件hold実験は、proposal単位の判定として成立していないため、この報告のフィルター性能の根拠には含めていない。
