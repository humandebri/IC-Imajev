# Imajev-4BをInternet Computerで動かすための実験計画

2026-10-02追加：token tile32/16/8、出力tile8、INT8直接ロードを比較。改善した出力tile8＋直接ロードを採用し、通常708 queryを0.711兆命令へ削減。共通prefix45 tokenのclient-held conv/DeltaNet/KV state cacheを実装し、準備済み617は611 query・0.493兆命令。初回準備578 query・0.256兆命令を別計上。3入力の全32層/state/判断bit一致。50 query未達。最新はdocs/DIRECTIONS.md。


2026-10-02追加：整数dot256列の正確なI32加算、gate/up/SwiGLU融合とwork予算に基づく分割を実装。通常query全32層で708 query、0.739兆命令、1.248 GB、単回78.490秒。以前のINT8方式と全層/state/判断bit一致。最大45.80億handler命令の設定は明示した実験オプションで、既定の保守的分割は残す。50 queryは未達。最新はdocs/MLP_FUSION.md。


2026-10-01追加：query内stageまで計測し、整数scale/reduction・BF16 codec・RNE量子化をSIMD化、token分割を8境界へ整列、残差/normを融合。0.847兆命令・868 query・全層/state/判断bit保存。50 query未達。最新はdocs/BOTTLENECKS.md。

作成日・更新日: 2026-10-01（日本時間）
状態: 固定モデル・公式ホスト比較・Rust/Wasm・INT8 packの全32層local query推論まで完走。2026-10-01、数値維持の追加改善で1.762兆命令・1,116 query、演算方式変更の許可後は整数base＋F32 LoRAで1.022兆命令・932 queryへ削減。ホスト23件の同じpackのF32/整数ラベル一致を確認したが、確率は変わり、校正維持や一般精度を保証しない。50 queryは未達。最新実測・精度評価はdocs/INTEGER_ARITHMETIC.md。以下の当初計画を残す。mainnet/push/PRは未実施。

## 目的と最初の到達点

Layaより大きい公開の判定モデルを、Internet Computerのcanisterで分割queryとして実行できるか検証する。採用候補はImajev-4Bに決定した。まずテキスト入力だけを対象にする。画像の認識はモデルの能力として存在するが、初回の移植範囲には含めない。2B版は4Bの移植が容量や実行環境で難しい場合の後続候補とし、最初から2Bの移植を必須にしない。

最初の到達点は、短いテキストと2〜7個の選択肢、1質問を入力し、固定した公式実装と比較可能なlogits・確率・unknownの扱いを、ローカルcanisterのqueryだけで返すことである。公式評価に合わせ、1回のforwardに相当する計算を複数queryへ分割し、選択肢順の4回rotation平均は初回には使わない。モデルのuploadと読み込み準備にはupdateを使う。推論途中の状態はクライアントが保持する。

BOOM DAO提案の評価で改善を示せることと、canister上で実行できることを別々の達成条件にする。モデルを大きくしただけで判定精度が上がるとは仮定しない。

## 採用モデルと評価条件

Imajev-4Bを選ぶ根拠は、未公開問題を含むJevBench v1.4.2.2の総合評価で67.37、1位となった比較結果である。総合点は判断能力・校正・速度・費用を組み合わせた指標であり、正解率でも、あらゆる業務で最も優秀という証明でもない。画像対応を採用理由の中心には置かない。

| 対象 | 位置付け | 固定・確認する内容 |
|---|---|---|
| mohit67890/imajev-4b | 採用候補 | 評価済みrevision c9e5f132から開始。base、LoRA、専用readout、校正を一組で固定 |
| Qwen/Qwen3.5-4B | 基盤モデル | 採用adapterと対応するbase revision、config、tokenizer、processor |
| mohit67890/imajev | 参照実装 | 評価済みserver commit a0134749e0900189c129cd6bb5000969f3b64bb5を起点に確認 |
| convaiinnovations/laya-typed-decisions | 比較基準 | IC-Laya側の固定モデル・Wasm・既存結果 |
| JevK5-4B / Plumb-4B | 必要時の比較候補 | ホスト上で同じ問題を評価。移植対象はImajevに絞る |

公式評価の条件はrotations=1とcalibration.jsonである。移動するmainや4回rotationの推奨例を、この条件と混ぜない。実験開始時にadapter revisionの短縮表記を完全なcommitへ解決し、ファイルのSHA256も保存する。必要な参照環境が再現できなければ、その差分を明記する。

ImajevはTypeSafeのJevとは独立した公開モデルであり、Qwen3.5の言語モデルへLoRAと専用decision readoutを加えた構成である。画像も入力できるが、初回はテキストだけを扱う。公式APIのunknown_probabilityとabstainedを含め、情報不足をどう返すかも参照する。判定の確率や棄権は正しさを保証しない。

LocalLLaMA/typed-decisionsを補助ベンチに使う場合は、trainで学習・調整したモデルとzero-shotのモデルを分ける。このデータのgoldは教師モデルの回答分布であり、スコアを実世界の正解率として扱わない。今回の採用はそのベンチへの適応点だけで決めていない。

- Imajev公式実装: https://github.com/mohit67890/imajev
- 4Bモデルカード: https://huggingface.co/mohit67890/imajev-4b
- 固定評価の説明: https://github.com/fstandhartinger/jevbench/blob/main/docs/RELEASE-v1.4.2.2.md
- Layaモデルカード: https://huggingface.co/convaiinnovations/laya-typed-decisions
- Typed Decisions: https://huggingface.co/datasets/LocalLLaMA/typed-decisions
- ICリソース上限: https://docs.internetcomputer.org/references/resource-limits/
- IC料金表: https://docs.internetcomputer.org/references/cycle-costs/

## 独立した作業領域

作業領域は`/Volumes/KINGSTON/ICP/IC-Imajev`とし、リポジトリ名は`IC-Imajev`にする。既存のIC-Laya-Standaloneのソース、Git状態、モデル、稼働中のローカルcanisterを変更しない。既存実装は参照元として読み、再利用するコードは依存関係とライセンスを記録して新しいリポジトリへ取り込む。

実装開始後の配置案:

```text
PLAN.md                     この計画書
README.md                   再現可能な実行手順
Cargo.toml                  Wasm向け実行エンジンのworkspace
crates/imajev-runtime/        演算・モデルpack・分割状態
canisters/inference/        upload・準備・query API
client/                     tokenizer・query実行・途中保存
scripts/                    export・変換・比較・測定
fixtures/                   小さな合成tensorと固定入力
benchmarks/                 問題定義・評価手順・正解根拠
checkpoints/                重み・変換pack（Git対象外）
artifacts/                  生の測定結果（Git対象外）
docs/                       設計判断・要約・採用理由
```

Gitを初期化するときに重み、生成pack、ビルド成果物、キャッシュ、秘密鍵、生の測定ログをignoreする。再現用の小さなfixtureと結果要約だけを明示的に管理する。

## 段階1: モデルと実行条件を固定する

重みを取得する前に、固定revisionのbase config、adapter_config、safetensorsのヘッダー、公式のprompt生成と専用readoutコードを調べる。必要な演算、層数、hidden幅、語彙数、tensor配置、LoRAの対象とrank、readout、unknownと校正の設定を記録する。モデルカードの現在のmainと評価済みrevisionの構成を混ぜない。

Qwen3.5の演算を通常のattentionだけと仮定しない。対象checkpointでfull attentionと線形attention・recurrent系ブロックがどう構成されるか確認し、convolution、gating、正規化、位置表現などの必要演算を一覧にする。既存のCandleやCPU実装が対象revisionに対応しているかは、小さな入力で動かして確認する。

テキスト入力で不要なvision encoderをpackから除外できるか、公式のテキスト出力との一致で確認する。base全体のファイルサイズと、canisterへ載せる言語部分・readoutのサイズを分け、LoRAを統合する場合と実行時に適用する場合の容量・精度・命令数を比較する。

全重みの実バイト数、最大tensor、量子化の補助情報、tokenizer、推論時の状態を見積もる。4Bの単純な重み概算はINT8で約4GB、INT4で約2GBだが、これは実行メモリでも最終packサイズでもない。GiBとGBを区別して報告する。

成果物: MODEL_LOCK.jsonの形式設計、演算一覧、容量見積もり、利用する実装とライセンスの一覧。ロックにはbase・adapter・readout・tokenizer・processor・校正ファイル・公式serverの識別情報を個別に保存する。

## 段階2: canisterへ移す前に精度を比較する

固定した公式Imajev実装で4Bの参照出力を取得する。公式のMLXまたはPyTorch経路からこのマシンで利用可能な方法を選び、vLLMやCandleへそのまま読み込めるとは仮定しない。1質問・テキストのみ・rotations=1・対応するcalibration.jsonありを基準にする。baseのBF16、adapterとreadoutの実精度を記録し、量子化結果と分ける。

BOOM DAOの617など既存の問題を含め、最低投票期間、最大ロック期間、stake、mint、変更なし、情報不足を扱う評価集合を作る。提案ごとの事実、旧値の取得時点、質問、選択肢、正解の根拠を保存する。既存結果と同じ質問の比較と、情報を増やした質問の比較は別の列で報告する。

主な評価は、項目別の判定一致、重大変更の見逃し、誤警告、情報不足の扱い、logits差、確率差である。正解が定義できる集合だけで精度や校正を評価し、Layaとの一致率を正解率として扱わない。量子化やpromptの調整に使った問題と、最終評価用の問題を分離する。少数の提案だけで一般的な精度向上を主張しない。

成果物: 固定入力、参照logits、評価手順、Imajev-4B・Layaの比較表。JevK5-4BとPlumb-4Bは必要に応じてホスト上の比較へ加えるが、同時にcanisterへ移植しない。Imajevに改善が見られない場合は、全面移植より先に原因を調査して採用判断を見直す。

## 段階3: Rust/Wasmで演算を再現する

まず小さな合成tensorで個々の演算を検証する。その後、対象モデルの1ブロック、数ブロック、全体の順に参照実装と比較する。比較する中間tensor、許容誤差、対象の重み精度を事前に記録する。

最初は正しさを優先し、量子化誤差と移植誤差を混ぜない。全モデルを高精度でcanisterへ載せる必要はなく、ネイティブ側や小さなfixtureで高精度の比較を行ってから量子化版へ移る。

判定は公式promptのprefillと専用decision readoutで完了させる。Imajevの選択肢コード・unknown・読み出すhidden state位置・候補マスク・校正・APIへの変換を公式コードどおりに再現する。JevK5の回答文字の次token logit方式へ置き換えない。公開資料のreadoutは255個の選択肢コードとunknownからなる256出力であるが、採用revisionのshapeを実ファイルで検証する。必要なreadout行だけ計算する最適化は全readoutの参照結果と一致してから採用する。

型安全な出力も検証対象にする。許可した選択肢だけを返すこと、確率が有限かつ有効な範囲に収まること、確率の正規化とunknownの扱いが公式実装と一致すること、Scoreの値が定義したrubricに対応することを確認する。型の正しさを判定の正しさとして扱わない。

成果物: ネイティブ実行器、演算テスト、参照比較、Wasmでの小さな演算ベンチ。

## 段階4: 重みの配置と分割単位を決める

次の構成を順に実測して選ぶ。最初から複数canisterへ固定しない。

1. 4Bの量子化した言語部分の重みと専用readoutをheapへ保持する構成。LoRAの統合後の容量と作業領域まで含めて収まるか確認する。
2. 一つのcanisterのstable memoryへ全重みを置き、必要な層またはtensorタイルだけquery内でheapへ読む構成。
3. 層を複数canisterへ配置し、クライアントがactivationを次のcanisterへ渡す構成。

stable memoryに重みが収まっても、query内の読み出し上限、コピー、量子化の展開、演算の命令数が制約になる。対象subnetの現行上限と実測を使い、常に全重みをheapへ複製する実装を避ける。

分割は層境界を第一候補とする。1層だけでquery上限へ達する場合は、行列積のタイルやトークンブロックなど、層の内部を区切る必要がある。recurrent系ブロックの分割では内部状態と位置の継続も設計する。クライアントへ保持する情報を不要に増やさないため、まず層を順に進める方式を調べ、同じpromptに対する不要なdecode用cacheを作らない。

分割queryの状態には、モデル・revision・形式・位置・shape・入力の識別情報を持たせる。サイズ上限、展開上限、checksum、有限値、scale、進捗を検証する。別モデルや別入力の状態混在を拒否する。ただし、checksumや入力hashだけでクライアントによる改変を防げるとは扱わない。初期段階はowner専用の補助判定に限定する。

成果物: 構成ごとのheap・stable使用量、最大読み出し量、query命令数、通信量と、採用構成の理由。

## 段階5: 通信と計算を最適化する

重みの精度とactivationの通信形式を別々に比較する。重みはINT8を基準候補にし、容量が必要な場合はINT4を評価する。GGUFなど公開の量子化結果はホスト上の候補比較に使えるが、既存canisterの演算へそのまま読み込めるとは仮定しない。

中間状態はF32を比較基準にし、可逆圧縮、INT8、必要に応じた他の形式を比較する。重み量子化とactivation量子化を同時に導入せず、どちらが判定差を生んだか追跡する。

行列積での積和からscale適用、bias、丸めまでを可能な範囲でまとめ、不要なINT32・F32配列の確保を削減する。固定scaleや整数の乗算・シフトは校正データと未使用の評価データで検証してから採用する。Wasm SIMDの効果は命令数と時間の両方で測る。

通信が切れた場合に同じ状態から再開できるクライアントを作る。状態はバイナリで保存し、モデル識別情報とchecksumを付け、保存途中のファイルを有効なcheckpointとして扱わない。一時的な通信エラーは同じ要求で上限付き再試行し、命令上限では分割幅を縮小する。推論を自動的にupdateへ切り替えない。

成果物: 精度差・通信量・命令数・時間の比較と、採用した設定。

## 段階6: ローカルcanisterで通し測定する

IC-Imajev専用のローカルidentity、canister、ネットワーク設定を使う。既存のIC-Layaのローカル環境を流用して上書きしない。入力長はまず32・64・128 tokens、2〜7選択肢で確認し、成功後に256・512以上へ広げる。token数は自然文の文字数ではなく、公式promptと特殊tokenを含む実入力で数える。

各測定で、ソースcommit、Wasm hash、モデルrevision、pack hash、tokenizer hash、量子化設定、入力hash、分割幅、cache条件を保存する。

| 測定項目 | 報告する内容 |
|---|---|
| 判定 | 参照との差、選択肢順位、項目別の正解・誤り |
| 命令数 | 各query、最大値、全体合計、失敗分。handler内だけか全体かも明記 |
| 時間 | 各queryと全体、warm/coldの条件、複数回の分布 |
| 通信 | バイナリ要求・返信、成功分と失敗分、外側の通信を含むか |
| メモリ | heapとstable、準備中と推論中、観測ピークの測定間隔 |
| 初期準備 | upload量、update回数、warmup命令数、cycles差分 |

query上限の直前まで使い切る設定を標準にしない。初期の目安は各queryのhandlerを40億命令以下に抑え、Candidのdecode・返信encodeを含む実呼び出しの成功で確認する。分割後の最大命令数に加え、合計命令数や通信増加も評価する。

成果物: 再現可能なローカル推論、境界入力と再送・再開の検証、構成選択の報告。

## 後続の画像対応

テキスト経路が完走し、品質と費用を評価できてから画像対応を検討する。画像前処理、vision encoder、画像tokenへの変換、言語部分への接続、追加重みの配置を別の実装段階にする。画像tokenが増やす計算・中間状態・通信を測る。テキストだけの結果で画像経路の品質や費用を推定しない。

画像対応は今回のテキスト移植の完了条件に含めない。実施する場合は画像と構造化情報の矛盾、情報不足、二画像の比較など、実際の用途に対応した評価集合を作る。

## mainnetへ進む条件

ローカルで対象入力が完走し、参照との誤差と量子化による判定差が把握できてからmainnetの試験を検討する。実装開始時点ではmainnetデプロイを行わない。

mainnetでは、モデルを保持する費用、初回upload・warmup費用、必要なcycles残高と凍結閾値を見積もる。通常queryの現在の無料扱いだけを理由に費用や負荷を無視しない。ネットワークの待ち時間、同時実行、再試行の挙動はmainnetで別途測る。

最初の公開範囲はowner専用とする。upgrade後はstable memoryのモデルから実行用の状態を作り直せるようにする。補助判定をそのまま資金移動やDAO操作の承認へ使わず、その用途では認証・結果の検証・既存のupdate経路との関係を別途設計する。

## 最初に着手する作業

1. Imajev-4Bの評価済みrevisionと公式serverを固定し、base・adapter・readout・校正の組合せと容量表を作る。
2. このマシンで動く参照実装を選び、短い入力の選択肢logitsを取得する。
3. BOOM DAOの固定した比較問題で、Layaからの改善を確認する。
4. 最も移植が難しいブロックを小さなtensorでWasm実行し、命令数を測る。
5. その結果で4Bの重み常駐・stableからの部分読み込み・複数canisterのどれへ進むか決める。2Bへ縮小する場合は別の採用判断として記録する。

完了の判断は、モデルが大きいことではなく、判定品質・待ち時間・通信量・運用費用を実測して選べる状態になったことで行う。

## 2026-10-01 実装到達点

固定モデルの公式ホスト比較、Rust/Wasm移植、全層INT8 pack、client-held通常query分割とcheckpoint再開を実装した。BOOM DAO 617・132 token・rotations=1がローカルcanisterで完走し、参照likelyと一致。3,908 query・604秒・通信5.38 GB。詳細は `docs/FULL_INFERENCE.md`、各queryの実測は `docs/full-results.json`。全32層BF16 A/B、複数問題のINT8精度、通信圧縮は後続評価。画像/mainnet/push/PRは実施していない。

可逆BF16通信と大型queryにより、同じ質問を3,108 query・560.522秒・通信2.549 GBへ改善（通信52.64%減）。全層・最終hidden・判断logit/確率が改善前とビット一致。`docs/COMMUNICATION.md` と `docs/efficient-results.json` に実測を記録。
