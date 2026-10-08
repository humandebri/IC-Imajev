# 有料update推論の116-token対応と512-token分割版

以下の実測は変更時点の通常116-token構成と512-token分割構成によるもの。現在の標準ビルドは512 tokens受付であり、最新の処理選択は [入力長に応じたスケジューラ](ADAPTIVE_TOKEN_SCHEDULER.md) を参照する。

互換性確認用の116-token構成は、固定27-token prefixを含む116 tokensまで受け付ける。workerの区切りを30B命令、回数上限を8に合わせた。公開済みcanisterへの適用は、このソース変更とは別操作になる。

`experimental-update-token-chunks` featureを有効にすると、全入力512 tokensまで受け付ける実験構成になる。worker上限は116以下で8回、117〜256で32回、257〜512で64回。外部の `infer` は1回で、内部の自己呼び出しが最後まで進めて結果を返す。Candidの型は変更しない。

embedding・初期norm・各層のDeltaNet／Attention・MLPを最大57 tokensずつ処理する。DeltaNetの再帰状態とconv履歴、Attentionのprefix＋処理済みsuffix KVを引き継ぐ。入力を切り捨てず、全suffixのhiddenを次の層へ渡す。再帰状態はjobが所有し、immutable prefix cacheを書き換えない。

256以下では従来の計算経路を維持する。256超でも、Attentionの入力・Q＋KV・RoPE後のQ＋KVの配列上限、または75Mの演算量制限に当たる区間だけ16 headsを8＋8へ分割する。全KV履歴を1つの配列へまとめず、head groupごとにprefixとsuffixを順に詰める。K/V投影、Qの量子化、QのLoRA A積、最後のoutput投影はそれぞれ1回に保つ。

重い演算の間で命令数を確認し、次のworkerへ状態を引き継ぐ。途中で失敗したjobは状態を破棄し、既存の返金処理を使う。進捗の検証にはjob IDと分割後のstage上限を使う。

## ビルドと検証

実機の性能比較には、既存の凍結済み最適化runtimeと34個の検証済み投影kernelを使う。`build_paid_token_chunks.py` は状態継続部分を追加し、kernel本体のhash一致とWasm validationを確認する。通常のCargo buildと同じ性能とは扱わない。

`prove_paid_token_limit_local.py` は、明示したローカルcanisterをスナップショットで保護し、upgrade・準備・測定後に元のmodule、重みcache、packを復元する。作成したcallerは停止する。`audit_paid_token_proof.py` は保存したCandid返信と数値配列を、ネットワークを使わず再検証する。

長い入力の参照計算は、prefix cacheを使わず、通常queryから全入力を計算する。DeltaNetは最大64 tokens・16 headsの別経路で状態を引き継ぐ。全32層のsuffix hidden・conv／suffix KV、最終norm hidden、判断・logits・確率を比較する。

通常canisterの17 tests、分割feature付きcanisterの18 tests、key-major状態を含むruntimeの70 testsが通過した。57＋33 tokensへ分割した再帰演算の出力・最終状態のbit一致、512位置までのKV履歴の並び、256以下で追加groupを使わないこと、Attention内部配列の境界、異なるlayer／入力／offsetの拒否、失敗時のcarry消費を含む。実際のモデル全層の一致は別途local実測で確認する。

測定入力はBOOM 653の84-token入力に普通のtokenを挿入した長さ試験用のfixtureであり、長文に対する精度評価ではない。時間はローカル1回の値。cyclesは推論前後のcanister status残高差に受領料金を足した概算で、待機費用とstatus呼び出し費用も含む。mainnet料金の設定根拠にはそのまま使わない。

## 116-token通常構成の実測

`artifacts/paid-116-v1/build-v3` を使用し、84／116 tokensが完走した。116 tokensは内部worker 7回、43.87秒、187.06B命令、最大heap 4,188,995,584 bytes（約3.90 GiB、4 GiBまで101.06 MiB）。cycles残高差による概算は187.54B。各入力1回の値で、準備費用を除く。

各66 queryによる独立計算で、84／116 tokensの全32層のsuffix hidden・conv／suffix KV、最終norm hidden、判定・確率・logitsがbit一致した。117 tokensはquote／inferで拒否し、添付100B cyclesの全額未受領返却を確認した。成功IDの再送も追加受領なし。元状態へ復元し、試験callerを停止した。

証跡は `artifacts/paid-116-v1/proof-v3/report.json` と `audit.json`。保存した16件のCandid返信を再decodeし、配列と判定も再検証した。module SHA256は `7628665efcf403b72be59dbf16afb54892e1d15b850e6dbb36f1ebb77dcfbb91`。

最初の試行は計測スクリプトの残高取得先を修正するため中断し、2回目は試験callerへ割り当てるlocal cycles残高が足りず中断した。いずれもsnapshotから復元済み。完了した測定は3回目の `proof-v3` であり、不完全な試行を性能集計へ含めていない。

## 旧256-token実験構成の実測

`artifacts/paid-256-v1/build-v3` を使ったSSD上の再測定 `proof-v2` が完了した。module SHA256は `d987f125e0f7195e67d7eb7710c7c66e04bcacd71f9420fd59edcd9aeafbb77d`。

| 全入力tokens | 内部worker | 時間（秒） | 合計命令（B） | 最大heap（bytes） | cycles概算（B） |
| --- | --- | --- | --- | --- | --- |
| 84 | 4 | 32.57 | 121.67 | 4,177,199,104 | 121.85 |
| 117 | 7 | 49.83 | 192.83 | 4,177,199,104 | 193.28 |
| 132 | 8 | 52.32 | 223.61 | 4,177,199,104 | 223.71 |
| 256 | 17 | 125.04 | 496.19 | 4,189,782,016 | 497.18 |

256 tokensのheapは約3.90 GiB、4 GiBまで100.31 MiB残った。通常116-token版からの増加は0.75 MiB。これはWasmの確保済みページ数から求めた最大値であり、生存中の配列だけのサイズではない。256の最大worker命令数は31.86Bで、実際の各update messageが命令上限内で成功した。記録用処理と返信の小さい末尾はcounterの測定範囲外。

通常queryによる独立計算は84 tokensで66 query、117 tokensで610 query、256 tokensで1,259 query。各入力で全32層のsuffix hidden・conv／suffix KV、最終norm hidden、判定・確率・logitsがbit一致した。132は容量・性能測定のみで、独立計算との全層比較は行っていない。

257／500 tokensはquoteとinferの双方で上限超過として拒否し、添付100B cycles全額が未受領で返った。成功IDの再送も追加受領なし。workerは外部から呼ぶと `self only` で拒否した。

256-token jobの途中stage 21でtrapを注入し、返金 `Done` とcallerの実残高を確認した。787Bの料金に対して、残高差はrelay等の実行費用約24.03Mのみ。失敗IDの再送は追加受領せず同じ失敗結果を返した。faultを解除して新しい84-token jobを実行し、以前の判定・logits・確率との一致も確認した。

保存した37件のCandid返信を再decodeし、全比較配列と判定をオフラインで再検証した。`proof-v2/report.json` と `audit.json` に記録している。最後にbaselineのmodule・全cache・全packをsnapshotから復元し、一致確認後に試験snapshotを削除、caller `4fbx2-kt777-77775-aaabq-cai` を停止した。公開canisterには適用していない。

## 512-token実験構成の実測

`artifacts/paid-512-v1/build-v2` による `proof-v2` が完了した。module SHA256は `651320f6b33d83e6c6a47d05050e301d59bd2e83c98447bb1d81d1e308da8d3a`。34個の投影kernel本体は従来の検証済み版とhash一致を確認している。

| 全入力tokens | 内部worker | 時間（秒） | 合計命令（B） | 最大heap（bytes） | cycles概算（B） |
| --- | --- | --- | --- | --- | --- |
| 84 | 4 | 27.88 | 121.67 | 4,177,199,104 | 121.73 |
| 116 | 7 | 50.26 | 187.06 | 4,188,995,584 | 187.60 |
| 256 | 17 | 121.61 | 496.16 | 4,188,995,584 | 496.37 |
| 359 | 24 | 188.89 | 723.86 | 4,192,796,672 | 724.57 |
| 512 | 35 | 275.17 | 1,074.61 | 4,202,561,536 | 1,077.06 |

512 tokensは固定prefix 27＋suffix 485。外部inferは1回のまま、内部worker 35回で完了した。最大heapは約3.91 GiB、4 GiBまで88.125 MiB残った。最大worker命令数は31.97Bで、各update messageは実際に成功した。359はRoPE後Q＋KVの配列サイズだけが先に制限へ達する区間を含む境界測定。

独立した通常queryによる計算で、84／116／256／512の全32層のsuffix hidden・conv／suffix KV、最終norm hidden、判定・確率・logitsがbit一致した。query数は順に66／66／1,259／2,456。359は容量・性能・境界測定のみで、独立した全層比較は行っていない。513 tokensはquote／inferで拒否し、添付100B cycles全額の未受領返却を確認した。

512-token jobのstage 20でtrapを注入し、返金Doneとcaller実残高を照合した。料金1,555Bに対して、残高差はrelay等の実行費用約27.23Mのみ。失敗IDの再送は追加受領なし。fault解除後の新しい84-token jobも、以前の判定・確率・logitsと一致した。外部からのworker呼び出し拒否と、成功ID再送時の追加受領なしも確認済み。

保存した41件のCandid返信と全比較配列・判定をオフラインで再検証した。証跡は `proof-v2/report.json` と `audit.json`。最後にbaselineのmodule・全cache・全packをsnapshotから復元し、一致確認後にsnapshotを削除、caller `4xhad-gd777-77775-aaacq-cai` を停止した。公開canisterには適用していない。

最初の512-token試行は、内部Q＋KVの境界チェックを補強したbuildへ切り替えるため、cache準備中に意図的に中断した。snapshotから復元・一致確認済みで、callerも停止済み。完了した測定は2回目の `proof-v2`。

### 短い入力への影響

旧256版の84／256、通常116版の116と同じfixtureで比較した。内部worker回数はそれぞれ4／7／17のまま。全32層の中間値・conv／KV、最終hidden、判定・確率・logitsも以前の証跡とbit一致した。

| tokens | 以前の合計命令 | 512対応後の合計命令 | 差 |
| --- | --- | --- | --- |
| 84 | 121,668,215,831 | 121,668,215,831 | 0 |
| 116 | 187,058,449,377 | 187,058,454,082 | +4,705（約0.0000025%） |
| 256 | 496,187,964,495 | 496,155,627,027 | −32,337,468（約0.0065%減） |

追加head分割は256以下で使わず、短い入力の重い計算量は維持できた。時間は84が32.57→27.88秒、116が43.87→50.26秒、256が125.04→121.61秒。各1回の測定で環境負荷を統制した反復benchmarkではないため、時間の改善・悪化の因果関係は断定しない。比較記録は `artifacts/paid-512-v1/short-input-comparison.json`。

## 再実行

凍結済みkernel・モデル・認証済みlocal networkを持つ元checkoutを `--source-root` に指定する。出力先は未作成のディレクトリを指定する。snapshotと実行中の状態を保持するため、local networkの保存先に30 GiB以上の空きを事前確認する。

```sh
python3 scripts/build_latest_common_prefix27.py \
  --source-root /Volumes/KINGSTON/ICP/IC-Imajev --legacy-116 \
  --directory artifacts/paid-116-v1/rebuild
python3 scripts/build_paid_token_chunks.py \
  --source-root /Volumes/KINGSTON/ICP/IC-Imajev \
  --directory artifacts/paid-512-v1/rebuild --fixed-token-tiles --diagnostics
python3 scripts/prove_paid_token_limit_local.py \
  --source-root /Volumes/KINGSTON/ICP/IC-Imajev \
  --build artifacts/paid-512-v1/rebuild \
  --directory artifacts/paid-512-v1/reproof \
  --lengths 84,116,256,359,512,513 \
  --expected-limit 512 --reference-lengths 84,116,256,512 --fault-recovery
python3 scripts/audit_paid_token_proof.py \
  --directory artifacts/paid-512-v1/reproof \
  --helper artifacts/paid-update-v1/tools/args
```

通常116-token版の検証は、対応するbuildを指定し、`--lengths 84,116,117 --expected-limit 116 --reference-lengths 84,116` を使う。通常版にはfault endpointを含めないので `--fault-recovery` を付けない。

旧256-token版は500を拒否した。現在のソースでは追加head分割により512まで受け付ける。512超は、追加の演算・位置範囲・メモリ・命令数の検証が必要で、今回の対象に含めない。

## ローカル保存先の復旧

256-token版の初回試行は推論完了後、外付けドライブへのstate書き込みで `No space left on device` が発生し、PocketICが停止した。これはcanister heapの上限到達とは別の、ホストのディスク容量不足だった。参照計算が終わっていない初回の `artifacts/paid-256-v1/proof/report.json` は不完全な証跡として残す。

復旧時に使ったicp-cli 1.0.2のnetwork startはstateを再作成するため、既存network内のsnapshotは使えなくなった。事前に保存されていたportable baselineのWasm・heap・stable memoryのサイズとSHA256を検証して転送し、同じモデルcanister IDへ復元した。module、全weight cache、全packの一致を確認済み。旧network内の試験callerは復元していない。

保存先をワークツリーから独立した内蔵SSDの `/Users/0xhude/.codex/local-network-data/imajev-8001-20261008` へ移し、元projectのlocal networkパスからsymlinkで参照する。別名で保存した旧ディレクトリやportable backupは削除していない。復旧の証跡は `artifacts/paid-256-v1/recovery`、保存先の記録は `artifacts/paid-256-v1/storage-relocation.json`。再測定ではnetworkを再起動せず、canisterのsnapshotだけで保護・復元する。
