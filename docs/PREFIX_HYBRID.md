# クライアント保持prefixを一部だけログから復元する候補

2026-10-03。専用canisterで全24層の復元だけを実装・実測。全モデルへは未採用。主問題67 query、50/32回には未達。

## 毎質問の再構成を省く

採用済みDelta logは、各質問の通常queryで全32 headのprefix状態を再構成する。全F32状態をそのまま送ると入力と合わせて2 MBを超える。今回の候補は、クライアントが20 headの正確なF32状態を可逆圧縮して保持し、残り12 headだけ従来のK・innovation・decay logとして保持する。canister側には質問の状態を保存しない。

F32の符号・mantissaは3 byteに格納し、exponentだけcanonical Huffmanで可逆符号化する。量子化を追加せず、F32の全bitを復元する。残すlogのKは元のBF16を2 byteで送り、innovation/decayは元F32のまま。隣接するhead pairは同じ形式にし、残ったlogを元の順序で既存SIMD復元へ渡す。head間の状態計算は独立し、乗算・加算の順序を変えない。prefix圧縮・選択はクライアントで一度行う想定で、主推論側への接続は未実装。

全24層の保存済み45-token prefixと87-token入力で、最大16,424-byte header、入力・convのBF16 bytesを含む見積もりは1,971,502〜1,978,453 bytes。20/32 head分のprefix outer-product再構成を省ける。これは動的命令の削減率ではない。全32 headを送る形式は、既存INT8入力の可逆圧縮と元F32 A積を仮定しても2,078,111〜2,124,681 bytesとなり、現在の2 MB制限を超えた。生結果`artifacts/prefix-dense/hybrid-capacity.json`。

## 復元だけの通常query測定

専用`7hukf-2d777-77775-aaakq-cai`、module `1c2ffa5266ec232591473ba07b8c1d5e1d8eab049c6982b2c82f63dcc3c7f9f9`。24層×従来/候補の48通常queryすべてで、独立した保存済みF32状態とnative decoderとWasmのdigestが一致。実行前後のmodule・source hashを確認した。質問依存のupdateは行っていない。

従来復元は各層174,175,935命令。候補は復号・入力の検査・残る12 headの復元・出力配置を含め、28.4448〜28.5702%減。24層合計1,191,016,209命令減った。主全推論265,931,336,448命令に対して約0.448%相当の規模だが、全推論の実測削減率ではない。新しいframe処理と検証、通信費用を接続後に評価する必要がある。

計測counterはdigestとCDKのCandid decode/encodeを除外する。単回時間は同時ビルド等の負荷を統制していない。性能改善の根拠は復元部分の命令数に限る。全モデルのhidden・判断・確率・query数はまだ比較していない。

通信との交換条件がある。診断の従来入力はlog全体をF32で送る形式であり、採用推論のBF16 K codecと異なる。診断payload差は24層合計8,477,771 bytes増。採用推論への接続では、元のBF16 Kを含むlogよりさらに増える見込みで、診断差を全推論の通信差とは扱わない。既存のframe checksumと900,000 logical float制限も維持して実接続を検証する。

## 実装と再現

`scripts/prefix_codec_bench/src/codec.rs`はtable長、canonical codeの過剰割当、payload/mantissa長、maskのpair境界、token上限、有限値、切れたcode、余分なbyte/paddingを拒否する。共通codeは10-bit lookupで復号し、長いcodeだけfallbackする。canisterはownerだけを保持し、全状態はquery引数で受け取る。

`scripts/explore_prefix_dense.py`は実データの容量と全bit復元を検査する。`scripts/quantized_input.rs`は入力圧縮の容量調査専用で、ホスト計算結果を推論へ渡さない。`scripts/check_prefix_codec.py`は保存状態をoracleとして48 queryを比較する。sourceとWasmは`artifacts/prefix_codec/source.zip`等、raw結果は`artifacts/prefix_codec/check/report.json`。生成物はGit ignore対象。

ビルドは`build_prefix_codec_cached.py --runtime <固定rlib>`で固定したcompiled runtimeを使い、flags・compiler・依存hash・sourceの前後一致を記録する。この診断ビルドを全推論Cargoビルドの同一最適化結果とは扱わない。native helperはstandalone Cargoの`prefix_args`を使う。測定前に専用canisterへ診断Wasmをinstallし、`check_prefix_codec.py --canister <診断ID>`を実行する。

次は復号の各部分をさらに測り、通信増加と削減命令の交換条件を確認してから全推論へ接続する。採用済み主canister・Laya・他タスクのcanisterは変更していない。

## 4-bit exponentとSIMD復号への変更

2026-10-03追加。Huffman lookupを毎floatで行う処理を省いた。`NPF1`は各headについて連続15 exponentを4 bitで表し、それ以外だけ元の8-bit exponentを付ける。符号・mantissaは元の3 byteで保存し、4 floatずつ標準Wasm SIMDで復号する。窓の選択と例外の格納は初回の準備で行う。追加量子化はない。

全24層・48通常queryはすべて保存状態/native/Wasmとbit一致。module `ced1aef0ad7cafc6b11d5899128b7b64339089b39015ad7631430a2a3394a539`、Wasm 1,028,489 bytes。18/32 headをdense形式、14 headを元logで保持した。復元は174,175,927から89,463,463〜89,501,183命令へ **48.6145〜48.6361%減**。全24層合計 **2,032,791,617命令減**。主推論の命令数に対する規模は約0.7644%であり、主推論の実測改善率やquery数の削減ではない。

packetは1,438,785〜1,440,278 bytes。元の87-token BF16入力・conv・最大header見積もりを加えて2 MB以内。診断のF32 log入力との差は24層合計7,856,849 bytes増で、採用推論のBF16 K codecとの差や全通信量ではない。状態の再構成を省く代わりに通信が増える。

rawは`artifacts/prefix_codec/nibble-check/report.json`、実測sourceは`nibble-validated-source.zip`、前後hashは`nibble-source-hashes.json`。過去のHuffman証拠を上書きしていない。nativeでは符号付き0、subnormal、最大の有限exponent、例外の不足・過剰、非有限値、予約byte・base上限を検査した。

## canisterで一度だけ準備し、返したpacketを再利用する検証

追加の`prepare_prefix`通常queryを実装した。元prefix logからcanister内で状態を復元し、exponentの最適な15-value窓と例外を決め、サイズの小さい9 head pairを選んで可逆packetを作る。全packetをクライアントへ返し、次の通常queryがそのpacketを受け取る。canisterはowner以外の状態を保持しない。ホストが推論して作った状態を入力する経路ではない。prefix log自体は既存のcanister全層検証が生成したものを使用した。

module `8ef15af7f8e97db0dc28d8ad7b6e32f0ed191c26bc84bdd85d7989dd104cd737`、1,041,883-byte Wasm。24層×（準備・旧復元・再利用）の **72通常query** が通過した。canisterが返したpacketは独立した保存状態から組み立てたpacketともbyte一致し、nativeとWasmの再利用出力も元状態と全bit一致した。復元部分の削減は上記2,032,791,617命令と同じ。

この独立診断で一度だけ払う準備は、1層262,548,741〜263,946,665命令、24層合計 **6,326,348,500命令**。Candid request+reply合計 **61,219,865 bytes**。digestとCDK Candid処理は命令counterに含まない。準備費用を省いて改善を報告しない。この診断では4質問以上再利用すると、準備を含む命令数が旧復元の反復より少なくなる。ただし通信の損益は別であり、全推論の時間・通信が改善する証拠ではない。

実行は`check_prefix_codec.py --codec nibble --prepare-on-canister`。raw `artifacts/prefix_codec/preparation-check/report.json`、固定ビルド `preparation-build/report.json`、ソース `preparation-validated-source.zip` / `preparation-source-hashes.json`。nativeの新しいprepare round-trip、不正token数、非BF16 K、非有限値の拒否を含む3 testsが通過した。

残る本体接続は、prefix queryが既に計算した状態から直接packetを作ること、既存のmodel/pack/input hashに結び付けるframeとcache versionを追加すること、通常suffix queryでの入力・返答上限、token IDからの新しいcold prefix生成を含む全5条件を検証すること。現時点で主canisterの全推論query数は67のまま、50/32回には未達。独立診断を本体採用済みとは扱わない。

## 再準備を呼ばないクライアント経路（2026-10-03）

`prepare_prefix_reuse.py`を追加。24層のcodec準備結果をopaqueなNPF1 packetとしてクライアントのディスクへ保存する。model・pack・input・元canister module・元report・各層の元状態hash・codec moduleをcache identityに結び付ける。再利用時はidentityとpacketの長さ/hash/headerを検査し、`prepare_prefix`を呼ばない。初回は24 packetと元状態/moduleの前後検査を終えてからmanifestを公開する。異なる入力、異なるmodule、改変packet、再準備を呼ばないcache hitの4 testsが通過した。

```sh
.venv/bin/python scripts/prepare_prefix_reuse.py \
  --prefix-directory artifacts/output-pairs-v2-prefix \
  --directory artifacts/prefix_codec/reuse-cache \
  --canister 7hukf-2d777-77775-aaakq-cai \
  --run-report artifacts/prefix_codec/reuse-first.json
```

同じcache directoryで2回目を実行し、別の`--run-report`を指定する。専用canisterには上記preparation moduleをinstallしておく。初回24通常query、6,326,348,500命令、61,219,865 Candid bytes。2回目は**準備query 0・準備命令0・準備通信0**。module確認のstatus呼び出し、クライアントのファイル検査、将来のsuffix推論のpacket送信/復元はこのゼロに含まない。

初回の全24 packetは、過去の独立encoder/native/Wasm一致検証のpacketとbyte一致し、state digestも一致した。raw `artifacts/prefix_codec/reuse-first.json` / `reuse-second.json`、実行sourceと証拠hash `reuse-source.zip` / `reuse-proof.json`。入力は既存canisterが生成した45-token prefix log。ホスト推論を入力せず、状態をcanisterへ永続化しない。

この保存/再利用経路は独立codec用で、`TextGraph`のsuffix全推論へまだ接続していない。既存本体も共通prefixの計算結果を保存しており、今回の0準備queryを既存本体のquery削減として加算しない。全推論の改善値は引き続き未測定。
