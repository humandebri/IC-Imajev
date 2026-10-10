> Historical prefix27 procedure. Use [PAID_PREFIX5_VERIFICATION.md](PAID_PREFIX5_VERIFICATION.md) for current execution; the commands below are retained as a record.

# 27-token common prefix専用構成

現行ソースの有料update推論は、全入力に共通する固定27 tokenの状態だけを保持する。`update_prefix`は27-token状態だけを受け付け、32層が揃うまで新規推論は受け付けない。38-token voting bankの準備と選択は不要。

新規リクエストは固定prefix込みで28〜84 token、suffixは1〜57 token。以前の38-token prefixと一致する入力も27-token prefixを使うため、差分の11 tokenはsuffixの計算・料金に含まれる。以前は受け付けていた85〜95 tokenの入力は短くする必要がある。frontendのカウント詳細にも同じ条件を表示する。

既存receiptの保存形式は変えていない。旧構成で完了・失敗した95 tokenまでの入力は、receiptが残っている間、同じcaller・入力・request IDで再取得できる。upgrade後は重みとprefixを再準備する。固定重みの4,702,451,200 bytesは変わらない。

## ビルドと準備

以下はローカル対象への手動適用手順。対象canisterの受付を停止し、推論と返金の完了を確認してからupgradeする。スクリプトは指定したWasmのmodule hashと対象を照合する。出力ディレクトリは毎回新しい場所を指定する。

```sh
.venv/bin/python scripts/build_common_prefix27.py \
  --directory artifacts/common-prefix27/build
```

このビルドは最新の全推論検証済み `paid-stack-carry-projection-v1` を基に、現行の有料APIと27-token制限、ordinary queryの公開を組み合わせる。34個の投影kernel本体を検証済み親と照合し、最適化済みruntime・固定Delta prefixの事前準備・hash一括更新・コピー削減を維持する。アーカイブの依存ファイルが必要。ビルドだけでcanisterを変更しない。replicated実行と管理updateは引き続きowner権限を要求する。

以前の6-kernel版のビルドとは異なる。最新の削減を含む親の測定値を、新しい27-token構成の命令数として流用しない。SHA256四並列の部品実験とworker返却後の計測probeは、全推論への採用が未検証のため含めない。

生成した`full.wasm`をupgradeし、既存の`prepare_weight_cache.py`による重み等の準備を済ませた後、common prefixだけを準備・登録する。

```sh
.venv/bin/python scripts/register_common_update_prefix.py \
  --canister <LOCAL_CANISTER_ID> \
  --wasm artifacts/common-prefix27/build/full.wasm \
  --directory artifacts/common-prefix27/registration

.venv/bin/python scripts/prepare_fixed_prefix_states.py \
  --canister <LOCAL_CANISTER_ID> \
  --wasm artifacts/common-prefix27/build/full.wasm \
  --directory artifacts/common-prefix27/fixed-states
```

先にcommon prefixのcache・全32層・packetのhashと形状を検証してから、32回の`update_prefix`で登録する。最適化版の最初のprefix登録はquery用の固定状態cacheをclearするため、その後に24層の固定Delta状態を準備する。順序を逆にするとquery用cacheが失われる。途中で失敗した登録を同じcanisterで重ねて実行すると、既存層は重複登録として拒否される。再upgradeと再準備を行う。

料金設定は版を増やして適用し、呼び出し側も新しい`quote`を取得する。以前の38-token入力はsuffix数と料金が変わるため、旧見積もりを再利用しない。

## 測定記録との関係

`PAID_UPDATE_INFERENCE_MEASURED.md`と既存proofのdual-bank記録は当時の測定として保持する。今回の構成でのcanister推論、命令数、heap使用量は改めて測定する必要がある。準備対象は減るが、以前38-token bankを使っていた入力の計算量は増えるため、過去の命令数やworker数をそのまま適用しない。

2026-10-07に本番 `xis3j-paaaa-aaaai-axumq-cai` を最新最適化・公開query版へupgradeした。raw Wasm SHA256は `9b90e968ece56eaddef5a53c4ab5e55d07cd7b8fc4adaa87321642f0ac1b8bba`、実際のgzip module hashは `3291a064976fb13df548582871ae96a3347f052569eceb7b9807dc23ec5b2495`。stable memoryのモデル4,702,451,200 bytesを保持し、721項目の重みcacheと全32層のcommon prefix、24層のquery用固定状態を再準備した。有料推論はdisabledのまま。

BOOM 653の84-token入力で匿名の32 queryを新規実行し、各層の保存hidden/state・最終hidden・判断が以前の参照とbit一致した。報告命令合計は149,773,886,050から126,961,172,113へ15.23%減、最大queryは4,097,821,474命令。これは同じ入力のordinary queryを各1回比較した結果で、全メッセージ費用や有料updateの削減率ではない。証跡は `artifacts/mainnet-prefix27-upgrade-20261007/report.json` と `anonymous-query-653/report.json`。
