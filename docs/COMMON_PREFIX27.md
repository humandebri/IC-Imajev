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

このビルドは既存のquery32 runtimeとWAT kernelの検証済みアーカイブを使い、現行の有料APIと27-token schedulerを組み合わせる。アーカイブの依存ファイルが必要。ビルドだけでcanisterを変更しない。

生成した`full.wasm`をupgradeし、既存の`prepare_weight_cache.py`による重み等の準備を済ませた後、common prefixだけを準備・登録する。

```sh
.venv/bin/python scripts/prepare_fixed_prefix_states.py \
  --canister <LOCAL_CANISTER_ID> \
  --wasm artifacts/common-prefix27/build/full.wasm \
  --directory artifacts/common-prefix27/fixed-states

.venv/bin/python scripts/register_common_update_prefix.py \
  --canister <LOCAL_CANISTER_ID> \
  --wasm artifacts/common-prefix27/build/full.wasm \
  --directory artifacts/common-prefix27/registration
```

前者は24層の固定Delta状態を準備する。後者はcommon prefixのcache・全32層・packetのhashと形状を検証してから、32回の`update_prefix`で登録する。途中で失敗した登録を同じcanisterで重ねて実行すると、既存層は重複登録として拒否される。再upgradeと再準備を行う。

料金設定は版を増やして適用し、呼び出し側も新しい`quote`を取得する。以前の38-token入力はsuffix数と料金が変わるため、旧見積もりを再利用しない。

## 測定記録との関係

`PAID_UPDATE_INFERENCE_MEASURED.md`と既存proofのdual-bank記録は当時の測定として保持する。今回の構成でのcanister推論、命令数、heap使用量は改めて測定する必要がある。準備対象は減るが、以前38-token bankを使っていた入力の計算量は増えるため、過去の命令数やworker数をそのまま適用しない。
