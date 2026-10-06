# 後段stream APIのレビューと検証

2026-10-04。既存の全体推論は51queryのまま。今回の接続APIは全体未接続であり、50/32query到達や実canisterでの精度一致はまだ主張しない。

レビューで、canisterの`experimental-mlp-attention-finish`がruntimeだけを有効にし、canister自身の型付きqueryデコーダを有効にしない問題を修正した。`experimental-mlp-stream`を依存関係に追加し、`experimental-terminal-stream`だけの構成でも契約テストを実行した。通常の全feature構成ではこの欠落が隠れていた。

後段の要求には、hidden BF16の2-plane可逆Huffman圧縮（tag12）、MLP全体→Delta先頭（tag13）、MLP stream完了→Attention全体、MLP30 stream完了→terminal/readoutを追加した。重みの量子化、F32 carry、演算順序、専用readoutとcalibrationは変更しない。従来の要求方向とreply形式を維持し、新しい方向と範囲を明示的に検査する。最終判断は既存の通常query `terminal_step_decision`を利用する。

Pythonテストでは1/87/89 tokenでhidden圧縮を独立したcanonical decoderで復元し、従来packetの全bytesと比較した。signed zeroを含む。MLP入力のBF16精度、不正進捗、reply方向、サイズ、選択肢順序とquery modeを変えたcheckpointの拒否も確認した。型付き終端結果をcheckpointから復元でき、判断欠落を拒否する。

検証結果：runtimeは131 unit（1 ignored）、12 integration、10 docが通過。追加のstream境界テストも修正後通過。canister契約はterminal-stream単独構成とhalf/tail構成で各3件通過。Python codec一式35件、追加4件、transport3件、journal6件、joined/roll graph各3件通過。保存済みMLP payload315個はbyte一致。`git diff --check`通過。生成物は`artifacts/`に置きGit対象外。

次は新Wasmを専用実験canisterへ導入し、2MB超過していた第6queryと後段の接続を実測する。主canisterとLayaは変更しない。命令数・通信量・全状態/判断の参照一致を確認するまで既存51query経路を維持する。
