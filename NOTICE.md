The text runtime primitives in this repository are written for this experiment.
The authoritative model/prompt/readout/calibration behavior comes from
mohit67890/imajev commit a0134749e0900189c129cd6bb5000969f3b64bb5 (Apache-2.0).
Its source is downloaded into ignored vendor/ and source hashes are in MODEL_LOCK.json.
Qwen/Qwen3.5-4B and mohit67890/imajev-4b declare Apache-2.0 licenses.

Historical benchmark inputs in benchmarks/cases.json and retained prompt/field
metadata in tools/proposal_assessment/prompt_contract.json originate from
IC-Laya-Standalone (MIT). The MIT notice remains in docs/licenses/Laya-MIT.txt.
The external Laya classifier, benchmark drivers and derived unrolled INT8 kernel
implementation have been removed or replaced. Historical experiment archives
retain their original source provenance and notices.
Python and Rust dependency versions are in requirements.lock and Cargo.lock;
these projects retain their respective licenses.
