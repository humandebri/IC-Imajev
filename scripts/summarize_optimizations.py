from pathlib import Path
import json,hashlib
import argparse
from repository_paths import existing_directory
parser=argparse.ArgumentParser()
parser.add_argument('--laya-root', type=existing_directory, required=True)
laya=parser.parse_args().laya_root
root=Path(__file__).resolve().parents[1];r=json.loads((root/'artifacts/optimization-check/report.json').read_text());transport=json.loads((root/'artifacts/optimization-check/transport-exploration.json').read_text());scheduler=json.loads((root/'artifacts/optimized-scheduler/report.json').read_text());contracts=json.loads((root/'artifacts/fast-readout-contracts/report.json').read_text())
paths=['tools/client_held_query.py','tools/query_transport.py','crates/laya-candle/src/int8.rs','crates/laya-candle/src/epilogue.rs','canisters/decision-engine/src/lib.rs','canisters/decision-engine/src/query_transport.rs']
r['reference_source_hashes']={str(laya/p):hashlib.sha256((laya/p).read_bytes()).hexdigest() for p in paths};r['implementation_hashes']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['crates/imajev-runtime/src/lib.rs','canisters/inference/src/lib.rs','crates/imajev-client/src/main.rs','client/transport.py','scripts/benchmark_optimizations.py']}
r['scheduler']=scheduler;r['readout_contracts']=contracts;r['transport_exploration']=transport
(root/'docs/optimizations.json').write_text(json.dumps(r,indent=2)+'\n')
