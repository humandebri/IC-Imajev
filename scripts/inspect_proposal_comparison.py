import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'client'))
from transport import Transport
m=json.loads((R/'checkpoints/full-int8.manifest.json').read_text())
t=Transport(m['model'],'http://localhost:8001/','2hlkr-gl777-77775-aaaxa-cai',str(R/'artifacts/imajev-local.pem'),R/'artifacts/proposal-full-161-20261007/preflight',m['pack_hash'])
try: print(json.dumps(t.command({'op':'module_hash'})))
finally:t.close()
