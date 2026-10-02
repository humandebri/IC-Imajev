import json,pathlib,sys,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import TextGraph
from transport import Transport
from prefix_inference import verify_module,file_hash
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());t=Transport(m['model'],'http://localhost:8001/','4caro-hl777-77775-aaaba-cai',str(ROOT/'artifacts/imajev-local.pem'),ROOT/'artifacts/review-head-boundaries',m['pack_hash'],'bf16-exact');cases=[]
try:
 verify_module(t,file_hash(ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm'))
 g=TextGraph(t,m,arithmetic='int8')
 for n in [147,512]:
  heads=g.bounded_heads(8,n*768)
  # Exactly BF16 inputs with nonzero Q/K/V, same causal kernel for both layouts.
  values=np.arange(heads*3*n*256,dtype=np.float32).reshape(heads,3,n,256)%7/np.float32(8)
  a=t.run('attention_heads_bf16',values,[n,256,heads]).reshape(heads,n,256)
  b=np.stack([t.run('attention_bf16',values[h],[n,256]).reshape(n,256) for h in range(heads)])
  assert np.array_equal(a.view(np.uint32),b.view(np.uint32))
  cases.append(dict(tokens=n,heads=heads,bitwise_equal=True))
 (ROOT/'docs/review-head-boundaries.json').write_text(json.dumps(dict(cases=cases),indent=2)+'\n');print(cases)
finally:t.close()
