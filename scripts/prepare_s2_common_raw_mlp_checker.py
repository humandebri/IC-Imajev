#!/usr/bin/env python3
"""Freeze MLP checker; does not install, stage or call any canister."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'artifacts/s2-common-raw-probe-v1/frozen-checker.py';d=ROOT/'artifacts/s2-common-raw-mlp-probe-v1';s=old.read_text().replace('s2-common-raw-probe-v1','s2-common-raw-mlp-probe-v1')
 s=s.replace("a = ap.parse_args()","ap.add_argument('--shape',choices=['gate','down'],required=True)\n    a = ap.parse_args()\n    ROWS,COLS=(9216,2560)if a.shape=='gate'else(2560,9216)")
 s=s.replace("d = ROOT/'artifacts/s2-common-raw-mlp-probe-v1/check'","d = ROOT/'artifacts/s2-common-raw-mlp-probe-v1'/('check-'+a.shape)")
 s=s.replace('project:(vec nat8,nat8)->(Measurement) query;', 'project:(vec nat8,nat8)->(Measurement) query; stage_input:(nat32,vec nat8)->(Prep);')
 s=s.replace("name = 'model.language_model.layers.3.self_attn.q_proj.weight'","name = 'model.language_model.layers.3.mlp.'+('gate_proj'if a.shape=='gate'else'down_proj')+'.weight'")
 s=s.replace('(8192,2560,\'int8\')',"(ROWS,COLS,'int8')").replace('8192*2564','ROWS*(COLS+4)')
 s=s.replace("assert weight_hash == json.loads((ROOT/'artifacts/column32/check/report.json').read_text())['tensor_bytes_sha256']","assert len(weights)==ROWS*(COLS+4)")
 s=s.replace('for start in range(0,8192,512):','for start in range(0,ROWS,512 if COLS==2560 else 128):').replace('end = start+512','end = min(ROWS,start+(512 if COLS==2560 else 128))')
 s=s.replace('8192*2560','ROWS*COLS').replace('start*2560','start*COLS').replace('end*2560','end*COLS').replace('rows*2560','rows*COLS')
 a=s.index("    previous = ROOT/");z=s.index('    results = []',a)
 s=s[:a]+'''    cases=[]
    for n in [1,8,48,56,57,87]:
        values=np.resize(np.array([-127.,127.,0.,-0.,-1.,1.,0.5,-0.5,2**-126,-2**-126],dtype='<f4'),n*COLS)
        cases.append((f'boundary-{n}',values,dict(scope='Synthetic signed extremes/zero/tiny at real MLP shape; no full hidden-state claim')))
'''+s[z:]
 s=s.replace('n = values.size//2560; rows = 4096 if n>109 else 8192','n = values.size//COLS; rows = ROWS').replace("'2560',str(wp)","str(COLS),str(wp)").replace('cols=2560','cols=COLS')
 before="        measured = {}";assert s.count(before)==1
 s=s.replace(before,'''        staged=[]
        for offset in range(0,ip.stat().st_size,1_000_000):
            part=d/f'{label}-part-{offset}.bin';part.write_bytes(ip.read_bytes()[offset:offset+1_000_000]);arg=d/f'{label}-stage-{offset}.args.bin'
            subprocess.run([str(HELPER),'chunk',str(offset),str(part),str(arg)],check=True);staged.append(call('stage_input',arg))
        empty=d/'empty.bin';empty.write_bytes(b'')
'''+before)
 s=s.replace("'query',str(method),str(ip),str(arg)","'query',str(method),str(empty),str(arg)").replace('provenance=provenance,measurements=measured','provenance=provenance,input_staging=staged,measurements=measured')
 s=s.replace("scope='Only Q projection with quantization/input preparation included; digest/Candid excluded. No full-model/query-count claim.'","scope='MLP component quantize/prepare/project counters; identical staged F32 bytes for both methods. Input staging updates reported separately. Digest/Candid excluded from projection interval. No full-paid/goal claim.'")
 p=d/'frozen-mlp-checker.py';assert not (d/'checker-entry-hashes.json').exists();compile(s,str(p),'exec');p.write_text(s)
 entry=ROOT/'scripts/check_s2_common_raw_mlp_probe.py';entry.write_text("#!/usr/bin/env python3\nfrom pathlib import Path\nROOT=Path(__file__).resolve().parents[1]\np=ROOT/'artifacts/s2-common-raw-mlp-probe-v1/frozen-mlp-checker.py'\nexec(compile(p.read_text(),str(p),'exec'),dict(__file__=__file__,__name__='__main__'))\n")
 compile(s,str(p),'exec');(d/'checker-entry-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),old,p,entry]},indent=2)+'\n');print('MLP checker frozen for two shapes, six full-token inputs each')
if __name__=='__main__':main()
