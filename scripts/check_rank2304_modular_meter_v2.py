#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    original=ROOT/'scripts/check_rank2304_modular_meter.py'
    prior=ROOT/'artifacts/rank2304-modular-meter-v1'
    r=json.loads((prior/'report.json').read_text())
    assert sha(original)==r['source_hashes'][str(original.relative_to(ROOT))]
    source=original.read_text().replace('rank2304-modular-meter-v1','rank2304-modular-meter-v2')
    source=source.replace("OLD='1bfed03d74aea873ed78e1f4568a04a49e29ad587b6effdf0b01cd0dd539d0e9'",'OLD='+repr(r['module']))
    source=source.replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path('+repr(str(ROOT))+')')
    source=source.replace("checked16_vs_direct_percent=(r['checked16']/r['direct']-1)*100", "checked16_vs_direct_percent=(r['checked16']/r['direct']-1)*100,universal32_vs_direct_unrolled_percent=(r['universal32']/r['direct_unrolled']-1)*100")
    source=source.replace('all36_update_outputs_exact','all48_update_outputs_exact')
    d=ROOT/'artifacts/rank2304-modular-meter-v2'
    frozen=d/'frozen-checker.py';assert not frozen.exists();frozen.write_text(source)
    (d/'checker-entry.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),original,frozen]},indent=2)+'\n')
    exec(compile(source,str(frozen),'exec'),dict(__file__=str(frozen),__name__='__main__'))
if __name__=='__main__':main()
