#!/usr/bin/env python3
"""Reuse exact memory/IC tests with current eight-lane control and new variant."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    original=ROOT/'scripts/check_rank7_prepare8_probe.py'
    old=ROOT/'artifacts/rank7-prepare8-probe-v1';prior=json.loads((old/'report.json').read_text())
    assert sha(original)==prior['source_hashes'][str(original.relative_to(ROOT))]
    code=original.read_text().replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path('+repr(str(ROOT))+')')
    code=code.replace("ROOT/'artifacts/rank2304-modular-meter-v2/report.json'","ROOT/'artifacts/rank7-prepare8-probe-v1/report.json'")
    code=code.replace("d=ROOT/'artifacts/rank7-prepare8-probe-v1'","d=ROOT/'artifacts/rank7-prepare8-unrolled-probe-v1'")
    code=code.replace('[4,8]','[8,16]')
    before="prepare4=values[4],prepare8=values[8],saved=values[4]-values[8],percent=(values[8]/values[4]-1)*100"
    assert code.count(before)==1
    code=code.replace(before,"prepare8=values[8],prepare16=values[16],saved=values[8]-values[16],percent=(values[16]/values[8]-1)*100")
    code=code.replace('all14_node_operand_buffers_exact','all22_node_operand_buffers_exact').replace('all14_ic_operand_buffers_exact','all22_ic_operand_buffers_exact')
    d=ROOT/'artifacts/rank7-prepare8-unrolled-probe-v1';frozen=d/'frozen-checker.py';assert not frozen.exists();frozen.write_text(code)
    files=[Path(__file__),original,old/'report.json',frozen]
    (d/'checker-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
    exec(compile(code,str(frozen),'exec'),dict(__file__=str(frozen),__name__='__main__'))
if __name__=='__main__':main()
