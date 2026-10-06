#!/usr/bin/env python3
"""Verify frozen proof sources and summarize the isolated W4 experiments."""
import argparse,hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    reports={}
    for version in [3,4,5,6]:
        base=ROOT/f'artifacts/w4-lut/check-v{version}';p=base/'report.json';r=json.loads(p.read_text());reports[version]=r
        b=ROOT/f'artifacts/w4-lut/build-v{version}'
        assert sha((b/'diagnostic.wasm').read_bytes())==r['module_sha256']
        with zipfile.ZipFile(b/'source.zip')as z:
            for name,digest in r['build']['source_hashes'].items():assert sha(z.read(name))==digest
        with zipfile.ZipFile(base/'checker-source.zip')as z:
            for name,digest in r['checker_source_hashes'].items():assert sha(z.read(name))==digest
        for c in r['cases']:
            assert c['integer_lut_exact'] and c['native_w4_digest_exact']
            previous=next(x for x in reports[3]['cases']if x['label']==c['label'])
            assert previous['input_sha256']==c['input_sha256']
            assert previous['projection_error_vs_int8']==c['projection_error_vs_int8']
            assert [t['digest']for t in previous['measurements']['w4_lut_simd']['tiles']]==[t['digest']for t in c['measurements']['w4_lut_simd']['tiles']]
    rows=[]
    for c in reports[6]['cases']:
        m=c['measurements'];full=m['w4_lut_full_query'];old=c.get('historical_current_s1_instructions')
        rows.append(dict(label=c['label'],tokens=c['tokens'],full_query_instructions=full['total_instructions'],full_query_project_instructions=full['project_instructions'],historical_int8_instructions=old,ratio_to_historical=full['total_instructions']/old if old else None,relative_l2_error=c['projection_error_vs_int8']['relative_l2'],max_abs_error=c['projection_error_vs_int8']['max_abs'],split_totals={v:next(x for x in r['cases']if x['label']==c['label'])['measurements']['w4_lut_simd']['total_instructions']for v,r in reports.items()},wall_seconds=full['wall_seconds']))
    final=reports[6];packed=final['packed_weight_bytes']+final['group_scale_bytes'];original=final['original_weight_and_scale_bytes']
    result=dict(scope='Isolated Q projection only. Native/Wasm agreement proves W4 kernel correctness, not decision accuracy. Historical INT8 comparator uses different module but identical locked W8 tensor/input and measurement boundaries.',completed_queries=sum(r['ordinary_queries']for r in reports.values()),completed_preparation_updates=sum(sum(not c['query']for c in r['calls'])for r in reports.values()),packed_plus_scales=packed,original_plus_scales=original,weight_storage_reduction_percent=100*(1-packed/original),cases=rows,reports={v:sha((ROOT/f'artifacts/w4-lut/check-v{v}/report.json').read_bytes())for v in reports})
    out=ROOT/a.output
    if out.exists():raise ValueError('Fresh output required')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
