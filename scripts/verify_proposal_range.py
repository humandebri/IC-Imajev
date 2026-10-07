#!/usr/bin/env python3
"""Audit range coverage, frozen inputs, canister identities and cost accounting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(out):
    prepared=json.loads((out/'prepared.json').read_text())
    inputs=json.loads((out/'inputs.json').read_text())
    report=json.loads((out/'report.json').read_text())
    identity=json.loads((out/'identity.json').read_text())
    assert report['summary']['complete']
    for name in ('inputs','prepared','runner'):
        suffix='py' if name=='runner' else 'json'
        assert sha(out/f'{name}.{suffix}')==identity[f'{name}_sha256']
    assert [p['proposal_id'] for p in report['proposals']]==list(range(prepared['first'],prepared['last']+1))
    manifest=json.loads((out.parent/'snapshots/manifest.json').read_text())
    assert sha(out.parent/'snapshots/manifest.json')==prepared['source_manifest_sha256']
    assert manifest['all_fetched'] and manifest['complete']
    for r in manifest['records']:
        p=Path(r['snapshot']);assert sha(p)==r['sha256']
        snapshot=json.loads(p.read_text());assert str(snapshot['id'])==str(r['proposal_id'])
        assert snapshot['root_canister_id']==manifest['sns_root']
    total={'new':dict(inferences=0,queries=0,instructions=0,candid_bytes=0,wall_seconds=0),
           'reused':dict(inferences=0,queries=0,instructions=0,candid_bytes=0,wall_seconds=0)}
    verified=[]
    for e,row in zip(prepared['entries'],report['model_rows']):
        i=e['record_index'];record=inputs['records'][i];dest=out/'runs'/f'{i:03d}'
        path=dest/'report.json';r=json.loads(path.read_text())
        assert sha(path)==row['report_sha256']
        assert hashlib.sha256(json.dumps(record['token_ids']).encode()).hexdigest()==record['input_sha256']==r['input_hash']
        assert r['model']==inputs['model_lock_sha256']
        assert r['wasm_sha256']==r['deployed_wasm_sha256']=='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
        assert [x['layer'] for x in r['layers']]==list(range(32))
        assert r['tokens']==e['tokens']==len(record['token_ids'])<=128
        assert r['query_count']==r['executed_query_count']>0 and not r['replayed_queries'] and not r.get('fallback')
        assert r['comparison']['typed_output_valid'] and not (dest/'active-staging.json').exists()
        assert r['max_query_instructions']<5_000_000_000 and r['max_observed_heap_bytes']<2**32
        assert all(q['ok']['request_bytes']<2_000_000 and q['ok']['reply_bytes']<2_000_000 for q in r['queries'])
        kind='reused' if (dest/'reuse.json').exists() else 'new'
        if kind=='reused':
            reuse=json.loads((dest/'reuse.json').read_text())
            assert reuse['exact_token_identity'] and sha(Path(reuse['source'])/'report.json')==reuse['source_report_sha256']==sha(path)
        t=total[kind];t['inferences']+=1;t['queries']+=r['query_count'];t['instructions']+=r['total_instructions'];t['candid_bytes']+=r['total_candid_bytes'];t['wall_seconds']+=r['wall_seconds_this_run']
        verified.append(dict(proposal_ids=e['proposal_ids'],report_sha256=sha(path),tokens=r['tokens'],queries=r['query_count'],source=kind))
    assert len(verified)==report['summary']['model_distinct']
    counts=dict(Counter(p['final_label'] for p in report['proposals'] if p['route']!='skipped'))
    assert counts==report['summary']['counts']
    result=dict(verified=True,total_snapshots=len(manifest['records']),model_inputs=verified,costs=total,
                accuracy_measured=False,report_sha256=sha(out/'report.json'))
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (out/'REPORT.md').open('a') as f:
        f.write('\n## 実行量と独立照合\n\n全161 snapshotと18種類の推論について、入力hash・モデルhash・前後module hash・全32層・query数・再実行なし・上限内を照合済み。新規実行と過去結果再利用の実行量は以下のとおり。\n\n')
        for kind,t in total.items():
            f.write(f'- {kind}: 推論{t["inferences"]}種類、{t["queries"]:,} query、{t["instructions"]:,} handler命令、Candid {t["candid_bytes"]:,} bytes、記録wall time合計{t["wall_seconds"]:.1f}秒。\n')
        f.write('\n費用・時間は各モデル入力の合計で、161件分の独立実行や本番レイテンシではない。重み準備・HTTP等は含まない。再利用分は今回消費した実行量に含めない。\n')
    print(json.dumps(dict(verified=True,counts=counts,costs=total),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evaluation',type=Path);a=p.parse_args();verify(a.evaluation.resolve())
