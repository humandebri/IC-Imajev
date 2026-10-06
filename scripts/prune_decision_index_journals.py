#!/usr/bin/env python3
"""Remove completed-run intermediate binary frames after preserving their hashes."""
import hashlib, json, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/decision-index-v1'

def main():
    for report_path in sorted((D/'runs').glob('*/report.json')):
        run=report_path.parent
        marker=run/'intermediate-frame-hashes.json'
        if marker.exists():
            continue
        report=json.loads(report_path.read_text())
        # Only finished reports with a typed decision may be pruned.
        assert report['comparison']['typed_output_valid']
        assert report['decision_query']['ok']['decision']['probabilities']
        frames=sorted(run.rglob('queries/*.bin'))
        entries=[]
        for path in frames:
            content=path.read_bytes()
            entries.append(dict(path=str(path.relative_to(run)),size=len(content),sha256=hashlib.sha256(content).hexdigest()))
        value=dict(purpose='Intermediate request/reply frames discarded after completed inference; source, reports, metrics, hidden outputs and final state retained',
                   report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),frames=entries)
        marker.write_text(json.dumps(value,indent=2)+'\n')
        for path in frames:
            path.unlink()
        print(json.dumps(dict(run=run.name,frames=len(frames),bytes_removed=sum(x['size'] for x in entries))),flush=True)

if __name__=='__main__':main()
