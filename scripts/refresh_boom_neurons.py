#!/usr/bin/env python3
"""Fetch public indexed BOOM neurons for the #617 current-state counterfactual."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
SNS = 'xjngq-yaaaa-aaaaq-aabha-cai'
BASE = f'https://sns-api.internetcomputer.org/api/v1/snses/{SNS}'


def read(url):
    response = subprocess.run(['curl', '-fsSL', '--max-time', '30', url], check=True, capture_output=True)
    return json.loads(response.stdout)


def main():
    started = datetime.now(timezone.utc)
    reference = int(started.timestamp())
    def url(offset, maximum=None):
        query = dict(limit=100, offset=offset, sort_by='id')
        if maximum is not None:
            query['max_neuron_index'] = maximum
        return BASE + '/neurons?' + urlencode(query)
    first = read(url(0))
    maximum, total = first['max_neuron_index'], first['total_neurons']
    # Pin the membership boundary and stable ID order, but not mutable state.
    first = read(url(0, maximum))
    if first['total_neurons'] != total:
        raise ValueError('neuron count changed; retry the snapshot')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pages = [first] + list(pool.map(lambda offset: read(url(offset, maximum)), range(100, total, 100)))
    rows = [neuron for page in pages for neuron in page['data']]
    if any(page['total_neurons'] != total for page in pages) or len(rows) != total or len({row['id'] for row in rows}) != total:
        raise ValueError('incomplete or changing neuron membership; no snapshot saved')
    projection = []
    for row in rows:
        if row['root_canister_id'] != SNS:
            raise ValueError('wrong SNS')
        projection.append(dict(id=row['id'], dissolve_state=row['dissolve_state'],
                               voting_power=str(row['voting_power']), stake_e8s=str(row['stake_e8s']),
                               updated_at=row['updated_at']))
    snapshot = dict(rootCanisterId=SNS, sourceUrl=BASE + '/neurons',
                    sourceKind='Public indexed API, not certified governance responses',
                    startedAt=started.isoformat(), completedAt=datetime.now(timezone.utc).isoformat(),
                    referenceTimestampSeconds=reference, maxNeuronIndex=maximum,
                    totalNeurons=total, pageSize=100, pages=len(pages), neurons=projection,
                    scenario='Apply #617 thresholds to current neurons without relocking. Not proposal-time reconstruction. Pages are not atomic.')
    encoded = (json.dumps(snapshot, indent=2) + '\n').encode()
    (ROOT / 'frontend/data/boom-617-neurons.json').write_bytes(encoded)
    print(json.dumps(dict(neurons=total, pages=len(pages), sha256=hashlib.sha256(encoded).hexdigest(), bytes=len(encoded))))


if __name__ == '__main__':
    main()
