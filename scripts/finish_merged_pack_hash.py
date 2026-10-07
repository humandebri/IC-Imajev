#!/usr/bin/env python3
"""Finish an uploaded diagnostic pack using concurrent owner hash requests.

hash_pack is a synchronous, atomic canister update: each call hashes the next
contiguous 8MB using the stored cursor. IC serializes these updates. The original
uploader may also advance that cursor; ready calls are idempotent. This changes
only preparation throughput, never the Wasm or numerical inference.
"""
import argparse
import concurrent.futures
import json
import re
import subprocess
import time

from measure_merged_adapter_ic import DIRECTORY, ROOT, VARIANTS, Transport


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=VARIANTS, required=True)
    args = ap.parse_args()
    canister, manifest_path = VARIANTS[args.variant]
    m = json.loads((ROOT / manifest_path).read_text())
    t = Transport(m['model'],'http://localhost:8001/',canister,str(ROOT/'artifacts/imajev-local.pem'),DIRECTORY/f'{args.variant}-hash',m['pack_hash'])
    try:
        initial = t.command(dict(op='pack_status'))['ok']
        if initial['pack_hash'] != m['pack_hash'] or initial['received'] != m['bytes']:raise ValueError('Upload must be complete and match the selected experiment pack')
        start = time.monotonic()
        calls, ready = 0, initial['ready']
        def advance(_):
            result = subprocess.run(['icp','canister','call',canister,'hash_pack','(8000000 : nat64)',
                                     '--network','local','--identity','imajev-local','--output','candid'],
                                    cwd=ROOT,check=True,capture_output=True,text=True)
            cursor = re.search(r'([0-9_]+) : nat64', result.stdout)
            if cursor is None or not re.search(r'\b(?:true|false)\b',result.stdout):raise ValueError(result.stdout)
            return int(cursor[1].replace('_','')),bool(re.search(r'\btrue\b',result.stdout))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            while not ready:
                rows = list(pool.map(advance,range(8)))
                calls += len(rows)
                ready = any(row[1] for row in rows)
                print(json.dumps(dict(calls=calls,hashed=max(row[0] for row in rows),ready=ready)),flush=True)
        final = t.command(dict(op='pack_status'))['ok']
        if not final['ready'] or final['hashed'] != m['bytes'] or final['pack_hash'] != m['pack_hash']:raise ValueError('Final pack identity')
        report = dict(scope='Diagnostic owner preparation only; concurrent hash updates serialized by IC, original uploader may also advance cursor.',
                      initial=initial,final=final,extra_update_calls=calls,wall_seconds=time.monotonic()-start)
        (DIRECTORY/f'{args.variant}-hash/report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        t.close()


if __name__ == '__main__':
    main()
