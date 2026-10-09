#!/usr/bin/env python3
"""Compare shorter unknown descriptions on the prepared local update canister."""
import argparse, hashlib, json, pathlib, subprocess
from tokenizers import Tokenizer
ROOT = pathlib.Path(__file__).resolve().parents[1]
LONG = 'unknown — cannot be determined from the available evidence, the premise is false, or no listed option is correct'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--indices', default='0,19,11')
    ap.add_argument('--variants', default='bare')
    a = ap.parse_args(); d = ROOT / a.directory; d.mkdir(parents=True, exist_ok=False)
    fixture_path = ROOT / 'artifacts/reference-serving.json'
    tokenizer_path = ROOT / 'checkpoints/base/tokenizer.json'
    fixture = json.loads(fixture_path.read_text()); tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prefix = json.loads((ROOT / 'artifacts/single-quad/full-proof-v2/prefix/queries/cache.json').read_text())['token_ids']
    variants = {'original': LONG, 'bare': 'unknown', 'short': 'unknown — insufficient evidence, false premise, or no correct option'}
    inputs = []
    for index in map(int, a.indices.split(',')):
        r = fixture['records'][index]; text = tokenizer.decode(r['token_ids'], skip_special_tokens=False)
        assert tokenizer.encode(text, add_special_tokens=False).ids == r['token_ids']
        old = '\n' + chr(65 + len(r['options'])) + ': ' + LONG + '<|im_end|>'
        assert text.count(old) == 1
        for variant in a.variants.split(','):
            new = '\n' + chr(65 + len(r['options'])) + ': ' + variants[variant] + '<|im_end|>'
            rewritten = text.replace(old, new)
            ids = tokenizer.encode(rewritten, add_special_tokens=False).ids
            assert ids[:len(prefix)] == prefix and 1 <= len(ids)-len(prefix) <= 89
            inputs.append(dict(index=index, case=r['id'], offset=r['offset'], variant=variant, gold=r['gold'], options=r['options'], token_ids=ids, text=rewritten))
    bridge = ROOT / 'artifacts/update-inference/client-target/release/imajev-client'
    wasm = ROOT / 'artifacts/update-inference/build-v2/full.wasm'
    report = dict(module_sha256=sha(wasm), bridge_sha256=sha(bridge), fixture_sha256=sha(fixture_path), tokenizer_sha256=sha(tokenizer_path), script_sha256=sha(pathlib.Path(__file__)), cases=[])
    (d/'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
    p = subprocess.Popen([str(bridge), 'http://localhost:8001/', '6eydd-o3777-77775-aaama-cai', str(ROOT/'artifacts/imajev-local.pem')], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    def call(cmd):
        p.stdin.write(json.dumps(cmd)+'\n'); p.stdin.flush(); line=p.stdout.readline()
        if not line: raise RuntimeError('bridge exited')
        r=json.loads(line)
        if 'error' in r: raise RuntimeError(r['error'])
        return r
    try:
        assert call(dict(op='module_hash'))['ok']['module_hash']==report['module_sha256']
        for inp in inputs:
            cd=d/f"{inp['index']:02d}-{inp['variant']}"; cd.mkdir(); rows=[]
            cmd=dict(diagnostics=True,op='update_infer_start', ids=inp['token_ids'][len(prefix):], options=inp['options'])
            while True:
                r=call(cmd); rows.append(r); progress=r['ok']['progress']
                (cd/f'{len(rows):02d}.json').write_text(json.dumps(r,indent=2)+'\n')
                print(json.dumps(dict(index=inp['index'],variant=inp['variant'],call=len(rows),stage=progress['stage'])),flush=True)
                if progress['done']: break
                cmd=dict(diagnostics=True,op='update_infer_continue',id=progress['id'],stage=progress['stage'])
            decision=progress['decision']; prediction='__unknown__' if decision['abstained'] else decision['value']
            row=dict(index=inp['index'],case=inp['case'],offset=inp['offset'],variant=inp['variant'],gold=inp['gold'],prediction=prediction,correct=None if inp['gold'] is None else prediction==inp['gold'],tokens=len(inp['token_ids']),suffix_tokens=len(inp['token_ids'])-len(prefix),update_calls=len(rows),total_handler_instructions=sum(x['ok']['progress']['instructions'] for x in rows),max_handler_instructions=max(x['ok']['progress']['instructions'] for x in rows),wall_seconds=sum(x['wall_seconds'] for x in rows),decision=decision)
            report['cases'].append(row); (d/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(row),flush=True)
        assert call(dict(op='module_hash'))['ok']['module_hash']==report['module_sha256']
    except Exception as error:
        (d/'failure.json').write_text(json.dumps(dict(error=str(error),partial=report),indent=2)+'\n'); raise
    finally:
        p.stdin.close(); p.wait(timeout=15)
if __name__=='__main__': main()
