#!/usr/bin/env python3
"""Reuse the frozen update verifier, recording inclusive spans after every batch."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'scripts/measure_update_templates.py'


def main():
    # Pin the measurement code before adding the diagnostic reads.
    build = __import__('json').loads((ROOT/'artifacts/update-instruction-profile-v1/measurement-source.json').read_text())
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == build['sha256']
    source = SOURCE.read_text()
    anchor = "r=call(cmd);progress=r['ok']['progress'];"
    assert source.count(anchor) == 1
    replacement = '''r=call(cmd);progress=r['ok']['progress'];
    did=d/'profile.did';did.write_text('service : {last_instruction_profile:()->(text) query;}')
    raw=subprocess.check_output(['icp','canister','call',report['canister'],'last_instruction_profile','()','--query','--network','local','--identity','imajev-local','--candid',str(did),'--output','candid'],text=True,cwd=ROOT)
    profile_path=directory/f'{len(rows)+1:02d}.profile.candid';profile_path.write_text(raw)
    decoded,_=json.JSONDecoder().raw_decode(raw[raw.index('"'):])
    spans=json.loads(decoded);r['instruction_profile']=spans;r['profile_reply_sha256']=sha(profile_path)
    '''
    source = source.replace(anchor, replacement)
    exec(compile(source, str(SOURCE), 'exec'), dict(__file__=__file__, __name__='__main__'))


if __name__ == '__main__':
    main()
