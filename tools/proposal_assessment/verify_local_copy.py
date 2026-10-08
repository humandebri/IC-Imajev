"""Verify retained prompt metadata; full replacement comparison uses archived sources."""
import argparse
import json
from pathlib import Path
from .token_sweep import sha

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources-only',action='store_true')
    args = parser.parse_args()
    package = Path(__file__).parent
    provenance = json.loads((package/'PROVENANCE.json').read_text())
    data = provenance['retained_data']
    if sha(package/data['path'])!=data['sha256']:
        raise ValueError('retained prompt contract hash mismatch')
    if not (package/data['notice']).is_file():
        raise ValueError('retained data license notice missing')
    if not args.sources_only:
        parser.error('Use scripts/check_proposal_replacement.py with a frozen baseline and snapshot archive for conformance.')
    print(json.dumps(dict(retained_contract_verified=True,external_classifier_required=False)))

if __name__=='__main__':
    main()
