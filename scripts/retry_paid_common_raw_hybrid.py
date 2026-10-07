#!/usr/bin/env python3
"""Prepare or run a fresh proof without overwriting the failed trial."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--recovery-verification', required=True, type=Path)
    parser.add_argument('--proof-storage', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    verification = args.recovery_verification.resolve()
    proof = (args.proof_storage or directory / 'proof').resolve()
    if args.proof_storage:
        storage_root = Path.home() / '.codex/goal-artifacts/01a10ffe-1dc8-7931-b3de-a3e4d35bd910'
        assert proof.is_relative_to(storage_root.resolve()), 'external proof storage must belong to this task'
    assert not proof.exists(), 'preserve previous proof storage'
    assert directory.is_relative_to(ROOT / 'artifacts')
    assert not directory.exists(), 'preserve previous trial directories'
    source = ROOT / 'artifacts/paid-common-raw-hybrid-v1/frozen-proof.py'
    expected = json.loads((source.parent / 'proof-entry-hashes.json').read_text())
    for path, digest in expected.items():
        assert sha(ROOT / path) == digest, path
    code = source.read_text()
    old_root = 'ROOT=Path(__file__).resolve().parents[1];'
    assert code.count(old_root) == 1
    code = code.replace(old_root, 'ROOT=Path(' + repr(str(ROOT)) + ');')
    old = "D=ROOT/'artifacts/paid-common-raw-hybrid-v1/proof-storage/proof'"
    assert code.count(old) == 1
    # PaidTransport records repository-relative paths. Keep its lexical path
    # under the repository while the symlink stores the bytes on the SSD.
    code = code.replace(old, 'D=ROOT/' + repr(str((directory / 'proof').relative_to(ROOT))))
    if args.proof_storage:
        mkdir = ' D.mkdir(parents=True,exist_ok=False);'
        assert code.count(mkdir) == 1
        code = code.replace(mkdir, ' Path(' + repr(str(proof)) + ').mkdir(parents=True,exist_ok=False);')
    guard = " callers=[];snapshot=None;stopped=False;results=[];checks=[]"
    assert code.count(guard) == 1
    code = code.replace(guard, " baseline=json.loads((ROOT/'artifacts/local-goal-recovery-v1/baseline-state.json').read_text())\n assert dict(module=BASELINE,cache=cache,pack=pack)==baseline,'live baseline differs before trial'\n" + guard)
    compile(code, str(directory / 'frozen-proof.py'), 'exec')
    if args.run:
        verified = json.loads(verification.read_text())
        assert verified['complete'] and verified['full_state_equal']
        assert verified['target'] == '4caro-hl777-77775-aaaba-cai'
    directory.mkdir(parents=True)
    if args.proof_storage:
        (directory / 'proof').symlink_to(proof, target_is_directory=True)
    frozen = directory / 'frozen-proof.py'
    frozen.write_text(code)
    files = [Path(__file__), source, frozen, source.parent / 'proof-entry-hashes.json',
             ROOT / 'artifacts/local-goal-recovery-v1/baseline-state.json']
    if args.run:
        files.append(verification)
    (directory / 'entry-hashes.json').write_text(json.dumps({str(p): sha(p) for p in files}, indent=2) + '\n')
    (directory / 'entry.json').write_text(json.dumps(dict(prepared=True, run=args.run,
        recovery_verification=str(verification), original_trial_preserved=True,
        proof_storage=str(proof),
        requires_live_baseline_equality_before_mutation=True), indent=2) + '\n')
    if args.run:
        exec(compile(code, str(frozen), 'exec'), dict(__file__=str(frozen), __name__='__main__'))
    else:
        print('Prepared proof only; no canister mutation or recovery claim')


if __name__ == '__main__':
    main()
