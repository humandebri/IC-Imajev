"""Read verified proposal snapshots only from the explicitly selected archive."""
import hashlib
import json
import pathlib


def read_snapshot(archive, record, sns_root):
    """Rebase historical paths onto the selected archive and verify its bytes."""
    stored = pathlib.Path(record['snapshot'])
    relative = pathlib.Path('snapshots') / f"proposal-{record['proposal_id']}.json"
    if '..' in stored.parts or (stored.is_absolute() and pathlib.Path(*stored.parts[-2:]) != relative) or \
            (not stored.is_absolute() and stored != relative):
        raise ValueError(f"invalid snapshot path for proposal {record['proposal_id']}")
    archive = pathlib.Path(archive).resolve()
    path = (archive / relative).resolve(strict=True)
    if not path.is_relative_to(archive):
        raise ValueError('snapshot resolves outside the selected archive')
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError(f"snapshot size mismatch for proposal {record['proposal_id']}")
    data = path.read_bytes()
    if len(data) > 2 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError(f"snapshot size or hash mismatch for proposal {record['proposal_id']}")
    proposal = json.loads(data)
    if str(proposal['id']) != str(record['proposal_id']) or proposal['root_canister_id'] != sns_root:
        raise ValueError(f"snapshot identity mismatch for proposal {record['proposal_id']}")
    return path, proposal

