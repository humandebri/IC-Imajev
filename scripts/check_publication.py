#!/usr/bin/env python3
"""Check publishable working files and optionally reachable Git blobs; redact values."""
import argparse
import gzip
import io
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
# Construct directory markers so this checker does not flag its own rules.
PERSONAL_PATH = rb"/(?:" + b"Users|Volumes|home" + rb")/[^\s\x22\x27`]+"
RULES = {
    "personal_absolute_path": PERSONAL_PATH,
    "private_key": rb"-----BEGIN (?:[A-Z ]*PRIVATE KEY|OPENSSH PRIVATE KEY)-----",
    "provider_token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{20,})\b",
    "credential_url": rb"https?://[^\s/:]+:[^\s/@]+@",
    "credential_assignment": rb"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password)\s*[:=]\s*[\x22\x27][^\x22\x27\n]{12,}[\x22\x27]",
}
COMPILED = {name: re.compile(pattern) for name, pattern in RULES.items()}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def scan(data, path, blob=None):
    if path.endswith('.wat.gz'):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as source:
                data = source.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise ValueError('source size bound')
        except (OSError, EOFError, ValueError):
            item = {'rule': 'invalid_compressed_source', 'file': path, 'line': 1}
            if blob:
                item['blob'] = blob
            return [item]
    found = []
    for name, pattern in COMPILED.items():
        for match in pattern.finditer(data):
            item = {"rule": name, "file": path, "line": data.count(b"\n", 0, match.start()) + 1}
            if blob:
                item["blob"] = blob
            found.append(item)
    return found


def working_files():
    # Includes untracked non-ignored files; deleted tracked files are omitted.
    paths = sorted(set(git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")) - {""})
    findings, count = [], 0
    for path in paths:
        file = ROOT / path
        if file.is_symlink():
            findings.append({"rule": "symlink_requires_review", "file": path, "line": 1})
            continue
        if not file.is_file():
            continue
        count += 1
        if file.name.startswith(".env") and not file.name.endswith(".example") or file.suffix in {".pem", ".seed", ".key", ".p12", ".pfx"} or file.name in {"id_rsa", "id_ed25519"}:
            findings.append({"rule": "credential_filename", "file": path, "line": 1})
        findings.extend(scan(file.read_bytes(), path))
    return count, findings


def history():
    findings, count = [], 0
    rows = git("rev-list", "--objects", "--all").decode().splitlines()
    # Stream one object at a time, without shell interpolation or secret output.
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        for row in rows:
            oid, _, path = row.partition(" ")
            proc.stdin.write((oid + "\n").encode())
            proc.stdin.flush()
            header = proc.stdout.readline().decode().split()
            if len(header) != 3:
                raise RuntimeError("Unable to read Git object " + oid)
            data = proc.stdout.read(int(header[2]))
            proc.stdout.read(1)
            if header[1] == "blob":
                count += 1
                findings.extend(scan(data, path, oid))
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("git cat-file failed")
    return count, findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Also scan blobs reachable from local refs; findings cause failure")
    args = parser.parse_args()
    count, findings = working_files()
    result = {"working_files_scanned": count, "working_findings": findings}
    if args.history:
        blobs, old = history()
        result.update(history_blobs_scanned=blobs, history_findings=old)
    all_findings = findings + result.get("history_findings", [])
    result["candidate_counts"] = dict(Counter(item["rule"] for item in all_findings))
    result["limits"] = "Pattern-based candidate detection, including bounded expansion of WAT gzip sources; excludes ignored local files, reflogs, dangling objects, remote refs not fetched, and commit metadata. No clean result certifies absence of secrets."
    print(json.dumps(result, indent=2))
    return int(bool(all_findings))


if __name__ == "__main__":
    raise SystemExit(main())
