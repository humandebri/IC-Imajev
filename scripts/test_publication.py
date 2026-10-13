"""Exercise redaction, ignored credentials, deleted files and historical exposure."""
import json
import gzip
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import check_publication as checker


class PublicationTests(unittest.TestCase):
    def test_compressed_sources_are_scanned_and_corruption_is_rejected(self):
        raw = ('/none\n/' + 'Users' + '/example/private\n').encode()
        hits = checker.scan(gzip.compress(raw, mtime=0), 'kernel.wat.gz')
        self.assertEqual(hits, [{'rule': 'personal_absolute_path', 'file': 'kernel.wat.gz', 'line': 2}])
        self.assertNotIn('example/private', json.dumps(hits))
        for packed in [b'bad gzip', gzip.compress(b'x' * (4 * 1024 * 1024 + 1)), gzip.compress(b'public')[:-4]]:
            self.assertEqual(checker.scan(packed, 'kernel.wat.gz')[0]['rule'], 'invalid_compressed_source')
        self.assertEqual(checker.scan(gzip.compress(b'(module)'), 'kernel.wat.gz'), [])

    def test_matches_report_locations_without_values(self):
        path = "/" + "Users" + "/example/private"
        token = "gh" + "p_" + "a" * 30
        raw = ("ordinary text\n" + path + "\n" + token).encode()
        hits = checker.scan(raw, "fixture.txt")
        self.assertEqual({h["rule"] for h in hits}, {"personal_absolute_path", "provider_token"})
        self.assertEqual({h["line"] for h in hits}, {2, 3})
        self.assertNotIn(token, json.dumps(hits))
        self.assertNotIn(path, json.dumps(hits))

    def test_key_and_credential_candidates(self):
        samples = {
            "private_key": b"-----BEGIN " + b"PRIVATE KEY-----",
            "credential_url": b"https://" + b"user:example-password@host.invalid/path",
            "credential_assignment": b"api_key = " + b"'example-secret-value'",
        }
        for rule, raw in samples.items():
            with self.subTest(rule=rule):
                self.assertIn(rule, {h["rule"] for h in checker.scan(raw, "fixture")})
        self.assertEqual(checker.scan(b"http://localhost:8001/\ncheckpoints/base/tokenizer.json", "fixture"), [])

    def test_working_tree_and_reachable_history_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL)
            git("init", "-q")
            (root / ".gitignore").write_text(".env\n")
            (root / ".env").write_text("ignored local credential")
            (root / "tracked.txt").write_text("/" + "Volumes" + "/example/private\n")
            git("add", ".gitignore", "tracked.txt")
            git("-c", "user.name=Publication test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
            (root / "tracked.txt").unlink()
            (root / "untracked.txt").write_text("safe\n")
            (root / "credential.key").write_text("synthetic placeholder")
            (root / "link").symlink_to(root / ".env")
            with patch.object(checker, "ROOT", root):
                count, working = checker.working_files()
                blobs, old = checker.history()
            self.assertEqual(count, 3)
            self.assertEqual({h["rule"] for h in working}, {"credential_filename", "symlink_requires_review"})
            self.assertNotIn(".env", {h["file"] for h in working})
            self.assertEqual(blobs, 2)
            self.assertEqual(len(old), 1)
            self.assertEqual(old[0]["file"], "tracked.txt")
            self.assertIn("blob", old[0])


if __name__ == "__main__":
    unittest.main()
