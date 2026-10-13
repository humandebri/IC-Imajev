"""Publication bundles must reject unsafe inputs and have deterministic metadata."""
import io
import gzip
from pathlib import Path
import tarfile
import tempfile
import unittest

from package_public_release import read_entry, write_archive


class PublicReleaseTests(unittest.TestCase):
    def test_archive_rejects_secret_inside_compressed_wat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = b'gh' + b'p_' + b'a' * 30
            (root / 'kernel.wat.gz').write_bytes(gzip.compress(token))
            with self.assertRaises(ValueError) as error:
                read_entry(root, 'kernel.wat.gz')
            self.assertNotIn(token.decode(), str(error.exception))

    def test_rejects_traversal_symlinks_and_candidate_without_exposing_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'safe').write_bytes(b'public')
            (root / 'link').symlink_to(root / 'safe')
            (root / 'folder').symlink_to(root, target_is_directory=True)
            for name in ['../outside', '/outside', '.git/config', 'link', 'folder/safe']:
                with self.assertRaises(ValueError): read_entry(root, name)
            secret = b'gh' + b'p_' + b'a' * 30
            (root / 'candidate').write_bytes(secret)
            with self.assertRaises(ValueError) as error: read_entry(root, 'candidate')
            self.assertNotIn(secret.decode(), str(error.exception))

    def test_archive_is_deterministic_has_no_host_identity_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            one, two = [Path(directory) / name for name in ['one.tar.gz', 'two.tar.gz']]
            entries = {'b.txt': (b'B', 0o644), 'a.py': (b'A', 0o755)}
            write_archive(one, entries); write_archive(two, dict(reversed(list(entries.items()))))
            self.assertEqual(one.read_bytes(), two.read_bytes())
            with tarfile.open(fileobj=io.BytesIO(one.read_bytes())) as archive:
                self.assertEqual(archive.getnames(), ['ic-imajev/a.py', 'ic-imajev/b.txt'])
                for member in archive:
                    self.assertEqual((member.uid, member.gid, member.uname, member.gname, member.mtime), (0, 0, '', '', 0))
            with self.assertRaises(FileExistsError): write_archive(one, entries)


if __name__ == '__main__': unittest.main()
