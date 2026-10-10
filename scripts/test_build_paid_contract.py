import tempfile
import unittest
from pathlib import Path

from build_paid_contract import (PAID_SOURCES, OLD_FOOTER, CURRENT_FOOTER,
                                 METADATA_END, METADATA_BOUND,
                                 copy_paid_sources, align_upgrade_metadata)

ROOT = Path(__file__).resolve().parents[1]


class PaidBuildContractTests(unittest.TestCase):
    def test_legacy_and_current_sources_use_the_same_upgrade_contract(self):
        current = (ROOT / 'canisters/inference/src/lib.rs').read_text()
        # Cover both historical footer layouts without unpublished build artifacts.
        legacy = current.replace(CURRENT_FOOTER, OLD_FOOTER).replace(
            METADATA_BOUND + '\n            ', '')
        sources = {'legacy': legacy, 'bounded-legacy': current.replace(CURRENT_FOOTER, OLD_FOOTER),
                   'current': current}
        for name, original in sources.items():
            with self.subTest(source=name), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                (directory / 'lib.rs').write_text(original)
                copy_paid_sources(directory, ROOT / 'canisters/inference/src')
                self.assertTrue(all((directory / name).is_file() for name in PAID_SOURCES))
                align_upgrade_metadata(directory)
                updated = (directory / 'lib.rs').read_text()
                self.assertNotIn(OLD_FOOTER, updated)
                self.assertEqual(updated.count(CURRENT_FOOTER), 1)
                self.assertEqual(updated.count(METADATA_BOUND), 1)
                # Every other part, including numerical code, must be preserved.
                expected = original.replace(OLD_FOOTER, CURRENT_FOOTER)
                if METADATA_BOUND not in original:
                    expected = expected.replace(METADATA_END, METADATA_BOUND + '\n            ' + METADATA_END)
                self.assertEqual(updated, expected)
                align_upgrade_metadata(directory)
                self.assertEqual((directory / 'lib.rs').read_text(), updated)

    def test_missing_archive_is_rejected_before_copying_any_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical, directory = root / 'source', root / 'build'
            canonical.mkdir(); directory.mkdir()
            for name in PAID_SOURCES[:-1]:
                (canonical / name).write_text('source')
            with self.assertRaisesRegex(ValueError, 'receipt_archive.rs'):
                copy_paid_sources(directory, canonical)
            self.assertEqual(list(directory.iterdir()), [])

    def test_unrecognized_or_ambiguous_metadata_is_rejected_without_changes(self):
        original = (ROOT / 'canisters/inference/src/lib.rs').read_text()
        for text in [original.replace(CURRENT_FOOTER, 'unknown_footer();'),
                     original + '\n' + OLD_FOOTER,
                     original.replace(METADATA_END, 'let end = m.bytes;'),
                     original + '\n' + METADATA_BOUND]:
            with self.subTest(text=text[-80:]), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                path = directory / 'lib.rs'; path.write_text(text)
                with self.assertRaises(ValueError):
                    align_upgrade_metadata(directory)
                self.assertEqual(path.read_text(), text)


if __name__ == '__main__':
    unittest.main()
