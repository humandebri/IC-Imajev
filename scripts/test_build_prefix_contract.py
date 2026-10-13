import tempfile
import unittest
from pathlib import Path
from build_prefix_contract import align_prefix_contract


class PrefixContractTests(unittest.TestCase):
    def test_paid_prefix_binds_registration_stream_and_chunk_history(self):
        for count in [5, 27]:
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                (d / 'paid_inference.rs').write_text(f'const COMMON_PREFIX:[u32;{count}]=[' + ','.join(['1'] * count) + '];')
                (d / 'update_inference.rs').write_text('if p!=27 {return Err("prefix token bounds");}\nif b.tokens!=27 {return Err("stream prefix identity");}')
                (d / 'chunked_update.rs').write_text('const PREFIX:usize=27;')
                (d / 'runtime').mkdir()
                (d / 'runtime/prefix_state_cache.rs').write_text('const TOKENS:usize=27;')
                self.assertEqual(align_prefix_contract(d), count)
                self.assertIn('pub(super) const COMMON_PREFIX', (d / 'paid_inference.rs').read_text())
                self.assertEqual((d / 'update_inference.rs').read_text().count('super::paid_inference::COMMON_PREFIX.len()'), 2)
                self.assertIn('PREFIX:usize=super::paid_inference::COMMON_PREFIX.len()', (d / 'chunked_update.rs').read_text())
                self.assertEqual((d / 'runtime/prefix_state_cache.rs').read_text(), f'const TOKENS:usize={count};')

    def test_rejects_invalid_paid_prefix_before_building(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'paid_inference.rs').write_text('const COMMON_PREFIX:[u32;5]=[1,2];')
            with self.assertRaisesRegex(ValueError, 'length mismatch'):
                align_prefix_contract(d)
