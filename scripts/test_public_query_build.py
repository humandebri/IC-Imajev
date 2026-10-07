import tempfile
import unittest
from pathlib import Path

from public_query_build import GUARD, PUBLIC, expose, expose_public_queries

ROOT = Path(__file__).resolve().parents[1]

class PublicQueryBuildTests(unittest.TestCase):
    def test_current_handlers_match_public_policy_and_admins_keep_owner(self):
        text = (ROOT / 'canisters/inference/src/lib.rs').read_text()
        for method in PUBLIC:
            body = text.split('fn ' + method + '(', 1)[1].split('{', 1)[1]
            self.assertTrue(body.lstrip().startswith('query_access();'), method)
        for method in ['prepare', 'upload_chunk', 'hash_pack', 'warm_weights', 'clear_weight_cache', 'profile_step']:
            body = text.split('fn ' + method + '(', 1)[1].split('{', 1)[1]
            self.assertTrue(body.lstrip().startswith('owner();'), method)
        self.assertIn(GUARD, text)

    def test_transform_does_not_expose_admin_or_diagnostic_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            text = 'fn owner() {}\n' + '\n'.join('fn '+m+'() -> Result<(),String> { owner(); Ok(()) }' for m in [*PUBLIC, 'prepare', 'profile_step'])
            (directory / 'lib.rs').write_text(text)
            for name, method in [('query_mlp_delta.rs','mlp_delta_front'), ('query_attention_mlp_front.rs','attention_mlp_front')]:
                (directory / name).write_text('fn '+method+'() { owner(); }')
            expose_public_queries(directory)
            result = (directory / 'lib.rs').read_text()
            self.assertIn('if ic_cdk::api::in_replicated_execution() { owner(); }', result)
            self.assertIn('fn prepare() -> Result<(),String> { owner();', result)
            self.assertIn('fn profile_step() -> Result<(),String> { owner();', result)
            self.assertEqual(result.count('query_access();'), len(PUBLIC))

    def test_unknown_archive_shape_fails_instead_of_silently_keeping_private_queries(self):
        with self.assertRaises(ValueError):
            expose('fn step() { changed_guard(); }', ['step'])

if __name__ == '__main__':
    unittest.main()
