import unittest
from pathlib import Path

from build_latest_common_prefix27 import prefix27_scheduler


class SchedulerTests(unittest.TestCase):
    def test_preserves_optimized_scheduler_except_registration_limits(self):
        parent = Path(__file__).resolve().parents[1] / 'artifacts/paid-stack-carry-projection-v1/build/update_inference.rs'
        old = parent.read_text()
        new = prefix27_scheduler(old)
        self.assertEqual(new.replace('if p!=27', 'if !(1..=132).contains(&p)')
                         .replace('if !g.banks.is_empty()', 'if g.banks.len()>=2'), old)
        self.assertIn('prepare_server_delta_prefix', new)
        self.assertIn('server_delta_bound_input', new)
        self.assertNotIn('if !(1..=132).contains(&p)', new)

    def test_rejects_unexpected_archive(self):
        with self.assertRaises(ValueError):
            prefix27_scheduler('fn changed_scheduler() {}')


if __name__ == '__main__':
    unittest.main()
