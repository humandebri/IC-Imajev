"""Exercise the proof's actual setup/finally without importing model tools."""
import ast
import contextlib
import copy
import io
import json
import unittest
from pathlib import Path


class CleanupTests(unittest.TestCase):
    def run_failure(self, failed_call, cleanup_failure=None):
        path = Path(__file__).with_name('prove_paid_token_limit_local.py')
        main = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        trial = copy.deepcopy(next(n for n in main.body if isinstance(n, ast.Try)
                                   and any(isinstance(c, ast.Assign)
                                           and any(isinstance(t, ast.Name) and t.id == 'stop_attempted'
                                                   for t in c.targets) for c in n.body)))
        # Stop before loading weights or calling inference. All failure and
        # restoration handlers are the unmodified source nodes.
        end = next(i for i, n in enumerate(trial.body)
                   if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                   and isinstance(n.value.func, ast.Name) and n.value.func.id == 'icp'
                   and isinstance(n.value.args[0], ast.Constant)
                   and n.value.args[0].value == 'start')
        trial.body = trial.body[:end + 1]
        calls = []
        reports = {}
        cache, pack = {'weights': 721}, {'ready': True}

        def icp(*args):
            calls.append(args)
            if args[:len(failed_call)] == failed_call or (
                    cleanup_failure and args[:len(cleanup_failure)] == cleanup_failure):
                raise RuntimeError('simulated CLI failure')
            return 'protected-snapshot' if args[:2] == ('snapshot', 'create') else ''

        class Bridge:
            def command(self, request):
                return {'ok': {'cache': cache}} if request['op'] == 'weight_cache_status' else {'ok': pack}

            def close(self):
                pass

        scope = dict(icp=icp, write=lambda p, v: reports.update({p.name: copy.deepcopy(v)}),
                     snapshot=None, caller=None, restored=False, stop_attempted=False,
                     report={'complete': False}, d=Path('/unused'), build=Path('/unused/build'),
                     TARGET='local-test', BASELINE='baseline', before={'cache': cache, 'pack': pack},
                     bridge=lambda _: Bridge(), verify_module=lambda *_: None, json=json)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                exec(compile(ast.Module(body=[trial], type_ignores=[]), str(path), 'exec'), scope)
        return calls, reports['report.json']

    def test_snapshot_failure_restarts_and_verifies_unchanged_baseline(self):
        calls, report = self.run_failure(('snapshot', 'create'))
        self.assertEqual(calls, [('stop', 'local-test'),
                                ('snapshot', 'create', 'local-test', '--quiet'),
                                ('start', 'local-test')])
        self.assertTrue(report['baseline_restored'])
        self.assertFalse(report['complete'])
        self.assertNotIn('snapshot_deleted', report)

    def test_uncertain_stop_failure_also_restarts(self):
        calls, report = self.run_failure(('stop',))
        self.assertEqual(calls[-1], ('start', 'local-test'))
        self.assertTrue(report['baseline_restored'])

    def test_install_failure_restores_snapshot_before_deleting_it(self):
        calls, report = self.run_failure(('install',))
        self.assertEqual(calls[-4:], [('stop', 'local-test'),
                                     ('snapshot', 'restore', 'local-test', 'protected-snapshot'),
                                     ('start', 'local-test'),
                                     ('snapshot', 'delete', 'local-test', 'protected-snapshot')])
        self.assertTrue(report['baseline_restored'])
        self.assertTrue(report['snapshot_deleted'])

    def test_restart_failure_is_reported_without_claiming_restoration(self):
        _, report = self.run_failure(('snapshot', 'create'), ('start',))
        self.assertFalse(report['baseline_restored'])
        self.assertIn('restoration_error', report)

    def test_restore_failure_preserves_snapshot(self):
        calls, report = self.run_failure(('install',), ('snapshot', 'restore'))
        self.assertFalse(any(c[:2] == ('snapshot', 'delete') for c in calls))
        self.assertFalse(report['baseline_restored'])
        self.assertIn('restoration_error', report)


if __name__ == '__main__':
    unittest.main()
