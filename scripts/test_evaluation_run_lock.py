"""Verify exclusion across processes and automatic release after worker exit."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = str(Path(__file__).resolve().parent)
PROGRAM = '''
import pathlib, sys
sys.path.insert(0, sys.argv[1])
from evaluation_run_lock import run_lock
directory = pathlib.Path(sys.argv[2])
with run_lock(directory):
    print('locked', flush=True)
    if sys.argv[3] == 'hold':
        sys.stdin.readline()
    if not (directory / 'report.json').exists():
        with (directory / 'executions.txt').open('a') as f:
            f.write('inference\\n')
        (directory / 'report.json').write_text('{}')
'''


class Tests(unittest.TestCase):
    def launch(self, directory, mode):
        process = subprocess.Popen([sys.executable, '-B', '-c', PROGRAM, SCRIPTS, str(directory), mode],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        def cleanup():
            if process.poll() is None:
                process.kill()
            process.communicate()
        self.addCleanup(cleanup)
        return process

    def test_competing_process_waits_and_rechecks_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = self.launch(directory, 'hold')
            self.assertEqual(first.stdout.readline(), 'locked\n')
            second = self.launch(directory, 'run')
            with self.assertRaises(subprocess.TimeoutExpired):
                second.communicate(timeout=.2)
            self.assertFalse((directory / 'report.json').exists())
            first.communicate(input='\n', timeout=5)
            second.communicate(timeout=5)
            self.assertEqual((first.returncode, second.returncode), (0, 0))
            self.assertEqual((directory / 'executions.txt').read_text().splitlines(), ['inference'])

    def test_killed_worker_releases_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = self.launch(directory, 'hold')
            self.assertEqual(first.stdout.readline(), 'locked\n')
            first.kill()
            first.communicate(timeout=5)
            second = self.launch(directory, 'run')
            second.communicate(timeout=5)
            self.assertEqual(second.returncode, 0)
            self.assertTrue((directory / 'report.json').exists())


if __name__ == '__main__':
    unittest.main()
