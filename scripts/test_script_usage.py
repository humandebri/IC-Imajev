"""Historical CLI gates must stop before optional imports or experiment side effects."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import script_usage as usage


class HistoricalCliTests(unittest.TestCase):
    def test_every_historical_entry_blocks_without_opt_in_before_imports(self):
        entries = list(usage.HISTORICAL) + [
            "reproduce_proposal_assessment.py", "reproduce_prefix_inference.py",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "must-not-exist"
            for name in entries:
                with self.subTest(script=name):
                    result = subprocess.run(
                        [sys.executable, "-S", str(usage.ROOT / "scripts" / name), "--directory", str(directory)],
                        cwd=temporary, capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("HISTORICAL:", result.stderr)
                    self.assertIn("--historical", result.stderr)
                    self.assertNotIn("ModuleNotFoundError", result.stderr)
                    self.assertFalse(directory.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_historical_help_is_available_without_model_dependencies(self):
        for name in list(usage.HISTORICAL) + ["reproduce_proposal_assessment.py", "reproduce_prefix_inference.py"]:
            with self.subTest(script=name):
                result = subprocess.run(
                    [sys.executable, "-S", str(usage.ROOT / "scripts" / name), "--help"],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("HISTORICAL:", result.stdout)

    def test_opt_in_is_consumed_without_changing_original_arguments(self):
        args = ["driver", "run", "--historical", "--directory", "retained-run"]
        with patch.object(sys, "argv", args), contextlib.redirect_stderr(io.StringIO()):
            usage.require_historical_cli("run_fresh_proposal_assessment.py")
        self.assertEqual(args, ["driver", "run", "--directory", "retained-run"])

    def test_only_known_local_children_receive_opt_in(self):
        child = str(usage.ROOT / "scripts/prepare_prefix_reuse.py")
        cmd = [sys.executable, "-B", child, "--canister", "local-test"]
        self.assertEqual(usage.historical_command(cmd), cmd + ["--historical"])
        self.assertEqual(usage.historical_command(cmd + ["--historical"]), cmd + ["--historical"])
        self.assertNotIn("--historical", cmd)
        frozen = [sys.executable, "/retained/archive/prepare_prefix_reuse.py"]
        self.assertEqual(usage.historical_command(frozen), frozen)
        preferred = [sys.executable, str(usage.ROOT / "scripts/reproduce_prefix_inference.py")]
        self.assertEqual(usage.historical_command(preferred), preferred + ["--historical"])

    def test_legacy_builder_aliases_are_guarded_before_build_inputs(self):
        for name in ["build_common_prefix27.py", "build_latest_common_prefix27.py", "reproduce_frozen_optimized.py"]:
            with self.subTest(script=name):
                result = subprocess.run(
                    [sys.executable, "-S", str(usage.ROOT / "scripts" / name), "--legacy-116"],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("HISTORICAL:", result.stderr)

    def test_automatic_historical_fallback_keeps_explicit_opt_in(self):
        with patch.object(sys, "path", [str(usage.ROOT / "client"), *sys.path]):
            from limit_fallback import standard_command
        cmd = standard_command(["--tail-start", "--canister", "local-test"], Path("new-journal"))
        self.assertIn("--historical", cmd)
        self.assertNotIn("--tail-start", cmd)
        self.assertIn("local-test", cmd)
        repeated = standard_command(["--historical", "--canister", "local-test"], Path("new-journal"))
        self.assertEqual(repeated.count("--historical"), 1)


if __name__ == "__main__":
    unittest.main()
