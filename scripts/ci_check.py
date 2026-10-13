#!/usr/bin/env python3
"""Run the same asset-independent checks locally and in GitHub Actions."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Deliberately enumerate the standard-library suite. Other test_*.py files
# exercise models, recorded inference, local replicas, or historical artifacts.
PYTHON_TESTS = [
    "test_paid_update_transport",
    "test_public_query_build",
    "test_build_paid_contract",
    "test_build_prefix_contract",
    "test_canister_api_names",
    "test_browser_examples",
    "test_optimized_build",
    "test_optimized_paid",
    "test_paid_instruction_accounting",
    "test_paid_instruction_evidence",
    "test_paid_state_evidence",
    "test_paid_contract_evidence",
]


def run(args, cwd=ROOT):
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=["rust", "wasm", "python", "frontend", "all"])
    suite = parser.parse_args().suite
    if suite in ("rust", "all"):
        run(["cargo", "test", "--workspace", "--locked"])
        run(["cargo", "test", "-p", "imajev-client", "--example", "paid_update_args", "--locked"])
    if suite in ("wasm", "all"):
        run(["cargo", "build", "--release", "--locked", "--target", "wasm32-unknown-unknown", "-p", "imajev-inference"])
    if suite in ("python", "all"):
        # Disable site packages to ensure this suite needs no ML environment.
        run([sys.executable, "-S", "-B", "-m", "unittest", "-v", *PYTHON_TESTS], ROOT / "scripts")
    if suite in ("frontend", "all"):
        run(["node", "scripts/check-ci.mjs"], ROOT / "frontend")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode if error.returncode > 0 else 1)
