"""Describe historical CLI boundaries without importing models or opening artifacts."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = {
    "build_paid_token_chunks.py": "Historical paid build requiring frozen runtime, compiler and donor Wasm",
    "reproduce_frozen_optimized.py": "Historical optimized build from frozen compiler/donor archives",
    "build_latest_common_prefix27.py": "Historical optimized build from frozen compiler/donor archives",
    "build_common_prefix27.py": "Historical compatibility alias for frozen optimized builds",
    "assess_proposal_range.py": "Frozen 500–660 proposal assessment and archived runner",
    "benchmark_proposal_assessment.py": "Frozen 600–660 benchmark with historical local canister/module",
    "evaluate_prompt_accuracy.py": "Original/short-prompt comparison on a frozen local module",
    "run_fresh_proposal_assessment.py": "October 7, 2026 frozen-module reassessment; fresh means fresh queries",
    "run_prefix_canister.py": "Historical ordinary-query graphs and prefix/carry contracts",
    "prepare_prefix_reuse.py": "Historical prefix codec preparation using owner updates",
    "prepare_all_proposal_windows.py": "Historical multi-window proposal input preparation",
    "prepare_all_proposal_query32.py": "Historical prefix27/query32 input preparation",
    "prepare_all_proposal_compact86.py": "Historical prefix27/86-token input preparation",
    "build_paid_message_checkpoint.py": "Archived paid candidate build with fixed artifact destinations",
}
PREFERRED = {
    "build_paid_token_chunks.py": "reproduce_frozen_optimized.py",
    "run_fresh_proposal_assessment.py": "reproduce_proposal_assessment.py",
    "run_prefix_canister.py": "reproduce_prefix_inference.py",
    "build_latest_common_prefix27.py": "build_optimized_inference.py",
}


def require_historical_cli(script, detail=None):
    """Require an explicit CLI opt-in before imports, file writes or remote calls.

    The consumed flag belongs to this wrapper, not to a frozen driver's parser.
    This protects command-line entry points; importing a helper does not run it.
    """
    name = Path(script).name
    purpose = detail or HISTORICAL[name]
    preferred = PREFERRED.get(name, name)
    help_requested = "--help" in sys.argv[1:] or "-h" in sys.argv[1:]
    if "--historical" not in sys.argv[1:]:
        stream = sys.stdout if help_requested else sys.stderr
        print(f"HISTORICAL: {purpose}", file=stream)
        print("This is not an entry point for the pinned public browser release.", file=stream)
        print(f"Use: python3 scripts/{preferred} --historical <original arguments>", file=stream)
        print("Original options: add --historical --help (requires the driver's Python dependencies).", file=stream)
        print("See scripts/SCRIPT_INDEX.md for targets, data and side effects.", file=stream)
        raise SystemExit(0 if help_requested else 2)
    sys.argv.remove("--historical")
    print(f"HISTORICAL: {purpose}; verify the original module, local target and retained data.", file=sys.stderr)


def historical_command(command):
    """Forward opt-in only to known retained historical children in this checkout."""
    result = list(command)
    for argument in result:
        path = Path(str(argument))
        names = set(HISTORICAL) | {"reproduce_proposal_assessment.py", "reproduce_prefix_inference.py"}
        if path.name in names and path.resolve() == ROOT / "scripts" / path.name:
            if "--historical" not in result:
                result.append("--historical")
            break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", nargs="?", help="Historical script filename to describe")
    args = parser.parse_args()
    if args.script:
        name = Path(args.script).name
        if name not in HISTORICAL:
            parser.error("not a guarded historical entry; see scripts/SCRIPT_INDEX.md")
        print(f"{name}: {HISTORICAL[name]}")
        if name in PREFERRED:
            print(f"Preferred name: {PREFERRED[name]}")
    else:
        for name, purpose in sorted(HISTORICAL.items()):
            print(f"{name}: {purpose}")
    print("Historical CLI execution requires --historical; no commands were executed.")


if __name__ == "__main__":
    main()
