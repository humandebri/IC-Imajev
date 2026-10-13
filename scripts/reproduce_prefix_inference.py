#!/usr/bin/env python3
"""Explicit entry point for historical prefix/query graphs and retained artifacts."""
if __name__ == "__main__":
    from script_usage import require_historical_cli
    require_historical_cli("run_prefix_canister.py")
    from run_prefix_canister import main
    main()
