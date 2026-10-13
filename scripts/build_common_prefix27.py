#!/usr/bin/env python3
"""Compatibility alias; prefer build_optimized_inference.py."""
from build_latest_common_prefix27 import main

if __name__ == '__main__':
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).with_name('build_latest_common_prefix27.py')), run_name='__main__')
