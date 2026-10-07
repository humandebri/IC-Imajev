#!/usr/bin/env python3
"""Build the current 27-token scheduler with the archived query32 kernels."""
import argparse
from pathlib import Path

import build_paid_update_full_candid as build

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    build.D = (build.ROOT / args.directory).resolve()
    # The shared builder hashes the generated sources, including this entry point.
    build.__file__ = str(Path(__file__).resolve())
    build.__doc__ = __doc__
    build.main()

if __name__ == '__main__':
    main()
