"""Validate explicit external inputs before any experiment side effects."""
import argparse
from pathlib import Path


def existing_directory(value):
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f'directory not found: {value}; specify an existing input directory')
    return path


def existing_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f'file not found: {value}; specify an existing input file')
    return path
