"""Small helpers shared by the example scripts (imports and printing)."""

import os
import sys


def add_repo_root_to_path():
    """Put the repo root on sys.path so the hypothesis_tests package imports.

    The examples live in examples/ but the package sits at the repo root, so
    this lets them run directly (python examples/worked_examples.py) without
    installing anything.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_result(result, width=22):
    """Print a result dict as aligned key: value rows, rounding floats."""
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key:>{width}}: {value:.4f}")
        else:
            print(f"{key:>{width}}: {value}")
