#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path


def run_pytest(test_file=None, verbose=True):
    """Run pytest on specified test file or all tests"""
    cmd = ["python", "-m", "pytest"]

    if test_file:
        cmd.append(f"tests/{test_file}")
    else:
        cmd.append("tests/")

    if verbose:
        cmd.append("-v")

    # Run pytest
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def main():
    """Main test runner"""
    if len(sys.argv) < 2:
        print("Running all tests...")
        return run_pytest()

    test_suite = sys.argv[1].lower()

    test_map = {
        "data": "test_data.py",
        "rnns": "test_rnns.py",
        "rnn": "test_rnns.py",
        "lstm": "test_rnns.py",
        "transformer": "test_transformer.py",
        "training": "test_training.py",
        "train": "test_training.py",
        "generation": "test_text_generation.py",
        "text": "test_text_generation.py",
        "sampling": "test_text_generation.py",
        "all": None
    }

    if test_suite not in test_map:
        print(f"Unknown test suite: {test_suite}")
        print(f"Available options: {', '.join(test_map.keys())}")
        return 1

    test_file = test_map[test_suite]
    if test_file:
        print(f"Running {test_suite} tests ({test_file})...")
    else:
        print("Running all tests...")

    return run_pytest(test_file)


if __name__ == "__main__":
    sys.exit(main())
