"""
Record golden outputs for the standard sampling pipeline.

Re-capturing blesses whatever the sampler does now, so only run it after a
change to the output that was intended. Existing files are kept unless --force
is given.

    ~/ComfyUI/venv/bin/python tests/capture_golden.py [--force] [case ...]
"""
import argparse
import os

import torch

from golden_cases import CASES, GOLDEN_DIR, golden_path, run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="case names (default: all)")
    parser.add_argument("--force", action="store_true", help="overwrite existing golden files")
    args = parser.parse_args()

    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for name in args.cases or sorted(CASES):
        path = golden_path(name)
        if os.path.exists(path) and not args.force:
            print(f"skip   {name} (exists)")
            continue
        recording = run_case(name)
        torch.save(recording, path)
        print(f"wrote  {name}: {len(recording['calls'])} sampler call(s)")


if __name__ == "__main__":
    main()
