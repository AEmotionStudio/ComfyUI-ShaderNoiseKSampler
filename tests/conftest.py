"""Put the tests directory on sys.path so test modules can import the shared helpers."""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import helpers  # noqa: E402,F401  (loads ComfyUI + the pack before any test module)
