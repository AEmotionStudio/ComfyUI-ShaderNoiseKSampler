"""Legacy mode must reproduce the pre-2.0 pipeline bit for bit."""
import os

import pytest
import torch

from golden_cases import CASES, golden_path, run_case
from helpers import assert_same


@pytest.mark.parametrize("name", sorted(CASES))
def test_legacy_matches_golden(name):
    path = golden_path(name)
    if not os.path.exists(path):
        pytest.fail(f"missing golden file {path}; run tests/capture_golden.py on known-good legacy code")
    expected = torch.load(path, weights_only=True)
    assert_same(run_case(name), expected)
