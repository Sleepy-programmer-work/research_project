"""
tests/test_tass.py — Compatibility shim. All tests have been split into:

  tests/test_tass_structural.py  — output schema and invariants
  tests/test_tass_stage1.py      — degenerate filter + pHash + fallback
  tests/test_tass_stage2.py      — Greedy FPS + fixed/adaptive modes
  tests/test_tass_errors.py      — error handling + constructor validation

Re-imports all test classes so 'pytest tests/test_tass.py' still collects them.
"""

from tests.test_tass_structural import TestTASSStructuralContract       # noqa: F401
from tests.test_tass_stage1 import (                                     # noqa: F401
    TestTASSStage1DegenerateFilter,
    TestTASSStage1PHash,
    TestTASSStage1Fallback,
)
from tests.test_tass_stage2 import TestTASSStage2GreedyFPS              # noqa: F401
from tests.test_tass_errors import TestTASSErrorHandling                 # noqa: F401


