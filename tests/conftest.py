"""Test configuration, and one deliberate divergence from interpretune worth stating up front.

**`IT_RUN_CUDA_TESTS` is ADDITIVE here, not a selector.** Interpretune's conftest treats it as a
selector -- `pytest_collection_modifyitems` does `items[:] = [...]`, REPLACING the collected list with
the CUDA-marked subset. That is right for a 2300-test suite where the CUDA leg is a separate job with
its own budget.

Copying it here would be a silent regression. This repo's Azure GPU lane sets `IT_RUN_CUDA_TESTS=1` and
is the ONLY lane that installs circuit-tracer, so it is the only place the
`(core, interp_engine, circuit_tracer)` composition tests run at all. Under selector semantics that lane
would collect the device tests and drop the other 55, and the composition coverage that lane exists to
provide would vanish -- reported as a smaller green number, which is not an error anyone would chase.

So here the variable UN-SKIPS rather than selects: the GPU lane runs the full suite PLUS the device
tests. `test_cuda_selection_is_additive` asserts this, so a future edit that copies interpretune's
semantics turns the suite red instead of quietly narrowing the one lane that matters.

The pipeline prints `collected` beside `passed` for the same reason, from the other direction. Neither
guard is redundant: that one catches a narrowed SELECTION, this one prevents it.
"""

from __future__ import annotations

#: Interpretune's conformance suite ships as a pytest plugin, and a repository opts in by naming it.
#: The suite owns case selection: it reads what the composed module's backends DECLARE at runtime and
#: runs the cases for each declared surface, with a refusal case for each undeclared one, so declaring
#: less is never a way to pass more. See `tests/test_conformance.py` for this repository's half.
pytest_plugins = ["interpretune.testing.conformance.plugin"]


import os

import pytest


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cuda: needs a CUDA device. Skipped unless a GPU is present AND IT_RUN_CUDA_TESTS=1. "
        "Both conditions, because a GPU being present is not a request to spend it.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip CUDA-marked tests unless asked for. Never narrow the collection.

    Note what this does NOT do: it does not reassign `items`. Adding a `items[:] = ...` here is the
    exact regression the module docstring describes, and the accompanying test exists to catch it.
    """
    requested = os.getenv("IT_RUN_CUDA_TESTS", "0") == "1"
    if requested and _cuda_available():
        return

    reason = (
        "needs IT_RUN_CUDA_TESTS=1 and a CUDA device"
        if not requested
        else "IT_RUN_CUDA_TESTS=1 but no CUDA device is visible"
    )
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)
