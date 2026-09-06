"""This repository's adapter example notebook, under interpretune's notebook-form contract.

Adapter-level examples became a standard because an adapter delivered through the hub needs one artifact
that exercises the HUB path end to end. Every other test here reaches this component through pip or a
staged tree, so none of them would notice the published component becoming unloadable.

The suite is glob-based: it discovers `*.ipynb` under the configured publish directory, so a notebook
added later is covered without touching a list. That is deliberate on interpretune's side and worth not
undoing here -- a roster of notebook names passes green while asserting nothing about a notebook missing
from it (speediedan/interpretune#476 is that bug in their tree).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("interpretune.testing.conformance", reason="needs interpretune with the conformance suite")

from interpretune.testing.conformance import NotebookFormConformance  # noqa: E402


class TestNotebookForm(NotebookFormConformance):
    root = Path(__file__).resolve().parents[1]

    #: The PIP form of this adapter, which the published notebook must not import.
    #:
    #: The notebook's subject is the HUB path -- `it.hub.pull` then `load_hub_adapter` behind the trust
    #: gate -- so importing the installed package would demonstrate the one delivery mechanism a reader
    #: does not need shown, and would hide the trust gate entirely.
    #:
    #: `interp_engine_adapter`, not `it_interp_engine_adapter`: the import name, not the distribution
    #: name `it-interp-engine-adapter`. The suite refuses a name absent from this environment precisely
    #: because that misspelling forbids nothing and passes forever.
    forbidden_imports = ("interp_engine_adapter",)
