"""This component under finetuning-scheduler's ``TunableView`` contract, through interpretune's tuning conformance
(#48).

interpretune attaches a view to every module after model initialization (speediedan/interpretune#758), and
``TuningConformance`` runs finetuning-scheduler's view cases against it, selected by what the view declares. The target
is the same one ``test_conformance.py`` uses, so the session tuned here is the session the backend conformance measures.
"""

from __future__ import annotations

import pytest

pytest.importorskip("interpretune.testing.conformance", reason="needs interpretune with the conformance suite")

from interpretune.testing.conformance import ConformanceTarget  # noqa: E402
from interpretune.testing.conformance.tuning import TuningConformance  # noqa: E402

from tests.test_conformance import COMPOSITION, build_session_cfg, load_component  # noqa: E402


class TestInterpEngineTuning(TuningConformance):
    target = ConformanceTarget(
        composition=COMPOSITION,
        session_cfg_factory=build_session_cfg,
        forward_family="hf_native",
        load=load_component,
    )
