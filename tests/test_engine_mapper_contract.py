"""The ENGINE's hook-name contract, asserted through the engine's own mapper.

Deliberately different from `test_seam.py`, and the difference is the point. Those tests assert what
THIS component does: `_Seam.point_for` refuses the ambiguous block-level hooks, and it does so whatever
the engine underneath believes. These assert what the ENGINE does, by routing through
`interp_engine.mappers.tlens_hook_to_point` on purpose.

**This file is the executable form of an upstream defect report.** Released interp-engine (through
1.5.1) resolves TransformerLens' pre-norm `hook_mlp_in` / `hook_attn_in` onto the post-norm `mlp_in` /
`attn_in` points, silently. The two tensors are a whole normalization apart: cosine 0.175 and a norm
ratio of 231x at gemma-3-1b-it layer 5, past 3000x deeper in the stack. So these tests are RED on the
released engine and GREEN on the fork branch carrying the refusal, which is what makes them evidence
rather than an assertion.

WHY THE FORK IS A TEST EXTRA AND NOT A DEPENDENCY. Proving the defect needs the fork; shipping it to
consumers does not. `_Seam.point_for` refuses those names regardless of the engine, so a consumer on the
released engine is already protected by this component. See the comment on the pin in `pyproject.toml`.

WHAT TO DO WHEN THE FIX LANDS UPSTREAM: drop the fork line from the test extra. Nothing here changes.
These assert the correct behaviour, not the presence of a fork, so the released engine satisfies them
the moment it carries the fix. That is deliberate: a fork pin that cannot be retired without editing
tests is how a fork becomes permanent.

NOT A PARITY TEST. Cross-backend convergence parity must NOT route through this mapper -- comparing a
post-norm tensor against a pre-norm one reports a spurious divergence, and the natural response to a
spurious divergence is to loosen the tolerance until it passes, which buries the real defect under a
number chosen to hide it. Parity compares semantically identical points through the seam. This file
compares the mapper against itself, which is a different question with a different answer.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def mappers():
    return pytest.importorskip("interp_engine.mappers", reason="interp-engine is a declared dependency")


def _point(mappers, hook: str):
    """Resolve a hook to `(name, layer)`, or None where the engine refuses it."""
    try:
        addr = mappers.tlens_hook_to_point(hook)
    except mappers.UnmappedHook:
        return None
    return (addr.name, addr.layer)


class TestPreAndPostNormInputsAreNotTheSamePoint:
    """The defect, stated as the property it violates.

    TransformerLens fires `hook_mlp_in` on `resid_mid` and `hook_attn_in` on `resid_pre`, both BEFORE
    the block norm. The engine's `mlp_in` / `attn_in` are the norm's OUTPUT, gain multiply included --
    the engine's own `_TLENS_STABLE` comment says so while introducing the entries that violate it.
    """

    @pytest.mark.parametrize(
        ("block_hook", "sublayer_hook"),
        [("hook_mlp_in", "mlp.hook_in"), ("hook_attn_in", "attn.hook_in")],
    )
    def test_the_block_hook_does_not_resolve_to_the_sublayer_point(self, mappers, block_hook, sublayer_hook):
        """The load-bearing assertion. If both spellings answer with the same point, one of them is
        wrong by a whole normalization, and nothing in the engine says which."""
        block = _point(mappers, f"blocks.5.{block_hook}")
        sublayer = _point(mappers, f"blocks.5.{sublayer_hook}")

        assert sublayer is not None, f"{sublayer_hook!r} is the correct spelling and must still resolve"
        assert block != sublayer, (
            f"{block_hook!r} and {sublayer_hook!r} both resolve to {sublayer}. They are different "
            "tensors: the first is the residual stream before the block norm, the second is what the "
            "sublayer actually receives. On gemma-3-1b-it layer 5 they share cosine 0.175 and differ in "
            "norm by 231x. Resolving them alike reads an artifact off the wrong activations, silently."
        )

    @pytest.mark.parametrize("block_hook", ["hook_mlp_in", "hook_attn_in"])
    def test_the_block_hook_is_refused_rather_than_remapped(self, mappers, block_hook):
        """Refusal specifically, not a remap to `resid_mid`/`resid_pre`.

        A remap would be defensible and is still wrong to do silently: the caller's intent is genuinely
        ambiguous between the pre-norm residual and the sublayer's argument, and picking one is the
        substitution this whole line of work exists to stop. The engine should say it cannot tell.
        """
        assert _point(mappers, f"blocks.5.{block_hook}") is None, (
            f"{block_hook!r} resolves to a point instead of being refused. Even a correct remap would be "
            "wrong here: the caller may mean either side of the norm, and only they can say which."
        )

    @pytest.mark.parametrize("block_hook", ["hook_mlp_in", "hook_attn_in"])
    def test_the_refusal_names_both_alternatives(self, mappers, block_hook):
        """A refusal that does not say what to ask for instead sends the caller back to guessing, which
        is the behaviour being removed."""
        with pytest.raises(mappers.UnmappedHook) as excinfo:
            mappers.tlens_hook_to_point(f"blocks.5.{block_hook}")
        message = str(excinfo.value)
        sublayer = "mlp.hook_in" if "mlp" in block_hook else "attn.hook_in"
        residual = "hook_resid_mid" if "mlp" in block_hook else "hook_resid_pre"
        assert sublayer in message, f"refusal must name {sublayer!r}, the sublayer's argument"
        assert residual in message, f"refusal must name {residual!r}, the pre-norm residual"


class TestTheReverseTableAgreesWithItself:
    """`_POINT_TO_TLENS` emitted the block-level hook for `attn_in` and the sublayer hook for `mlp_in`.

    A round trip cannot catch this: while the forward table collapses both spellings onto one point,
    `attn_in -> hook_attn_in -> attn_in` returns what it started with, closing over a name that denotes
    the wrong tensor. So this asserts the emitted NAME, not the round trip.
    """

    @pytest.mark.parametrize(("point", "expected"), [("mlp_in", "mlp.hook_in"), ("attn_in", "attn.hook_in")])
    def test_the_sublayer_points_emit_the_sublayer_hook(self, mappers, point, expected):
        emitted = mappers._POINT_TO_TLENS.get(point)
        assert emitted == expected, (
            f"{point!r} emits {emitted!r}, but that point is the norm's OUTPUT and {expected!r} is the "
            f"TransformerLens name for it. Emitting a block-level hook names the pre-norm tensor instead."
        )

    def test_the_four_sublayer_entries_are_consistent(self, mappers):
        """The asymmetry was visible without any measurement: three of four used the sublayer form."""
        entries = {p: mappers._POINT_TO_TLENS.get(p) for p in ("mlp_in", "mlp_out", "attn_in", "attn_out")}
        block_level = {p: h for p, h in entries.items() if h and not h.startswith(("mlp.", "attn."))}
        assert not block_level, f"these sublayer points emit block-level hooks: {block_level}"
