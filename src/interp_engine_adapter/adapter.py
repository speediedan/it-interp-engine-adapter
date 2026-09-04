"""Hub-delivered Interpretune adapter for interp-engine (decoderesearch/interp-engine).

Loaded by ``interpretune.hub.adapters.load_hub_adapter``: a single self-contained module, executed in
the caller's process behind the speediedan/interpretune#255 trust gate, which registers
``Adapter.interp_engine`` and its compositions into ``CompositionRegistry``.

**Self-contained by requirement, not by preference.** The rails import this file by path rather than
as a package, so it cannot import sibling modules from its own repo. Importing from the *installed*
``interpretune`` and ``interp_engine`` is expected.

NOTE [the seam]: every ``interp_engine`` symbol in this adapter is reached through the ``_Seam`` class
below, for two reasons. The first is version insulation: interp-engine's 1.x minors change signatures
with no deprecation aliases, deliberately, preferring load-time errors over silently wrong engines.
The second is sharper and was measured rather than anticipated -- see NOTE [hook names lie].

NOTE [hook names lie]: hook naming is not a shared vocabulary across interpretability projects, and
the disagreements are silent. ``tlens_hook_to_point`` maps TransformerLens' block-level
``hook_mlp_in`` onto the engine's post-norm ``mlp_in`` point, but TransformerLens fires that hook on
``resid_mid.clone()`` BEFORE the block norm. On google/gemma-3-1b-it layer 5 the two tensors share a
cosine similarity of **0.088**, and published SAE dashboards have been generated from the wrong one.
So this seam **refuses** the legacy block-level sublayer-input hooks rather than resolving them to the
tensor the caller probably meant. Substituting silently is exactly how this defect class spreads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from interpretune.adapters.registration import CompositionRegistry
from interpretune.analysis.backends.capabilities import (
    BackendCapability,
    InterventionMode,
    InterventionSupport,
    PositionScope,
    get_model_backend,
)
from interpretune.analysis.backends.interventions import normalize_position_scope
from interpretune.analysis.ops.base import get_batch_input
from interpretune.base import CoreHelperAttributes, BaseITModule
from interpretune.protocol import Adapter


ADAPTER_NAME = "interp_engine"


@dataclass(kw_only=True)
class InterpEngineConfig:
    """Configuration for the interp-engine execution adapter.

    Args:
        backend: Execution tier. Only ``"eager"`` is supported here. The vLLM tier is deliberately out
            of scope: it is Linux/CUDA-only, has sharp request-correctness rules (enforce_eager for
            capture, prefix-cache salting, no gradients), and those details belong to whoever owns the
            engine, not to an adapter.
        requires_grad: Build the wrapped model with gradients enabled. Off by default because capture
            for activation endpoints detaches, and holding the graph costs memory a dashboard run does
            not need.
        attn_implementation: Passed through to the engine when attention internals are wanted;
            attention capture requires an eager attention implementation.
        dtype: Passed through to ``EagerModel``. ``None`` leaves the already-loaded model's dtype
            alone, which is the point of wrapping in place.
    """

    backend: str = "eager"
    requires_grad: bool = False
    attn_implementation: str | None = None
    dtype: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend != "eager":
            raise ValueError(
                f"InterpEngineConfig.backend={self.backend!r} is not supported by this adapter; only "
                "'eager' is. The vLLM tier is intentionally out of scope for the first slice: it is "
                "Linux/CUDA-only and its request-correctness rules belong with the engine rather than "
                "an adapter."
            )


class UnmappableHookError(ValueError):
    """A TransformerLens hook name has no honest counterpart among the engine's points."""


class _Seam:
    """The one place this adapter touches ``interp_engine``.

    Every method imports the engine locally rather than at module import. That keeps loading this
    component cheap (the rails execute it during registration, before any model exists) and matches
    interpretune's own convention for heavy optional dependencies. The manifest's ``requires.pip``
    block has already guaranteed the package is present and in range by the time any of this runs.
    """

    #: Bridge-style names that genuinely denote the sublayer's argument, i.e. the block norm's OUTPUT.
    #: ``ln2.hook_out`` and ``mlp.hook_in`` are the same tensor (measured identical, cos 1.000000 on
    #: gemma-3-1b-it layer 5), so both are accepted and mean the same point.
    _SUBLAYER_INPUT_ALIASES = {"mlp.hook_in": "mlp_in", "ln2.hook_out": "mlp_in", "attn.hook_in": "attn_in"}

    #: Legacy block-level names that LOOK like sublayer inputs and are not. Refused with the name to
    #: use instead; see NOTE [hook names lie].
    _REFUSED = {
        "hook_mlp_in": "mlp.hook_in",
        "hook_attn_in": "attn.hook_in",
    }

    @staticmethod
    def _split(hook_name: str) -> tuple[int | None, str]:
        parts = hook_name.split(".")
        if len(parts) >= 3 and parts[0] == "blocks" and parts[1].isdigit():
            return int(parts[1]), ".".join(parts[2:])
        return None, hook_name

    @classmethod
    def refuse_if_ambiguous(cls, bare_hook: str) -> None:
        """Raise if a BARE hook name (no ``blocks.N.`` prefix) is one this adapter refuses to translate.

        The guard circuit-tracer's resolver is wrapped with. It asks this seam the same question the two-way
        compositions ask, so a hook the adapter refuses in one composition is not quietly accepted in another.

        Only the adapter's OWN refusal propagates. Anything else -- a name needing a model to resolve, a name the engine
        simply does not know -- is left to the caller's own resolution path, because this is a guard against a specific
        silent substitution, not a second opinion on every hook name.
        """
        try:
            cls.point_for(f"blocks.0.{bare_hook}", None)
        except UnmappableHookError:
            raise
        except Exception:
            return

    @classmethod
    def point_for(cls, hook_name: str, model: Any) -> Any:
        """Translate a TransformerLens hook name to an engine ``Address``, refusing the ambiguous ones.

        ``model`` is always passed to the engine's mapper. Translating by string alone silently
        differs on sandwich-norm architectures (Gemma-2/3, OLMo-2/3), where the block-level
        ``hook_mlp_out``/``hook_attn_out`` are the POST-norm contributions. On gemma-3-1b-it layer 5
        the string-only reading is cos 0.109 / 0.221 against the model-aware one.
        """
        from interp_engine import Address, tlens_hook_to_point

        layer, base = cls._split(hook_name)
        if base in cls._REFUSED:
            raise UnmappableHookError(
                f"{hook_name!r} is TransformerLens' block-level hook, which fires on the residual "
                f"stream BEFORE the block norm, not on the sublayer's argument. Ask for "
                f"{cls._REFUSED[base]!r} instead if you want the tensor the sublayer receives, or "
                f"'blocks.N.hook_resid_mid' if you genuinely want the pre-norm residual. This adapter "
                "refuses rather than choosing for you: the two differ by a whole normalization (cos "
                "0.088 on gemma-3-1b-it layer 5), and guessing is how published artifacts end up "
                "encoding a tensor they were not trained on."
            )
        if base in cls._SUBLAYER_INPUT_ALIASES and layer is not None:
            return Address(cls._SUBLAYER_INPUT_ALIASES[base], layer)
        return tlens_hook_to_point(hook_name, model)

    @staticmethod
    def wrap_model(model: Any, tokenizer: Any, cfg: InterpEngineConfig, model_id: str) -> Any:
        """Wrap an ALREADY-LOADED HF model in place.

        No reload, no second copy.
        """
        from interp_engine import EagerModel

        kwargs: dict[str, Any] = dict(cfg.model_kwargs)
        if cfg.attn_implementation is not None:
            kwargs["attn_implementation"] = cfg.attn_implementation
        return EagerModel(
            model_id,
            hf_model=model,
            tokenizer=tokenizer,
            requires_grad=cfg.requires_grad,
            dtype=cfg.dtype,
            **kwargs,
        )

    @staticmethod
    def capture(engine_model: Any, tokens: Any, points: list[Any], detach: bool = True) -> Any:
        """One forward pass, capturing the requested points.

        The sync free function is used deliberately: interpretune's ``*_step`` and analysis-op call
        sites are synchronous. It REFUSES inside a running event loop by design, which is why the
        notebook path uses the async surface instead of reaching for nest_asyncio.
        """
        from interp_engine import run_with_cache

        return run_with_cache(engine_model, tokens, points, detach=detach)

    @staticmethod
    def steer(engine_model: Any, tokens: Any, specs: list[Any]) -> Any:
        """Run one forward pass with steering applied AT THE LAST PROMPT POSITION, returning logits.

        ``steer`` is a CONTEXT MANAGER yielding a hook manager, not a call, so the forward has to
        happen inside it. Keeping that shape here rather than in the backend is the point of the seam:
        if the engine changes how steering is entered, exactly one method moves.

        ``specs`` is a list of ``(PositionScope, SteerSpec)`` pairs. The scope is not carried on the
        engine's ``SteerSpec`` -- it has no position field -- so it is grouped here and expressed as the
        engine's ``position_mask``.

        NOTE [the position mask is not an optimisation]: interp-engine steers EVERY prompt position by
        default, while interpretune's default scope is ``LAST_TOKEN``. Passing the engine's default
        through applies the caller's vector at all five positions of a five-token prompt when the caller
        asked for one, and the forward completes and returns a plausible result. Nothing raises, the
        shapes agree, and only the meaning differs, which is this adapter's own defect class reached
        through a DEFAULT rather than through a hook name.

        ``position_mask`` EXCLUDES positions, so ``LAST_TOKEN`` is "exclude all but the last" and
        ``ALL_POSITIONS`` is no mask at all. Measured on gpt2 layer 5: without a mask 5 of 5 positions
        change; with one, exactly 1, and the logits match a hand-built HuggingFace reference to 0.0.

        MIXED SCOPES ARE EXPRESSED, NOT REFUSED, and that was measured rather than assumed. The mask is
        per CALL, so the obvious reading is that one payload cannot carry two scopes. But ``steer``
        contexts NEST, and nesting gives a mask per group: a payload mixing scopes runs as one context
        per group. Verified against two HuggingFace forward pre-hooks applying the same two vectors at
        the same two scopes, agreeing to 0.0.

        That nesting composes correctly here because every declared mode is ADDITIVE, and addition
        commutes, so group order cannot matter. A future mode that overwrites rather than accumulates
        (``replace`` is the obvious one) would make group order significant, and this would then need to
        define the order or refuse the mixed payload. `_MODE_TO_STEER_METHOD` is the guard: it declares
        exactly one mode, and `intervention_support` is asserted to agree with it.
        """
        from interp_engine import run_with_cache, steer
        from contextlib import ExitStack

        n_positions = _prompt_length(tokens)
        last_only = list(range(n_positions - 1)) if n_positions and n_positions > 1 else None

        grouped: dict[Any, list[Any]] = {}
        for scope, spec in specs:
            grouped.setdefault(scope, []).append(spec)

        with ExitStack() as stack:
            for scope, group in grouped.items():
                mask = last_only if scope is PositionScope.LAST_TOKEN else None
                stack.enter_context(steer(engine_model, group, prompt_token_ids=tokens, position_mask=mask))
            return _logits_of(run_with_cache(engine_model, tokens, []))

    @staticmethod
    def steer_spec(
        vector: Any, layer: int, coeff: float = 1.0, method: str = "additive", point: str = "resid_post"
    ) -> Any:
        """Build one engine ``SteerSpec``."""
        from interp_engine import SteerSpec

        return SteerSpec(vector=vector, layer=layer, coeff=coeff, method=method, point=point)

    @staticmethod
    def forward(engine_model: Any, tokens: Any) -> Any:
        """A plain forward pass, no capture and no steering.

        ``EagerModel`` has no ``forward`` attribute and its instances are not callable. This said
        ``engine_model.forward(tokens)`` from the first commit and therefore never worked, against 1.3.6
        or 1.5.1 alike: not a regression, a path nothing exercised. The seam's contract tests read
        signatures and registry membership, which is the surface a minor bump moves, and never CALLED
        this, so a method that does not exist passed every one of them.

        A forward is a capture of NO points: `run_with_cache` returns a ``Cache`` whose ``.output`` is
        the model output. Verified equal to the HF forward to 0.0 on gpt2, which is the expected result
        rather than a tolerance, since the engine wraps the HF model in place and runs its forward.
        """
        from interp_engine import run_with_cache

        return _logits_of(run_with_cache(engine_model, tokens, []))

    async def capture_async(engine_model: Any, tokens: Any, points: list[Any], detach: bool = True) -> Any:
        """The notebook path: one forward pass capturing the requested points, awaited.

        ``run_with_cache`` REFUSES inside a running event loop by design, and Jupyter always has one,
        so a notebook cannot use the sync surface this adapter's ops path uses. ``EagerModel.capture``
        is the async equivalent. Kept here rather than reached around the seam so the invariant holds
        for the notebook too: exactly one place in this adapter touches the engine, and if the async
        entry point moves, one method moves with it.

        Deliberately NOT nest_asyncio: patching the running loop to make a sync API work in a notebook
        hides the reason the sync API refused, and the engine offers a real async surface.
        """
        # Same argument ORDER as the sync free function, despite the different parameter name:
        # `run_with_cache(model, tokens, points)` and `model.capture(prompt_token_ids, points)` line up
        # position for position once `self` is the model. Worth stating because both arguments are
        # positional and interchangeable to the interpreter, so a future reorder in the engine would be
        # SILENT here -- the call would keep working and capture at addresses derived from token ids.
        # `test_argument_order_matches_the_engine` pins the engine's order so that surfaces as a test
        # failure rather than as a wrong tensor.
        return await engine_model.capture(tokens, points, detach=detach)

    @staticmethod
    def capabilities() -> Any:
        """The engine's capability set, readable before warmup.

        Capability is queried, never inferred from a backend name: unsupported requests raise
        ``CapabilityUnsupported`` naming the working call, and nothing downgrades to a warning.
        """
        from interp_engine import CAPABILITIES

        return CAPABILITIES


def _tokens_from_batch(batch: dict[str, Any]) -> Any:
    """Pull token ids out of an interpretune batch, using interpretune's own key resolution.

    ``get_batch_input`` already handles the cross-backend key difference (TransformerLens ``input`` vs
    HuggingFace/NNsight ``input_ids``) and is what the bundled circuit-tracer adapter uses. Duplicating
    that logic here would diverge the moment interpretune recognises another key, and batch shapes are
    interpretune's to define, not this component's to guess at. Verified importable in a venv holding
    interpretune and interp-engine only: it pulls no framework.
    """
    return get_batch_input(batch)


def _names_filter_to_list(names_filter: Any) -> list[str]:
    """Normalize interpretune's ``NamesFilter`` to a concrete list of hook names.

    A CALLABLE filter cannot be enumerated without the model's hook inventory, which the engine does not expose as
    TransformerLens names, so it is refused rather than silently under-capturing.
    """
    if names_filter is None:
        raise UnmappableHookError("a names_filter is required: this backend cannot capture 'everything'")
    if isinstance(names_filter, str):
        return [names_filter]
    if callable(names_filter):
        raise UnmappableHookError(
            "callable names_filters are not supported by this backend: it cannot enumerate the model's "
            "hook inventory in TransformerLens vocabulary. Pass an explicit list of hook names."
        )
    return list(names_filter)


#: interpretune ``InterventionSpec.mode`` -> interp-engine ``SteerSpec.method``, for the modes whose
#: semantics were MEASURED to match. Only ``add`` is here; each omission has its own reason, and the
#: reasons are not interchangeable.
#:
#: ``add`` maps because the operations are the same one: interpretune's ``add`` is
#: ``h + tensor * scale`` and the engine's ``additive`` delta is ``coeff * vector``.
#:
#: ``project`` is NOT mapped to ``orthogonal``, DESPITE THE NAME, and this is the substitution most
#: likely to be made by someone reading quickly. They are COMPLEMENTARY projections:
#:
#:     engine ``orthogonal``   h -> (I-P)h + coeff * P h    keeps the orthogonal part, scales parallel
#:     interpretune ``project``   h -> P h                  keeps the parallel part, discards orthogonal
#:
#: ``orthogonal`` always retains ``(I-P)h``, so NO coeff produces ``Ph`` (measured: coeff 0, 1, 2, -1
#: all fail; ``test_orthogonal_can_never_produce_project_onto`` pins it). ``projection_cap`` is not it
#: either: it clamps a projection into ``[min, max]`` rather than replacing an activation.
#:
#: ``replace`` overwrites the whole activation. Both activation-reading engine methods move only the
#: component along one direction and leave the complement untouched by construction, so neither can
#: express an overwrite.
#:
#: ``patch`` is a lens-coordinate SWAP, ``h <- h + V(sigma(c) - c)``, whose targets are a permutation of
#: the coordinates MEASURED during the forward. ``projection_cap`` takes static ``min``/``max``, so a
#: data-dependent target cannot be expressed. This is the mode the J-space steering path uses.
#:
#: NOTE on a reason that was wrong here before: "additive steering never observes the current
#: activation" is true of ``additive`` and false of ``orthogonal`` and ``projection_cap``, which both
#: read it. Stating it as the general justification would have licensed exactly the ``project ->
#: orthogonal`` mapping refused above. Each refusal now names its own mechanism.
#:
#: MEASURED CANDIDATES for modes interpretune does not define yet (jlens lane, k=1 only -- see
#: ``_REJECT_SPAN_CAVEAT``): ``reject`` (h - VV^+h) == ``orthogonal`` with ``coeff=0``, exactly;
#: ``clamp`` (set the coordinate along v to a target) == ``projection_cap`` with ``min == max ==
#: target``, exactly. Both verified numerically before being written down.
_MODE_TO_STEER_METHOD = {"add": "additive"}

#: Why the measured ``reject``/``clamp`` mappings are qualified by ``k=1``. The engine's projector is
#: RANK ONE (``P = v_hat v_hat^T``, one vector per ``SteerSpec``), while the jlens modes are defined
#: over a ``(k, d_model)`` span. Composing k per-row specs reproduces the span operation ONLY when the
#: rows are orthonormal: measured, stacked rank-1 rejections equal ``h - VV^+h`` exactly for an
#: orthonormal ``V`` and NOT for a generic one (row cosine -0.240 was enough to break it). J-lens pole
#: pairs are not orthonormal by construction, so a caller wanting a span operation must orthonormalize
#: first and understand that this changes which subspace is being removed.
_REJECT_SPAN_CAVEAT = "rank-1 per spec; span operations need orthonormal rows"


def _steer_method_for(mode: str) -> str:
    """Translate an intervention mode, refusing the ones this backend cannot express faithfully."""
    method = _MODE_TO_STEER_METHOD.get(mode)
    if method is None:
        raise UnmappableHookError(
            f"intervention mode {mode!r} cannot be expressed through interp-engine's steering surface "
            f"from this backend (supported: {sorted(_MODE_TO_STEER_METHOD)}). See the mapping table's "
            "notes for why each refused mode is refused -- the reasons differ, and 'project' in "
            "particular is NOT interp-engine's 'orthogonal' despite the name: they are complementary "
            "projections. Refused rather than approximated: a steering call that silently applied the "
            "wrong mode would complete and return plausible logits for an intervention nobody asked for."
        )
    return method


def _require_expressible_operand(tensor: Any, hook_name: str) -> Any:
    """Refuse a span operand, because the engine's projector is rank-one and additive BROADCASTS.

    interpretune's ``intervention_tensor`` is deliberately polymorphic: it may be ``(d_model,)`` or a
    higher-rank slice, and the J-space modes are defined over a ``(k, d_model)`` span. This backend
    cannot take that shape, and the two ways it fails are different kinds of bad:

    * ``orthogonal``/``projection_cap`` raise, but with a tensor-size message naming neither the span
      nor the rank-one projector, so the caller learns nothing about the actual constraint.
    * ``additive`` does NOT reliably raise, and the silent set has a general form worth stating instead
      of an enumeration: **it is exactly torch's BROADCAST-COMPATIBLE set** -- ``k == 1``, or
      ``seq_len == 1``, or ``k == seq_len``. Verified as an exact match across ``k, seq in 1..6``.

      Two of those three branches are wrong, and they are wrong differently:

        ``k == seq_len``   the shape is PRESERVED and the values are corrupted: a different vector is
                           applied at each position, with no span semantics anywhere in the result.
        ``seq_len == 1``   worse. The SEQUENCE AXIS IS REPLACED by ``k``: a 1-token prompt steered with
                           a 3-row span yields ``(1, 3, d_model)``, which flows downstream as though
                           the prompt had three tokens. And ``seq_len == 1`` is not exotic -- it is a
                           single-token prompt, and every step of incremental decoding, so the case
                           most likely to be hit in generation is the one that fails worst.

      This is why it can never be caught reliably downstream: **broadcasting is designed to succeed
      silently, so the engine is not failing to validate so much as inheriting a rule whose entire
      purpose is to not raise.** Refusing ``ndim > 1`` here is therefore not defensive coding against a
      list of known cases; it is the only way to leave the broadcast rule's domain, and it predicts
      cases rather than enumerating them.

      It also means the behaviour is DATA-DEPENDENT: the same call site is safe with a 4-token prompt
      and silently wrong with a 3-row span at 3 tokens. A test fixing one prompt length can only
      establish that THAT length is not the bad one, which is the argument for a guard over a test.

    A caller wanting a span operation composes per-row specs, which reproduces ``h - V V^+ h`` only for
    ORTHONORMAL rows (measured exact for orthonormal ``V``, wrong at row cosine -0.240). Orthonormalizing
    preserves the SPAN, so a rejection survives it, but coordinates expressed in the original basis do
    not -- so a clamp target stated in pole coordinates is not preserved by that change of basis.
    """
    ndim = getattr(tensor, "ndim", None)
    if ndim is not None and ndim > 1:
        shape = tuple(getattr(tensor, "shape", ()))
        raise UnmappableHookError(
            f"intervention tensor for {hook_name!r} has shape {shape}: this backend takes a single "
            "direction per spec, because interp-engine's projector is rank-one. A span operand would "
            "not raise reliably -- with additive steering it broadcasts whenever the leading dimension "
            "equals the sequence length, applying a different vector per position and returning a "
            "plausible result that is not a span operation. Compose one spec per direction instead, "
            "and note that this reproduces a span operation only for ORTHONORMAL rows."
        )
    return tensor


def _logits_of(cache: Any) -> Any:
    """The logits tensor from an engine ``Cache``, raising rather than defaulting when absent.

    NOT ``getattr(cache, "logits", None)``. That spelling was here and it could never work: the engine's
    ``Cache`` exposes ``get``, ``output`` and ``tensors``, and the logits live at ``cache.output.logits``.
    Because the attribute simply does not exist, the default fired on EVERY call and `fwd_w_cache`
    returned ``None`` for logits always, in every version, since it was written.

    A `getattr` default is worse here than a wrong method name: a wrong name raises the first time it
    runs, while a default returns a plausible ``None`` forever. Interpretune's protocol declares
    ``tuple[torch.Tensor, Any]``, so downstream `answer_logits` was silently ``None`` rather than an
    error anyone could act on.

    Hence the raise. If a future engine returns an output without ``logits``, that should stop a caller
    rather than hand them a shape the protocol says cannot happen.
    """
    output = getattr(cache, "output", None)
    if output is None:
        raise UnmappableHookError(
            "the engine returned a cache with no `output`, so there are no logits to return; the "
            "protocol declares a tensor here and this backend will not substitute None for one."
        )
    logits = getattr(output, "logits", None)
    if logits is None:
        raise UnmappableHookError(
            f"the engine's output ({type(output).__name__}) exposes no `logits`, so this backend "
            "cannot satisfy the protocol's declared return type. Refused rather than returned as None."
        )
    return logits


def _prompt_length(tokens: Any) -> int:
    """Number of prompt positions in ``tokens``, or 0 when it cannot be told.

    Returning 0 rather than guessing 1 matters: the caller uses this to build a last-position mask, and
    a wrong length would mask the wrong positions silently. When the shape cannot be read, no mask is
    applied and the engine's own behaviour stands, which is visible rather than subtly wrong.
    """
    shape = getattr(tokens, "shape", None)
    if shape is None:
        return 0
    return int(shape[-1]) if len(shape) >= 1 else 0


def _intervention_targets(interventions: Any, model: Any) -> list[tuple[int, str, Any]]:
    """Yield ``(layer, engine point, spec)`` for each intervention, via the same seam capture uses.

    ``model`` is REQUIRED, and passing it is not a formality. ``tlens_hook_to_point`` accepts a null
    model and silently returns the string-only reading, which differs on sandwich-norm architectures
    for exactly ``hook_mlp_out``/``hook_attn_out`` (cos 0.109 / 0.221 on gemma-3-1b-it layer 5). Those
    two are not in the alias table, so a model-blind call here would resolve them to the wrong point.
    Steering is the worse direction to get this wrong in: capture at a wrong point returns a tensor you
    might notice, while steering WRITES somewhere the caller did not name, the forward completes, and
    the result looks plausible. That is this adapter's own defect class arriving through an optional
    argument.
    """
    if model is None:
        raise UnmappableHookError(
            "a model is required to resolve intervention targets: hook translation is model-aware by "
            "design, and the model-blind reading silently differs on sandwich-norm architectures."
        )
    targets = []
    items = interventions.items() if hasattr(interventions, "items") else interventions
    for hook_name, specs in items:
        address = _Seam.point_for(hook_name, model)
        layer = getattr(address, "layer", None)
        if layer is None:
            raise UnmappableHookError(f"{hook_name!r} does not name a layer, so it cannot be steered")
        # A NamedTuple IS a tuple, so a bare `InterventionSpec` would otherwise be treated as a SEQUENCE
        # of specs and iterated into its own FIELDS. The first field is a tensor, which then reaches
        # `spec.intervention_tensor` as an AttributeError naming nothing useful. `_fields` is what
        # distinguishes the two, and interpretune's `InterventionSpec` is exactly that shape.
        #
        # The bundled backends never meet this because they normalize to a list before iterating; this
        # seam takes the caller's value directly, so telling a spec from a sequence of specs is its job.
        for spec in [specs] if hasattr(specs, "_fields") or not isinstance(specs, (list, tuple)) else specs:
            targets.append((layer, getattr(address, "name", "resid_post"), spec))
    return targets


class InterpEngineAttributeMixin:
    """Config access and the wrapped-engine handle."""

    it_cfg: Any

    @property
    def interp_engine_cfg(self) -> InterpEngineConfig:
        cfg = getattr(self.it_cfg, "interp_engine_cfg", None)
        return cfg if cfg is not None else InterpEngineConfig()

    @property
    def engine_model(self) -> Any:
        """The ``EagerModel`` wrapping this session's model, or ``None`` before init."""
        return getattr(self, "_engine_model", None)


class InterpEngineCaptureMixin(InterpEngineAttributeMixin):
    """Capture through the seam, in the vocabulary interpretune ops already speak."""

    def post_auto_model_init(self, *args: Any, **kwargs: Any) -> Any:
        """Wrap ``self.model`` after Interpretune has loaded it.

        This is where the TL and nnsight adapters also act, and it is what makes the wrap free: the
        model already exists, so ``EagerModel`` takes it in place.
        """
        result = super().post_auto_model_init(*args, **kwargs)  # type: ignore[misc]
        model_id = getattr(getattr(self, "it_cfg", None), "model_name_or_path", "") or ""
        # ATTACH, DO NOT OVERRIDE. Attaching the backend the way bundled adapters do (`self._model_backend`)
        # is right; assigning it unconditionally was not. In a composition where another adapter also
        # attaches one -- which is how a third adapter composes with this one -- an unconditional assignment
        # makes MRO ORDER silently decide which backend wins, and the loser's capabilities disappear with no
        # error at the point of loss. Interpretune's circuit-tracer adapter states this contract explicitly
        # and honours it; a hub adapter that did not would be the reason a composition including it behaved
        # differently from one that did not, which is a defect nobody would look for here.
        #
        # There is no model-backend REGISTRY to register into (backends are adapter-attached by
        # construction), so a hub adapter reaches this seam exactly as a bundled one does.
        if get_model_backend(self) is None:
            self._model_backend = InterpEngineModelBackend()
        self._engine_model = _Seam.wrap_model(
            self.model,  # type: ignore[attr-defined]
            getattr(self, "tokenizer", None),
            self.interp_engine_cfg,
            model_id,
        )
        return result

    def engine_capture(self, tokens: Any, hook_names: list[str], detach: bool = True) -> Any:
        """Capture at TransformerLens hook names, translated through the seam."""
        points = [_Seam.point_for(name, self.engine_model) for name in hook_names]
        return _Seam.capture(self.engine_model, tokens, points, detach=detach)


class BaseInterpEngineModule(BaseITModule):
    """Base module for interp-engine compositions."""


class InterpEngineModule(InterpEngineCaptureMixin, CoreHelperAttributes, BaseInterpEngineModule):
    """The ``(core, interp_engine)`` module composition."""


#: Cache for the composed circuit-tracer pairing class. Built once, lazily, because constructing it
#: imports circuit-tracer, and this module MUST import without it: the two core compositions are the
#: ones that have to work in an environment that never installed the optional package.
_CT_PAIRING: Any = None


def _circuit_tracer_pairing() -> Any:
    """The ``(core, interp_engine, circuit_tracer)`` module class, or ``None`` where unavailable.

    LAZY BY NECESSITY, not by preference. `interpretune.adapters.circuit_tracer.adapter` imports
    circuit-tracer eagerly at module level -- correct for a module whose classes subclass circuit-tracer's
    -- so touching it at import time here would make an OPTIONAL dependency mandatory for the two
    compositions that do not need it, and the failure would land at hub load rather than at registration.
    """
    global _CT_PAIRING
    if _CT_PAIRING is not None:
        return _CT_PAIRING
    try:
        from interpretune.adapters.circuit_tracer.adapter import (
            BaseCircuitTracerModule,
            CircuitTracerAdapter,
            CircuitTracerAnalysisMixin,
        )
    except Exception:
        return None

    class CircuitTracerInterpEngineModuleMixin(InterpEngineAttributeMixin):
        """The pairing-specific layer, and the reason it exists is a refusal rather than a feature.

        Circuit-tracer resolves a transcoder set's TransformerLens hook names through
        ``interp_engine.mappers.tlens_hook_to_point``. That mapper resolves TL's BLOCK-LEVEL
        ``hook_mlp_in``/``hook_attn_in`` -- which fire on the residual stream BEFORE the block norm -- to
        the engine's ``mlp_in``/``attn_in``, which are the norm's OUTPUT. Those are a whole normalization
        apart: measured on gemma-3-1b-it, cosine 0.175 at layer 5 and a magnitude ratio reaching ~3000x
        deeper in the stack.

        This adapter refuses that substitution in its two-way compositions. A three-way composition that
        silently regained it would be worse than not shipping one at all, because the refusal in the other
        two is what tells a user this adapter is careful about the distinction. So the guard is reinstated
        HERE, at the boundary where the caller named the hook.

        Circuit-tracer special-cases ``ln1/ln2.hook_normalized`` (the hooks the mapper refuses) but not
        these two (the ones it silently mis-resolves), so nothing downstream catches it.
        """

        def post_auto_model_init(self, *args: Any, **kwargs: Any) -> Any:
            result = super().post_auto_model_init(*args, **kwargs)  # type: ignore[misc]
            self._guard_feature_hook_resolution()
            return result

        def _guard_feature_hook_resolution(self) -> None:
            """Route circuit-tracer's hook resolution through this adapter's seam first.

            Wraps the INSTANCE rather than subclassing, because circuit-tracer validates the replacement model by class
            NAME (`type(m).__name__ != CT_BACKEND_REGISTRY[backend]`), so a subclass would be rejected as the wrong
            model for the backend.
            """
            model = getattr(self, "_replacement_model", None)
            resolve = getattr(model, "_resolve_feature_hook", None)
            if resolve is None or getattr(resolve, "_it_seam_guarded", False):
                return

            def guarded(hook: str, _inner: Any = resolve) -> Any:
                _Seam.refuse_if_ambiguous(hook)
                return _inner(hook)

            guarded._it_seam_guarded = True  # type: ignore[attr-defined]
            model._resolve_feature_hook = guarded

    class CircuitTracerInterpEngineModule(
        CircuitTracerInterpEngineModuleMixin,
        CircuitTracerAnalysisMixin,
        CircuitTracerAdapter,
        CoreHelperAttributes,
        BaseCircuitTracerModule,
        BaseInterpEngineModule,
    ):
        """The ``(core, interp_engine, circuit_tracer)`` module composition.

        Base order mirrors the bundled pairings exactly (``CircuitTracerNNsightModule`` puts
        ``BaseNNsightModule`` last, where ``BaseInterpEngineModule`` sits here): the pairing mixin wins,
        then circuit-tracer's own layers, then the partner backend's base. The only difference from the
        bundled cases is WHO defines it -- circuit-tracer owns the TL and nnsight pairings because it
        arrived after both, and this component owns its own because it arrived after circuit-tracer. Same
        shape, same precedence, different owner.
        """

    _CT_PAIRING = CircuitTracerInterpEngineModule
    return _CT_PAIRING


class InterpEngineModelBackend:
    """The engine as an interpretune ``ModelBackend``: required core plus ``INTERVENTION``.

    Interpretune's ``ModelBackend`` was decomposed into a required core and capability-gated method
    groups precisely so a component like this one can implement what it actually does and SAY so
    (interpretune R1, speediedan/interpretune#401). This implements:

      * ``ModelBackendCore`` -- ``fwd``, ``fwd_w_cache``, ``wrap_activation_cache``, and the capability
        introspection ops gate on.
      * ``SupportsIntervention`` -- ``fwd_w_intervention``, via the engine's steering.

    and deliberately NOT ``SupportsLatentModels`` or ``SupportsGradients``: the eager tier has no
    SAE-attachment design and this adapter does not claim a gradient surface. An op that needs either
    now fails at ``require_backend_capability`` with a message naming the op, the missing capability
    and what this backend does claim -- rather than as an ``AttributeError`` somewhere inside the op.

    Every engine symbol goes through ``_Seam``, like the rest of this file; interpretune symbols are
    imported at module level, since those are the framework this component extends rather than the
    dependency it insulates against.
    """

    @property
    def capabilities(self) -> frozenset[Any]:
        return frozenset({BackendCapability.INTERVENTION})

    @property
    def intervention_support(self) -> InterventionSupport:
        """Which configurations of ``INTERVENTION`` this backend honours.

        BOTH scopes, because the engine expresses them one argument apart: ``position_mask`` excludes
        positions, so ``LAST_TOKEN`` is a mask and ``ALL_POSITIONS`` is its absence. Declaring only
        ``LAST_TOKEN`` would have been the easy answer and the wrong one -- it would refuse an operation
        this backend can perform exactly, which is the mirror image of silently narrowing one.

        ONE mode, and the asymmetry is real rather than an omission. ``replace``, ``patch`` and
        ``project`` all need the CURRENT activation, and the engine's additive steering primitive never
        observes it. `_steer_method_for` refuses those three by name; this declaration is the same fact
        stated ahead of the call, so the gate can refuse before reaching this backend at all.
        """
        return InterventionSupport(
            position_scopes=frozenset({PositionScope.LAST_TOKEN, PositionScope.ALL_POSITIONS}),
            modes=frozenset({InterventionMode(mode) for mode in _MODE_TO_STEER_METHOD}),
        )

    def supports(self, capability: Any) -> bool:
        # Compare by VALUE: interpretune's own gate does, because a suite can load the capabilities
        # module twice and leave value-equal, identity-distinct members on either side.
        supported = {getattr(c, "value", c) for c in self.capabilities}
        return getattr(capability, "value", capability) in supported

    # -- ModelBackendCore ---------------------------------------------------------------------

    def fwd(self, model: Any, batch: dict[str, Any]) -> Any:
        return _Seam.forward(model, _tokens_from_batch(batch))

    def fwd_w_cache(self, model: Any, batch: dict[str, Any], names_filter: Any) -> tuple[Any, Any]:
        """Forward with activation caching, at TransformerLens hook names.

        ``names_filter`` is interpretune's vocabulary; the seam translates each name to an engine
        address and REFUSES the ambiguous ones rather than guessing, which is the behaviour this
        adapter exists to get right.
        """
        hook_names = _names_filter_to_list(names_filter)
        points = [_Seam.point_for(name, model) for name in hook_names]
        cache = _Seam.capture(model, _tokens_from_batch(batch), points)
        by_hook = {name: cache[point] for name, point in zip(hook_names, points)}
        return _logits_of(cache), self.wrap_activation_cache(by_hook, model)

    def wrap_activation_cache(self, cache_dict: dict[str, Any], model: Any) -> Any:
        """Return the mapping as-is.

        The protocol permits this: it asks for "a cache object suitable for indexed access by hook
        name", and a dict already is one. Wrapping in a TransformerLens ``ActivationCache`` would make
        this adapter depend on TransformerLens, which is exactly the coupling it avoids.
        """
        return cache_dict

    # -- SupportsIntervention -----------------------------------------------------------------

    def fwd_w_intervention(
        self,
        model: Any,
        batch: dict[str, Any],
        interventions: Any,
        latent_model_handles: list[Any] | None = None,
    ) -> tuple[Any, Any]:
        """Baseline and steered logits, from the engine's steering surface.

        ``latent_model_handles`` is refused rather than ignored: this backend does not claim
        ``LATENT_MODELS``, so accepting handles and silently dropping them would produce a result that
        looks like an intervention over latents and is not.
        """
        if latent_model_handles:
            raise UnmappableHookError(
                "this backend does not support latent-model handles (it claims INTERVENTION but not "
                "LATENT_MODELS); attach latents through a backend that does, rather than having them "
                "silently ignored here."
            )
        tokens = _tokens_from_batch(batch)
        # Paired with its scope rather than flattened: the engine's `SteerSpec` has no position field,
        # so the scope has to travel beside the spec to reach the mask that expresses it.
        specs = [
            (
                normalize_position_scope(getattr(spec, "position_scope", PositionScope.LAST_TOKEN)),
                _Seam.steer_spec(
                    vector=_require_expressible_operand(spec.intervention_tensor, point),
                    layer=layer,
                    coeff=spec.scale_factor,
                    method=_steer_method_for(getattr(spec, "mode", "replace")),
                    point=point,
                ),
            )
            for layer, point, spec in _intervention_targets(interventions, model)
        ]
        pre = _Seam.forward(model, tokens)
        post = _Seam.steer(model, tokens, specs)
        return pre, post


class InterpEngineAdapter:
    """Registers this adapter's compositions.

    Discovered by the rails via ``register_adapter_ctx``.
    """

    #: Every composition this component can register, with what each needs. Data rather than branches so
    #: the manifest and the code can be compared entry for entry: the manifest's `requires` decides which
    #: of these the RAILS consider satisfiable, and `_needs_circuit_tracer` is how the same question gets
    #: answered on the direct-import path, where no manifest was consulted.
    _COMPOSITIONS = (
        ((Adapter.core, ADAPTER_NAME), False, "interp-engine eager-tier adapter composed with core."),
        (
            (Adapter.lightning, ADAPTER_NAME),
            False,
            "interp-engine eager-tier adapter composed with the Lightning adapter.",
        ),
        (
            (Adapter.core, ADAPTER_NAME, Adapter.circuit_tracer),
            True,
            "interp-engine composed with circuit-tracer attribution, on top of core.",
        ),
    )

    @staticmethod
    def _composition_key(combination: tuple[Any, ...]) -> tuple[str, ...]:
        """The rails' identity for a composition: component plus its adapter names, value-sorted.

        `getattr(a, "value", a)` rather than `str()`, mirroring the rails' own normalization exactly, and
        that is not a style choice: `str(Adapter.core)` is ``'Adapter.core'``, so a key built with `str()`
        never matches the rails' key built from manifest STRINGS. Every composition is then judged
        unsupported and skipped SILENTLY, because skipping an unsatisfiable composition is the documented
        correct outcome. The failure surfaces later as "satisfiable composition did not register", which
        points at registration when the fault is one character of key construction.
        """
        names = sorted(str(getattr(a, "value", a)) for a in combination)
        return ("module", *names)

    @classmethod
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
        """Register the compositions this environment can actually support.

        TWO PATHS, and the difference between them is the whole point of one published component serving
        every environment:

        * Loaded through the hub rails, `supported_compositions()` returns the set the loader determined
          is satisfiable from the manifest's per-composition `requires`. Register exactly that -- the
          rails then check that every satisfiable composition registered, so under-registering is an
          error rather than a quiet loss.
        * Imported directly (`pip install`), it returns ``None``, meaning no manifest governs this
          invocation. That is a named case, not a fallback: nothing evaluated the conditionality, so this
          code answers the same question itself and registers what it knows it can support.
        """
        try:
            from interpretune.hub.adapters import supported_compositions

            supported = supported_compositions()
        except ImportError:  # interpretune predating the per-composition `requires` rails
            supported = None

        for combination, needs_ct, description in cls._COMPOSITIONS:
            if supported is None:
                if needs_ct and _circuit_tracer_pairing() is None:
                    continue
            elif cls._composition_key(combination) not in supported:
                continue
            classes = (InterpEngineModule,) if not needs_ct else (_circuit_tracer_pairing(),)
            if classes[0] is None:
                # The rails called this composition satisfiable but the class will not build. Raising
                # names the disagreement; skipping would leave the rails' own invariant to report it as
                # a registration mismatch, which points at the wrong thing.
                raise UnmappableHookError(
                    f"composition {cls._composition_key(combination)} was reported satisfiable, but the "
                    "circuit-tracer pairing class could not be built in this environment."
                )
            adapter_ctx_registry.register(
                ADAPTER_NAME,
                component_key="module",
                adapter_combination=combination,  # type: ignore[arg-type]
                composition_classes=classes,
                description=description,
            )
