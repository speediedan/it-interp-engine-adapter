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
    ModelBackendCapability,
    InterventionMode,
    InterventionSupport,
    PositionScope,
    get_model_backend,
)
from interpretune.analysis.backends.interventions import normalize_position_scope
from interpretune.analysis.ops.base import get_batch_input
from interpretune.adapters.core import CoreAdapter
from interpretune.base import CoreHelperAttributes, BaseITModule
from interpretune.config.module import ITConfig
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


@dataclass(kw_only=True)
class ITInterpEngineConfig(ITConfig):
    """The composed module config for this adapter: an ``ITConfig`` that DECLARES ``interp_engine_cfg``.

    Two config classes, as every bundled adapter has. :class:`InterpEngineConfig` above is the payload,
    holding this adapter's settings; this one is the composed config a session's module actually
    receives, and its job is to make ``interp_engine_cfg`` a real dataclass field.

    That distinction is the whole point rather than a formality. Without a class declaring the field,
    interpretune's auto-composition has nothing to compose: the setting is set on the seed config as a
    stray attribute after construction and read back through a `getattr` default, so the module looks
    configured while its config never was. A stray attribute also cannot be validated, serialized with
    the config, or distinguished from a misspelling.
    """

    interp_engine_cfg: InterpEngineConfig = field(default_factory=InterpEngineConfig)


class UnmappableHookError(ValueError):
    """A TransformerLens hook name has no honest counterpart among the engine's points."""


class EngineHasNoPointError(UnmappableHookError):
    """The ENGINE has no point for this name -- as distinct from this adapter refusing it.

    A subclass, so every capture path that catches `UnmappableHookError` still refuses: from the caller's
    seat "cannot capture here" is one outcome and the reason does not change what they can do.

    It exists for the one place where the reason IS the whole question. `_Seam.refuse_if_ambiguous`
    guards circuit-tracer's own resolver, and it must propagate only THIS adapter's refusals; a name the
    engine merely does not know has to fall through to circuit-tracer, which may well resolve it. Before
    this class existed the two were the same exception, so widening the message on engine gaps silently
    converted "interp-engine has no point for `ln2.hook_normalized`" into "this adapter refuses
    `ln2.hook_normalized`", and circuit-tracer stopped seeing a hook it resolves perfectly well.

    That regression was invisible in every environment without circuit-tracer installed, which is every
    environment except the GPU lane.
    """


class _Seam:
    """The one place this adapter touches ``interp_engine``.

    Every method imports the engine locally rather than at module import. That keeps loading this
    component cheap (the rails execute it during registration, before any model exists) and matches
    interpretune's own convention for heavy optional dependencies. The manifest's ``requires.pip``
    block has already guaranteed the package is present and in range by the time any of this runs.
    """

    #: Bridge-style names that genuinely denote the sublayer's argument, i.e. the block norm's OUTPUT.
    #: ``ln2.hook_out``/``mlp.hook_in`` and ``ln1.hook_out``/``attn.hook_in`` are each ONE tensor under two
    #: spellings, measured identical at max|d| = 0.0 and cos 1.000000 on gpt2 layer 5 and on gemma-3-1b-it
    #: layer 5 -- a pre-norm trunk and a sandwich-norm one, since that is the architecture where a naive
    #: reading of a block norm's output stops being the sublayer's argument.
    #:
    #: BOTH spellings of a pair belong here, and the attn pair was half-populated for a while, so it is
    #: worth saying why the missing half is not merely redundant. Interpretune's vocabulary canonicalizes
    #: the two spellings to a single base, so a capture declaration derived by probing each spelling
    #: independently answers one question two ways -- and at lookup the canonical spelling wins. Where
    #: that is the refused one, the accepted spelling becomes unreachable rather than redundant, and the
    #: declaration promises a point the capture path then refuses.
    _SUBLAYER_INPUT_ALIASES = {
        "mlp.hook_in": "mlp_in",
        "ln2.hook_out": "mlp_in",
        "attn.hook_in": "attn_in",
        "ln1.hook_out": "attn_in",
    }

    #: Block-level names interp-engine's mapper has no entry for, though the tensor they name is one
    #: the engine addresses under another name. NOT the same category as `_REFUSED` below, and keeping
    #: the two apart is the entire discipline of this file:
    #:
    #:   `_REFUSED` names tensors that DIFFER. `blocks.N.hook_mlp_in` and `mlp.hook_in` are a whole
    #:   normalization apart (cos 0.088 on gemma-3-1b-it layer 5). Mapping one to the other would answer
    #:   a question the caller did not ask, and that substitution is why this adapter exists.
    #:
    #:   This table names tensors that are IDENTICAL, and identical by measurement rather than by
    #:   argument. `blocks.5.hook_in` is what interpretune's own resolver resolves to a forward pre-hook
    #:   on block 5; captured that way it agrees with the engine's `resid_pre.5` at max|d| = 0.0, and
    #:   with `blocks.4.hook_out` (which the engine DOES accept, as `resid_post.4`) at 0.0 and cos
    #:   1.000000. So the engine accepts a block's output under this spelling and refuses its input, one
    #:   side of a symmetric pair. That is an alias gap upstream, reported there; the entry here is what
    #:   keeps this adapter usable in the meantime, and it should be deleted when the engine grows it.
    #:
    #: A future session adding to this table owes the same measurement. "These obviously name the same
    #: thing" is precisely the reasoning `_REFUSED` exists to refuse.
    _MEASURED_ALIASES = {"hook_in": "resid_pre"}

    #: The same category, for names carrying no layer. `unembed.hook_in` is the input to the unembedding,
    #: which interpretune's resolver resolves to a forward pre-hook on `lm_head`; captured that way it
    #: agrees with the engine's global `final_norm` point at max|d| = 0.0, cos 1.000000 on gpt2. The
    #: engine HAS the tensor and lists `final_norm` in `UNMAPPED_TLENS`, meaning "no TransformerLens
    #: spelling" rather than "cannot capture", so this is the same one-sided gap as `hook_in` and not a
    #: capability this backend lacks.
    _MEASURED_GLOBAL_ALIASES = {"unembed.hook_in": "final_norm"}

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
        except EngineHasNoPointError:
            # The engine not knowing a name is not this adapter's refusal, and circuit-tracer resolves
            # names the engine has no point for -- `ln2.hook_normalized` among them. Ordered before the
            # base class deliberately: it IS an `UnmappableHookError`, so the broad clause would catch it.
            return
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
        from interp_engine import Address, UnmappedHook, tlens_hook_to_point

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
        if base in cls._MEASURED_ALIASES and layer is not None:
            # GUARDED on the architecture that makes the identity false. The engine treats the mirror
            # name `hook_out` as stream-dependent -- `resid_post` on an ordinary trunk, the whole stream
            # bundle on a hyper-connection model -- so a block's input is equally stream-dependent, and
            # `resid_pre` is only the block's input where there is ONE residual stream to be pre of.
            # Refused rather than mapped there: the measurement behind this table was taken on gpt2, and
            # a measurement does not generalise to an architecture it was not taken on.
            from interp_engine.mappers import has_hyper_connections

            if has_hyper_connections(model):
                raise UnmappableHookError(
                    f"{hook_name!r} names a block input, and this model carries hyper-connections, so it "
                    "has several residual streams rather than one. This adapter maps that name to "
                    "'resid_pre' only on single-stream models, where the two were measured identical. "
                    "Ask for the specific stream you mean."
                )
            return Address(cls._MEASURED_ALIASES[base], layer)
        if base in cls._MEASURED_GLOBAL_ALIASES and layer is None:
            return Address(cls._MEASURED_GLOBAL_ALIASES[base])
        try:
            return tlens_hook_to_point(hook_name, model)
        except UnmappedHook as exc:
            # The engine's own text, given this adapter's voice. A caller who reaches here is holding an
            # interpretune hook name and needs to know what THIS component can do with it; the engine's
            # message lists ITS point vocabulary, which is not the vocabulary the caller wrote in.
            #
            # `EngineHasNoPointError`, NOT a bare `UnmappableHookError`: see that class. Capture paths
            # treat them alike; the circuit-tracer guard must not.
            raise EngineHasNoPointError(
                f"{hook_name!r} is a hook name interpretune recognises and this backend cannot capture "
                f"at: interp-engine's mapper has no point for it ({exc}). Ask for a name this backend "
                "maps, or capture through a backend that covers it. Not resolved to a nearby point: the "
                "neighbouring tensor is a different tensor."
            ) from exc

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
    def capture(
        engine_model: Any, tokens: Any, points: list[Any], detach: bool = True, attention_mask: Any = None
    ) -> Any:
        """One forward pass, capturing the requested points.

        The sync free function is used deliberately: interpretune's ``*_step`` and analysis-op call
        sites are synchronous. It REFUSES inside a running event loop by design, which is why the
        notebook path uses the async surface instead of reaching for nest_asyncio.
        """
        from interp_engine import run_with_cache

        return run_with_cache(engine_model, tokens, points, detach=detach, attention_mask=attention_mask)

    @staticmethod
    def steer(engine_model: Any, tokens: Any, specs: list[Any], attention_mask: Any = None) -> Any:
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

        That nesting composes correctly here while every spec in a mixed payload commutes, and
        addition is what commutes: several ``add`` specs in any order, and several ``reject`` specs
        from one orthonormalised operand (orthogonal projectors commute), are order-free. An ``add``
        and a ``reject`` split across scopes at one point do NOT commute -- rejecting after adding
        differs from adding after rejecting, while nested per-scope contexts fire in an order that
        is unmeasured relative to declaration order -- so that mixture is refused by name in
        ``fwd_w_intervention`` (see ``_refuse_mixed_modes``). Within one scope, declaration order
        flows through untouched. A future mode that overwrites rather than accumulates (``replace``
        is the obvious one) extends the same refusal. `_MODE_TO_STEER_METHOD` declares the mapped
        modes, and `intervention_support` is asserted to agree with it.
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
            return _logits_of(run_with_cache(engine_model, tokens, [], attention_mask=attention_mask))

    @staticmethod
    def steer_spec(
        vector: Any,
        layer: int,
        coeff: float = 1.0,
        method: str = "additive",
        point: str = "resid_post",
        minimum: float | None = None,
        maximum: float | None = None,
        explicit: bool = False,
    ) -> Any:
        """Build one engine ``SteerSpec``.

        ``minimum``/``maximum`` are the band for ``method="projection_cap"`` and are ignored by the
        others, matching ``SteerSpec.min``/``.max``. They exist because without them this constructor
        cannot express a working clamp at all: the delta is
        ``(clamp(p, min, max) - p) * unit``, so a spec carrying neither bound collapses to zero for
        every input rather than for the inputs that happen not to need clamping.

        ``explicit`` records that the caller provided every parameter deliberately, as opposed to
        leaving a default unset. The identity refusal below can only fire on an omission: a mapped
        ``reject`` at ``scale_factor=0.0`` builds ``orthogonal`` at ``coeff=1.0`` with every
        parameter set, so refusing it would mistake an explicit identity (which the conformance
        contract requires) for a missing one. The intervention path always passes ``True`` -- a
        missing ``scale_factor`` already raises in ``_spec_field`` before this is reached, so an
        omission cannot arrive that way. Direct callers leave the default, keeping the refusal.
        """
        from interp_engine import SteerSpec

        # Refused because the request cannot be told apart from a parameter nobody set, and the result
        # would be a forward that completes and returns plausible logits for an intervention nobody
        # performed. Two other cases are deliberately NOT refused, and the boundaries matter:
        #   * a spec that does nothing to THIS input but could to another -- a min-only clamp on data
        #     already above the floor -- is a clamp working, and refusing it would be the mirror defect;
        #   * a no-op the caller asked for EXPLICITLY, at a value that is not the default, is a request
        #     rather than an omission. `additive` at coeff=0.0 is the instance, and it is required to
        #     behave as an identity by the conformance suite.
        # NOT every parameterization whose delta is zero. `additive` at coeff=0.0 is also identically
        # zero, and is deliberately ALLOWED: 0.0 is never this signature's default, so it is always an
        # explicit request, and the conformance contract requires a zero-scale intervention to be an
        # IDENTITY rather than an error (`test_zero_intervention_is_identity`). Refusing it broke that
        # suite, which is how this distinction was found.
        #
        # What is refused is the case where the no-op is indistinguishable from an OMISSION: a caller
        # who never set the parameter and one who set it to the value that does nothing produce the
        # same spec, so the intent cannot be recovered. That is `orthogonal` at the default coeff, and
        # `projection_cap` with no band at all.
        identity = None
        if method == "orthogonal" and coeff == 1.0 and not explicit:
            identity = "coeff=1.0, so the delta is (coeff-1)*projection, which is zero"
        elif method == "projection_cap" and minimum is None and maximum is None:
            identity = "neither a minimum nor a maximum, so the delta is (clamp(p) - p) with no clamp"
        if identity is not None:
            raise UnmappableHookError(
                f"method={method!r} was requested with {identity}. That is not an intervention that "
                "happens to do nothing to this input, which would be fine -- it can do nothing to any "
                "input, so it would complete normally and return plausible logits for an intervention "
                "nobody performed. Refused rather than built. Pass a coefficient that steers, or for "
                "projection_cap at least one of minimum/maximum."
            )
        return SteerSpec(vector=vector, layer=layer, coeff=coeff, method=method, point=point, min=minimum, max=maximum)

    @staticmethod
    def forward(engine_model: Any, tokens: Any, attention_mask: Any = None) -> Any:
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

        return _logits_of(run_with_cache(engine_model, tokens, [], attention_mask=attention_mask))

    async def capture_async(
        engine_model: Any, tokens: Any, points: list[Any], detach: bool = True, attention_mask: Any = None
    ) -> Any:
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
        _refuse_padding_without_mask_support(tokens, attention_mask)
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


def _layer_count(engine_model: Any) -> int:
    """Layer count for the wrapped model, read from the HuggingFace config.

    `num_hidden_layers` is the canonical field and `n_layer` the GPT-2-era alias; both are checked
    because the engine wraps the HF model IN PLACE and does not normalize its config. Zero rather than a
    guess when neither is present: an empty inventory refuses every callable filter visibly, where a
    guessed depth would build a SHORT inventory and quietly drop the model's upper layers -- a filter
    over layers 0..11 on a 32-layer model returns a cache that looks entirely ordinary.
    """
    cfg = getattr(getattr(engine_model, "hf_model", None), "config", None) or getattr(engine_model, "config", None)
    for depth_field in ("num_hidden_layers", "n_layer"):
        value = getattr(cfg, depth_field, None)
        if isinstance(value, int):
            return value
    return 0


def _hook_vocabulary(engine_model: Any) -> list[str]:
    """Every hook name a caller could legitimately NAME for this model, in interpretune's vocabulary.

    Built from `HookNameResolver(architecture).supported_hooks`, expanded over layers for the components
    that HAVE layers. Interpretune's vocabulary, not the engine's, and that choice is the whole point: a
    `names_filter` is a predicate over the names interpretune uses.

    THE OBVIOUS ALTERNATIVE IS WRONG, and quietly so. Enumerating FORWARD from the engine instead --
    `EagerModel.points()` spelled through `point_to_tlens_hook` -- yields 252 names on gpt2 and looks
    complete. It is not: the engine's reverse table holds one CANONICAL spelling per point while its
    forward table accepts aliases, so `blocks.5.hook_resid_post` is in that inventory and
    `blocks.5.hook_out` is not, though both name the same tensor and interpretune uses the second. A
    predicate over the forward-built inventory matched NONE of the conformance suite's capture points.

    THE LAYER SPLIT COMES FROM `component_map_for`, AND GUESSING IT WAS A REAL BUG. This first tried
    every base name both bare and layer-qualified and kept whichever forms `resolver.resolve` accepted,
    to avoid reading a private attribute. `resolve` is more permissive than that assumed: it accepts
    `blocks.5.unembed.hook_out` and a bare `attn.hook_z`, neither of which names anything. While
    interpretune's vocabulary was 13 base names that produced a little junk and no visible harm. When it
    grew to 34, the junk became 252 of 433 entries and FOUR OF FIVE ordinary filters -- `endswith
    hook_out`, `endswith hook_in`, everything at one layer, anything mlp -- began refusing outright,
    because each selected some nonsense name this seam then could not map.

    `ComponentMap.block_components()` and `.global_components()` answer it properly and publicly, so a
    block component is expanded over layers and a global one is offered bare, and neither is offered as
    the other. Note the empty-string component: `hook_resid_post` and friends belong to the block itself,
    which `block_components()` reports as `''`.
    """
    from interpretune.analysis.backends.hook_mapping import HookNameResolver
    from interpretune.analysis.points import UnknownPointError, component_map_for, parse

    architecture = type(getattr(engine_model, "hf_model", engine_model)).__name__
    try:
        resolver = HookNameResolver(architecture)
        component_map = component_map_for(architecture)
    except (KeyError, ValueError) as exc:
        raise UnmappableHookError(
            f"a callable names_filter needs this backend's hook inventory, and interpretune has no hook "
            f"mapping for {architecture!r} ({exc}). Pass an explicit list of hook names instead."
        ) from exc

    per_layer = set(component_map.block_components())
    global_only = set(component_map.global_components())
    layers = range(_layer_count(engine_model))

    names: list[str] = []
    for base in resolver.supported_hooks:
        try:
            component = parse(base).component
        except UnknownPointError:
            # The resolver names a hook the point vocabulary cannot parse. Skipped rather than guessed
            # at: a name whose component is unknown cannot be placed in or out of a block, and either
            # placement invents a hook.
            continue
        if component in per_layer:
            names.extend(f"blocks.{layer}.{base}" for layer in layers)
        elif component in global_only:
            names.append(base)
    return names


def _partial_selection_message(selected: list[str], mappable: list[str], unmappable: list[str]) -> str:
    """Explain a refused predicate WITHOUT reciting it.

    A broad filter over interpretune's vocabulary can select a hundred names here, so listing every
    unmappable one produces a message nobody finishes reading -- and an unreadable refusal is a refusal
    that gets worked around rather than understood. Summarised by BASE NAME instead, which is what a
    predicate is actually written against: `blocks.0.ln1.hook_out` through `blocks.11.ln1.hook_out` are
    one fact about this backend, not twelve.

    The counts are given because they carry the decision. A caller told "8 of 100 selected names are
    unmappable" can weigh narrowing the predicate against passing an explicit list; a caller told only
    "refused" cannot.
    """

    def base(name: str) -> str:
        parts = name.split(".")
        return ".".join(parts[2:]) if len(parts) >= 3 and parts[0] == "blocks" and parts[1].isdigit() else name

    bases = sorted({base(name) for name in unmappable})
    shown = ", ".join(bases[:8]) + (f", and {len(bases) - 8} more" if len(bases) > 8 else "")
    return (
        f"the callable names_filter selected {len(selected)} hook names, {len(unmappable)} of which this "
        f"backend cannot capture at. interp-engine has no point for these: {shown}. "
        f"Refused rather than captured PARTIALLY: a cache holding the other {len(mappable)} is "
        "indistinguishable from a complete one at the call site, and a caller iterating it would compute "
        "over a filter that silently meant less than it said. Narrow the predicate to exclude those "
        "names, or pass an explicit list of the hooks you want."
    )


def _names_filter_to_list(names_filter: Any, engine_model: Any) -> list[str]:
    """Normalize interpretune's ``NamesFilter`` to a concrete list of hook names.

    A CALLABLE filter is a predicate over TransformerLens names, so it is applied to this backend's
    mappable inventory rather than refused. This reversed an earlier refusal, and the reasoning matters
    more than the change: the refusal said the engine's inventory could not be enumerated in
    TransformerLens vocabulary, and by the time it was checked that was no longer true. What makes
    applying the predicate honest is not that the inventory is complete in absolute terms -- it is not,
    6 point names have no TransformerLens spelling -- but that those points cannot be NAMED by a caller
    in either direction, so no predicate can be written that ought to have matched them. See
    `_mappable_hook_names`.

    Interpretune's core turns an explicit list into a callable at setup, so the suite's explicit list
    arrives here as a predicate; refusing callables therefore refused ordinary use, not an exotic one.
    """
    if names_filter is None:
        raise UnmappableHookError("a names_filter is required: this backend cannot capture 'everything'")
    if isinstance(names_filter, str):
        return [names_filter]
    if callable(names_filter):
        vocabulary = _hook_vocabulary(engine_model)
        selected = [name for name in vocabulary if names_filter(name)]
        if not selected:
            raise UnmappableHookError(
                f"the callable names_filter matched none of the {len(vocabulary)} hook names interpretune "
                "recognises for this model. Nothing was captured, and an empty cache is refused rather "
                "than returned, because a filter that silently matches nothing is indistinguishable from "
                "one that ran and found nothing to capture."
            )
        mappable, unmappable = [], []
        for name in selected:
            try:
                _Seam.point_for(name, engine_model)
            except UnmappableHookError:
                unmappable.append(name)
            else:
                mappable.append(name)
        if unmappable:
            raise UnmappableHookError(_partial_selection_message(selected, mappable, unmappable))
        return selected
    return list(names_filter)


#: interpretune ``InterventionSpec.mode`` -> interp-engine ``SteerSpec.method``, for the modes whose
#: semantics were MEASURED to match. Each omission has its own reason, and the reasons are not
#: interchangeable.
#:
#: ``add`` maps because the operations are the same one: interpretune's ``add`` is
#: ``h + tensor * scale`` and the engine's ``additive`` delta is ``coeff * vector``.
#:
#: ``reject`` maps to ``orthogonal`` with a coefficient rule, not just a method name: the engine's
#: ``coeff`` is the strength KEPT along the vector while interpretune's ``scale_factor`` is the
#: fraction REMOVED, so ``coeff = 1 - scale_factor`` (full removal at the default scale of 1.0,
#: graded ablation bands below it, and the identity at an explicit 0.0). The mapping is exact at
#: every ``k`` because the adapter orthonormalises the span first (see ``_reject_basis_rows``):
#: ``VV^+`` is the orthogonal projector onto the span, so an orthonormal basis ``Q`` of the same
#: span gives ``QQ^T == VV^+`` identically, and each basis vector gets one rank-one engine spec.
#:
#: ``clamp`` maps to ``projection_cap`` with the band scaled by the row's norm: interpretune clips
#: the pseudoinverse coordinates ``c = V^+ h`` into ``[lo, hi]`` while the engine clips the projection
#: onto the UNIT direction ``p = h . v_hat``, and for one row ``p = c ||v||``, so an engine band of
#: ``[lo*||v||, hi*||v||]`` reproduces the interpretune band exactly (each side only when supplied).
#: Exact at ``k=1`` and for mutually orthogonal rows (one cap per row, either order -- orthogonal
#: caps commute). Coupled rows are refused by name (see ``_clamp_cap_rows``): the coordinates are
#: pseudoinverse coefficients depending on every row, so neither per-row caps nor an orthonormalised
#: basis reproduces the band. Orthonormalising for clamp is therefore refused along with
#: approximating, unlike ``reject`` where it is the mapping.
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
_MODE_TO_STEER_METHOD = {"add": "additive", "reject": "orthogonal", "clamp": "projection_cap"}

#: Why per-row engine specs reproduce a span operation only for orthonormal rows -- and why that
#: constrains ``clamp`` but no longer ``reject``. The engine's projector is RANK ONE
#: (``P = v_hat v_hat^T``, one vector per ``SteerSpec``). Composing one spec per row reproduces the
#: span operation ONLY when the rows are orthonormal (measured: exact for orthonormal ``V``, wrong
#: at row cosine -0.240). ``reject`` orthonormalises first (see ``_reject_basis_rows``), so the
#: restriction does not touch it: rejection depends on the span alone. ``clamp`` targets are stated
#: in pole coordinates, which an orthonormalisation does not preserve, so coupled rows are refused
#: by name (see ``_clamp_cap_rows``) rather than orthonormalised or approximated.
_CLAMP_SPAN_CAVEAT = "rank-1 per spec; span operations need orthonormal rows"

#: Explicit refusal reasons for the modes ``_MODE_TO_STEER_METHOD`` does not map, keyed by
#: interpretune mode value. Each reason names its own mechanism (see the mapping table's notes):
#: a shared justification previously licensed the ``project -> orthogonal`` substitution this
#: file refuses. A mode in NEITHER the table NOR this dict is undecided, and the seam test fails
#: naming it until someone rules. That red is the deliverable, not a side effect.
_MODE_REFUSAL_REASONS = {
    "replace": (
        "an overwrite replaces the whole activation, while every engine steering method moves "
        "only the component along one direction and leaves the complement untouched"
    ),
    "patch": (
        "a lens-coordinate swap whose targets are measured during the forward, while "
        "``projection_cap`` takes static min/max fixed when the spec is built"
    ),
    "project": (
        "the complementary projection of the engine's ``orthogonal``: it keeps the parallel "
        "component where ``orthogonal`` keeps the orthogonal one, so no coefficient converts"
    ),
}


def _uncovered_modes(documented: set[str], table: dict[str, str], reasons: dict[str, str]) -> set[str]:
    """Modes with neither a mapping nor an explicit refusal reason.

    Factored so the coverage guard's ability to fire is itself testable: the live seam test
    derives ``documented`` from ``InterventionMode`` and asserts the result is empty, while the
    positive controls call this helper with synthetic sets in both directions.
    """
    return set(documented) - set(table) - set(reasons)


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


def _coeff_for(mode_value: str, scale_factor: float) -> float:
    """The engine coefficient for a mapped mode and intervention scale.

    ``add`` passes the scale through (the operations are the same one). ``reject`` complements it:
    the engine keeps ``coeff`` along the vector while interpretune removes ``scale_factor``, so
    ``coeff = 1 - scale_factor``. A non-mapped mode never reaches here (``_steer_method_for``
    refuses it first); the table, not this function, decides what maps.
    """
    if mode_value == "reject":
        return 1.0 - scale_factor
    return scale_factor


def _reject_basis_rows(tensor: Any, hook_name: str) -> list[Any]:
    """Orthonormal basis rows for a ``reject`` operand, one engine vector per row.

    ``reject`` is defined over a ``(k, d_model)`` span for any ``k >= 1`` while the engine takes
    one vector per spec, so the adapter orthonormalises and fans out. The basis comes from the
    row space via SVD, keeping the numerical rank rather than ``k``: dependent rows shrink the
    span exactly as the pseudoinverse does, and an all-zero operand rejects nothing (also what
    the pseudoinverse computes). Signs and row order are immaterial -- every per-row projector
    is sign-invariant and orthonormal projectors commute -- so no canonicalisation is needed.
    SVD rather than unpivoted QR deliberately: without pivoting, QR's first ``r`` columns need
    not span a rank-deficient row space, while the SVD basis does by construction.
    """
    import torch

    rows = torch.as_tensor(tensor, dtype=torch.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    elif rows.ndim != 2:
        raise UnmappableHookError(
            f"intervention tensor for {hook_name!r} has shape {tuple(getattr(tensor, 'shape', ()))}: "
            "a reject operand is one direction or a (k, d_model) span. Anything else is not a span "
            "operation this backend can decompose into per-vector specs."
        )
    if rows.shape[0] == 0:
        return []
    if not bool(torch.isfinite(rows).all()):
        raise UnmappableHookError(
            f"intervention tensor for {hook_name!r} carries NaN or inf: the reference computes NaN "
            "through the pseudoinverse, so an identity here would be a silent divergence rather than "
            "a span operation. Clean the operand instead."
        )
    _, singular, vh = torch.linalg.svd(rows, full_matrices=False)
    tolerance = max(rows.shape) * torch.finfo(torch.float32).eps * singular[0].item()
    rank = int((singular > tolerance).sum().item())
    basis = vh[:rank]
    input_dtype = getattr(tensor, "dtype", torch.float32)
    return [row.to(dtype=input_dtype) for row in basis]


def _clamp_cap_rows(tensor: Any, hook_name: str) -> list[tuple[Any, float]]:
    """One ``(row, row norm)`` per clamp cap: exact only for orthogonal rows.

    ``clamp`` bounds pseudoinverse coordinates, which depend on every row at once, so unlike
    ``reject`` the operand must already be orthogonal -- orthonormalising would change what the
    band bounds. Pairwise row cosines above ``_CLAMP_ORTHOGONALITY_TOLERANCE`` are refused by
    name rather than approximated, as is a zero row (no unit direction to cap along).
    """
    import torch

    rows = torch.as_tensor(tensor, dtype=torch.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    elif rows.ndim != 2:
        raise UnmappableHookError(
            f"intervention tensor for {hook_name!r} has shape {tuple(getattr(tensor, 'shape', ()))}: "
            "a clamp operand is one direction or a (k, d_model) span of mutually orthogonal rows. "
            "Anything else is not a band operation this backend can decompose into per-vector caps."
        )
    if rows.shape[0] == 0:
        return []
    if not bool(torch.isfinite(rows).all()):
        raise UnmappableHookError(
            f"intervention tensor for {hook_name!r} carries NaN or inf: clean the operand instead."
        )
    norms = rows.norm(dim=1)
    if not bool((norms > 0).all()):
        raise UnmappableHookError(
            f"intervention tensor for {hook_name!r} has a zero row: a clamp caps the projection "
            "onto the row's UNIT direction, and a zero row has none. Remove it instead."
        )
    unit = rows / norms.unsqueeze(1)
    cosines = unit @ unit.T
    off = cosines - torch.eye(rows.shape[0])
    worst = off.abs().max().item()
    if worst > _CLAMP_ORTHOGONALITY_TOLERANCE:
        raise UnmappableHookError(
            f"intervention tensor for {hook_name!r} has coupled rows (worst pairwise cosine "
            f"{worst:.3g} above {_CLAMP_ORTHOGONALITY_TOLERANCE:g}): clamp coordinates are "
            "pseudoinverse coefficients that depend on every row, and a rank-one cap bounds a "
            "different quantity, so neither per-row caps nor an orthonormalised basis reproduces "
            "the band. Pass mutually orthogonal rows instead."
        )
    input_dtype = getattr(tensor, "dtype", torch.float32)
    return [(row.to(dtype=input_dtype), float(norm)) for row, norm in zip(rows, norms.tolist())]


#: The largest pairwise row cosine a clamp operand may carry: orthogonal caps commute, so order is
#: free, while anything above this is a coupled span the rank-one caps cannot reproduce.
_CLAMP_ORTHOGONALITY_TOLERANCE = 1e-6


def _refuse_mixed_modes(targets: list[tuple[int, str, Any]]) -> None:
    """Refuse non-commuting modes mixed ACROSS scopes at one point, naming the unmeasured order.

    Interpretune's contract is sequential application in declaration order, and within one scope
    this backend honours it exactly: specs are grouped by scope preserving declaration order, and
    the engine applies each group's specs in list order. Across scopes on the same point the
    adapter enters one nested steering context per scope, and the order in which nested contexts'
    hooks fire relative to declaration order is unmeasured -- so an ``add`` and a ``reject`` split
    across scopes has no defined meaning yet and is refused. Same scope (even mixed) flows
    through in declaration order; same-mode multiplicity anywhere is fine.
    """
    scopes_by_point: dict[tuple[int, str], dict[str, set[Any]]] = {}
    for layer, point, spec in targets:
        mode = _spec_field(spec, "mode", "replace")
        mode_value = getattr(mode, "value", mode)
        scope = _spec_field(spec, "position_scope", PositionScope.LAST_TOKEN)
        scope_value = getattr(scope, "value", scope)
        scopes_by_point.setdefault((layer, point), {}).setdefault(mode_value, set()).add(scope_value)
    for (layer, point), by_mode in scopes_by_point.items():
        if "add" in by_mode and "reject" in by_mode:
            scopes = by_mode["add"] | by_mode["reject"]
            if len(scopes) > 1:
                raise UnmappableHookError(
                    f"interventions at layer {layer}, point {point!r} mix 'add' and 'reject' "
                    "across position scopes: rejecting after adding differs from adding after "
                    "rejecting, and this backend enters one nested steering context per scope, "
                    "whose firing order relative to declaration order is unmeasured. Keep "
                    "non-commuting modes in one scope, where declaration order is honoured "
                    "exactly, or split them across calls."
                )


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


def _attention_mask_from_batch(batch: dict[str, Any]) -> Any:
    """The batch's attention mask, or ``None`` when it carries none.

    THIS IS NOT AN OPTIONAL EXTRA, and the measurement that says so belongs in front of the next reader.
    ``run_with_cache`` takes ``attention_mask`` and defaults it to ``None``, which runs the forward
    ATTENDING TO PAD POSITIONS. On gpt2 with a two-row left-padded batch, the padded row's last-position
    logits -- the ones a classification consumer reads -- differ from HuggingFace's by 54.9, and captured
    ``resid_post`` differs by 44.7. With the mask threaded, both agree to 0.0.

    Nothing raises either way. Shapes agree either way. Only the numbers are wrong, and only for rows
    shorter than the longest in their batch, which is why a suite running at batch size 1 cannot see it:
    an unpadded batch has no mask to ignore. This adapter shipped that defect behind a batch refusal that
    was itself a misreading, so the refusal was hiding a real bug rather than standing in for one.

    Read directly rather than through `get_batch_input`, which resolves INPUT keys across backends and
    has no equivalent for the mask.
    """
    if not isinstance(batch, dict):
        return getattr(batch, "attention_mask", None)
    return batch.get("attention_mask")


def _refuse_padding_without_mask_support(tokens: Any, attention_mask: Any) -> None:
    """Refuse a PADDED batch on a path that cannot carry the mask, rather than running it unmasked.

    ``EagerModel.capture`` -- the async surface the notebook path uses -- has no ``attention_mask``
    parameter, while the sync ``run_with_cache`` does. So the async path runs one prompt, or an evenly
    sized batch, correctly, and cannot run a padded one correctly.

    The refusal is scoped to what is actually broken: an unpadded batch goes through, because there is no
    mask to lose. It replaced a blanket single-prompt refusal that refused far more than the real
    limitation. That one came from misreading the engine's protocol-arm message, which said "takes one
    prompt at a time on the GPT2LMHeadModel backend" and was naming the WRONG MODEL rather than reporting
    a batching limit -- the backend was being handed the raw HuggingFace model instead of the engine one.
    """
    if attention_mask is None:
        return
    try:
        padded = bool((attention_mask == 0).any())
    except (AttributeError, TypeError):
        return
    if padded:
        raise UnmappableHookError(
            "the async capture path cannot carry an attention mask, and this batch is padded. Running it "
            "here would attend to pad positions and return numbers that look right and are not (measured "
            "at 54.9 on a gpt2 padded row's last-position logits). Use the sync capture path, which "
            "threads the mask, or pass an unpadded batch."
        )


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


_SPEC_UNSET = object()


def _spec_field(spec: Any, name: str, default: Any = _SPEC_UNSET) -> Any:
    """Read one field off an intervention spec, whatever shape it arrived in.

    Specs reach a backend in TWO shapes and both are normal. A caller building them in process passes
    interpretune's `InterventionSpec`, a NamedTuple. A spec that has been through the analysis store --
    which is every spec the conformance runner produces, and every spec a saved analysis replays --
    comes back as a plain DICT, because that is what the columnar store round-trips to. The bundled
    circuit-tracer backend reads both for exactly this reason: attributes on its write path, subscripts
    on its read path.

    Reading only attributes fails on the dict shape with `'dict' object has no attribute
    'intervention_tensor'`, which names neither this backend nor the shape mismatch, and sends the
    reader looking for a missing field rather than a missing accessor. Six conformance cases failed
    that way.

    A missing field with no default RAISES rather than returning None: `scale_factor` defaulting to None
    reaches the engine as a coefficient and multiplies to a TypeError three frames away, and
    `intervention_tensor` defaulting to None steers with nothing while completing normally.
    """
    if hasattr(spec, "keys"):
        if name in spec:
            return spec[name]
    else:
        value = getattr(spec, name, _SPEC_UNSET)
        if value is not _SPEC_UNSET:
            return value
    if default is _SPEC_UNSET:
        raise UnmappableHookError(
            f"an intervention spec is missing {name!r} (spec is a {type(spec).__name__}). Refused rather "
            "than defaulted: a missing coefficient or tensor completes the forward and returns a result "
            "that is not the intervention the caller asked for."
        )
    return default


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
        # `keys` catches the dict shape for the same reason `_fields` catches the NamedTuple one: both
        # are SINGLE specs that would otherwise be iterated into their own parts.
        single = hasattr(specs, "_fields") or hasattr(specs, "keys") or not isinstance(specs, (list, tuple))
        for spec in [specs] if single else specs:
            targets.append((layer, getattr(address, "name", "resid_post"), spec))
    return targets


class InterpEngineAttributeMixin:
    """Config access and the wrapped-engine handle."""

    it_cfg: Any

    @property
    def interp_engine_cfg(self) -> InterpEngineConfig:
        """This session's engine config, refusing rather than defaulting when it did not arrive.

        The default this replaced was not a convenience, it was a mask. This property is read from
        `post_auto_model_init`, so by the time it runs a composition declaring `interp_engine` is being
        built and a config was expected; absence means one did not reach the module, not that the caller
        declined to supply one.

        And absence has a known cause worth naming in the message rather than leaving to be rediscovered.
        Auto-composition finds an adapter's config class by importing `interpretune.adapters.<name>.config`
        (`AUTOCOMP_SEARCH_TEMPLATES`), which cannot resolve for a hub-delivered adapter: its module is
        executed from a cache under a revision-scoped synthetic name, so no import path follows from the
        adapter name. `interpretune.config.shared` says discovery for those goes through a registry, and no
        such lookup exists. So a settings object passed for this adapter through a composition's
        constructor is dropped, and defaulting here turned that drop into plausible behaviour on an engine
        the caller never asked for.
        """
        cfg = getattr(self.it_cfg, "interp_engine_cfg", None)
        if cfg is None:
            raise ValueError(
                "no `interp_engine_cfg` reached this module, and this adapter will not run on a default "
                "in its place: the settings it would silently substitute (backend, attention "
                "implementation, dtype) change what the engine computes, and nothing downstream could "
                "tell the difference. This module only exists inside a composition that declares "
                "`interp_engine`, so a config was expected here.\n\n"
                "The usual cause is that the composition's config class cannot carry it. Auto-composition "
                "locates an adapter's config by import path, which does not resolve for a hub-delivered "
                "adapter, so a value passed through the constructor is dropped rather than carried. Set "
                "it on the module config the session is built from and confirm it survives to "
                "`module.it_cfg.interp_engine_cfg`."
            )
        return cfg

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
        self._engine_model = _Seam.wrap_model(
            self.model,  # type: ignore[attr-defined]
            getattr(self, "tokenizer", None),
            self.interp_engine_cfg,
            model_id,
        )
        # INTERPRETUNE'S HELPERS READ `module.model.tokenizer`, and this adapter is the first
        # composition where `module.model` is the raw HuggingFace model rather than a wrapper.
        # `get_module_dims` does `module.model.tokenizer.vocab_size`, which every other adapter satisfies
        # because its wrapper carries one: TransformerLens' `HookedTransformer` and nnsight's
        # `LanguageModel` both do.
        #
        # The engine model CANNOT take that slot. `EagerModel` is not an `nn.Module` -- its MRO is
        # `[EagerModel, object]` -- so assigning it raises `cannot assign as child module`, unlike
        # circuit-tracer's replacement models which are typed `nn.Module` precisely so they can. So the
        # convention is met where it can be met: the model the module already owns carries the tokenizer
        # it was loaded with.
        #
        # SOURCED FROM THE ENGINE MODEL, not from `self.tokenizer`, and that is measured rather than
        # preferred. In a real `ITSession` the module carries NO tokenizer at this point: it lives on the
        # datamodule, and `self.tokenizer` is None when this hook runs. `EagerModel` resolves its own from
        # the model id, so after the wrap there is exactly one tokenizer in scope that is guaranteed to
        # exist and to be the one the engine will actually use.
        #
        # Attaching rather than replacing, and only when absent, so a composition whose model already
        # satisfies the convention is left alone.
        # ATTACHED AFTER THE WRAP, and the order is load-bearing now rather than incidental: the backend
        # is constructed WITH the engine model, so it cannot be built before one exists. Constructing it
        # first bound `engine_model=None` and every backend call fell through to the argument it was
        # handed, which is the raw HuggingFace model -- reintroducing the exact defect this change fixes.
        if get_model_backend(self) is None:
            self._model_backend = InterpEngineModelBackend(engine_model=self._engine_model)

        tokenizer = getattr(self._engine_model, "tokenizer", None)
        if tokenizer is not None and getattr(self.model, "tokenizer", None) is None:  # type: ignore[attr-defined]
            self.model.tokenizer = tokenizer  # type: ignore[attr-defined]
        return result

    def engine_capture(
        self, tokens: Any, hook_names: list[str], detach: bool = True, attention_mask: Any = None
    ) -> Any:
        """Capture at TransformerLens hook names, translated through the seam.

        ``attention_mask`` is optional here and NOT optional in effect: omitting it on a padded batch
        runs the forward attending to pad positions, which is wrong by 44.7 on gpt2's ``resid_post``
        without raising anything. Callers holding an interpretune batch should pass
        ``batch["attention_mask"]``; a single prompt has nothing to pass.
        """
        points = [_Seam.point_for(name, self.engine_model) for name in hook_names]
        return _Seam.capture(self.engine_model, tokens, points, detach=detach, attention_mask=attention_mask)


class BaseInterpEngineModule(BaseITModule):
    """Base module for interp-engine compositions."""


class InterpEngineModule(InterpEngineCaptureMixin, CoreAdapter, CoreHelperAttributes, BaseInterpEngineModule):
    """The ``(core, interp_engine)`` module composition.

    `CoreAdapter` IS PART OF A CORE COMPOSITION, and leaving it out is why this module could not complete
    a runner step. It contributes `batch_to_device`, which the analysis runner calls on every batch, and
    without it the composed module raised `AttributeError` from inside dataset generation -- surfacing as
    a `DatasetGenerationError` that named neither the method nor this component.

    TransformerLens does not inherit it because `TransformerLensAdapter` defines its own
    `batch_to_device` for a converted model. interp-engine runs the HuggingFace model in place, so core's
    implementation is the correct one rather than merely the available one, which is the same reason the
    datamodule slot is plain `ITDataModule`.
    """


def _datamodule_classes(combination: tuple[Any, ...]) -> tuple[Any, ...]:
    """The datamodule composition for one adapter combination.

    Lightning contributes its own datamodule base and the others do not, matching how the bundled
    adapters register: `(ITDataModule,)` for a core composition, `(ITDataModule, LightningDataModule)`
    for a Lightning one.

    The Lightning import is LOCAL and deliberate. This entrypoint must import in an environment that
    never installed Lightning -- the core compositions are the ones that have to work there -- so
    reaching for it at module scope would make an optional dependency mandatory for compositions that
    do not use it. Same reasoning as `_circuit_tracer_pairing`, and the same failure if ignored: an
    import error at hub load rather than at registration.
    """
    from interpretune.base import ITDataModule

    if Adapter.lightning not in combination:
        return (ITDataModule,)

    from interpretune.adapters.lightning import LightningDataModule

    return (ITDataModule, LightningDataModule)


#: Cache for the composed circuit-tracer pairing class. Built once, lazily, because constructing it
#: imports circuit-tracer, and this module MUST import without it: the two core compositions are the
#: ones that have to work in an environment that never installed the optional package.
_CT_PAIRING: Any = None


def _hf_model_device(model: Any) -> Any:
    """The device a loaded HF model's parameters live on (``None`` when it has none to read)."""
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None


def _hf_model_dtype(model: Any) -> Any:
    """The dtype a loaded HF model's parameters carry (``None`` when it has none to read)."""
    try:
        return next(model.parameters()).dtype
    except (AttributeError, StopIteration):
        return None


def _expose_hf_surface(replacement_model: Any) -> None:
    """Give circuit-tracer's interp-engine replacement model the HF surface interpretune reads off
    ``module.model``.

    Interpretune's core treats ``module.model`` as an HF model: hyperparameter capture reads ``.config``,
    the dims mixin reads ``.base_model`` and ``.config.vocab_size``, generation reads ``.generation_config``.
    The bundled replacement models carry those (nnsight's proxies its ``LanguageModel``); the interp-engine
    one keeps them on ``hf_model`` and exposes none. Attached as PLAIN attributes via ``object.__setattr__``,
    deliberately: ``nn.Module.__setattr__`` would register ``base_model`` as a submodule and double-count
    every parameter in ``parameters()`` and the state dict. Instance ``__dict__`` wins over ``nn.Module``'s
    ``__getattr__``, so reads resolve without registration. The clean home for this is circuit-tracer
    itself (properties on ``InterpEngineReplacementModel``); until then this pairing owns the shim.
    """
    hf_model = getattr(replacement_model, "hf_model", None)
    if hf_model is None:
        return
    for name in ("config", "generation_config", "base_model"):
        if not hasattr(replacement_model, name) and hasattr(hf_model, name):
            object.__setattr__(replacement_model, name, getattr(hf_model, name))


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
        from interpretune.adapters.circuit_tracer.registry import CT_BACKEND_REGISTRY
    except Exception:
        return None
    # The third-party registration interpretune's rails expect: `CircuitTracerConfig` refuses a backend
    # name that is not in this registry, and `_load_replacement_model` validates the loaded model against
    # the class name recorded here. circuit-tracer ships the backend (`InterpEngineReplacementModel`);
    # this component owns the pairing, so it owns the registration. Idempotent by construction.
    CT_BACKEND_REGISTRY.setdefault(ADAPTER_NAME, "InterpEngineReplacementModel")

    class CircuitTracerInterpEngineModuleMixin(InterpEngineCaptureMixin):
        # `InterpEngineCaptureMixin`, not the attribute-only mixin: the capture mixin owns the
        # `post_auto_model_init` that wraps the engine and attaches this adapter's model backend, and
        # the two-way module inherits it. This pairing inherited only the attribute mixin, so the chain
        # from its own hook skipped the engine entirely: no engine, no model backend, and every steering
        # op would have failed on this composition before any attribution ran (measured: the module had
        # no `_engine_model` after init). Engine first, replacement model second, both over one HF model.
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
            self._load_interp_engine_replacement_model()
            self._guard_feature_hook_resolution()
            return result

        def _load_interp_engine_replacement_model(self) -> None:
            """Load circuit-tracer's replacement model over the engine, so attribution can run.

            The bundled pairings do this from their TL and nnsight mixins, handing circuit-tracer the HF
            model the module already loaded through ``pretrained_kwargs``. This pairing had no such call,
            so ``replacement_model`` stayed ``None`` and every attribution-driven op failed at first use
            (measured: `'NoneType' object has no attribute 'tokenizer'` at session setup).

            ONE SET OF WEIGHTS, TWO ENGINES OVER IT. The adapter's engine (built by the base hook above)
            drives steering; circuit-tracer builds its own ``EagerModel`` over the SAME ``hf_model`` for
            attribution, with ``requires_grad`` and the frozen-attention implementation it needs. Both
            wrap in place without owning the module, and circuit-tracer's replay taps are installed only
            inside its attribution contexts, so outside those contexts the shared model is the one the
            adapter's engine steers.
            """
            cfg = getattr(self, "circuit_tracer_cfg", None)
            if cfg is None:
                return
            if cfg.backend != ADAPTER_NAME:
                raise UnmappableHookError(
                    f"the (core, interp_engine, circuit_tracer) composition runs circuit-tracer over the "
                    f"engine, so circuit_tracer_cfg.backend must be {ADAPTER_NAME!r}; got {cfg.backend!r}. "
                    "Pick the bundled pairing for that backend instead of this component."
                )
            tokenizer = getattr(getattr(self, "_engine_model", None), "tokenizer", None)
            if tokenizer is None:
                tokenizer = getattr(self.model, "tokenizer", None)  # type: ignore[attr-defined]
            # THE HF MODEL'S CURRENT DEVICE, not circuit-tracer's default. At this hook the module's model
            # is still where interpretune loaded it (CPU in a real session; the move to the accelerator
            # comes later and carries the replacement model along, since it is an ``nn.Module`` holding the
            # transcoders). Left to its default, circuit-tracer loads the transcoders on CUDA and its
            # configuration forward fails on the first splice with tensors on two devices (measured).
            device = _hf_model_device(self.model)  # type: ignore[attr-defined]
            # THE HF MODEL'S DTYPE TOO. `CircuitTracerConfig.dtype` defaults to bfloat16 and sizes the
            # transcoders; the module loads its model at whatever its own config says (float32 in the
            # seed configs). The bundled pairings never meet this because circuit-tracer loads a fresh
            # model at the config dtype; here one set of weights serves both engines, so the transcoders
            # must match the weights that exist. Measured on the first pass: `float != BFloat16` at the
            # first transcoder splice. Realigned on the config so the recorded dtype is the one used.
            hf_dtype = _hf_model_dtype(self.model)  # type: ignore[attr-defined]
            if hf_dtype is not None and cfg.dtype != hf_dtype:
                cfg.dtype = hf_dtype
            self._load_replacement_model(  # type: ignore[attr-defined]
                pretrained_kwargs={"hf_model": self.model, "tokenizer": tokenizer, "device": device}  # type: ignore[attr-defined]
            )
            _expose_hf_surface(self._replacement_model)  # type: ignore[attr-defined]

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
    """The engine as an interpretune ``ModelBackend``: required core plus ``ACTIVATION_INTERVENTION``.

    CARRIES ITS ENGINE MODEL, because the one it is HANDED is the wrong object. Interpretune's ops call
    ``backend.fwd(model=module.model, batch=batch)``, and every bundled adapter satisfies that because
    `module.model` IS its wrapper -- a `HookedTransformer`, an nnsight `LanguageModel`, a circuit-tracer
    `ReplacementModel`. This component cannot use that convention: `EagerModel` is not an `nn.Module`, so
    assigning it to `module.model` raises `cannot assign as child module`.

    Handed the raw HuggingFace model instead, interp-engine falls back to its PROTOCOL arm and refuses
    with "takes one prompt at a time on the GPT2LMHeadModel backend" -- a message about batching that is
    really a message about being given the wrong model. That symptom cost an hour of chasing a batching
    limitation this component does not have.

    So the backend is constructed with the engine model in `post_auto_model_init` and uses it. The `model`
    argument is not ignored so much as SATISFIED: `wrap_model` wraps in place, so the engine model and the
    passed HF model are the same weights, and acting on the former is acting on the latter. This mirrors
    circuit-tracer's `CT_MODEL_BACKEND_FACTORIES`, which are handed the module for the same reason.

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

    def __init__(self, engine_model: Any = None) -> None:
        self._engine_model = engine_model

    def _engine(self, model: Any) -> Any:
        """The engine model to act on: the one this backend was built with, else the argument.

        Falling back rather than requiring it keeps direct construction working for the tests that build a
        backend without a module, and honours a caller who DOES pass an engine model.
        """
        return self._engine_model if self._engine_model is not None else model

    @property
    def capabilities(self) -> frozenset[Any]:
        return frozenset({ModelBackendCapability.ACTIVATION_INTERVENTION})

    def capture_support(self, model: Any) -> Any:
        """Which vocabulary points this backend can capture on ``model``, derived rather than declared.

        Every base in the architecture's inventory is put through the seam's own resolver, and the split
        is whatever it answers: resolved bases are capturable, refused ones carry the seam's refusal text
        as their reason. Deriving it means the record cannot drift from what the adapter actually does --
        a written-down list would be a second source of truth for a question the resolver already
        answers, which is the shape this component has had to remove elsewhere.

        The reasons are the `UnmappableHookError` messages verbatim rather than a summary. Those already
        distinguish the two gaps this adapter is careful about: a name the engine's mapper has no entry
        for, and a block-level name that is ambiguous across sandwich-norm architectures and is refused
        rather than silently resolved to the wrong tensor.

        Derived on every call and never cached. The record is instance-derived and interpretune re-queries
        it after latent models are attached, so a cached answer would describe a model that has since
        changed.
        """
        from interpretune.analysis.backends.capabilities import CaptureSupport
        from interpretune.analysis.points import component_map_for
        from interpretune.analysis.points.inventory import inventory, spelled_at

        if model is None:
            raise ValueError(
                "capture_support needs the wrapped HuggingFace model to read its architecture and layer "
                "count; it was given None. This backend is constructible without a module for unit tests, "
                "and this is the one method that cannot answer in that state."
            )
        if self._engine_model is None:
            raise ValueError(
                "capture_support needs the EagerModel this backend wraps, and none is attached. Unlike the "
                "seam's other methods it does NOT fall back to the HuggingFace model it was handed: the "
                "engine's mapper reads architecture quirks off the EagerModel (sandwich norms, residual "
                "stream count), and given a bare HF model it refuses every point that depends on them "
                "rather than answering. The record would come back understating what this backend can "
                "capture -- 7 of 30 base points on gpt2 rather than 11 -- and understating is the "
                "direction no capture case catches, because an omitted point is neither captured nor "
                "refused, only unknown."
            )
        architecture = self._architecture_of(model)
        cmap = component_map_for(architecture)
        engine = self._engine_model

        capturable: set[str] = set()
        uncapturable: dict[str, str] = {}
        for base in inventory(cmap):
            try:
                _Seam.point_for(spelled_at(base, cmap), engine)
                capturable.add(base)
            except ValueError as exc:
                uncapturable[base] = str(exc)

        return CaptureSupport(
            capturable=frozenset(capturable),
            uncapturable=uncapturable,
            n_layers=self._n_layers_of(model),
            architecture=architecture,
        )

    @staticmethod
    def _architecture_of(model: Any) -> str:
        """The HF class name `component_map_for` keys on, from the model's own config."""
        architectures = getattr(getattr(model, "config", None), "architectures", None)
        if not architectures:
            raise ValueError(
                f"cannot read an architecture from {type(model).__name__}: its config declares no "
                "`architectures`, and the component map is keyed by that HuggingFace class name."
            )
        return str(architectures[0])

    @staticmethod
    def _n_layers_of(model: Any) -> int:
        """Layer count from the HF config, which is canonical where the engine's count could disagree."""
        cfg = getattr(model, "config", None)
        for attr in ("num_hidden_layers", "n_layer", "n_layers"):
            value = getattr(cfg, attr, None)
            if value is not None:
                return int(value)
        raise ValueError(
            f"cannot read a layer count from {type(model).__name__}'s config (tried num_hidden_layers, "
            "n_layer, n_layers); the capture record needs it to refuse a layer beyond the model."
        )

    @property
    def intervention_support(self) -> InterventionSupport:
        """Which configurations of ``ACTIVATION_INTERVENTION`` this backend honours.

        BOTH scopes, because the engine expresses them one argument apart: ``position_mask`` excludes
        positions, so ``LAST_TOKEN`` is a mask and ``ALL_POSITIONS`` is its absence. Declaring only
        ``LAST_TOKEN`` would have been the easy answer and the wrong one -- it would refuse an operation
        this backend can perform exactly, which is the mirror image of silently narrowing one.

        TWO modes. The reason the others stay out is NOT that the engine cannot see the current
        activation, which an earlier version of this docstring claimed: two of its three steering
        methods read it (``orthogonal`` and ``projection_cap`` both project the residual onto the
        vector), and only ``additive`` does not. The reason is that the three are a closed set of
        specific arithmetic forms rather than a general "compute a delta from h" hook, and the missing
        modes need arithmetic none of them computes -- except ``reject``, which ``orthogonal``
        computes exactly once the span is orthonormalised (see the mapping table):

        * ``reject`` maps with ``coeff = 1 - scale_factor`` over an orthonormalised span basis.
        * ``project`` keeps the component PARALLEL to the vector where ``orthogonal`` keeps the
          ORTHOGONAL one. Complementary projections, so no coefficient converts one into the other.
        * ``patch`` needs targets measured during the forward, per token, while ``projection_cap``'s
          bounds go through ``float()`` and so cannot vary by position.
        * ``replace`` needs an arbitrary target, which no method computes.

        Each refusal is a missing OPERATION or a missing per-token PARAMETER, never a missing
        observation. The distinction matters to a reader deciding whether some future engine release
        closes the gap: activation-blindness would be fixed by a new hook, and these are fixed by
        new arithmetic. `_steer_method_for` refuses all three by name; this declaration is the same
        fact stated ahead of the call, so the gate can refuse before reaching this backend at all.
        """
        # Derived from the mapping rather than restated, so the two cannot disagree. That also means a
        # test comparing this declaration against the mapping compares the mapping to itself and can
        # never fail; what actually needs asserting is that each declared mode STEERS, which is
        # `TestEveryDeclaredModeCanActuallySteer` in the seam tests.
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
        tokens = _tokens_from_batch(batch)
        return _Seam.forward(self._engine(model), tokens, attention_mask=_attention_mask_from_batch(batch))

    def fwd_w_cache(self, model: Any, batch: dict[str, Any], names_filter: Any) -> tuple[Any, Any]:
        """Forward with activation caching, at TransformerLens hook names.

        ``names_filter`` is interpretune's vocabulary; the seam translates each name to an engine
        address and REFUSES the ambiguous ones rather than guessing, which is the behaviour this
        adapter exists to get right.
        """
        engine = self._engine(model)
        hook_names = _names_filter_to_list(names_filter, engine)
        tokens = _tokens_from_batch(batch)
        points = [_Seam.point_for(name, engine) for name in hook_names]
        cache = _Seam.capture(engine, tokens, points, attention_mask=_attention_mask_from_batch(batch))
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
                "this backend does not support latent-model handles (it claims ACTIVATION_INTERVENTION but not "
                "LATENT_MODELS); attach latents through a backend that does, rather than having them "
                "silently ignored here."
            )
        tokens = _tokens_from_batch(batch)
        # Paired with its scope rather than flattened: the engine's `SteerSpec` has no position field,
        # so the scope has to travel beside the spec to reach the mask that expresses it. One
        # InterventionSpec fans out to one or more engine specs: `add` is one-to-one, while `reject`
        # over a (k, d) span becomes one spec per orthonormalised basis vector (see
        # `_reject_basis_rows`). Mixed `add`/`reject` at one point is refused before any of that, in
        # `_refuse_mixed_modes`, because the two do not commute.
        targets = list(_intervention_targets(interventions, self._engine(model)))
        _refuse_mixed_modes(targets)
        specs: list[tuple[Any, Any]] = []
        for layer, point, spec in targets:
            mode = _spec_field(spec, "mode", "replace")
            mode_value = getattr(mode, "value", mode)
            method = _steer_method_for(mode)
            scale = _spec_field(spec, "scale_factor")
            scope = normalize_position_scope(_spec_field(spec, "position_scope", PositionScope.LAST_TOKEN))
            tensor = _spec_field(spec, "intervention_tensor")
            if mode_value == "reject":
                rows = _reject_basis_rows(tensor, point)
                for row in rows:
                    specs.append(
                        (
                            scope,
                            _Seam.steer_spec(
                                vector=row,
                                layer=layer,
                                coeff=_coeff_for(mode_value, scale),
                                method=method,
                                point=point,
                                explicit=True,
                            ),
                        )
                    )
                continue
            if mode_value == "clamp":
                lo = _spec_field(spec, "clamp_min", None)
                hi = _spec_field(spec, "clamp_max", None)
                for row, norm in _clamp_cap_rows(tensor, point):
                    specs.append(
                        (
                            scope,
                            _Seam.steer_spec(
                                vector=row,
                                layer=layer,
                                method=method,
                                point=point,
                                minimum=None if lo is None else lo * norm,
                                maximum=None if hi is None else hi * norm,
                                explicit=True,
                            ),
                        )
                    )
                continue
            specs.append(
                (
                    scope,
                    _Seam.steer_spec(
                        vector=_require_expressible_operand(tensor, point),
                        layer=layer,
                        coeff=_coeff_for(mode_value, scale),
                        method=method,
                        point=point,
                        explicit=True,
                    ),
                )
            )
        engine = self._engine(model)
        attention_mask = _attention_mask_from_batch(batch)
        pre = _Seam.forward(engine, tokens, attention_mask=attention_mask)
        post = _Seam.steer(engine, tokens, specs, attention_mask=attention_mask)
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
        # Register the composed config class BEFORE any composition, so a session constructing a config
        # can find it. Nothing derives an import path to this module -- it executes under a synthetic
        # name -- so this registration is the only way auto-composition can see the class.
        try:
            adapter_ctx_registry.register_module_cfg_class(ADAPTER_NAME, ITInterpEngineConfig)
        except AttributeError:
            # An interpretune predating registry-backed config discovery. The component still works;
            # its settings compose the older way, which is what the version floor is for.
            pass

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
            # THE DATAMODULE SLOT, and its absence is why this component could not form a session.
            #
            # `ITSession` resolves a composition per COMPONENT KEY: it looks up `("datamodule", *adapters)`
            # as well as `("module", *adapters)`. Registering only the module slot registers successfully,
            # passes every test that builds a module directly, and fails at session construction with a
            # KeyError naming the datamodule key -- which reads as a missing adapter rather than as a
            # component that registered half of itself.
            #
            # Nothing in this adapter has been exercised through a real `ITSession` until now; the seam
            # tests and the parity module both host the module class by hand, which is precisely the shape
            # of test that cannot see this. Interpretune's conformance suite builds the session, so it
            # found it on first contact.
            #
            # `ITDataModule` unchanged, because interp-engine changes how a forward is OBSERVED and not how
            # data is prepared. That is the same answer the TransformerLens and circuit-tracer adapters
            # give for the same reason, rather than a default taken for convenience.
            adapter_ctx_registry.register(
                ADAPTER_NAME,
                component_key="datamodule",
                adapter_combination=combination,  # type: ignore[arg-type]
                composition_classes=_datamodule_classes(combination),
                description=description,
            )
