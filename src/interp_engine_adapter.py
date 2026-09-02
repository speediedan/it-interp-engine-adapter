"""Hub-delivered Interpretune adapter for interp-engine (decoderesearch/interp-engine).

Loaded by ``interpretune.hub.adapters.load_hub_adapter``: a single self-contained module, executed in
the caller's process behind the #255 trust gate, which registers ``Adapter.interp_engine`` and its
compositions into ``CompositionRegistry``.

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
from interpretune.analysis.backends.capabilities import BackendCapability
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
        """Run one forward pass with steering applied, returning logits.

        ``steer`` is a CONTEXT MANAGER yielding a hook manager, not a call, so the forward has to
        happen inside it. Keeping that shape here rather than in the backend is the point of the seam:
        if the engine changes how steering is entered, exactly one method moves.
        """
        from interp_engine import steer

        with steer(engine_model, specs):
            return engine_model.forward(tokens)

    @staticmethod
    def steer_spec(vector: Any, layer: int, coeff: float = 1.0, method: str = "additive", point: str = "resid_post") -> Any:
        """Build one engine ``SteerSpec``."""
        from interp_engine import SteerSpec

        return SteerSpec(vector=vector, layer=layer, coeff=coeff, method=method, point=point)

    @staticmethod
    def forward(engine_model: Any, tokens: Any) -> Any:
        """A plain forward pass, no capture and no steering."""
        return engine_model.forward(tokens)

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

    A CALLABLE filter cannot be enumerated without the model's hook inventory, which the engine does
    not expose as TransformerLens names, so it is refused rather than silently under-capturing.
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
        for spec in (specs if isinstance(specs, (list, tuple)) else [specs]):
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
        # Attach the model backend the way bundled adapters do (`self._model_backend`). There is no
        # model-backend REGISTRY to register into -- backends are adapter-attached by construction --
        # so a hub adapter reaches this seam exactly as a bundled one does.
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


class InterpEngineModelBackend:
    """The engine as an interpretune ``ModelBackend``: required core plus ``INTERVENTION``.

    Interpretune's ``ModelBackend`` was decomposed into a required core and capability-gated method
    groups precisely so a component like this one can implement what it actually does and SAY so
    (interpretune R1, #401). This implements:

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
        return getattr(cache, "logits", None), self.wrap_activation_cache(by_hook, model)

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
        specs = [
            _Seam.steer_spec(
                vector=spec.intervention_tensor,
                layer=layer,
                coeff=spec.scale_factor,
                method=_steer_method_for(getattr(spec, "mode", "replace")),
                point=point,
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

    @classmethod
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
        for combination, description in (
            ((Adapter.core, ADAPTER_NAME), "interp-engine eager-tier adapter composed with core."),
            (
                (Adapter.lightning, ADAPTER_NAME),
                "interp-engine eager-tier adapter composed with the Lightning adapter.",
            ),
        ):
            adapter_ctx_registry.register(
                ADAPTER_NAME,
                component_key="module",
                adapter_combination=combination,  # type: ignore[arg-type]
                composition_classes=(InterpEngineModule,),
                description=description,
            )
