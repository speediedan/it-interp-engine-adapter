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
    def capabilities() -> Any:
        """The engine's capability set, readable before warmup.

        Capability is queried, never inferred from a backend name: unsupported requests raise
        ``CapabilityUnsupported`` naming the working call, and nothing downgrades to a warning.
        """
        from interp_engine import CAPABILITIES

        return CAPABILITIES


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
