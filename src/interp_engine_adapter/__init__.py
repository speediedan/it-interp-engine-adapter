"""Package surface for the pip-install path. **The hub loader never executes this file.**

`adapter.py` is the entrypoint: the hub loader runs it straight out of the snapshot by path, so IT
imports no siblings and stays self-contained. This module does import a sibling, which is fine and is
not an exception to that rule -- package context is real on the pip path and absent on the hub path,
and the file the loader runs is the other one.

The package exists so `it_component.yaml` can ship as PACKAGE DATA. That is what lets the pip path and
the hub path resolve the same manifest through the same predicate rather than one of them falling back
to a weaker question. Shipping it via `[tool.setuptools.data-files]` instead would land it at the
install root, where every component publishing a manifest would overwrite the last (measured).
"""

from interp_engine_adapter.adapter import (  # noqa: F401
    ADAPTER_NAME,
    InterpEngineAdapter,
    InterpEngineConfig,
    InterpEngineModule,
    UnmappableHookError,
    _Seam,
)
