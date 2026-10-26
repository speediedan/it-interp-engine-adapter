# Memory-profiling deltas for this component

interpretune measures nothing about a hub adapter, so this component measures itself: each composition
runs beside the same work driven through the wrapped library's own API, one subprocess per arm, and the
delta (interpretune arm minus native arm) in peak and retained accelerator bytes plus a per-step leak
slope is what composing this adapter costs. The protocol is interpretune's profiling-delta design
(interpretune#811/#823); `stats.py` and `runner.py` are vendored from interpretune's
`tests/profiling_delta` because that harness does not ship in the interpretune package (see their
headers for provenance and the sync rule).

| composition | interpretune arm | native arm |
| --- | --- | --- |
| `interp_engine_test` | `(core, interp_engine)` composition capturing `blocks.5.hook_resid_pre` per step on gpt2 | `interp_engine.run_with_cache` on an `EagerModel` wrapping the same HF model |
| `interp_engine_circuit_tracer_test` | `(core, interp_engine, circuit_tracer)` composition attributing one prompt per step (gemma-3-1b-pt) | a bare circuit-tracer `ReplacementModel` (nnsight backend) on the same model and transcoder set |

Run them where an accelerator is present:

```bash
IT_RUN_CUDA_TESTS=1 python -m pytest tests/profiling_delta -v
IT_PROFILING_DELTA_DIR=/tmp/artifacts IT_RUN_CUDA_TESTS=1 python -m pytest tests/profiling_delta -v  # keep artifacts
```

A composition with no entry in `delta_bounds.yaml` for the running vendor is report-only: measured and
printed, never gated. Bounds freeze only from spread measured on both vendors.

Measured on NVIDIA (RTX 4090, 2026-10-25, 5 repeats): `interp_engine_test` delta is exactly zero on
every metric (the engine wraps the HF model in place, so this is the expected result, now pinned by a
test rather than asserted in prose). `interp_engine_circuit_tracer_test` reads +1.36 GiB peak and
+0.53 GiB retained with a +192 KiB/step slope: the three-way composition runs two engines over one set
of weights, and what that costs beyond a bare replacement model is now a recorded number instead of an
assumption. It stays report-only until that overhead is judged acceptable as a baseline or reduced.
