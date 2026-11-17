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
printed, never gated. Bounds freeze only from spread measured on both vendors, with `freeze.py` (vendored
beside `stats.py`):

```bash
python -m tests.profiling_delta.freeze --artifacts <nvidia artifacts> <amd artifacts> --out tests/profiling_delta/delta_bounds.yaml
```

Both pairs are frozen on both vendors. Measured 2026-11-17 against interpretune `05cd5b1f`, two runs of five
repeats per vendor; every repeat gave the same delta, so each tolerance is the floor in `stats.py` (8 MiB for
peak and retained bytes, 64 KiB per step for the slope):

| composition | vendor | peak_alloc delta | end_current delta | slope |
| --- | --- | --- | --- | --- |
| `interp_engine_test` | NVIDIA RTX 4090 | 0 | 0 | 0 |
| `interp_engine_test` | AMD Radeon AI PRO R9700 | 0 | 0 | 0 |
| `interp_engine_circuit_tracer_test` | NVIDIA RTX 4090 | +1264 MiB | +565 MiB | -368 KiB/step |
| `interp_engine_circuit_tracer_test` | AMD Radeon AI PRO R9700 | +1229 MiB | +554 MiB | -352 KiB/step |

`interp_engine_test` is exactly zero on both vendors: the engine wraps the HF model in place, so capturing
through the composition costs nothing. `interp_engine_circuit_tracer_test` records what the three-way
composition costs over a bare replacement model; the two vendors agree to within 3%. The bound is a regression
gate on that cost as it stands, not a judgement that it is the right cost: reducing it is a change to the
baseline, and freezing it means growth is caught rather than absorbed.
