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

Then record the environment the artifacts were measured in, beside the bounds:

```bash
python -m tests.profiling_delta.environment --artifacts <nvidia artifacts> <amd artifacts>
```

**A bound is valid only in the environment it was frozen in.** A delta moves with the libraries either arm
allocates through, not only with interpretune. Measured once: this repository's GPU gate, then on an image with torch
2.13.0+cu130, reported the three-way pair's peak 24 MiB over a bound frozen under torch 2.15.0+cu132. interpretune had
not moved it (re-measured with the gate's interpretune commit in the freeze environment, it matched to the byte), and
under torch 2.15 the gate's own package set reproduces the bound exactly, so torch alone made the difference. So
`frozen_environments.yaml` records each vendor's versions of torch, transformers, accelerate, transformer-lens,
nnsight, circuit-tracer and interp-engine (`environment.py`), every pair artifact records the versions it ran under,
and the test refuses to compare against a bound frozen in a different environment, naming each version that differs.
When the gate image or a resolved dependency moves, re-freeze in the gate's environment.

Both vendors are frozen in the GPU gate's environment as of 2026-11-18 (torch 2.15.0 on both images, transformer-lens
4.4.0, accelerate 1.16.0, transformers 5.22.2, circuit-tracer `6f7bc827`), reproduced locally from the package set the
gate resolves: two runs of five repeats per vendor. Every repeat gave the same delta, so each tolerance is the floor
in `stats.py` (8 MiB for peak and retained bytes, 64 KiB per step for the slope):

| composition | vendor | peak_alloc delta | end_current delta | slope |
| --- | --- | --- | --- | --- |
| `interp_engine_test` | NVIDIA RTX 4090 | 0 | 0 | 0 |
| `interp_engine_test` | AMD Radeon AI PRO R9700 | 0 | 0 | 0 |
| `interp_engine_circuit_tracer_test` | NVIDIA RTX 4090 | +1264 MiB | +565 MiB | -368 KiB/step |
| `interp_engine_circuit_tracer_test` | AMD Radeon AI PRO R9700 | +1229 MiB | +554 MiB | -352 KiB/step |

These are the values first frozen on 2026-11-17 under transformer-lens 4.0.0 and accelerate 1.15.0; neither upgrade
moved them. Under torch 2.13 the first repeat of each run differed from the other four, which would have widened the
pooled tolerances well past the floor; torch 2.15 shows no such repeat.

`interp_engine_test` is exactly zero on both vendors: the engine wraps the HF model in place, so capturing
through the composition costs nothing. `interp_engine_circuit_tracer_test` records what the three-way
composition costs over a bare replacement model; the two vendors agree to within 3%. The bound is a regression
gate on that cost as it stands, not a judgement that it is the right cost: reducing it is a change to the
baseline, and freezing it means growth is caught rather than absorbed.
