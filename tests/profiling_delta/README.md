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

**A bound is valid only in the environment it was frozen in.** A delta moves with every library either arm
allocates through, not only with interpretune: bounds once frozen under torch 2.15.0+cu132 read as a 24 MiB
regression in this repository's GPU gate, whose image carries torch 2.13.0+cu130, although interpretune had not
changed the delta at all (re-measured with the gate's interpretune commit in the freeze environment, it matched to
the byte). So `frozen_environments.yaml` records each vendor's versions of torch, transformers, accelerate,
transformer-lens, nnsight, circuit-tracer and interp-engine (`environment.py`), every pair artifact records the
versions it ran under, and the test refuses to compare against a bound frozen in a different environment, naming
each version that differs. When the gate image or a resolved dependency moves, re-freeze in the gate's
environment.

The cuda bounds are frozen in the GPU gate's environment (2026-11-18: torch 2.13.0+cu130, transformer-lens 4.4.0,
accelerate 1.16.0, circuit-tracer `6f7bc827`, interpretune `7e890500`), reproduced locally from the gate's resolved
package set; two runs of five repeats on an RTX 4090 reproduced the gate's own measurement exactly. The rocm bounds
are from 2026-11-17 (torch 2.15.0+rocm10.0, interpretune `05cd5b1f`, two runs of five repeats on a Radeon AI PRO
R9700); the gate does not run on AMD yet.

| composition | vendor | peak_alloc delta | end_current delta | slope |
| --- | --- | --- | --- | --- |
| `interp_engine_test` | NVIDIA RTX 4090 | 0 | 0 | 0 |
| `interp_engine_test` | AMD Radeon AI PRO R9700 | 0 | 0 | 0 |
| `interp_engine_circuit_tracer_test` | NVIDIA RTX 4090 (gate env) | +1288 MiB | +555 MiB | +192 KiB/step |
| `interp_engine_circuit_tracer_test` | AMD Radeon AI PRO R9700 | +1229 MiB | +554 MiB | -352 KiB/step |

Every repeat on AMD, and on NVIDIA under torch 2.15, gave the same delta. Under the gate's torch 2.13 the first
repeat of each run differs (slope delta -10.2 MiB/step, peak +5.4 MiB) while the other four are identical in both
runs. `freeze.py` pools every repeat's spread across vendors, so that first repeat sets the three-way pair's
tolerances on both vendors: 16.25 MiB for the peak and 31.3 MiB per step for the slope, where the floor is 8 MiB
and 64 KiB per step. The slope band is therefore loose until the first-repeat effect is understood or excluded.

`interp_engine_test` is exactly zero on both vendors: the engine wraps the HF model in place, so capturing
through the composition costs nothing. `interp_engine_circuit_tracer_test` records what the three-way
composition costs over a bare replacement model; the two vendors agree to within 3%. The bound is a regression
gate on that cost as it stands, not a judgement that it is the right cost: reducing it is a change to the
baseline, and freezing it means growth is caught rather than absorbed.
