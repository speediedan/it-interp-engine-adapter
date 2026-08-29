---
license: apache-2.0
library_name: interpretune
tags:
  - interpretune
  - interpretune-adapters
---

# interp-engine adapter for Interpretune

A hub-delivered execution adapter that composes
[interp-engine](https://github.com/decoderesearch/interp-engine) into an Interpretune session.

This is the first Interpretune adapter delivered as a hub component rather than bundled in the
framework: it is loaded through `it.hub`, gated by the same trust opt-in as any other hub-resident
code, and registers itself into `CompositionRegistry` on load.

## What it adds

`Adapter.interp_engine`, and the compositions `(core, interp_engine)` and
`(lightning, interp_engine)` for the `module` component.

The adapter wraps a model Interpretune has **already loaded**: `EagerModel(id, hf_model=...)` takes
the existing HF model in place, so there is no reload and no second copy in memory.

## Using it

```python
import os
import interpretune as it

os.environ["IT_TRUST_REMOTE_CODE"] = "1"      # this component is remote code; opt in deliberately
it.hub.pull("speediedan/it-interp-engine-adapter")
```

Loading executes the entrypoint in your process. Inspect it first: the `pull` above caches the repo
without executing anything, so the single file below is readable before you opt in.

## One thing worth knowing before you capture

**Hook names are not a shared vocabulary across projects, and this adapter refuses rather than
guesses.** TransformerLens fires `blocks.{i}.hook_mlp_in` on the residual stream *before* the block
norm, while a sublayer's actual argument is the norm's *output*. On gemma-3-1b-it layer 5 those two
tensors share a cosine similarity of **0.088**, and published SAE dashboards have been generated from
the wrong one.

So `seam.point_for()` maps `blocks.{i}.mlp.hook_in` and `blocks.{i}.ln2.hook_out` (which are the same
tensor) to the engine's `mlp_in` point, and **refuses** the legacy `blocks.{i}.hook_mlp_in` with an
error naming what to ask for instead. Silently substituting the tensor the caller probably meant is
how this class of defect propagates, and this adapter declines to add to it.

## Requirements

`interp-engine~=1.3`, and an Interpretune build with hub-delivered adapter support (#125).

The pin is deliberate: interp-engine's 1.x minors change signatures without deprecation aliases, as a
considered choice on their part. Every engine symbol is reached through one seam in the entrypoint,
so a future rename is a one-file change.
