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
tensor) to the engine's `mlp_in` point, maps `blocks.{i}.attn.hook_in` and `blocks.{i}.ln1.hook_out`
to `attn_in` for the same reason, and **refuses** the legacy `blocks.{i}.hook_mlp_in` with an error
naming what to ask for instead. Silently substituting the tensor the caller probably meant is
how this class of defect propagates, and this adapter declines to add to it.

## What this adapter declares, and what it refuses

The published card cannot answer this. It renders the validated manifest, and the publisher never
executes the entrypoint, so runtime declarations and hook refusals are not derivable from it. The card
says so and points here.

### Declared capabilities

`ACTIVATION_INTERVENTION`, and nothing else. Not `LATENT_MODELS`: passing latent-model handles is **refused**
rather than ignored, because a result that looks like an intervention over latents and is not is worse
than an error. Not `GRADIENTS`.

Interventions are **additive only**, at either position scope: `LAST_TOKEN` (the default) or
`ALL_POSITIONS`. A payload mixing the two scopes is expressed, not refused, by nesting one steering
context per scope group. Any other mode is refused by name on the mode axis.

### What it refuses, and why each refusal is not a gap

- **`blocks.{i}.hook_mlp_in` and `blocks.{i}.hook_attn_in`**, with the name to ask for instead. See the
  section above: those name a tensor a whole normalization away from the sublayer's argument.
- **Latent-model handles**, as above.
- **Undeclared intervention modes**, named on the mode axis rather than applied as the nearest declared
  one.
- **A callable `names_filter` that selects a hook this backend cannot map**, rather than capturing the
  mappable subset. A cache holding part of what was asked for is indistinguishable from a complete one
  at the call site.
- **A callable `names_filter` matching nothing at all**, because a filter that silently matches nothing
  looks exactly like one that ran and found nothing.
- **A padded batch on the async capture path**, which the engine's async surface has no `attention_mask`
  parameter to carry. The sync path threads it and takes padded batches normally.

### What it does NOT refuse, and one thing to know about batches

Batches are supported. An earlier revision refused them, on a misreading: the engine's message named the
model it had been handed, not a batching limit.

Batched results are exact against a HuggingFace forward given **the same batch and the same attention
mask** — measured at 0.0 rather than within a tolerance, because interp-engine wraps the HuggingFace
model in place and runs its forward. They are *not* equal to running each row alone, and no backend's
are: with left padding, a short row's real tokens sit at shifted absolute positions, so the two
computations legitimately differ (88.4 on gpt2). Compare against a batched reference, not a per-row one.

### Where the authoritative answer lives

`intervention_support` and `capabilities` on the registered backend. Load the component and ask it:

```python
backend = session.module.model_backend
backend.capabilities, backend.intervention_support
```

That is the answer this file is describing, and if the two ever disagree, the backend is right.

## Requirements

`interp-engine~=1.5.1`, and an Interpretune build with hub-delivered adapter support (speediedan/interpretune#125).

The pin is deliberate: interp-engine's 1.x minors change signatures without deprecation aliases, as a
considered choice on their part. Every engine symbol is reached through one seam in the entrypoint,
so a future rename is a one-file change.
