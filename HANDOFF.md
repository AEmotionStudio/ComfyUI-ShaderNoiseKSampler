# Handoff: directions for ShaderNoiseKSampler

Started 2026-09-12 after fixing MiniMax H3 support and running ~60 real
generations against H3 Max (int8 turbo) and SD 1.5 to characterise the node's
behaviour. Items 1 and 2 have since been worked; 3, 4 and 5 are untouched.

Everything here is grounded in measurement. Where a number appears it came from a
real run, and the section says how to reproduce it. Where a plan was **wrong**,
the section says so and why — two of the metrics originally proposed here were
measured and rejected, and one whole premise was inverted.

| # | Direction | State |
|---|---|---|
| 1 | Normalise strength across the knobs | **Done for blend modes** (`d391f08`); shader types and masks still uncalibrated |
| 2 | Auto-scale to the latent | **Premise disproved.** Real cause found and fixed (`0da8bb6`); follow-ups below |
| 3 | Lean into what the name promises | Not started |
| 4 | Shader noise on the audio stream | Not started |
| 5 | Per-stage shader parameters | Not started |

---

## Background: what the testing established

**1. The strength ceiling is set by the channel axis, not by latent size or step
count.** This inverted the original assumption — see item 2. Measured ceilings on
H3 (608x352, 8 steps, `res_multistep`/`simple`, cfg 1.0), all with
`decorrelate_channels` off:

| Setting | Usable | Fails by |
|---|---|---|
| `domain_warp` + `multiply` | ~0.25 | 0.75 (subject gone) |
| `tensor_field` | ~0.5 | — |
| `curl_noise` | ~0.2 | 0.5 (heavy colour casts) |
| `temporal_coherent` | ~0.35 | — (only faint striping at 0.35) |
| `soft_light`, `difference` | 0.3+ | — |
| `screen` | ~0.3 | 0.5 (edge artifacts) |
| `add` | ~0.15 | 0.5 (worst of all modes) |
| shape masks (`spiral`, `hexgrid`, …) | ~0.2 | 0.6 (mask drawn into the picture) |
| `use_temporal_coherence` | ~0.2 | 0.5 (swamps the frame) |

With `decorrelate_channels` on, `domain_warp` on H3 is clean at 0.5 where it
showed green blocking before.

**2. The audio stream moves even though nothing touches it.** Shader noise is
applied only to the spatial stream and the audio stream keeps its Gaussian noise
bit-identical (`test_the_shader_only_paints_the_spatial_stream`). Yet audio level
tracks strength monotonically: **-22.6 dB baseline -> -23.8 / -19.8 / -18.1 /
-17.0** across a 0.25/0.50/0.75/1.00 sweep. That is the DiT's joint attention
carrying a video perturbation into sound. It makes audio level a free,
independent check on any change to the noise — it confirmed item 1 from a
direction the calibration knew nothing about.

`temporal_coherent` is the exception: **-24.0 / -23.7 / -23.7 dB**, flat.

**3. Content sets the ceiling as much as settings do.** At an identical 0.5,
rain-on-glass, a steam locomotive and ocean surf stayed clean; a forge interior
and a concert hall showed green blocking. High-frequency scenes mask structured
noise, large dark regions let it survive denoising.

**4. Two plausible-sounding metrics do not predict any of this.** Recorded so
nobody spends the time again:

- *Low-frequency spectral energy* (this document's original proposal). Shape
  masks visibly wreck the image yet `hexgrid` (+0.114) and `rays` (+0.304) carry
  **less** low-frequency energy than no mask (+0.381). A radial power spectrum
  discards phase, and a mask modulates amplitude rather than spectral tilt.
- *Block mean-structure and non-stationarity.* Spearman against the observed
  ceilings: -0.20, +0.01, and +0.23 combined — one of them the wrong sign.

What did work was to stop looking for a universal "how visible is this noise"
statistic and measure something exactly defined instead: how much shader each
blend mode injects (item 1), and how many channels the draw spans (item 2).

---

## 1. Normalise strength across the knobs — done for blend modes

Shipped in `d391f08`. `BLEND_SHADER_FRACTION` in `core/noise_math.py` records the
measured shader share per mode across the 0..1 strength range, and
`normalized_strength()` re-expresses a strength on the reference mode's scale.
Exposed as `normalize_strength`, default off.

The measure is the cosine between the mixed noise and the shader. `mix_noise`
standardises both operands, so the result lies on the unit sphere they span and
that cosine is exactly the shader's share of it — model-free and unambiguous. The
modes differed by a factor of twenty-three at strength 0.5.

Verified on H3 across six modes at strength 0.30: audio spread across modes fell
from **4.6 dB to 2.1 dB** (std 1.76 -> 0.71), and the six normalised frames are
clean and comparably varied where the un-normalised ones range from untouched to
green-blocked.

One static table serves everything: the curves vary by at most 0.07 across four
shader types, a shape mask, and 4D/5D latents at three channel counts.
`test_blend_calibration_is_current` re-measures and fails on drift.

### What is left

**Shader types and shape masks are still uncalibrated.** They differ in the
*character* of the noise, not in how much is injected, so the cosine measure says
nothing useful about them and the two metrics in finding (4) both failed. The
tooltips carry the guidance instead.

If you want to try again, the thing to predict is not "how structured is this
noise" in the abstract but "how much of this structure survives denoising", which
may simply not have a model-free answer. A defensible fallback is an empirical
per-type factor fitted to observations on two or three architectures and labelled
as empirical — but do not fit it to one model and present it as general.

**Consider making `normalize_strength` the default** in a major version. It is a
breaking change for saved workflows, which is why it ships off.

---

## 2. Auto-scale to the latent — premise disproved

The original plan was to scale the default from the latent's spatial size and the
step count. **Both turned out to be the wrong variables.**

The test: SD 1.5 sweeps at 512px/20 steps, 512px/8 steps and 256px/20 steps, to
put a second measured ceiling beside H3's. SD 1.5 at a 64x64 latent with 20 steps
is a green-and-purple abstract by strength **0.25** — markedly worse than H3 at
38x22 with 8 steps, which was still a clean graded walk at that setting. Larger
latent, more steps, worse result. There was nothing to auto-scale.

### What it actually was

`shaders/base.py::_expand_channels` builds every channel past the first one or
two as a pointwise function (`sin`, `abs`) of a mixture of those two. The draw
therefore spans almost nothing however many channels are requested. Effective
rank, as the participation ratio of the channel covariance spectrum:

| | 4ch | 16ch | 24ch | 128ch |
|---|---|---|---|---|
| gaussian | 4.00 | 16.00 | 23.90 | 120.30 |
| `domain_warp` | **1.00** | 2.22 | 2.06 | **2.37** |
| `temporal_coherent` | **1.00** | 1.00 | 1.00 | **1.00** |
| `curl_noise` | 3.58 | 10.88 | 22.11 | 25.22 |
| `tensor_field` | 3.81 | 14.31 | 23.54 | 90.78 |

`domain_warp` — the default — is a two-dimensional signal copied across up to 128
channels. `temporal_coherent` is one channel repeated. Samplers expect i.i.d.
noise. This also explains the ordering in finding (1): `tensor_field` tolerated
the most strength and is the only near-full-rank generator.

Shipped in `0da8bb6` as `decorrelate_channels`, default off: fill the channel axis
from `DECORRELATION_BASIS` independent renders mixed through a seeded random
matrix. Rank becomes min(channels, basis) and every basis element is still that
shader, so the spatial character is preserved. SD 1.5 moves from an abstract
smear at 0.25 to a clean photoreal variation; H3 is clean at 0.5 where it showed
green blocking before.

It is guarded twice, because neither guess held alone. Generators already wider
than the basis are skipped — `curl_noise` looks correlated but spans 25 of 128
channels, so a correlation threshold was the wrong test. And the remix is kept
only when it actually widens the draw — `curl_noise` at four channels remixes
*narrower* than it started. Turning it on can never narrow the noise.

A separate bug fell out: `shaders/temporal_coherent_noise.py` read
`params["base_seed"]` unconditionally where `domain_warp` gates it on
`use_temporal_coherence`. The node always sets `base_seed`, so that generator
ignored its `seed` argument entirely — every stage and every decorrelation draw
produced an identical field, with only `time` varying.

### What is left

**Fix `_expand_channels` at source.** `decorrelate_channels` works around it from
`core/shader_noise.py`; the generators themselves still produce collapsed output
for every other caller, including the legacy path. Doing it properly inside
`shaders/base.py` would make the workaround unnecessary — but it changes legacy
output, which `tests/golden_cases.py` pins deliberately, so it needs the same
gating discussion as item 1.

**Tune `DECORRELATION_BASIS`.** It is 8, chosen to bound cost at LTXV's 128
channels. Nothing has tested whether 4 is as good or 16 is better. The cost is
roughly linear in the basis: 2-12x the noise-generation time depending on shape
and generator, which is small against sampling but not free.

**Consider making it the default**, with the same breaking-change caveat.

**Re-derive the ceilings with it on.** Every number in finding (1) was measured
with it off. The two knobs interact and nothing has mapped that.

---

## 3. Lean into what the name promises

Not started.

**The problem.** The README's pitch is exploring the neighbourhood around a seed,
and it genuinely works — a 0.00/0.05/0.10/0.15/0.20/0.25 sweep is a smooth graded
walk through the same scene, all six photoreal. But producing that meant
hand-authoring six workflows and stitching frames with ffmpeg. The core feature
is a manual grind.

**Approach.** A batch-walk node that emits N latents along a ramp.

- Inputs: the sampler's own model/conditioning/latent, plus which parameter to
  walk (`shader_strength`, `phase_shift`, `noise_scale`, seed), start, end, count.
- Output: a batched LATENT or image batch, feeding the existing
  `Advanced Image Comparer` / `Video Comparer` nodes.
- Reuse `pipelines.standard.run` per step rather than reimplementing sampling.
  Weights stay resident across the walk, which is what makes it cheap — an H3 run
  at 608x352/56 frames took **28-51s** once loaded, against several minutes for
  the initial 21 GB load.

**Where.** A new node in `nodes/`, registered in `__init__.py` beside the
comparers.

`phase_shift` is the most interesting axis after strength: documented as
revealing "different facets of the same core elements", and nothing has tested it.

---

## 4. Shader noise on the audio stream

Not started.

H3's audio stream gets plain Gaussian because it has no 2D spatial grid, and
`require_spatial_latent` refuses anything that is not rank 4 or 5. That is the
right conservative default and should stay the default. But the audio stream is
`[B, 32, 2, T_audio]` — channels x stereo x time — and a **1-D shader along the
time axis is well-defined**. Nobody has tried it.

Finding (2) is why it is interesting: perturbing video alone already moves audio
by up to 5 dB through joint attention. Driving audio directly, and letting that
propagate back into video, is unexplored in both directions.

**Approach.** Add a 1-D path to `core/shader_noise.py` treating `[B, C, L]` as a
1xL grid and `[B, C, 2, T]` as 2xT — the generators already honour
`target_channels` from 3 to 256 and produce exact shapes. Gate it behind a new
input, default off; do not change what `require_spatial_latent` refuses by
default. Start at very low strength: a 5 dB shift is already audible and there is
no equivalent of "a busy scene masks it".

**Where.** `core/shader_noise.py`, `pipelines/standard.py` (currently
`noise_streams[0]` only, deliberately), and a node input.

This also unblocks the sequence-latent models currently refused outright: Stable
Audio 1/3, ACE-Step 1.5, MiniMax Music 3, Hunyuan3D and TripoSplat.

---

## 5. Per-stage shader parameters

Not started.

Stages are segments of one trajectory, but every stage uses the same shader
config and differs only in strength via `schedule.stage_strengths`. The
trajectory is not uniform — early steps set composition, late steps set detail —
so identical noise character at both ends wastes structure the pipeline already
has.

**Approach.** Let each stage carry its own shader type and/or phase: coarse
structure early where it steers composition, fine structure late where it
perturbs detail without destroying the subject.

The plumbing is nearly there. `_shader_events` already builds a per-boundary list
of `(strength, seed)` tuples; widening to `(strength, seed, params)` and threading
it into `_apply_events` is contained. The node-side question is harder — exposing
per-stage config without a combinatorial explosion of widgets. A preset list
("coarse to fine", "fine to coarse", "uniform") is probably the right first cut,
mirroring how `sequential_distribution` already handles per-stage strength.

**Where.** `pipelines/standard.py` (`_shader_events`, `_apply_events`),
`core/schedule.py`, one node input.

**Sequence last.** It multiplies the parameter space, which is only tolerable now
that a strength value means one thing across blend modes.

---

## Reproducing the measurements

Scratch workflows and contact sheets live in the session scratchpad, not the
repository. To regenerate:

- Minimal H3 t2v graph: `UNETLoader` -> `MiniMaxH3SigmaShift(6.0, 3.0)`,
  `CLIPLoader(type="minimax")`, two `VAELoader`s (video + audio),
  `MiniMaxH3ImageToVideo` -> `ShaderNoiseKSamplerDirect` -> `VAEDecode` +
  `VAEDecodeAudio` -> `CreateVideo(24fps)` -> `SaveVideo`. `SaveVideo` needs
  `format.codec` as a dynamic-combo key, not `codec`.
- The 32B text encoder and the 21 GB UNet will not fit together on a 31 GB / 12 GB
  box. For sampling-only checks, skip the encoder and pass
  `[[torch.zeros(1, 16, 5120), {}]]` as both conditionings — `condition_proj` is
  `[5376, 5120]`, and a missing `text_token_tags` is handled.
- Audio level per clip: `ffmpeg -i X.mp4 -af volumedetect -f null -`. Use it as a
  free independent check on any change to the noise.
- SD 1.5 sweeps need no server: load via `comfy.sd.load_checkpoint_guess_config`
  and call `pipelines.standard.run` directly. Seconds per image.

`tests/helpers.py` has a `FakeModel("av")` carrying the real `MiniMaxH3AV`
format, a real dual-shift `ModelSamplingAV` and `MiniMaxH3`'s own
`process_latent_in/out` — enough to exercise the pipeline with no weights.

---

## Not on this list, deliberately

**The legacy pipeline stays frozen.** Its contract is reproducing pre-2.0 seeds
bit-for-bit. It has a known quirk — its inner-model class table sits inside
`if debugger.enabled:`, so detected channel counts depend on the debug level —
and fixing it would change output for existing workflows. `tests/golden_cases.py`
pins it.

**`core/sampler.py`, `core/blending.py` and `core/transforms.py` are uncalled by
any node** (`CODE_REVIEW.md:243`, which also lists `core/model_compat.py`, removed
in `363ac1e`). They carry three more copies of the wrong `[B,F,C,H,W]` layout
heuristic. Easy cleanup for whoever wants it.

---

## One live environment note

The Direct node's shader preview (`web/shader_renderer.js`) has GLSL for only
`domain_warp`, `tensor_field` and `curl_noise`. Selecting `temporal_coherent`
leaves the preview on its previous pattern and logs `Shader source not found` to
the console — sampling is unaffected, and the tooltip says so. A fourth GLSL
preview would close the gap.
