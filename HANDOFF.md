# Handoff: five directions for ShaderNoiseKSampler

Written 2026-09-12, after fixing MiniMax H3 support and running ~46 real
generations against H3 Max (int8 turbo) to characterise the node's behaviour on
a video model. This is the "what next" that came out of that, in the order I
would do it.

Everything below is grounded in measurements from that session. Where a number
appears, it came from a real generation, and the section says how to reproduce
it.

---

## Background: what the testing established

Three findings shape all five proposals.

**1. Shader noise is a much sharper instrument on video than on images.**
The 0.3 default is mild on SD 1.5. On H3 it is near the ceiling. The mechanism
is structural, not model-specific prejudice: H3's video latent is 24 channels at
a 16x spatial downscale, so a 608x352 frame is a **38x22 latent grid**. Shader
structure at that resolution is not fine texture, it is *large blocks* — and
with only 8 turbo steps there is little opportunity for the sampler to absorb
it. Coarser latents and fewer steps both push the usable ceiling down.

Measured ceilings on H3 (608x352, 8 steps, `res_multistep`/`simple`, cfg 1.0):

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

**2. The audio stream moves even though nothing touches it.**
Shader noise is applied only to the spatial stream; the audio stream keeps the
Gaussian noise `comfy.sample.prepare_noise` produced, bit-identical (asserted by
`test_the_shader_only_paints_the_spatial_stream`). Yet audio level tracks shader
strength monotonically: **-22.6 dB baseline -> -23.8 / -19.8 / -18.1 / -17.0**
across a 0.25/0.50/0.75/1.00 sweep. That is the DiT's joint attention carrying a
video perturbation into sound. H3 denoises `[text | audio | video]` as one
packed sequence, so the streams are genuinely coupled.

`temporal_coherent` is the exception — **-24.0 / -23.7 / -23.7 dB**, essentially
flat. Smooth temporal structure perturbs the joint trajectory far less than
spatially-blocky noise.

**3. Content sets the ceiling as much as settings do.**
At an identical strength of 0.5, rain-on-glass, a steam locomotive and ocean
surf all stayed clean and gave good variations. A forge interior and a concert
hall both showed green blocking. High-frequency scenes mask structured noise;
large dark regions let it survive denoising.

---

## 1. Normalise strength across the knobs

**The problem.** `shader_strength` means something different for every shader
type, blend mode and shape mask — see the ceilings table. A user who finds 0.25
works with `domain_warp`/`multiply`, then switches to `curl_noise` or enables a
shape mask, gets a ruined frame and no indication why. The slider is not one
control, it is eight controls sharing a label.

**Why this is first.** Every other item here is easier to reason about once the
dial means one thing. It is also the largest usability win available for the
smallest conceptual change.

**Approach.** Calibrate internally, so the public slider keeps its 0.0-1.0 range
and its meaning becomes consistent.

1. Define the measurable: the perturbation a configuration actually applies to
   the latent. `core.noise_math.mix_noise` already standardises both operands to
   mean 0 / std 1, so magnitude is not the differentiator — *spatial structure*
   is. A workable proxy is the noise's radially-averaged power spectrum:
   structured noise concentrates energy at low spatial frequencies, and it is
   exactly that low-frequency energy that survives denoising and shows up as
   blocking. Compute the low-frequency energy fraction of `generated` versus an
   i.i.d. Gaussian reference of the same shape.
2. For each (shader_type, blend_mode, shape_type) triple, measure that fraction
   at a fixed nominal strength. This is pure tensor work — no model, no GPU
   needed, and it belongs in `tests/` as a generated table rather than a
   hand-written one.
3. Scale the applied strength so equal slider values give equal low-frequency
   energy. Store the factors next to `core/constants.py`.

**Do not** normalise by post-hoc image comparison against a model. That would
bake H3's characteristics into a model-agnostic pipeline — precisely the mistake
the v2 rewrite removed.

**Where.** `core/noise_math.py` (`mix_noise`), a new calibration table beside
`core/constants.py`, and a test that regenerates it so drift is caught.

**Watch out.** This changes what existing `shader_strength` values do, so it is
a breaking change for saved workflows. Options: gate it behind the existing
`sampling_mode` input as a third mode, or accept the break in a major version
and say so loudly in `CHANGELOG.md`. The legacy pipeline must not be touched
either way — its contract is bit-exact seed reproduction.

---

## 2. Auto-scale to the latent

**The problem.** The node already knows everything it needs to pick a sensible
default and does not use any of it: the latent's channel count, spatial size,
rank, and the step count. A 38x22 video latent at 8 steps and a 128x128 SD
latent at 30 steps get the same 0.3.

**Approach.** Derive a strength ceiling from the sampling context and use it to
scale the slider, or at minimum to warn.

The two drivers, from finding (1):

- **Latent spatial size.** Smaller grids mean shader structure is coarser
  relative to the image. `primary.shape[-2:]` gives this directly.
- **Step count.** Fewer steps mean less opportunity to absorb structure. The
  pipeline already has `total_steps` from `schedule.build_sigmas`.

A first cut: scale down as the latent grid shrinks below a reference size and as
steps drop below a reference count, clamped to a sane floor. Calibrate the
reference against SD 1.5 at 512x512 / 20 steps, where the historical 0.3 default
is known good.

**Where.** `pipelines/standard.py` in `run()`, after `fix_empty_latent_channels`
and `build_sigmas` — both inputs are already in scope there. Note that
`_streams(samples)[0]` is the stream being painted, so use its shape, not
`NestedTensor.shape` (which silently reports only the first stream anyway).

**Sequence this after item 1** — auto-scaling an inconsistent unit just moves
the inconsistency somewhere harder to see.

**Also worth doing here:** make `temporal_coherent` the default `shader_type`
when the latent is 5D. It degrades most gracefully on video and keeps audio
stable, and it is now selectable (commit `7b40c55`).

---

## 3. Lean into what the name promises

**The problem.** The README's pitch is exploring the neighbourhood around a
seed, and it genuinely works — `scratchpad/sheets/7_lowsweep.png` from this
session shows 0.00 -> 0.05 -> 0.10 -> 0.15 -> 0.20 -> 0.25 as a smooth graded
walk through the same scene, all six photoreal. But producing that sheet meant
hand-authoring six workflows and stitching frames with ffmpeg. The core feature
is a manual grind.

**Approach.** A batch-walk node that emits N latents along a ramp.

- Inputs: the same model/conditioning/latent as the sampler, plus which
  parameter to walk (`shader_strength`, `phase_shift`, `noise_scale`, seed),
  start, end, and step count.
- Output: a batched LATENT, or an image batch, ready for the existing
  `Advanced Image Comparer` / `Video Comparer` nodes.
- Reuse `pipelines.standard.run` per step rather than reimplementing sampling.
  Model weights stay resident across the walk, which is what makes this cheap —
  an H3 run at 608x352/56 frames took **28-51s** once loaded, against several
  minutes for the initial 21 GB load.

That closes the loop the project already has the other half of. `phase_shift` is
the most interesting axis after strength: it is documented as revealing
"different facets of the same core elements" and nothing in this session tested
it.

**Where.** A new node in `nodes/`, registered in `__init__.py` alongside the
comparers.

---

## 4. Shader noise on the audio stream

**The problem — or rather, the opening.** H3's audio stream currently gets plain
Gaussian noise because it has no 2D spatial grid, and `require_spatial_latent`
refuses anything that is not rank 4 or 5. That is the correct conservative
default and should stay the default. But the audio stream is
`[B, 32, 2, T_audio]` — channels x stereo x time — and a **1-D shader along the
time axis is well-defined**. Nobody has tried it.

**Why it is interesting.** Finding (2) shows the two streams are strongly
coupled: perturbing video alone already moves the audio by up to 5 dB. Driving
the audio stream directly, and letting *that* propagate into the video through
the same joint attention, is unexplored in both directions. It is the most
novel thing on this list.

**Approach.**

1. Add a 1-D generation path to `core/shader_noise.py`. `latent_layout` raises on
   rank 3 today via `require_spatial_latent`; the natural extension is treating
   `[B, C, L]` as a 1xL grid and `[B, C, 2, T]` as 2xT, which the existing
   generators can already fill — they honour `target_channels` from 3 to 256
   (verified) and produce exact shapes.
2. Gate it behind a new input, default off. Do **not** change what
   `require_spatial_latent` refuses by default; the clear error is worth keeping
   for people who did not ask for this.
3. Start at very low strength. Audio is far less forgiving than video — a 5 dB
   level shift is already audible, and there is no equivalent of "busy scene
   masks the noise".

**Where.** `core/shader_noise.py` (the layout/generate path), `pipelines/standard.py`
(currently `noise_streams[0]` only — line references the *first* stream
deliberately), and a new node input.

**Note.** This also unblocks the sequence-latent models that item currently
refuses outright: Stable Audio 1/3, ACE-Step 1.5, MiniMax Music 3, Hunyuan3D and
TripoSplat. They are listed in the `UnsupportedLatentError` message.

---

## 5. Per-stage shader parameters

**The problem.** Stages are segments of one trajectory (that is the v2 fix), but
every stage uses the same shader config and differs only in strength, via
`schedule.stage_strengths`. The diffusion trajectory itself is not uniform —
early steps set composition, late steps set detail — so applying identical noise
character at both ends wastes the structure the pipeline already has.

**Approach.** Let each stage carry its own shader type and/or phase. Coarse,
low-frequency structure early where it can steer composition; fine structure
late where it perturbs detail without destroying the subject.

The plumbing is nearly there. `_shader_events` already builds a per-boundary list
of `(strength, seed)` tuples; widening that to `(strength, seed, params)` and
threading it into `_apply_events` is a contained change. The node-side question
is harder — exposing per-stage config without a combinatorial explosion of
widgets. A preset list ("coarse to fine", "fine to coarse", "uniform") is
probably the right first cut, mirroring how `sequential_distribution` already
handles per-stage strength.

**Where.** `pipelines/standard.py` (`_shader_events`, `_apply_events`),
`core/schedule.py` (alongside `stage_strengths`), and one node input.

**Sequence this last.** It multiplies the parameter space, which is only
tolerable once item 1 has made a single strength value mean one thing.

---

## Reproducing the measurements

The scratch workflows and contact sheets from this session are in the session
scratchpad, not the repository. To regenerate:

- Build a minimal H3 t2v graph: `UNETLoader` -> `MiniMaxH3SigmaShift(6.0, 3.0)`,
  `CLIPLoader(type="minimax")`, two `VAELoader`s (video + audio),
  `MiniMaxH3ImageToVideo` -> `ShaderNoiseKSamplerDirect` -> `VAEDecode` +
  `VAEDecodeAudio` -> `CreateVideo(24fps)` -> `SaveVideo`. `SaveVideo` needs
  `format.codec` as a dynamic-combo key, not `codec`.
- The 32B text encoder and the 21 GB UNet will not fit together on a 31 GB / 12 GB
  box. For sampling-only checks, skip the encoder and pass
  `[[torch.zeros(1, 16, 5120), {}]]` as both conditionings — `condition_proj` is
  `[5376, 5120]`, and a missing `text_token_tags` is handled.
- Audio level per clip: `ffmpeg -i X.mp4 -af volumedetect -f null -`.

`tests/helpers.py` has a `FakeModel("av")` that carries the real `MiniMaxH3AV`
format, a real dual-shift `ModelSamplingAV`, and `MiniMaxH3`'s own
`process_latent_in/out` — enough to exercise the pipeline without any weights.

---

## Two things not on this list, deliberately

**The legacy pipeline stays frozen.** Its contract is reproducing pre-2.0 seeds
bit-for-bit. It has a known quirk — its inner-model class table sits inside
`if debugger.enabled:`, so detected channel counts depend on the debug level —
and fixing that would change output for existing workflows. `tests/golden_cases.py`
pins it.

**`core/sampler.py`, `core/blending.py` and `core/transforms.py` are uncalled by
any node** (`CODE_REVIEW.md:243`). `core/model_compat.py` was removed in commit
`363ac1e`; these three were left alone because only that one was in scope. They
carry three more copies of the wrong `[B,F,C,H,W]` layout heuristic. Removing
them is easy cleanup for whoever wants it.

---

## One live environment note

The Direct node's shader preview (`web/shader_renderer.js`) has GLSL for only
`domain_warp`, `tensor_field` and `curl_noise`. Selecting `temporal_coherent`
leaves the preview showing its previous pattern and logs
`Shader source not found` to the console — sampling is unaffected, and the
tooltip says so. Adding a fourth GLSL preview would close the gap.
