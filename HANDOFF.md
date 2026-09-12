# Handoff: ShaderNoiseKSampler

Written 2026-09-12 across one session: fixing MiniMax H3 support, then working
through five directions it suggested. All five are implemented. What remains is
listed per item.

Everything here is grounded in measurement — roughly 80 real generations against
MiniMax H3 Max (int8 turbo) and SD 1.5. Where a number appears it came from a
real run. Where a plan was **wrong**, the section says so: two proposed metrics
were measured and rejected, and one whole premise was inverted. Those failures
are recorded on purpose, because they are the obvious-sounding ideas.

| # | Direction | State |
|---|---|---|
| 1 | Normalise strength across the knobs | Done for blend modes (`d391f08`); types and masks still uncalibrated |
| 2 | Auto-scale to the latent | Premise disproved; real cause found and fixed (`0da8bb6`) |
| 3 | Lean into what the name promises | Done (`86aac9e`) — `ShaderNoiseWalk` |
| 4 | Shader noise on the audio stream | Done (`36e9e8c`) — works; aesthetic value unproven |
| 5 | Per-stage shader parameters | Done (`08b7d3b`) — `stage_progression` |

Every new capability is an optional input defaulting to off, so no saved workflow
changes behaviour unless asked.

---

## Findings that shape everything else

**1. The strength ceiling is set by the channel axis**, not by latent size or
step count. See item 2. Ceilings measured on H3 (608x352, 8 steps,
`res_multistep`/`simple`, cfg 1.0), **all with `decorrelate_channels` off**:

| Setting | Usable | Fails by |
|---|---|---|
| `domain_warp` + `multiply` | ~0.25 | 0.75 (subject gone) |
| `tensor_field` | ~0.5 | — |
| `curl_noise` | ~0.2 | 0.5 (heavy colour casts) |
| `temporal_coherent` | ~0.35 | — (faint striping only) |
| `soft_light`, `difference` | 0.3+ | — |
| `screen` | ~0.3 | 0.5 (edge artifacts) |
| `add` | ~0.15 | 0.5 (worst mode) |
| shape masks | ~0.2 | 0.6 (mask drawn into the picture) |
| `use_temporal_coherence` | ~0.2 | 0.5 (swamps the frame) |

With `decorrelate_channels` on, `domain_warp` is clean at 0.5 on H3 and 0.25 on
SD 1.5 where both previously failed.

**2. The audio stream moves even when nothing touches it.** Shader noise reaches
only the spatial stream by default and the audio keeps its Gaussian noise
bit-identical, yet audio level tracks strength monotonically: **-22.6 dB baseline
-> -23.8 / -19.8 / -18.1 / -17.0** across a 0.25/0.50/0.75/1.00 sweep. That is
H3's joint attention carrying a video perturbation into sound.

This makes audio level a **free, independent check** on any change to the noise.
It confirmed item 1 from a direction the calibration knew nothing about (spread
across blend modes fell from 4.6 dB to 2.1 dB). Use it.

**3. Content sets the ceiling as much as settings do.** At an identical 0.5,
rain-on-glass, a steam locomotive and ocean surf stayed clean; a forge interior
and a concert hall showed green blocking. High-frequency scenes mask structured
noise; large dark regions let it survive denoising. Audio has no equivalent —
nothing to hide structure in.

**4. Two plausible metrics predict none of this.** Do not spend the time again:

- *Low-frequency spectral energy* (this document's own original proposal). Shape
  masks visibly wreck the image yet `hexgrid` (+0.114) and `rays` (+0.304) carry
  **less** low-frequency energy than no mask (+0.381). A radial power spectrum
  discards phase; a mask modulates amplitude, not spectral tilt.
- *Block mean-structure and non-stationarity.* Spearman against the observed
  ceilings: -0.20, +0.01, and +0.23 combined — one with the wrong sign.

What worked was abandoning the search for a universal "how visible is this noise"
statistic and measuring exactly-defined quantities instead: how much shader a
blend mode injects (item 1), and how many channels a draw spans (item 2).

---

## 1. Normalise strength — done for blend modes

`BLEND_SHADER_FRACTION` and `normalized_strength()` in `core/noise_math.py`,
exposed as `normalize_strength`.

The measure is the cosine between the mixed noise and the shader. `mix_noise`
standardises both operands, so the result lies on the unit sphere they span and
that cosine is exactly the shader's share — model-free and unambiguous. The modes
differed by a factor of twenty-three at strength 0.5. One static table serves
everything (curves vary by at most 0.07 across shader types, masks and latent
ranks); `test_blend_calibration_is_current` re-measures and fails on drift.

### Left to do

**Shader types and shape masks are still uncalibrated.** They differ in the
*character* of the noise rather than how much is injected, so the cosine says
nothing about them and both metrics in finding (4) failed. The tooltips carry the
guidance instead.

If you retry: the thing to predict is not "how structured is this noise" but "how
much of this structure survives denoising", which may have no model-free answer.
An empirical per-type factor fitted across two or three architectures and
labelled as empirical is defensible — fitting it to one model and presenting it
as general is not.

**Consider making it the default** in a major version.

---

## 2. Auto-scale to the latent — premise disproved

The plan was to scale from latent spatial size and step count. Both were the
wrong variables: SD 1.5 at a 64x64 latent with 20 steps is a green-and-purple
abstract by strength **0.25**, markedly worse than H3 at 38x22 with 8 steps.
Larger latent, more steps, worse result. Nothing to scale.

### What it actually was

`shaders/base.py::_expand_channels` builds every channel past the first one or
two as a pointwise function (`sin`, `abs`) of a mixture of those two, so the draw
spans almost nothing regardless of channel count:

| | 4ch | 16ch | 24ch | 128ch |
|---|---|---|---|---|
| gaussian | 4.00 | 16.00 | 23.90 | 120.30 |
| `domain_warp` | **1.00** | 2.22 | 2.06 | **2.37** |
| `temporal_coherent` | **1.00** | 1.00 | 1.00 | **1.00** |
| `curl_noise` | 3.58 | 10.88 | 22.11 | 25.22 |
| `tensor_field` | 3.81 | 14.31 | 23.54 | 90.78 |

`decorrelate_channels` fills the axis from `DECORRELATION_BASIS` independent
renders mixed through a seeded random matrix. Guarded twice, because neither
guess held alone: generators already wider than the basis are skipped
(`curl_noise` looks correlated but spans 25 of 128, so a correlation threshold
was the wrong test), and the remix is kept only when it actually widens the draw
(`curl_noise` at 4ch remixes *narrower*). It can never narrow the noise.

A separate bug fell out: `temporal_coherent` read `params["base_seed"]`
unconditionally where `domain_warp` gates it on `use_temporal_coherence`, so it
ignored its seed argument entirely.

### Left to do

**Fix `_expand_channels` at source.** `decorrelate_channels` works around it from
`core/shader_noise.py`; the generators still produce collapsed output for every
other caller, including legacy. Doing it properly changes legacy output, which
`tests/golden_cases.py` pins deliberately, so it needs the same gating
discussion as item 1.

**Tune `DECORRELATION_BASIS`.** It is 8, chosen to bound cost at LTXV's 128
channels. Nobody has tested whether 4 is as good or 16 better. Cost is roughly
linear: 2-12x the noise-generation time, small against sampling but not free.

**Re-derive the ceilings table with it on.** Every number in finding (1) was
measured with it off, and the knobs interact.

---

## 3. Lean into the name — done

`ShaderNoiseWalk` (`shader_noise_walk.py`) ramps one parameter across a batch in
a single run: `walk_parameter`, `walk_start`, `walk_end`, `walk_steps`. It
subclasses the Direct sampler and derives `INPUT_TYPES` from it, so a future
sampler input cannot silently go missing — there is a test for that. Output is a
batched LATENT for the comparer nodes.

Five H3 runs at 608x352/56 frames took **125s total** with the model resident,
against several minutes for the 21 GB load alone.

Multi-stream latents batch per stream through `cat_nested`. `batch_index` is
dropped — it picks a noise slot for one run and means nothing across several.

It produced the first evidence for the documented phase-shift claim. Distance
from the unshaded baseline across five points:

    shader_strength 0.00 -> 0.30   0.0000 .. 0.1761, monotonic
    phase_shift     0.00 -> 1.50   0.1126 .. 0.1695, spread 3.1x tighter

So phase holds roughly constant distance while rearranging detail, where strength
sweeps distance from zero. Not perfectly flat: distance drifts mildly downward as
phase rises.

### Left to do

`walk_steps` caps at 16 and each point is a full run, so a long walk is slow but
never surprising. Two obvious extensions nobody has needed yet: walking two
parameters as a grid, and emitting the parameter values as a text output for
labelling contact sheets.

---

## 4. Shader noise on the audio stream — done, value unproven

`shade_non_spatial`. Two things, which needed very different work.

**Audio streams needed no new layout code.** H3's audio is `[B, 32, 2, T]` —
already rank 4, read as 32 channels on a 2 x T grid, stereo as height and time as
width. Only the pipeline needed to stop painting `noise_streams[0]` exclusively.

**Sequence latents needed a new path.** `[B, C, L]` is painted as a one-row strip
and folded back, unblocking the five families `require_spatial_latent` refused:
Stable Audio 1/3, ACE-Step 1.5, MiniMax Music 3, Hunyuan3D, TripoSplat.

Measured on H3, 0.05 to 0.40 against a control:

    run            rms dB   centroid Hz   flatness
    audio off       -18.5          4244     0.1044
    on 0.05         -20.4          4618     0.0975
    on 0.10         -21.0          4702     0.0744
    on 0.20         -20.1          3801     0.0943
    on 0.40         -21.9          4730     0.0655

Spectral flatness is 1.0 for white noise and falls toward 0 for tonal content. It
drops in all four runs and roughly tracks strength — the model pushed toward
tonal sound and away from broadband texture. Video stayed clean at every setting
including 0.40.

### Left to do

**This is a hint, not a result.** One prompt, one sample per setting, and
centroid and level are not monotone. It needs several prompts with different
audio character (tonal, percussive, broadband) and repeats per setting before
anyone claims a direction. Nobody has *listened* to the output.

**The sequence-latent path is untested on a real audio model.** It generates
correct shapes and finite unit-variance noise, and the pipeline accepts it, but
no Stable Audio or ACE-Step checkpoint has been run through it.

**TripoSplat's second stream is camera parameters**, not audio. Enabling this
there paints the camera. A per-stream opt-in would be safer than the current
all-or-nothing.

---

## 5. Per-stage shader parameters — done

`stage_progression`: `uniform` (default), `coarse_to_fine`, `fine_to_coarse`.
Coarse-to-fine starts zoomed in on large features with fewer octaves and ends
zoomed out on small ones with more, matching what each part of the trajectory
decides. The span is 0.5x to 2x `noise_scale` and plus or minus one octave,
centred on the widget values.

Position comes from the boundary's place in the schedule, not the stage index:
sequential and injection stages interleave, and what matters is how far along the
trajectory the noise lands.

`_shaped()` copies the params dict rather than adjusting it — one dict is shared
across stages, so mutating it would compound silently. There is a test.

On H3 at strength 0.25 with three stages, all three progressions give clean,
coherent, visibly different neighbours.

### Left to do

**Nothing shows which progression is better**, only that they differ. That needs
a proper comparison across prompts, and probably a human judgement rather than a
metric.

**Only zoom and detail vary.** Per-stage *shader type* was the other half of the
original idea and is not implemented — the event tuple carries a params dict, so
it is a contained change if the abrupt character switch turns out to be useful.

---

## Reproducing the measurements

Scratch workflows and contact sheets live in the session scratchpad, not the
repository.

- Minimal H3 t2v graph: `UNETLoader` -> `MiniMaxH3SigmaShift(6.0, 3.0)`,
  `CLIPLoader(type="minimax")`, two `VAELoader`s (video + audio),
  `MiniMaxH3ImageToVideo` -> `ShaderNoiseKSamplerDirect` -> `VAEDecode` +
  `VAEDecodeAudio` -> `CreateVideo(24fps)` -> `SaveVideo`. `SaveVideo` needs
  `format.codec` as a dynamic-combo key, not `codec`.
- The 32B text encoder and the 21 GB UNet will not fit together on a 31 GB / 12 GB
  box. For sampling-only checks skip the encoder and pass
  `[[torch.zeros(1, 16, 5120), {}]]` as both conditionings — `condition_proj` is
  `[5376, 5120]`, and a missing `text_token_tags` is handled.
- Audio level: `ffmpeg -i X.mp4 -af volumedetect -f null -`. Spectral flatness and
  centroid need a short numpy script over `-f f32le` output.
- SD 1.5 sweeps need no server: `comfy.sd.load_checkpoint_guess_config` then call
  `pipelines.standard.run` directly. Seconds per image.
- Restarting ComfyUI: kill by exact pid from
  `ps -eo pid,cmd | awk '$2 ~ /venv\\/bin\\/python$/ && $3 ~ /main\\.py$/'`.
  A `pkill -f` pattern matching "main.py" also matches the shell running it.

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
heuristic. Easy cleanup.

---

## One live environment note

The Direct node's shader preview (`web/shader_renderer.js`) has GLSL for only
`domain_warp`, `tensor_field` and `curl_noise`. Selecting `temporal_coherent`
leaves the preview on its previous pattern and logs `Shader source not found` —
sampling is unaffected and the tooltip says so. A fourth GLSL preview would close
the gap.
