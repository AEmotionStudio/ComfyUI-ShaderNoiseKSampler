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

Re-derived on H3 with `decorrelate_channels` on at the current basis of 64, which
**inverts the ordering**:

| Generator | Ceiling without | With |
|---|---|---|
| `domain_warp` | ~0.25 | **~0.75** (degrades at 1.0) |
| `temporal_coherent` | ~0.35 | **~0.75** |
| `tensor_field` | ~0.5 | ~0.5 — unchanged, decorrelation skips it |
| `curl_noise` | ~0.2 | ~0.2 — unchanged, skipped |

The two generators decorrelation fixes now beat the two it skips, reversing the
earlier finding that `tensor_field` was the most tolerant: rank was what limited
`domain_warp`, and nothing else was. The generators that already span their
channels are limited by their spatial character instead, which decorrelation
does not touch.

That sweep ran with zero conditioning (no text encoder resident), so the content
comes from the model's prior rather than a prompt. Coherence is still
unambiguous, and the control in the same batch — `domain_warp` at 0.75 with
decorrelation off — reproduced the same green quilt seen earlier under real
prompts, so the comparison holds.

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

~~Tune `DECORRELATION_BASIS`.~~ Done (`7661d7c`): it shipped at 8 on a guess and
is now 64. On H3 at strength 0.75, basis 8 is still mostly destroyed, 16 is
coherent and 32 is clean. The cost the old value guarded against did not exist —
worst measured case is about a second per draw, one draw per stage boundary, on
runs of thirty to fifty seconds.

~~Re-derive the ceilings table with it on.~~ Done — see finding (1).

**Rank still tops out around 60% of channels**, because the basis draws are not
fully independent of each other either. Mixing genuinely orthogonal fields rather
than random combinations of correlated ones would close the rest of the gap.

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

**The rank-3 sequence-latent path is untested.** Note this is narrower than it
first appears: MiniMax H3 *is* an audio model — it generates music, speech and
sound jointly with video — so the flatness measurement above is a real
audio-model result, not a proxy. What has never run is the `[B, C, L]` shape
that Stable Audio and ACE-Step use, because no such checkpoint is installed. It
generates correct shapes and finite unit-variance noise and the pipeline accepts
it; nothing more is known.

~~TripoSplat's second stream is camera parameters.~~ Handled (`c938137`):
streams carrying fewer than 64 cells per batch item are skipped, which catches
the `[B, 1, 5]` camera while leaving H3's 414-cell audio painted.

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

## Next up: the agreed upgrade (decided, not yet started)

**Legacy compatibility is no longer a constraint.** The pack is upgrading; pre-2.0
workflows changing output is accepted. That supersedes commit `1b21bcb`
("mark the legacy pipeline as frozen"), the byte-exact promise in
`tests/golden_cases.py`, and the auto-switch in
`web/sampling_mode_migration.js:42`. Nothing below was blocked on anything but
that decision.

### What to do

1. **Fix `shaders/base.py::expand_channels` at source.** It builds every channel
   past the first one or two as a pointwise function (`sin`, `abs`) of a mixture
   of those two. `decorrelate_channels` currently compensates downstream from
   `core/shader_noise.py`; fixing the generator removes the need for the
   workaround and makes `domain_warp` mean one thing everywhere instead of two
   depending on sampling mode.
2. **Flip `decorrelate_channels` and `normalize_strength` to default on.** Both
   ship off purely to protect saved workflows. The measured case for each is in
   items 1 and 2 above.
3. **Re-purpose the golden suite as regression pins for the *standard* pipeline.**
   Re-capture from current code and repoint the cases away from legacy. The
   pre-2.0 reference is given up deliberately; what is kept is a fast bit-exact
   net against accidental future change, which is the only such net in the
   project.

### Facts already established, so this does not need re-deriving

- **Only `domain_warp` calls `expand_channels`.** `tensor_field`, `curl_noise`
  and `temporal_coherent` build their channels by other means and are untouched
  by a fix there.
- **Exactly one of the eleven goldens changes: `video_nested`.** The eight image
  cases call `expand_channels` but it returns early — at 4 channels there is
  nothing to grow — so they stay byte-identical. `image_styled`, `image_batch`,
  `video_curl` and `video_temporal` use other generators and never call it.
- **The suite badly under-covers the change.** One golden moves, but the affected
  population is every legacy workflow using `domain_warp` on a latent with five
  or more channels: Flux, SD3, WAN, Hunyuan, H3, LTXV. Only SD 1.5 and SDXL, at
  four channels, are unaffected. Do not read a small golden diff as a small
  change.
- **Rank still tops out near 60% of channels** even at basis 64, because the
  basis draws are not independent of each other either. Mixing genuinely
  orthogonal fields rather than random combinations of correlated ones would
  close the rest of the gap, and belongs with step 1.

### Why, in one paragraph

The rank finding is not a performance issue, it is a question of what the tool
is. If the shader spans one channel of four, what reaches the latent is not a
navigational field, it is a single pattern stamped across every channel at once.
Below about 0.25 that reads as a nudge and the node works as the README
describes. Above it the destination stops depending on where you started —
which is why 0.5 and 0.75 produced the same green quilt whatever the prompt. In
the README's own terms, rank-collapsed noise did not give you territory. It gave
you a different lottery with a strong house bias. Decorrelation roughly triples
the range over which the shader steers instead of overwrites.

### Three consequences to plan for

**Step 2 is the one that delivers.** Steps 1 and 3 are cleanup around it. Both
fixes currently default to off, so a user installing today gets rank-1 noise and
a strength dial that means eight different things depending on blend mode. The
project already knows better and does not act on it, which is further from the
stated goal than before the fixes existed.

**The migration becomes a lie.** `web/sampling_mode_migration.js` silently routes
any pre-2.0 workflow into `legacy`, and the 2.0.0 changelog gives the reason as
"so their seeds keep reproducing". Fix `expand_channels` and that migration still
fires but no longer delivers what it exists for. Either drop it, or redefine
`legacy` explicitly as "the old pipeline *structure*" rather than "the old
output", and say so in the changelog.

**A golden failure changes meaning.** Today it says "you broke backwards
compatibility". Afterwards it says "you changed the sampler". Both are worth
having and they are not the same signal. Re-capturing also gives up the only
bit-exact record of pre-2.0 behaviour: after it, reproducing the old output
needs a `git checkout`, not a test fixture.

### Watch the widget count

The Direct node now carries 25 required and 8 optional inputs. Each was
individually justified; the trend is still real. The README sells a compass and
the panel increasingly sells expertise. Before adding the next toggle, consider
whether presets over the existing knobs would serve better than another knob —
`stage_progression` is already shaped that way and is the pattern to copy.

### Then optionally

Removing legacy mode altogether — the pipeline, the `sampling_mode` input, the
migration JS, and the uncalled `core/sampler.py`, `core/blending.py` and
`core/transforms.py` — was considered and deliberately left out of the scope
above. It is a large deletion and a separate decision.

---

## Built: the collapse kept as a travel mode (`5434791`)

Do not simply delete the rank-1 behaviour when fixing `expand_channels`. It is a
second navigational primitive, and the project already speaks in travel
metaphors — vehicle, map, compass, driving between towns.

|  | rank-collapsed | decorrelated |
|---|---|---|
| what sets the destination | **shader parameters** | the seed |
| role of the seed | fades as strength rises | anchors throughout |
| coherent range | narrow, about 0.25 | wide, about 0.75 |
| push per unit strength | strong | gentle |

The two are a trade, not a ranking: collapse buys a harder push per unit of
strength at the cost of a narrower range before the picture stops being a
picture.

**It is controllable, which is the bar for calling it travel rather than
breakage.** At strength 0.6 the shape masks rendered `spiral`, `hexgrid`, `rays`
and `vignette` as four clearly distinct, recognisable images, in the prompt's own
palette. Shader parameters map to reproducible, meaningfully different
destinations. What they do *not* map to is scenes — in this regime the model is
being fed out-of-distribution input, so the destinations are texture and pattern
fields. That is a real limit, not a detail: this is not "jump to another town",
it is "jump somewhere that is not quite a town".

### As built

`travel_mode` replaces the `decorrelate_channels` boolean, over the same
`DECORRELATION_BASIS`: `walk` 64, `drift` 4, `jump` 1. Measured rank for
`domain_warp` on H3's 24 channels: native 2.06, jump 1.00, drift 3.73, walk
14.36.

Two things it needed. `jump` bypasses the widening guards, which exist precisely
to stop a remix narrowing the noise. And it forces rank 1 for *every* generator,
including `tensor_field` at its native 90 of 128 — otherwise `shader_type` would
not be a usable coordinate in jump-space.

`walk` is the default, so the standard pipeline now decorrelates unless told
otherwise. Legacy never reaches this code, so the golden suite is untouched.

On real H3, `jump` produces an orange-and-black texture field on the forge
prompt's own palette, and `stamp` at 0.90 with `hexgrid` draws an unmistakable
grid of glowing cells. Both behave as described.

### Still worth testing

- **Is `jump` reproducible across models?** The one that matters. If the shader
  sets the destination and the seed fades, the same parameters should land
  somewhere recognisably similar on SD 1.5 and on H3. If so it is a
  prompt-independent, model-independent coordinate system — a genuinely new
  thing. If not, a per-model curiosity. Cheap to answer: same parameters, `jump`,
  both models, compare.
- Does `drift` produce anything the other two do not, or is the behaviour
  bimodal? If bimodal, ship two modes rather than three.
- Do `jump` destinations stay distinct across shader *types* and `noise_scale`?
  Shape masks vary the destination clearly; the other axes are untested.

## Presets (`5434791`)

Seven bundles over the six settings that only mean anything together:
`custom`, `nudge`, `explore`, `roam`, `video`, `jump`, `stamp`. `apply_preset`
takes an `exclude` set and the Walk node passes the parameter it ramps, so a
preset pinning `shader_strength` cannot flatten a strength ramp.

**Calibrate preset values against a real prompt at a real working resolution.**
`roam` first shipped at 0.60, taken from a sweep with zero conditioning at
448x256 where `domain_warp` held to 0.75. Under an actual prompt at 608x352 it
showed colour bands at 0.55, and the value had to come down to 0.45. `stamp`
likewise needed 0.90 rather than 0.70, and `hexgrid` rather than `spiral` — at
0.70 a spiral mask just reads as a stylised subject, not as the mask.

### Left to do

**The preset overrides at execution time, so the widgets lie.** Select
`explore` and the `shader_strength` widget still reads whatever it read before,
while the run uses 0.30. The honest fix is to set the widgets from JS on
selection, the way `shader_renderer.js` already mirrors inputs with
`syncFromInputs`. Until then the tooltip names exactly which inputs a preset
takes over.

**Widget ordering.** `preset` is appended at the end of the optional block
because ComfyUI maps saved values by position. It is the front door and reads
last. A major version should move it to the top of the required block.

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
