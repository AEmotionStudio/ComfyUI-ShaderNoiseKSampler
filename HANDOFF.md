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
| + | Travel modes | Done (`5434791`) — `walk` / `drift` / `jump` |
| + | Presets | Done (`5434791`, tuned `ce4b1ec`) — seven bundles |
| + | The agreed upgrade | Done — generators fill their own channels, `normalize_strength` on, goldens on the standard pipeline, presets write the widgets |
| + | Blend keeps the base noise's statistics | Done (`f44aeb5`) — the road away from strength 0 starts at the seed's own image |
| → | Next up | Not started — see "Next up: what the measurements suggest" |

Most new capabilities are optional inputs defaulting to off. **Two are not:**
`travel_mode` defaults to `walk` and `normalize_strength` defaults on. Both are
part of the upgrade described below, which also changed the generators' own
output — for legacy too — and re-pointed the golden suite at the standard
pipeline. The last commit with the old behaviour is tagged `pre-collapse-fix`.

---

## Next up: what the measurements suggest

Written after the blend measurements further down ("Measuring the blend, not the
ceiling" and the near-end fix). These are interpretations of those results, marked by
how well they are supported, followed by what to try. None of it is started.

### How it works, in one pass

1. The seed makes the base Gaussian noise (`comfy.sample.prepare_noise`) and seeds the
   shader field, `seed + stage index` for each stage.
2. The generator renders a smooth spatial field for every latent channel, each its own
   draw (`BaseNoiseGenerator.fill_channels`).
3. `mix_noise` blends it into the base and gives the result the base's own statistics.
   Even `multiply` at strength 1.0 hands over only about two-thirds shader; the seed
   never leaves.
4. The sampler denoises from there, and the leftover noise at each stage boundary is
   blended again.

Strength moves the starting noise along a line from the seed toward seed-plus-shader;
the shader settings change the direction of that line.

### What is probably going on

**1. The model reads the noise's large-scale structure, and the shader is mostly
large-scale structure.** Likely; not measured directly. Gaussian noise has equal energy
at every spatial scale, while a shader field is smooth. Diffusion settles composition
and colour layout early, from the large-scale content, and removes fine structure. It
fits three results. `noise_scale` was the biggest lever: at 0.50, 0.5 turned every
SD 1.5 seed abstract and 2.0 gave clean portraits that still carried the field (overlap
0.38). The old rescale only shifted each channel's mean, the largest-scale component
there is, and that turned grayscale portraits into colour; SD 1.5's brightness and
colour are known to follow the noise's per-channel mean. And at high strength SD 1.5
lost the prompt once the shader owned the layout.

This is not the metric that finding (4) near the top rejected. That asked whether
spectral energy predicts how much shader a picture can hide; this asks which scales of
the shader reach the result.

**2. The path from noise to image is lumpy.** The behaviour is measured; the mechanism
is the usual one for diffusion samplers. The sampler makes early, discrete decisions
(pose, identity, framing), so a small change stays inside one outcome until it tips into
the next. SD 1.5 seed 1234 held one portrait from 0.05 to 0.25, seed 4242 changed man
at every step, and H3 moved visibly on a 0.06% change to its noise. Strength moves the
noise continuously; the images come in plateaus and jumps.

**3. How the channels relate sets the palette.** Likely, from the contact sheets.
`walk` gives every channel an independent field, so a strong result is full-colour.
`jump` gives every channel the same field, sign-flipped per channel, so colour collapses
onto one axis: the flat two- and three-colour shapes.

**4. H3 carries the shader weakly partly because each video frame gets an unrelated
pattern.** The code fact is certain; the size of its effect is untested.
`core/shader_noise.py::generate` uses `frame_seed = seed if temporal_coherence else seed
+ index`, so by default every latent frame draws a fresh field. A video model treats
structure that changes randomly between frames as noise and smooths it away. It fits the
older finding that holding one pattern across frames swamped H3 at 0.5. It also means
`temporal_coherent`, built with time as a real axis, is reseeded every frame unless
`use_temporal_coherence` is on, which defeats its design. H3's heavy conditioning and
8-step distilled sampling probably contribute; that part is a guess.

### What to try, in order

1. **Test the scale hypothesis on data already saved.** Split the saved latents and the
   reconstructed fields from `verification/blend/` into spatial-frequency bands and
   measure the overlap in each band. No renders needed. If the shader's influence sits
   in the large-scale bands, (1) holds and shapes everything below.
2. **Hold the shader pattern across video frames by default**, letting `time` evolve
   instead of reseeding, at least for `temporal_coherent`. Measure it on H3 with
   `drive.py` and `analyze.py`: overlap, contact sheets, and how much consecutive frames
   differ (flicker), which the tools do not measure yet.
3. **A scale control**: send the shader into the large scales only (layout, colour) or
   the fine scales only (texture). Better as a preset dimension or a clearer meaning for
   `noise_scale` than as another widget.
4. **A separate shader seed**: a new street map for the same town. Today, redrawing the
   pattern without moving the base noise takes `phase_shift` or `noise_scale`, which
   also change its character. Costs a widget; it could be offered on the Walk node only.
5. **Let the Walk node map the lumps.** Where neighbouring strengths give very different
   images, subdivide the step until the flip is found. That turns the hops into a map of
   each seed's plateaus, the "latent cartography" the README's roadmap promises.
6. **Finer control at the near end.** On SD 1.5 the whole close neighbourhood lies
   between 0 and about 0.25; a curved strength scale would give it more of the slider.

Housekeeping, from the upgrade's "Left to do": decide what `legacy` means and fix its
tooltip, make the golden tests' stub sampler call the progress callback so stage
boundaries are pinned, and re-tune the presets against the new noise.

### How much weight this carries

One prompt per model, three or four seeds, one stage, `domain_warp` only, and one middle
frame per H3 clip. Treat the explanations as well-supported hypotheses; items 1 and 2 are
the cheapest ways to confirm or break them.

---

## Findings that shape everything else

**1. The strength ceiling is set by the channel axis**, not by latent size or
step count. See item 2. Ceilings measured on H3 (608x352, 8 steps,
`res_multistep`/`simple`, cfg 1.0), **all at the noise the generators natively
produce** — what `travel_mode` now calls `jump`-ward, before any widening:

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

Re-derived on H3 at `travel_mode: walk` (basis 64), which **inverts the
ordering**:

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

Under a real prompt at 608x352 that same `walk` held only to about 0.5, so the
ceilings above are optimistic. The upgrade section below has real-prompt numbers
from before and after the generators were fixed.

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

~~Consider making it the default~~ Done in the upgrade below.

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

Widening fills the axis from `DECORRELATION_BASIS` independent renders mixed
through a seeded random matrix; `travel_mode` selects how many. Guarded twice, because neither
guess held alone: generators already wider than the basis are skipped
(`curl_noise` looks correlated but spans 25 of 128, so a correlation threshold
was the wrong test), and the remix is kept only when it actually widens the draw
(`curl_noise` at 4ch remixes *narrower*). It can never narrow the noise.

A separate bug fell out: `temporal_coherent` read `params["base_seed"]`
unconditionally where `domain_warp` gates it on `use_temporal_coherence`, so it
ignored its seed argument entirely.

### Left to do

~~Fix `_expand_channels` at source.~~ Done in the upgrade below — and it was
not only `expand_channels`.

~~Tune `DECORRELATION_BASIS`.~~ Done (`7661d7c`): it shipped at 8 on a guess and
is now 64. On H3 at strength 0.75, basis 8 is still mostly destroyed, 16 is
coherent and 32 is clean. The cost the old value guarded against did not exist —
worst measured case is about a second per draw, one draw per stage boundary, on
runs of thirty to fifty seconds.

~~Re-derive the ceilings table with it on.~~ Done — see finding (1).

~~Rank still tops out around 60% of channels, because the basis draws are not
independent.~~ Wrong diagnosis: 0.6 is what random mixing yields. Rendering one
field per channel reaches 22.7 of 24. See the upgrade below.

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

## The agreed upgrade — done

Legacy compatibility stopped being a constraint, so the three agreed steps were
carried out, plus the preset-to-widget sync. The `pre-collapse-fix` tag marks the
last commit with the old behaviour; reproducing pre-upgrade output now needs a
checkout of it, not a test fixture.

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

### What the plan got wrong

- **The collapse had three sources, not one.** `expand_channels` returned early at
  four channels, so fixing it alone would have changed nothing for SD 1.5 or SDXL:
  the rank 1.00 there came from `domain_warp.py`'s `repeat(1, 4, 1, 1)`.
  `temporal_coherent` broadcast one field to every channel with `.expand()` — rank
  1.00 at every count, and it is the shader the `video` preset picks. `curl_noise`
  padded its colour path with copies.
- **Seven goldens moved, not one:** every `domain_warp` case, plus `video_curl`.
  `image_batch`, `image_styled`, `video_temporal` and `image_zero_strength` held
  byte-identical, which is how the fix was checked before re-capturing.
- **"Rank tops out near 60% because the basis draws are not independent" was the
  wrong diagnosis.** One render per channel reaches 22.7 of 24. The 0.6 is what a
  random mixing matrix yields: 0.52 to 0.64 of the basis, measured across all four
  generators at 16, 24 and 128 channels.
- **`drift` was already broken for the wide generators.** `tensor_field` and
  `curl_noise` spanned enough that the widening guard returned their noise
  untouched, so `drift` silently equalled `walk` for them. Once every generator is
  wide it would have done so for all four.

### As built

1. **`BaseNoiseGenerator.fill_channels`** (`shaders/base.py`) replaces
   `expand_channels`. Each channel up to `CHANNEL_BASIS` (64) is its own render at
   `seed + 6151 * c`; channels past that are QR-orthogonalised mixtures of those
   renders. Channel 0 is the generator's own draw, so a one-channel request is
   unchanged. That keeps `jump` byte-identical to the tag for all four generators
   (verified), and with it the H3-calibrated `jump` and `stamp` presets. The extra
   renders run inside a forked RNG. 6151 was checked against the mod-10000 seed
   hashing inside `curl_noise`: none of the first 64 channels collide.
   `domain_warp`, `temporal_coherent` and `curl_noise` call it. `tensor_field`
   already rendered per channel and is untouched; `fast_high_channel_noise` still
   tiles, on purpose.

   | effective rank | before | after |
   |---|---|---|
   | `domain_warp` at 4 / 24 / 128 channels | 1.00 / 2.13 / 2.43 | 3.91 / 22.72 / 69.56 |
   | `temporal_coherent` at 4 / 24 / 128 | 1.00 / 1.00 / 1.00 | 3.93 / 22.67 / 66.08 |
   | `curl_noise` at 128 | 25.07 | 59.23 |

   Per-channel rendering was chosen over remixing a basis, although remixing was
   already in the code, for two reasons. It is cheaper at H3's 24 channels: 24
   renders instead of 64. And it keeps each channel a real shader field; a sum of
   many independent fields drifts toward Gaussian, so some of what remixing bought
   was the shader being washed out.

2. **The travel-mode guard is direction-aware** (`core/shader_noise.py::_maybe_decorrelate`).
   A basis below the widest (`drift`, `jump`) narrows unconditionally. Widening is
   skipped once the draw exceeds `_MIX_RANK_YIELD` (0.65, just above the best remix
   measured), so `walk` hands the generator's noise straight through instead of
   rendering 64 more draws and discarding them. On a four-channel latent `drift`
   and `walk` are the same by construction.
3. **`normalize_strength` defaults on.** `multiply`, the default blend mode, is the
   calibration reference and is unaffected.
4. **The goldens are pinned to the standard pipeline.** `test_legacy_golden.py`
   became `test_golden.py`, and every input only the standard pipeline reads is set
   explicitly in `NODE_DEFAULTS`, so a changed default shows up as an edit.
5. **Choosing a preset writes the widgets** (`web/src/preset_widgets.ts`). The
   table comes from `GET /shader_noise_ksampler/presets`, so it has one source.
   Editing a controlled widget to another value drops the preset back to `custom`.
   A saved workflow is never rewritten on load. The Python override stays the
   authority, because API submissions and the Walk node never run the JS.

`test_blend_calibration_is_current` no longer compares `difference` on an absolute
tolerance. Its whole curve is below the tolerance, and after the fix it sat 0.002
from failing by luck. Every other mode stays within 0.049 of the table, so the table
itself was not re-measured.

### Measured on real weights

These are ceilings: how much shader the picture can hide. That turned out not to
be what the project is for. "Measuring the blend, not the ceiling" below measures
what is.

**SD 1.5** at 512x512, 20 steps, cfg 7, `euler`/`normal`, seed 8888. `domain_warp`
on `walk`, before and after, same two prompts:

| | portrait holds to | landscape holds to |
|---|---|---|
| before | ~0.30 | ~0.30 |
| after | ~0.35 | ~0.45 |

A modest gain, not the large one predicted for four-channel models. Two other
changes are visible. Small strength steps now make small moves: after the fix, 0.25
and 0.30 are near-identical neighbours, where before 0.30 had already re-composed
the frame. And the failure mode changed from a red, black and cyan field to a
full-colour one, since all four channels now carry structure. `curl_noise` at four
channels is pixel-identical before and after, as it must be.

Those ceilings are judgements made by looking at one image per strength. One
exactly-defined number backs them. The mean pixel distance between the portrait
and the landscape at the same strength (64x64, RGB, 0 to 1) measures how much the
prompt still matters: 0.24 at strength 0, falling toward 0 once both prompts
become the same pattern.

| strength | 0.30 | 0.40 | 0.45 | 0.50 | 0.75 |
|---|---|---|---|---|---|
| before | 0.20 | 0.14 | 0.12 | 0.10 | 0.03 |
| after | 0.24 | 0.20 | 0.22 | 0.18 | 0.05 |

So the prompt keeps mattering further up the range, and both versions stop being
about the prompt by 0.75. Step-to-step distance agrees in the usable region
(0.25 to 0.30: 0.19 before, 0.09 after). Distance from the strength-0 image does
not separate the two versions at all, because it cannot tell a coherent new
picture from a broken one.

**MiniMax H3** under a real prompt: the forge scene, which is dark and so lets
structure survive. 608x352, 56 frames, 8 steps, `res_multistep`/`simple`, cfg 1.0,
two stages, seed 8888, `walk`. The before runs went through the ComfyUI server
while it still held the tagged code; the after runs went through the same server,
restarted. The strength-0 frame is pixel-identical between the two.

| | 0.50 | 0.75 | 1.00 |
|---|---|---|---|
| `domain_warp` before | clean | purple cast, striping, re-composed | rainbow field |
| `domain_warp` after | clean | **clean**, faint tint in one corner | degraded, subject still there |
| `temporal_coherent` before | clean | rainbow dot grid | rainbow dot grid |
| `temporal_coherent` after | clean, re-composed | rainbow dot grid | rainbow dot grid |

`domain_warp`'s ceiling moved from about 0.5 to about 0.75, and the seed keeps
anchoring the scene further up the range. `temporal_coherent` did not move: it
fails into a dot grid, which is the generator's spatial character and not
something channel width touches. The `video` preset's 0.35 sits below both.

That H3 ceiling rests on looking at the middle frame of each clip, from one
prompt and one seed, on a 0.25 grid. No metric backs it: distance from the
strength-0 frame is the same before and after for `domain_warp` (0.19 at 0.75 in
both), because what changed is whether the destination is still a picture, not
how far away it is. Temporal behaviour across frames was not assessed.

**Audio level is an unreliable ceiling signal here.** After the fix,
`temporal_coherent` rose steadily with strength (-24.3 to -15.6 dB), but
`domain_warp` did not track strength in either version, although its picture
changed the most. Judge ceilings from frames.

With `normalize_strength` on, all eight blend modes at 0.30 stay the same coherent
scene, but they are not the same image. Each sits 0.07 to 0.09 from the `multiply`
frame, about as far as `multiply` itself sits from strength 0 (0.08); `difference`,
which saturates, re-composes (0.17). There was no run with normalisation off, so
this shows only that no mode is over-driven at 0.30, not that the calibration works
on H3. The model-free table in item 1 is still the evidence for that. The audio
spread across the eight is 3.4 dB.

### Left to do

- ~~CHANGELOG~~ Done: an Unreleased section covers everything since 2.1.0.
- **The migration is now a lie.** `web/sampling_mode_migration.js` still routes
  pre-2.0 workflows to `legacy` "so their seeds keep reproducing", but `legacy`
  shares the generators, so its output changed too. Drop the migration, or redefine
  `legacy` as the old pipeline *structure*. The changelog lists it as a known issue;
  the `sampling_mode` tooltip still makes the old claim.
- **Re-calibrate the presets** against real prompts at a working resolution: the
  noise they were fitted to has changed. `jump` and `stamp` are exempt, being
  byte-identical.
- **Widget ordering.** `preset` still reads last; a major version should move it
  to the top of the required block.
- Removing legacy mode altogether remains a separate decision.
- **The golden suite never injects shader noise at a stage boundary.** Its stub
  sampler (`tests/helpers.py::recorded_sampling`) never calls the progress callback,
  and the pipeline applies boundary events only when it does, so the multi-stage
  goldens pin the first stage alone. `test_standard_pipeline.py` does cover
  boundaries.

### Watch the widget count

The Direct node now carries 25 required and 8 optional inputs. Each was
individually justified; the trend is still real. The README sells a compass and
the panel increasingly sells expertise. Before adding the next toggle, consider
whether presets over the existing knobs would serve better than another knob —
`stage_progression` is already shaped that way and is the pattern to copy.

---

## Measuring the blend, not the ceiling

The ceilings above answer "how much shader can the picture hide", which is not what
the project is for. The intent, as the author put it: a legitimate blend of the
seed's noise and shader noise, to create new visuals or to see how the shader alters
them. A seed is a town. Holding it fixed parks the car there, and the shader
settings explore its streets, near and far. How that looks is expected to differ
between models and between seeds, so nothing here compares models.

In the code, both halves of the town come from the seed: the base noise from
`comfy.sample.prepare_noise(seed)`, and the shader field from `seed + stage`.
Changing the seed moves both. Holding it fixes both, and the shader settings then
re-render only the field.

### Method

Everything is measured on the final latent, per model, over several seeds. Image
distance (mean absolute RGB difference at 64x64) is reported alongside, as a check on
whether latent distance means anything to the eye.

- **Town**: the mean distance between two seeds' strength-0 results. Every other
  distance is read in towns.
- **Far**: distance from your own seed's strength-0 result.
- **Home**: that distance over the mean distance to the other seeds' strength-0
  results. Below 1, you are nearer your own town than anyone else's.
- **Streets**: at one seed and strength, the distance between runs that differ only
  in `phase_shift` or `noise_scale`.
- **Overlap**: the correlation between a latent change and the shader field that
  caused it, against the same correlation with other seeds' fields, which is chance.
  The field is reconstructed by running the Direct node on CPU with a stub sampler
  and capturing what `core.shader_noise.generate` returned; that matches the CUDA
  render to 2.5e-6. Only the outermost `generate()` call counts: a travel mode that
  remixes calls it again for its one-channel basis draws, and recording one of those
  instead produced wrong `jump` numbers in a first pass, since corrected.

Every run is one stage: `domain_warp`, `multiply`, `normalize_strength` on, octaves
2, warp 0.7. Everything above "Fixed: the blend keeps the base noise's statistics"
was measured before that fix, which left SD 1.5 essentially unchanged from 0.25 up
and brought H3 nearer its seed's image across most of the range.

### SD 1.5

512x512, 20 steps, cfg 7, `euler`/`normal`, the fisherman portrait prompt, four seeds.

| | far (latent / image) | home (latent / image) | overlap: own field vs others' (max) |
|---|---|---|---|
| `walk` 0.25 | 0.77 / 0.73 | 0.71 / 0.65 | 0.28 vs 0.06 |
| `walk` 0.50 | 1.21 / 1.08 | 0.89 / 0.85 | 0.50 vs 0.07 |
| `walk` 0.75 | 1.73 / 1.26 | 0.95 / 0.94 | 0.69 vs 0.05 |
| `walk` 1.00 | 2.15 / 1.29 | 0.97 / 0.96 | 0.75 vs 0.05 |
| `jump` 0.50 | 1.52 / 1.30 | 0.94 / 0.97 | 0.72 vs 0.09 |

- **The shader is literally in the result**, in proportion to strength: the change in
  the final latent tracks its own field and no other.
- **Parking works.** In latent space every seed stayed nearest its own town at every
  strength. In image space that holds through 0.75; at 1.0, for one seed in four.
- **Close is not very close here.** 0.25 is already about 0.7 of a town away: a
  different fisherman, the same kind of picture. For what lies below 0.25, see "The
  near end of the road".
- **To the eye**: at 0.25 every seed is a photographic portrait, re-composed. At 0.50
  three of four are a genuine blend, the portrait painted through the shader's colour
  structure. At 0.75 and 1.0 the prompt is gone and each seed is its own reproducible
  shader field.
- **`jump`** gives flat two- and three-colour graphic shapes, and its latent change
  tracks its own field more strongly than `walk` does at the same strength (0.72
  against 0.50).

Streets at a parked seed, strength 0.50, in image towns: `phase_shift` 0.0 and 1.0
move 0.68 and 0.69; `noise_scale` 0.5 moves 1.36 and 2.0 moves 1.16. At 0.25, 0.50
and 0.75 alike, every street change tracks the change in the field that caused it
(0.19 to 0.78, against at most 0.12 for other seeds' fields), so the settings are
real, controllable levers.

`noise_scale` is the biggest of them, and it trades the shader's imprint against
staying home (overlap here pooled 4x):

| `noise_scale` at 0.50 | overlap: own vs others' (max) | home (image) | nearest own town |
|---|---|---|---|
| 0.5 (large features) | 0.77 vs 0.13 | 0.93 | 3/4 |
| 1.0 | 0.58 vs 0.08 | 0.85 | 4/4 |
| 2.0 (small features) | 0.38 vs 0.10 | 0.76 | 4/4 |

At 0.5 every seed turns abstract. At 2.0 every seed is a clean photographic portrait
again, still carrying its own field in the latent: the model absorbs fine structure
into the picture instead of drawing it.

### MiniMax H3

608x352, 56 frames, 8 steps, `res_multistep`/`simple`, cfg 1.0, the forge prompt,
three seeds; image measures use the middle frame.

| | far (latent / image) | home (latent / image) | overlap: own field vs others' (max) |
|---|---|---|---|
| `walk` 0.25 | 0.69 / 0.66 | 0.72 / 0.65 | 0.05 vs 0.01 |
| `walk` 0.50 | 0.86 / 0.87 | 0.83 / 0.78 | 0.08 vs 0.02 |
| `walk` 0.75 | 0.99 / 1.27 | 0.89 / 0.90 | 0.15 vs 0.03 |
| `walk` 1.00 | 1.24 / 1.77 | 0.96 / 0.98 | 0.26 vs 0.03 |
| `jump` 0.50 | 1.30 / 1.72 | 0.96 / 0.97 | 0.29 vs 0.02 |

- **The shader is in the result, but H3 transforms it far more than SD 1.5 does.**
  The direct trace of the field runs 0.05 to 0.26, against 0.28 to 0.75 there. It is
  above chance at every strength.
- **Parking works, in a tighter neighbourhood.** In latent space every seed stayed
  nearest its own town at every `walk` strength; in image space all three did at 0.25
  and two of three above that. The three seeds' forge scenes look alike (image town
  0.16, against 0.32 on SD 1.5), so image distance separates them weakly. By eye,
  each seed's scene is recognisable at every `walk` strength.
- **To the eye**: 0.25 and 0.50 are close neighbours of the seed's scene, the same
  anvil and fire with hands and tools moved. At 0.75 the shader's colour starts to
  show. At 1.00 it is a genuine blend: the seed's horseshoe and anvil with rainbow
  bokeh, dot grids and colour fields through them.
- **`jump`**: two of three seeds become new graphic visuals with the horseshoe inside
  them, glossy cyan bubbles on a textured red field and a navy and beige blob
  pattern; the third stays close to its scene. Its latent change carries its own
  field far more than `walk` does at the same strength (0.29 against 0.08).

Streets at 0.50, in image towns: `phase_shift` 0.0 and 1.0 move 0.68 and 0.85;
`noise_scale` 0.5 moves 1.24 and 2.0 moves 0.98. All of them stay the same seed's
forge scene, re-composed, with no visible shader pattern at this strength. Each
tracks its own field change (0.05 to 0.14, against at most 0.03 for other seeds').

`noise_scale` sets the imprint here as on SD 1.5 (0.16, 0.10 and 0.07 at 0.5, 1.0 and
2.0, pooled, against at most 0.03 to 0.04), but does not trade it cleanly against
staying home (image home 0.90, 0.78, 0.89).

### The near end of the road

Strengths 0.001, 0.05, 0.10, 0.15 and 0.20 on both models, with streets at 0.10. Each
step is the distance between neighbouring strengths, averaged over seeds, in image
towns; latent distances agree. Measured before the fix below.

| step | SD 1.5 | H3 |
|---|---|---|
| 0 to 0.001 | 0.32 | 0.48 |
| 0.001 to 0.05 | 0.30 | 0.41 |
| 0.05 to 0.10 | 0.27 | 0.47 |
| 0.10 to 0.15 | 0.30 | 0.40 |
| 0.15 to 0.20 | 0.28 | 0.28 |
| 0.20 to 0.25 | 0.24 | 0.40 |

- **Strength 0 was not the start of the road.** At exactly 0 the base noise reaches the
  sampler untouched; above 0, `mix_noise` rescaled every channel of it to mean 0 and
  deviation 1 before blending. At 0.001 that rescale was the whole change to the
  starting noise (on SD 1.5 the shader's part is 0.08, against 1.7 to 4.2 for the
  rescale), the result carried no trace of the field, and still the image moved as far
  as an ordinary step or further. Fixed below.
- **After that the road moves in hops, not a glide.** Each 0.05 moves a quarter to half
  a town on average, and per seed it alternates between near-identical runs and new
  pictures. SD 1.5 seed 1234 holds the same portrait from 0.05 to 0.25 (steps 0.11 to
  0.15), while seed 4242 changes man at every step from 0.10. That is the sampler's
  sensitivity to its starting noise, not something the shader adds.
- **Towns are not the same size on every model.** H3's steps look larger in towns, but
  its seeds render similar forge scenes (image town 0.16, against 0.32), and in
  absolute pixel difference its steps are no larger than SD 1.5's. By eye H3 keeps each
  seed's scene through 0.50 with hands, tools and horseshoe moved; SD 1.5 changes the
  man.
- **The field becomes detectable** from 0.05 on SD 1.5 (0.09 against 0.04 by chance)
  and from 0.10 on H3 (0.020 against 0.004). At 0.05 on H3 it is within chance.
- **Streets at 0.10** move 0.28 to 0.59 image towns on SD 1.5 and 0.37 to 0.73 on H3,
  `phase_shift` 1.0 the least on both.

### Fixed: the blend keeps the base noise's statistics (`f44aeb5`)

`mix_noise` now gives its result the base noise's own per-channel mean and deviation
instead of exactly 0 and 1. Strength 0 is still bit-identical, checked on every seed of
both models, and as strength approaches 0 the result approaches the base. Both models
were re-rendered from 0 to 0.50 with streets at 0.10, and compared with the runs above.

| image towns | SD 1.5 before | SD 1.5 after | H3 before | H3 after |
|---|---|---|---|---|
| starting noise moved at 0.001 | 1.3 to 3.2% | 0.06% | 1.0 to 1.1% | 0.06% |
| image step, 0 to 0.001 | 0.32 | 0.03 | 0.48 | 0.38 |
| far from home at 0.05 | 0.41 | 0.28 | 0.59 | 0.38 |
| far from home at 0.25 | 0.73 | 0.71 | 0.66 | 0.50 |
| far from home at 0.50 | 1.08 | 1.08 | 0.87 | 0.80 |

- **SD 1.5's road now starts at the seed's own image.** Per seed the first step fell to
  0.01, 0.07, 0.00 and 0.02, from 0.38, 0.51, 0.09 and 0.29; seed 8888 is the same man
  from 0 through 0.25. From 0.25 up it is essentially unchanged.
- **H3's first step shrank but did not vanish**: per seed 0.18, 0.69 and 0.26, from
  0.41, 0.60 and 0.44. Its starting noise moves 0.06 per cent at 0.001, which is the
  shader's own share, so the pipeline's jump is gone and the rest is H3 responding to a
  very small change. Two things inflate it: one middle frame is compared, where a
  slightly different hammer timing moves hands a long way, and H3's seeds look alike, so
  a town is small.
- **Unchanged on both**: how much of the field reaches the result, and the
  seed-dependent hops after the first step.

The first analysis of H3's after-runs silently reused the before-runs' video frames,
because frames were cached by run name for each model. `839ca1f` gives each manifest
its own cache, and the numbers here come from fresh frames.

### What the two models say together

Parking the seed does what the metaphor says on both: results stay nearest their own
town, and the shader settings are real levers whose effect traces back to the field.
What differs is the pace, and how the shader shows. SD 1.5 leaves the neighbourhood
fast, and by 0.75 the shader has replaced the prompt. H3 explores close streets
through 0.50 and blends the shader's look into the scene at 0.75 to 1.00, with the
scene still there. Each result is one prompt, one stage and three or four seeds, so
the strengths are illustrations, not calibration.

Not measured: more than one
stage; shader types other than `domain_warp`; flicker across frames. The tools are in
`verification/blend/`; see "Reproducing the measurements".

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
14.36. Since the upgrade the generator itself spans 22.7, `walk` passes that
through untouched, and `drift` narrows it to about 3.7.

Two things it needed. `jump` bypasses the widening guards, which exist precisely
to stop a remix narrowing the noise. And it forces rank 1 for *every* generator,
including `tensor_field` at its native 90 of 128 — otherwise `shader_type` would
not be a usable coordinate in jump-space.

`walk` is the default. Legacy never reaches the travel-mode code, though since
the upgrade it does share the wider generators.

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

~~The preset overrides at execution time, so the widgets lie.~~ Done in the
upgrade below: choosing a preset writes the widgets.

**Widget ordering.** `preset` is appended at the end of the optional block
because ComfyUI maps saved values by position. It is the front door and reads
last. A major version should move it to the top of the required block.

---

## Reproducing the measurements

Most scratch workflows live in the session scratchpad, not the repository. The
blend measurements are the exception. With the pack loaded in a running ComfyUI
server and nothing else queued:

    ~/ComfyUI/venv/bin/python verification/blend/drive.py sd15 --label far \
        --strengths 0.25,0.5,0.75,1.0 --street-strengths 0.25,0.5,0.75 --jump 0.5
    ~/ComfyUI/venv/bin/python verification/blend/analyze.py \
        ~/ComfyUI/output/snk_measure/sd15_far.jsonl --sheets ~/ComfyUI/output/snk_measure/sheets

`drive.py` records where each run's latent and image or video landed in a manifest,
and resumes from it; `h3` works the same way, at about 35 seconds a run against 1.5
for SD 1.5. Strength 0 is always included, since every distance is read against it.
`analyze.py` accepts several manifests for one model and checks that any run recorded
twice came out bit-identical. The server's output is deterministic, so a difference
there means something changed.

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
