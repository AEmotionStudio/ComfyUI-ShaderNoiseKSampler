# Changelog

All notable changes to this project will be documented in this file.

## [2.2.1] - 2026-09-23

### Removed
- **`CODE_REVIEW.md` from the package.** It was a working review that 2.2.0 shipped
  by mistake. Nothing else changes.

## [2.2.0] - 2026-09-23

Changes what a seed produces at any `shader_strength` above 0, in both sampling
modes. Strength 0 is unchanged.

The shader generators handed the sampler noise that spanned about one channel
however many the latent had, so the shader stamped one pattern across the whole
latent instead of blending into it. That is fixed at the source, the fixes that
were optional are now defaults, and blending keeps the base noise's own statistics
so the first step away from strength 0 is only as large as the shader makes it.

### Added
- **Every archetype the Shader Matrix documents is now a shader type**, eight of
  them new: `gaussian`, `fractal`, `perlin`, `heterogeneous_fbm`, `interference`,
  `projection_3d`, `cellular` and `waves`, beside the five that existed. They were
  marketed as supporter exclusives and never built; the README, the matrix modal and
  the Ko-fi cups stamped on nine of its thumbnails said otherwise, and that wording is
  gone. Each type is one module under `shaders/`, registered by import, selectable on
  every node, and pinned by a golden fixture for an image and a clip.

  What they are, and what the four shared knobs do to each -- the `shader_type`
  tooltip carries the same, one sentence per type:
  - `gaussian`: white noise, the same thing the sampler already starts from, so it is
    the control. Strength moves toward another seed's neighbourhood without adding
    structure. No pattern knob reaches it; shape masks still apply; with temporal
    coherence a clip drifts from one white field to a second.
  - `fractal`: the reference FBM, plain layered simplex. `warp_strength` is the
    spacing between the layers (a lacunarity dial, 1.5x to 4x), `phase_shift` slides
    each layer to a different part of the field; neither does anything at octaves 1.
  - `perlin`: classic gradient noise, smoother and more lattice-like than simplex,
    zero on every lattice point. `warp_strength` swirls the detail layers while the
    base layer keeps its shape; `phase_shift` is contrast, as in `domain_warp`.
  - `heterogeneous_fbm`: an FBM whose persistence varies across the frame, so it has
    rough patches and smooth ones. `warp_strength` is how different they are (0 is
    plain FBM), `phase_shift` how much of the frame is rough. Needs octaves above 1.
  - `interference`: two FBMs cut into cos and sin fringes that cross, banded and
    moire-like. `warp_strength` is the fringe density, from soft folds to dense
    moire; `phase_shift` retunes the second field.
  - `projection_3d`: a plane through a 3D simplex FBM. `phase_shift` is the depth of
    the slice, so it is the "different facet of the same field" knob literally, and
    `time` slides the plane, so a clip is coherent by construction. `warp_strength`
    bends the plane.
  - `cellular`: Worley cells. `octaves` picks the pattern -- 1 nearest distance, 2
    second-nearest, 3 edges, 4 product, a fractional value blends two -- and
    `phase_shift` the cell shape, 0 diamond, 0.5 round, 2 square. `warp_strength`
    bends the lattice. On a clip the cells live in 3D and time moves through them.
  - `waves`: `octaves` seeded plane waves summed, straight at `warp_strength` 0 and
    bent above it; `phase_shift` rearranges the same waves into a different
    interference pattern; `time` drifts each wave at its own rate.

  All of them fill every latent channel with a field of their own, keep channel 0 the
  same whether one channel or many were asked for, draw the channel axis in one
  batched call, and leave the global RNG alone. Like `spectral`, their strengths are
  **not** calibrated against real prompts the way the first four are.

  Draw cost and effective channel rank on `walk`, from `verification/benchmark_draw.py`
  (CPU, 4 threads, octaves 3), with `domain_warp` for reference:

  | type | H3 608x352/56f | H3 1344x768/124f | LTXV 128ch |
  |---|---|---|---|
  | `domain_warp` | 0.36s, 23.2/24 | 1.49s, 23.4/24 | 0.12s, 81/128 |
  | `gaussian` | 0.01s, 24.0/24 | 0.07s, 24.0/24 | 0.01s, 125/128 |
  | `fractal` | 0.15s, 23.4/24 | 0.54s, 23.7/24 | 0.04s, 64/128 |
  | `perlin` | 0.62s, 23.4/24 | 2.50s, 23.7/24 | 0.18s, 66/128 |
  | `heterogeneous_fbm` | 0.20s, 23.4/24 | 0.73s, 23.7/24 | 0.06s, 64/128 |
  | `interference` | 0.25s, 23.7/24 | 0.89s, 23.8/24 | 0.07s, 84/128 |
  | `projection_3d` | 0.31s, 23.6/24 | 1.31s, 23.8/24 | 0.09s, 76/128 |
  | `cellular` | 0.74s, 23.9/24 | 3.01s, 23.9/24 | 0.21s, 94/128 |
  | `waves` | 0.15s, 23.9/24 | 0.55s, 24.0/24 | 0.04s, 97/128 |

  `cellular` is the dearest, at twice `domain_warp`: 27 neighbouring cells per pixel on
  a clip. The five existing types draw exactly what they did before.

  Measured on Krea 2 (one prompt, three seeds, images only): `gaussian` moves the
  picture less than half as far as any structured type and never leaves its seed,
  which is what a control should do. `fractal`, `heterogeneous_fbm`, `cellular` and `projection_3d` behave like
  `domain_warp`, re-composing the scene from 0.25 without drawing their pattern into
  it, the two FBMs with more colour. `waves`, `interference` and `perlin` get drawn:
  from about 0.5 their bands appear as striped fabric and backdrop, `waves` soonest and
  hardest, so their useful range on that model sits below 0.5.

  Shared underneath: `shaders/fbm.py` (the draw-and-fill skeleton every scalar-field
  type uses, an FBM over the shared simplex, `simplex_warp`), `lattice_hash` and the
  four-corner `simplex_3d_full` in `shaders/simplex.py` (the latter moved verbatim
  from `temporal_coherent`, which now delegates; a new golden fixture pins it across
  the move), and `BaseNoiseGenerator.palette_channels`, one copy of the colour-scheme
  mapping `domain_warp` carries for itself.

- **The live shader display previews every shader type.** It had GLSL for only
  `domain_warp`, `tensor_field` and `curl_noise`; picking any other type left the
  preview on its previous pattern and logged `Shader source not found`. Each type now
  has a program written to mirror its Python generator knob for knob, not pixel for
  pixel. The deprecated node's preview dropdown is built from the same table, and a
  test requires that table to match the Direct node's `shader_type` list, so a type
  cannot ship without a preview again.

- **`Shader Noise Source`, a node that outputs a `NOISE` object**, so shader noise can
  start a run driven by `SamplerCustomAdvanced` -- with the guider, sampler and sigma
  schedule chosen separately. It answers a standing request for a custom-sampling
  version of the node, and reaches guidance the KSampler-shaped inputs cannot express:
  `BasicGuider` with no negative and no CFG, `DualCFGGuider`, guiders from other packs.
  `AddNoise` consumes one too.
  `example_workflows/MiniMaxH3_CustomSampling_SNK_Source.json` wires it to a
  `BasicGuider`, on core nodes and this pack and nothing else.

  It is the same shader inputs and exactly the noise the sampler starts from, pinned
  bit-for-bit against the Direct node so a seed means the same thing on both. It does
  not do stages: those re-enter the shader at segment boundaries partway through a run,
  and a `NOISE` object is asked for noise once, before sampling begins.

- **The sampler can run half a schedule**, through four inputs that mirror
  `KSampler (Advanced)`: `add_noise`, `start_at_step`, `end_at_step` and
  `return_with_leftover_noise`. A generation can now be split, with something else
  working on the latent in between -- which is what MiniMax H3 needs to fix faces,
  since the fix is a latent upscale partway through and the shader previously had to
  be dropped from the workflow to get one. Both halves keep their shader noise.
  `example_workflows/MiniMaxH3_Split_Upscale_SNK_Direct.json` wires it up.

  Stages spread across the steps a node actually samples rather than the whole
  schedule, so they divide the work it does; `stage_progression` still measures
  position in the whole trajectory, so a node running the tail of a schedule gets the
  fine end of `coarse_to_fine` instead of starting a fresh sweep. With `add_noise`
  off there is no starting noise to paint, so the shader only enters at an interior
  boundary. Every default is a no-op: saved workflows sample exactly what they did
  before, which the golden suite pins.

- **A fifth shader type, `spectral`.** It builds its field from a frequency band --
  the whole channel stack's Fourier coefficients drawn at once, shaped by a radial
  envelope, brought back with one inverse FFT -- instead of evaluating a procedural
  field per pixel per channel. Measured against `domain_warp` on the same latents:
  **0.02s against 1.16s** at MiniMax H3's 608x352/56 frames, **0.13s against 5.31s**
  at 1344x768/124 frames, **0.01s against 0.61s** at LTXV's 128 channels, where it
  also spans more of them (115 of 128 against 84).

  It is a different instrument, not a faster `domain_warp`: a shaped-Gaussian field
  is a cloud, with no filaments or swirls, and colour schemes do nothing to it. What
  it has instead is direct control over the large-scale structure the model actually
  reads -- `noise_scale` sets the band, `octaves` the roll-off, `warp_strength` the
  anisotropy -- and temporal coherence for free: holding the seed and advancing time
  turns each mode at its own rate rather than redrawing the field. Adjacent frames
  of a 24-channel, 8-frame draw correlate at 0.68, against `temporal_coherent`'s
  0.55 and `tensor_field`'s 0.80. The figure depends on clip length: `time` always
  spans 0 to 1 across the whole clip, so a longer clip takes smaller steps and
  comes out smoother frame to frame.

  Its strengths are **not** calibrated against real prompts the way the other four
  are. Treat the presets' numbers as not applying to it yet.

### Removed
- **The "supporter exclusive" framing of the missing shader types.** Nine Ko-fi
  badges under the matrix gallery's cards, the "Unlock Exclusive Shader Noise
  Palettes" card, the Ko-fi cup the thumbnail renderer stamped onto those nine
  previews (and the SVG it loaded for it), and the README's "additional advanced noise
  types are available to supporters" sentence. The general support link stays.
- **Dead noise code.** `utils/noise_utils.py` kept its own simplex, FBM, Perlin, Worley
  and value noise that no generator called; only `create_coordinate_grid` remains, and
  `utils` no longer exports `simplex_noise_2d`, `simplex_noise_3d`, `fbm_noise` or
  `random_gradient`. `web/glsl_shaders.js`, an unreferenced copy of the old GLSL
  sources, is gone too.

### Fixed
- **MiniMax H3 crashed with more than one shader boundary.** With `injection_stages`
  above 0 or `sequential_stages` of 2 or more, every H3 run failed in `pack_latents`.
  The noise at an interior boundary is recovered on the sampler's device, and the
  painted streams were moved to the device of the latent the run started with, so the
  video stream ended up on the CPU and the audio stream stayed on CUDA. Each stream is
  now painted on the device it is already on. Single-stream latents were not affected,
  beyond a needless round trip.
- **Legacy mode sampled `tensor_field` when asked for `temporal_coherent`.** The
  parameter whitelist was a hand-written set that never gained the name, and it
  rewrote anything it did not know to `tensor_field` with a printed warning. Standard
  mode was unaffected, since the node hands the pipeline `shader_type` directly; the
  legacy path reads it back out of the sanitised dictionary. The whitelist is now the
  generator registry, so a registered type cannot be missing from it.
- **`stage_progression` said it needed more than one stage to do anything, and that was
  not true.** A single stage sits at the start of the trajectory and is shaped from
  there, so `coarse_to_fine` was quietly drawing it at half the zoom and one octave
  down -- which is how the `roam` and `video` presets have always behaved. Only the
  tooltip changed; the sampling is what it was.

- **The first frame of a video was drawn from a different noise function than the
  rest.** `domain_warp` chose between a 2D and a 3D field by testing `time != 0`,
  and frame 0 is the only frame whose time is exactly 0, so it alone took the 2D
  path. With temporal coherence on, adjacent frames correlate at about 0.95 --
  except frame 0 against frame 1, which measured -0.01. The choice is now made once
  per draw from the latent's frame count, so a clip is one field throughout. Single
  images are unaffected, and only frame 0 of a video draw changes; the
  `video_nested` fixture was re-captured.

### Performance

- **The channel axis is drawn in one call instead of one per channel.** At MiniMax
  H3's default latent the draw used to perform 37 frames x 24 channels = 888
  separate renders of 4032 pixels each, where the time went to per-op dispatch
  rather than arithmetic. Generators now offer `fill_channels` a batched path and
  it takes it whenever more than one extra channel is wanted. Measured at
  1344x768/124 frames, before and after in one sitting: `domain_warp` **6.76s to
  1.90s**, `curl_noise` **6.54s to 1.94s**, `temporal_coherent` **5.76s to 1.66s**,
  `tensor_field` **5.37s to 2.08s**. Effective channel rank is unchanged in all four.

  `curl_noise` and `temporal_coherent` are byte-identical -- verify by running the
  suite, whose fixtures did not move. `domain_warp` is not: it turns its
  coordinates by an angle drawn from the seed, so the coordinates genuinely differ
  per channel and the batched draw has to materialise them, which changes how the
  tail of each elementwise op is vectorised. Five fixtures moved by 1.2e-07 to
  2.4e-07 -- float32 rounding, with effective channel rank identical to three
  decimal places. The previous behaviour is tagged `pre-batched-noise`.

  **`jump` and `stamp` are unaffected.** They build from one-channel draws, which
  never take the batched path; 32 draws across four generators and four latent
  shapes were checked byte-identical against the tag.

- **One simplex primitive instead of seven.** The 2D hash was duplicated across
  three generators and the 3D across four, with real differences hidden between
  them: three of the 3D copies sum a single simplex corner rather than four, and
  one of those computed three more corner hashes and discarded them. They are now
  `shaders/simplex.py`, named for what they do, with the differences kept
  deliberately rather than unified -- collapsing them would silently change three
  generators. `temporal_coherent`'s four-corner version stays with that generator:
  it picks corners by the real simplex ordering and reads gradients from a table,
  so it is a different function, not a parameterisation.

- **`tensor_field` draws its shape mask once** instead of once per channel, which
  at LTXV's 128 channels was 127 identical masks, and no longer clones the
  coordinate grid per channel. Byte-identical.

  Its batching works differently from the others': each of its channels takes one
  of four *visualisations*, its own scale, warp and time, and its own perturbation
  of the coordinate grid. The expensive part -- five simplex evaluations per
  channel -- does not depend on the visualisation, so it runs once for the whole
  channel axis and each channel then takes the cheap visualisation its index asks
  for. Its two fixtures moved: exact at vector-aligned shapes, up to 1.3e-05 at
  others, because tensor_field perturbs coordinates per channel and an ulp of
  coordinate can push a point across a simplex cell boundary. Holding the
  per-channel `time` in float64 until the point of use, where the scalar path's
  Python float was rounded, brought that down from 5.6e-04.

None of the rest of this changes what a seed produces. The golden fixtures are byte-identical
and the full suite passes untouched; that is the acceptance criterion for all of it.

- **A collapse no longer renders the draw it throws away.** `travel_mode: jump` (and
  the `jump` and `stamp` presets) rebuilds every channel from one-channel draws and
  never reads the wide draw, but the wide draw was rendered first anyway. Measured on
  `domain_warp` at MiniMax H3's latent: **5.99 s to 0.22 s** at 1344x768/124 frames,
  and 1.44 s to 0.05 s at 608x352/56 frames. `drift` is unaffected -- it needs the
  wide draw to measure, so it still pays for it.
- **The draw runs under `torch.inference_mode()`**, and clones on the way out so the
  noise leaves as an ordinary tensor rather than an inference tensor.
- **The pure-hash generators stopped reseeding the global RNG.** `domain_warp` did it
  once per channel render -- 888 times per draw at H3's default latent -- and
  `torch.manual_seed` reseeds every CUDA device as well as the CPU. `curl_noise` and
  `temporal_coherent` did it once per frame. All three are coordinate hashes of their
  seed argument, so the calls changed nothing. `tensor_field` still reseeds, because
  it draws `torch.randn_like` inside its channel loop and that value reaches its output.
- Together the last two are worth about **1.2x** on `walk` and `drift`.
- **New `verification/benchmark_draw.py`**, because none of the above was reproducible
  before: the only per-draw timing in the repo predated the per-channel fill and
  understated the cost roughly tenfold.
- **`SNK_TEST_CPU=1`** pins the test suite to the CPU, so it runs while a ComfyUI
  server on the same box is holding the GPU.

### Changed
- **Every latent channel gets a shader field of its own.** `domain_warp` copied one
  field across its four channels and built the rest from the first two;
  `temporal_coherent` broadcast one field to every channel; `curl_noise` padded its
  colour path with copies. Effective channel rank for `domain_warp` at 4 / 24 / 128
  channels goes from 1.00 / 2.13 / 2.43 to 3.91 / 22.72 / 69.56. `tensor_field`
  already did this and is unchanged. On MiniMax H3 under a real prompt,
  `domain_warp` now holds the seed's scene to about 0.75, against about 0.5 with the
  old after-the-fact decorrelation. This also changes `legacy` output; see Known
  issues.
- **Blending keeps the base noise's own mean and deviation**, per channel, instead of
  forcing exactly 0 and 1. Forcing them moved the starting noise by 1 to 3 per cent
  the moment strength left 0, before the shader contributed anything, and on SD 1.5
  that alone moved the image as far as a whole 0.05 step of shader. From 0 to 0.001
  the image now moves 0.03 of the typical distance between two seeds' images, where
  it moved 0.32. On H3 the same step fell from 0.48 to 0.38; the rest is H3
  responding to a very small change in its noise, not the node.
- **`normalize_strength` is on by default**, so one `shader_strength` value hands the
  sampler the same share of shader in every blend mode. `multiply`, the default
  mode, is the calibration reference and is unaffected.
- **`travel_mode` replaces `decorrelate_channels`.** `walk` (default) keeps the
  generator's own width, `drift` mixes it down to four directions, and `jump` folds
  it into one, so the shader's parameters set the destination and the seed stops
  mattering. The guard deciding when to remix is now direction-aware.
- **Tooltips describe what happens at each strength** rather than how much to avoid:
  the shader blending progressively into the picture, at a pace set by the model,
  the seed and `noise_scale`.
- **The golden test suite pins the `standard` pipeline** instead of `legacy`.

### Added
- **`preset` input.** `nudge`, `explore`, `roam`, `video`, `jump` and `stamp` set
  `shader_type`, `shader_strength`, `blend_mode`, `travel_mode`, `stage_progression`
  and `shape_type` together; `custom` leaves every widget alone.
- **Choosing a preset writes the widgets it controls**, and editing one of those
  widgets to another value switches the preset back to `custom`, so the panel shows
  what the run will use. The table is served by `GET /shader_noise_ksampler/presets`.
- **`verification/blend/`**: scripts that run a matrix of seeds, strengths and shader
  settings through a ComfyUI server and measure how the shader blends into the
  result: how far each setting moves you from the seed's own image, whether you stay
  nearer it than any other seed's, and how much of the shader pattern is in the
  final latent.

### Fixed
- **`drift` did nothing for `tensor_field` and `curl_noise`.** Those generators
  already spanned their channels, so the widening guard returned their noise
  untouched and `drift` behaved exactly like `walk`.

### Known issues
- **`legacy` no longer reproduces every earlier seed.** It keeps the pre-2.0 pipeline
  structure but shares the shader generators, so workflows using `domain_warp`,
  `temporal_coherent`, or `curl_noise` on latents wider than four channels now
  produce different images. Workflows saved before 2.0 are still switched to
  `legacy` when loaded, and the `sampling_mode` tooltip still describes it as
  reproducing their seeds.

## [2.1.0] - 2026-09-11

Compatibility release for ComfyUI 0.34.0's model roster, MiniMax H3 in
particular. The `standard` pipeline already took its noise shape from the latent
rather than from a table of model names, so most of the roster needed nothing;
what was wrong was how a multi-stream latent crossed a stage boundary.

### Fixed
- **MiniMax H3 multi-stage runs.** H3's latent is a `NestedTensor` of a video
  stream `[B,24,T,H,W]` and an audio stream `[B,32,2,T]`, and the model — not its
  latent format — carries audio scaled onto the video sigma schedule
  (`audio_scale` = `shift / audio_shift` = 4.0). The boundary split inverted
  through `latent_format.process_in`, which for `MiniMaxH3AV` is an identity, so
  the audio residual handed to the next segment was off by that factor of 4. It
  now inverts through the model's own `process_latent_in` / `process_latent_out`,
  which is what `CFGGuider.inner_sample` actually applies. No change for any
  other model: `BaseModel.process_latent_in` just calls the format.

  Measured on real H3 weights (MiniMax H3 Max, int8), two stages at
  `shader_strength` 0, where a segmented run must reproduce an uninterrupted one:

  | | video max error | audio max error |
  | --- | --- | --- |
  | before | 9.3e-01 (stream max 4.80) | 1.3e+00 (stream max 1.35) |
  | after | 4.8e-07 | 2.4e-07 |

  The audio stream was almost entirely wrong, and because H3 denoises both
  streams in one packed sequence the error reached the video through the DiT's
  joint attention — so this degraded picture as well as sound.
- **Shader noise at a stage boundary reads its shape from the noise it is about
  to paint**, instead of a shape captured before the run started.

### Changed
- **Latents with no spatial grid are refused by name.** Sequence latents
  (`[B, C, L]`: Stable Audio 1 / 3, ACE-Step 1.5, MiniMax Music 3, Hunyuan3D,
  TripoSplat) have no height and width for a shader to draw on. They now raise
  `UnsupportedLatentError` naming the shape, before sampling starts, rather than
  a bare `ValueError` from inside noise generation. At `shader_strength` 0.0
  there is nothing to paint, so those models sample through as a plain KSampler.
- **Verified across the roster, not assumed**: noise generation is exercised at
  every channel count ComfyUI ships — 3, 4, 8, 12, 16, 24, 32, 48, 64, 128 and
  256 — for both image and video latents.

### Added
- **`temporal_coherent` is selectable.** It shipped, was registered and passed
  its own tests, but was missing from the Direct node's `shader_type` list, so
  no workflow could reach it. It is 4D simplex noise with time as a real axis,
  built for animation — on MiniMax H3 it degrades more gracefully than the other
  three (faint striping at 0.35 where `domain_warp` shows blocks) and is the only
  type whose audio level barely moves with strength. Two tests now assert the
  dropdown and the generator registry agree in both directions.

### Changed
- **Tooltips rewritten from measurements.** `shader_strength`, `blend_mode`,
  `shader_type`, `shape_type` and `use_temporal_coherence` previously implied the
  full 0.0–1.0 range was usable. On video models it is not: H3 holds to about
  0.25 with `domain_warp`/`multiply` and is gone by 0.75, `curl_noise` and shape
  masks need roughly half that, and `use_temporal_coherence` — whose description
  claimed it "helps maintain frame-to-frame consistency" — swamps the picture at
  0.5 because holding one seed across frames reinforces the pattern instead of
  averaging it out. The tooltips now say so, and rank the blend modes by how
  aggressive they are.

- **`normalize_strength` (optional, default off).** One `shader_strength` value
  meant eight different things: measured as the shader's share of the mixed
  noise, the blend modes differ by a factor of twenty-three at 0.5. With this on,
  strength is read on `multiply`'s scale and the rest are rescaled to match. On
  H3 across six modes at 0.30, the spread in audio level fell from 4.6 dB to
  2.1 dB.
- **`Shader Noise Walk` node.** Ramps one parameter (`shader_strength`,
  `phase_shift`, `noise_scale`, `warp_strength`, `octaves`,
  `shape_mask_strength`, `color_intensity` or `seed`) across a batch in a single
  run, with the model resident throughout — five H3 runs at 608x352/56 frames
  took 125s total. Output is a batched latent for the comparer nodes. Multi-stream
  latents batch stream by stream.
- **`shade_non_spatial` (optional, default off).** Paints the streams that carry
  no picture: an audio stream across stereo x time, a sequence latent as a single
  row. Unblocks Stable Audio 1/3, ACE-Step 1.5, MiniMax Music 3, Hunyuan3D and
  TripoSplat, which were refused outright. On H3 it biases the sound toward tonal
  content (spectral flatness 0.104 -> 0.066) while leaving the picture clean.
  Streams with fewer than 64 cells per batch item are skipped even when it is on,
  since they carry metadata rather than content: TripoSplat's `[B, 1, 5]` camera would
  otherwise have its viewpoint moved. The smallest real content stream, H3's audio,
  has 414.
- **`stage_progression` (optional, default `uniform`).** Varies the shader across
  the run rather than drawing the same one at every stage: `coarse_to_fine`
  starts zoomed in on large features with fewer octaves and ends zoomed out on
  small ones with more, `fine_to_coarse` reverses it. Centred on your widget
  values, spanning 0.5x to 2x noise_scale and plus or minus one octave.
- **`decorrelate_channels` (optional, default off).** *Replaced by `travel_mode`
  before it shipped; see 2.2.0.* The generators built every
  channel past the first one or two as a pointwise function of those two, so
  `domain_warp` returned noise spanning a single channel at SD's four and about
  two at any larger count, and `temporal_coherent` returned identical channels.
  Samplers expect i.i.d. noise. Filling the channel axis from independent draws
  moves SD 1.5's usable strength from below 0.25 to about 0.5, and clears the
  blocking H3 showed at 0.5. Generators that already span their channels, such as
  `tensor_field`, are detected and left untouched.

### Fixed
- **`temporal_coherent` ignored its seed.** It read `params["base_seed"]`
  unconditionally where `domain_warp` gates that on `use_temporal_coherence`, and
  the node always sets `base_seed` — so every stage drew the identical field and
  only `time` varied. Nothing could select that generator from a workflow before
  this release, so no saved workflow changes.

### Removed
- **`core/model_compat.py`**, along with the `MODEL_CHANNEL_COUNTS`,
  `MODEL_NAME_PATTERNS` and `VIDEO_MODEL_CHANNELS` tables. Nothing called it. Its
  tables stopped at LTXV, its `model_type == "FLOW"` branch was unreachable
  (`str(ModelType.FLOW).upper()` is `"MODELTYPE.FLOW"`), and its 5-D layout guess
  defaulted to `[B,F,C,H,W]`, which ComfyUI never produces. The `legacy` mode's
  own detector is untouched, so pre-2.0 workflows still reproduce their seeds.

## [2.0.0] - 2026-09-11

Sampling correctness release. Stages, `denoise` and `custom_sigmas` now do what
the documentation says, and blended noise reaches the sampler in the
distribution the model was trained on. These change what an existing seed
produces, so the previous pipeline is preserved as a sampling mode.

### Added
- **`sampling_mode` input** on `Shader Noise KSampler (Direct)`: `standard`
  (default) or `legacy`. Workflows saved before 2.0 switch themselves to
  `legacy` when loaded, so their seeds keep reproducing.
- **Live shader preview on the Direct node.** It mirrors the node's real inputs
  rather than adding a duplicate set of widgets.
- **`sequential_distribution`, `injection_distribution` and
  `fast_high_channel_noise` are now real inputs.** They were declared as V1
  `hidden` tuple inputs, which ComfyUI never delivers, so they had been stuck at
  their defaults.
- **Python test suite** (186 tests), including 11 recorded "golden" runs that
  pin the legacy pipeline byte for byte.

### Fixed
- **Stages are segments of one sampling run.** Each stage used to build its own
  full schedule and restart from maximum noise. On flow models (Flux, SD3, WAN,
  Hunyuan, LTXV) `noise_scaling` is `sigma*noise + (1-sigma)*latent`, and sigma
  is 1.0 at the start, so the previous stage was multiplied by zero: two
  sequential stages were a half-length generation, not two halves of one.
- **`denoise` reaches the schedule.** It was hard-coded to 1.0 whenever a
  sequential stage ran, which is the default. Measured before the fix:
  img2img at denoise 0.6 and at 1.0 produced pixel-identical output, i.e. the
  input image was fully regenerated. This is the cause of the
  image-to-video / video-to-video weakness listed under Known Issues.
- **Custom sigmas are sampled, not just counted.** The model was wrapped to
  override `model_sampling`, but ComfyUI reads the schedule through
  `get_model_object("model_sampling")`, which resolves past the wrapper.
- **Blended noise keeps mean 0 / std 1.** The blend modes are image-compositing
  formulas that assume [0,1] data; on N(0,1) noise they shifted the
  distribution (overlay mean +0.39 at strength 0.3, soft_light std 4.2 at
  strength 1.0, the inverse transform reaching ~4e4). Compositing now happens in
  uniform space and the result is re-standardised.
- **Injection stages no longer end in a 1-step segment.** 20 steps with 3
  injection stages produced ranges (0,10) (10,19) (19,20), so the final image
  came out of a single step from full noise.
- **Video frames are no longer confused with channels.** The noise shape comes
  from the latent instead of being guessed; a 61-frame Wan or Hunyuan clip has
  16 latent frames and 16 channels, and time evolution was being applied across
  channels with no error raised.
- **`tensor_field` crashed for almost every configuration** (both modes): the
  eigenvalue-difference branch unsqueezed an already-4D tensor, so 46 of 48
  parameter combinations raised `RuntimeError`, including the node's defaults.
- **Inpainting masks, batch_index, and the progress bar / live preview** now
  behave as they do for a stock KSampler.
- **Fractional octaves** interpolate between integer renders; the 0.1-step
  slider previously did nothing until it crossed a whole number.
- **The image-side shape correction** raised `NameError` on a missing import and
  silently fell back to all-zero shader noise.

### Changed
- `Shader Noise KSampler` (the display node) is **deprecated**: hidden from node
  search, still loads in saved workflows, and always samples with `legacy`. Its
  cache now keys on the shader params file, so saving parameters re-runs it.
- The save_params API keeps only known keys and no longer returns exception
  text.
- Each shader generator registers once (four "Overwriting existing shader
  generator" warnings on every startup are gone).

### Removed
- Dead code: a duplicate comparer module, a parameter mapper that could only
  ever raise, stale compiled type stubs served to the browser, `__js_files__`,
  `CONTEXT_MENUS`, `has_preview`, and the redundant `IS_CHANGED` tuples.

### Verified
- Legacy mode is **pixel-identical** to 1.3.5 across 12 rendered configurations
  (SD1.5 and Wan 2.1, including 33-frame video).
- Standard mode at `shader_strength=0` is **pixel-identical to a stock
  KSampler** with the same seed.
- 186 Python tests and 91 web tests pass.

## [1.3.5] - 2026-06-16

### Changed
- Version bump for the Comfy registry (project 1.3.5, web frontend 1.0.4).

## [1.3.4] - 2026-06-02

### Security
- **Vitest Dependency (GHSA-5xrq-8626-4rwp / CVE-2026-47429)**: Bumped the `vitest` and `@vitest/coverage-v8` dev dependencies from `^1.2.2` (resolved `1.6.1`) to `^4.1.0` (resolved `4.1.8`) to address a critical (CVSS 9.8) arbitrary file read/write/execute vulnerability in the Vitest UI server for versions `< 4.1.0`. Dev-only dependency; the regenerated lockfile pulls `vite@8`. All 85 tests, typecheck, and coverage verified passing on the new major version.

## [1.3.3] - 2026-03-24

### Fixed
- **Shader Display Draw Order (Load-Order Independent)**: Fixed gradient title rendering over the shader display on page refresh. Made `onDrawForeground` chain load-order independent — both `gradient_title.ts` and `shader_renderer.ts` now ensure the gradient always draws as background and the shader canvas always renders on top, regardless of which extension registers first.

## [1.3.2] - 2026-03-24

### Fixed
- **Shader Display Draw Order**: Fixed gradient title background painting over the shader WebGL canvas by swapping the draw order in `gradient_title.ts` — gradient now renders before `origOnDrawForeground` so the shader sits on top.

## [1.3.1] - 2026-02-14

### Fixed
- **Complete GLSL Shader Restoration**: Restored all v260 shader code lost during TypeScript refactor — header grew from 17K→37K chars, with full FBM implementations, 16 shape masks, 24 color schemes, 4 domain warp modes, tensor field eigenvector visualization, and curl noise advection/particle simulation.
- **Chromium/Brave Shader Compatibility**: Fixed blank shader canvas in Chromium by adding `preserveDrawingBuffer: true` to WebGL context; fixed "basic-looking" shaders by upgrading fragment shader precision from `mediump` to `highp` with `#ifdef` fallback (Chromium's ANGLE enforces strict 16-bit mediump, losing noise detail).
- **GLSL Spec Compliance**: Fixed undefined `smoothstep` behavior where `edge0 >= edge1` in stripes, cross, and concentric shape masks — caused inconsistent rendering across GPU drivers.
- **HSV Color Scheme Discontinuity**: Fixed hue wrapping at `normalized=1.0` where `i=6` fell into wrong else branch, creating a visible color jump.
- **WebGL Resource Leak**: Added `gl.deleteProgram()` and `gl.deleteShader()` cleanup on shader link failure.
- **GLSL Normalize Safety**: Added zero-vector checks before `normalize(velocity)` in curl noise flow visualization and `applyWarpIntensity` to prevent undefined GLSL behavior.
- **Shader Debug Logs**: Removed `console.log` statements from shader compilation and loading that spammed the browser console.

### Security
- **API Input Validation Fix**: `validate_and_sanitize_params` now validates both camelCase frontend keys (`shaderScale`, `shaderType`, `shaderShapeType`, `shaderWarpStrength`, `shaderPhaseShift`) and snake_case internal keys — previously most validation was silently skipped because the frontend sends camelCase but validation only checked snake_case.

### Improved
- **Temporal Noise Optimization**: Optimized temporal coherent noise generation for better animation performance.
- **Accessibility**: Improved accessibility for shader matrix modal and copy button.

## [1.3.0] - 2026-01-30

### Added
- **API Endpoint for Parameter Saving**: Frontend shader parameter changes now save directly to the server via a new `/shader_noise_ksampler/save_params` endpoint, eliminating the need for manual file downloads.
- **Video Comparer Optimization**: Frames are now served as temporary files instead of base64 data URLs, resolving browser `QuotaExceededError` issues with longer videos.

### Fixed
- **Shader Import Paths**: Resolved import errors in shader modules (`domain_warp.py`, `curl_noise.py`, `tensor_field.py`) by switching to relative imports.
- **Video Comparer Duplicate Class**: Removed duplicate `VideoComparer` class that existed in two files.
- **Memory Threshold Priority**: Fixed memory threshold check order to ensure force cleanup runs when needed.
- **Metadata Cache Keys**: Fixed metadata key calculation mismatch between backend and frontend in Comparer nodes.

## [1.2.1] - 2026-01-28

### Added
- **TypeScript Migration**: Converted entire frontend codebase to TypeScript for improved type safety and maintainability.
- **Testing Infrastructure**: Added Vitest-based testing with 85+ unit tests covering core functionality.
- **Shared Rendering Utilities**: Extracted common golden eyeball and image scaling logic into reusable modules.

### Refactored
- **Frontend Build Pipeline**: Established `pnpm build` workflow with TypeScript compilation and automatic JS deployment.
- **Module Architecture**: Centralized shader registry and improved module organization.

### Fixed
- **PR Review Issues**: Addressed multiple rounds of code review feedback including dead code removal, module-private constants, and consistent shader registration.

## [1.2.0] - 2025-12-15

### Added
- **Auto-Fill Toggle**: Both `Advanced Image Comparer` and `Video Comparer` nodes now feature an `auto_fill` toggle for streamlined A/B testing.
- **Video Comparer Node**: New node for comparing two videos with six viewing modes (Playback, Side-by-Side, Stacked, Slider, Onion Skin, Sync Compare).
- **Advanced Image Comparer**: Eight comparison modes including Slider, Click, Side-by-Side, Grid, Carousel, and Onion Skin.
- **Shader Matrix Documentation**: Comprehensive in-app documentation accessible via "📊 Show Shader Matrix" button (Alt+M).
- **Temporal Coherence**: Frame-consistent noise generation for animations.

## [1.1.0] - 2025-11-20

### Added
- **Multi-Stage Shader Application**: Sequential and injection stages for applying shader noise at different points in the diffusion process.
- **Shape Masks**: Geometric overlays (Radial, Linear, Grid, Vignette, Spiral, Hexgrid) with adjustable strength.
- **Color Schemes**: Transformations (Inferno, Magma, Viridis, Jet, Turbo) applied before diffusion.
- **Blend Modes**: Multiply, Add, Overlay, Screen, Soft Light, Hard Light, Difference.

## [1.0.0] - 2025-10-01

### Initial Release
- **ShaderNoiseKSampler Node**: Advanced KSampler replacement with shader-based noise patterns.
- **ShaderNoiseKSampler (Direct)**: Variant without shader display for faster iteration.
- **Three Core Noise Types**: Domain Warp, Tensor Field, and Curl Noise.
- **Model Compatibility**: Support for SD 1.5, SDXL, Flux, WAN2.1, Hunyuan, and more.
