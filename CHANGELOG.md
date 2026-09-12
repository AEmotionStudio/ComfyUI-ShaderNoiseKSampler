# Changelog

All notable changes to this project will be documented in this file.

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
- **`decorrelate_channels` (optional, default off).** The generators built every
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
