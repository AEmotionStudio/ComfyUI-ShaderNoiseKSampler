# Changelog

All notable changes to this project will be documented in this file.

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
