# Changelog

All notable changes to this project will be documented in this file.

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
