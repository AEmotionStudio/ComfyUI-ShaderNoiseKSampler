## 2024-05-23 - Vectorized Color Interpolation
**Learning:** The codebase contained loop-based color interpolation logic in `ShaderParamsReader` and `CurlNoiseGenerator` which iterated over color stops and applied boolean masks for each segment. This is O(N*S) where N is pixels and S is stops.
**Action:** Replaced with `torch.bucketize` and vector indexing/gathering to perform interpolation in a single vectorized pass O(N). This pattern should be applied whenever discretizing continuous values into bins/segments for interpolation in PyTorch.

## 2026-01-17 - Vectorized Masked Blending
**Learning:** Masked tensor assignment (`tensor[mask] = val`) causes synchronization overhead and non-contiguous memory access. In `_blend_noises`, `overlay` and `hard_light` modes used this pattern.
**Action:** Replaced masked assignments with `torch.where` for element-wise conditional blending. This yielded ~10-13x speedup on CPU benchmarks while preserving exact functionality. Use `torch.where` for vectorization over boolean masking when possible.

## 2026-02-28 - Integer Bitwise Hashing for Simplex Gradient
**Learning:** In `TemporalCoherentNoiseGenerator.grad3d`, casting integers to floats to perform modulo operations (`h % 2`, `h % 4`) and then casting back is inefficient. Additionally, `torch.floor(x).to(int)` is slower than `x.int()` when `x` is already an integer-valued float.
**Action:** Replaced float arithmetic with integer bitwise operations (`h & 1`, `h & 2`) and utilized `.int()` for truncation on pre-floored inputs. This resulted in a ~1.4x speedup for the gradient calculation step, which is called heavily in the noise generation loop.
