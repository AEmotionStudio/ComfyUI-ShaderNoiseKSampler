
import torch
import time
import sys
import os

# Add repo root to path so we can import shaders if needed, though for this micro-benchmark
# we will define the functions locally to avoid dependency issues during the benchmark run.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def overlay_blend_masked(tensor, warp_tensor, blend_factor):
    tensor_norm = tensor * 2.0 - 1.0
    warp_norm = warp_tensor * 2.0 - 1.0

    dark_mask = (tensor_norm < 0)
    light_mask = (tensor_norm >= 0)

    result = torch.zeros_like(tensor)
    result[dark_mask] = (tensor_norm[dark_mask] * (1.0 + warp_norm[dark_mask] * blend_factor))
    result[light_mask] = tensor_norm[light_mask] + warp_norm[light_mask] * blend_factor * (1.0 - tensor_norm[light_mask])

    result = (result + 1.0) * 0.5
    return result

def overlay_blend_where(tensor, warp_tensor, blend_factor):
    tensor_norm = tensor * 2.0 - 1.0
    warp_norm = warp_tensor * 2.0 - 1.0

    # Calculate both branches for all elements
    # Note: For very expensive branches, torch.where evaluates both.
    # But here operations are simple arithmetic, so vectorization gain usually outweighs redundant calc.
    # However, to be strictly equivalent to the if/else logic of overlay:

    res_dark = tensor_norm * (1.0 + warp_norm * blend_factor)
    res_light = tensor_norm + warp_norm * blend_factor * (1.0 - tensor_norm)

    result = torch.where(tensor_norm < 0, res_dark, res_light)

    result = (result + 1.0) * 0.5
    return result

def hsv_to_rgb_masked(h, s, v):
    c = v * s
    h_prime = h * 6.0
    x = c * (1.0 - torch.abs(torch.fmod(h_prime, 2.0) - 1.0))
    m = v - c

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    mask0 = (h_prime < 1.0)
    r[mask0], g[mask0], b[mask0] = c[mask0], x[mask0], 0.0

    mask1 = (h_prime >= 1.0) & (h_prime < 2.0)
    r[mask1], g[mask1], b[mask1] = x[mask1], c[mask1], 0.0

    mask2 = (h_prime >= 2.0) & (h_prime < 3.0)
    r[mask2], g[mask2], b[mask2] = 0.0, c[mask2], x[mask2]

    mask3 = (h_prime >= 3.0) & (h_prime < 4.0)
    r[mask3], g[mask3], b[mask3] = 0.0, x[mask3], c[mask3]

    mask4 = (h_prime >= 4.0) & (h_prime < 5.0)
    r[mask4], g[mask4], b[mask4] = x[mask4], 0.0, c[mask4]

    mask5 = (h_prime >= 5.0)
    r[mask5], g[mask5], b[mask5] = c[mask5], 0.0, x[mask5]

    r, g, b = r + m, g + m, b + m
    return r, g, b

def hsv_to_rgb_where(h, s, v):
    c = v * s
    h_prime = h * 6.0
    x = c * (1.0 - torch.abs(torch.fmod(h_prime, 2.0) - 1.0))
    m = v - c

    # Use torch.where or similar vectorization
    # Since conditions are mutually exclusive ranges [0,1), [1,2), etc.
    # We can rely on h_prime floor mod 6

    # We can initialize r,g,b with zeros and add conditionally, or use nested torch.where
    # Nested torch.where might be deep.
    # Another approach: Precompute the 6 cases and gather? No, too much memory.
    # Simple nested torch.where is likely fastest for JIT/Kernel fusion, but Python overhead exists.

    # Optimization:
    # (r,g,b) values for sectors:
    # 0: (c, x, 0)
    # 1: (x, c, 0)
    # 2: (0, c, x)
    # 3: (0, x, c)
    # 4: (x, 0, c)
    # 5: (c, 0, x)

    # We can use logic:
    # r = c where (0,5), x where (1,4), 0 where (2,3)
    # g = c where (1,2), x where (0,3), 0 where (4,5)
    # b = c where (3,4), x where (2,5), 0 where (0,1)

    # Check conditions on h_prime
    # h_prime < 1: 0
    # h_prime < 2: 1
    # ...
    # But h_prime can be floats.

    sector = torch.floor(h_prime) % 6

    # R channel
    # sector 0 or 5 -> c
    # sector 1 or 4 -> x
    # else -> 0
    r = torch.where((sector == 0) | (sector == 5), c,
                    torch.where((sector == 1) | (sector == 4), x, torch.zeros_like(c)))

    # G channel
    # sector 1 or 2 -> c
    # sector 0 or 3 -> x
    # else -> 0
    g = torch.where((sector == 1) | (sector == 2), c,
                    torch.where((sector == 0) | (sector == 3), x, torch.zeros_like(c)))

    # B channel
    # sector 3 or 4 -> c
    # sector 2 or 5 -> x
    # else -> 0
    b = torch.where((sector == 3) | (sector == 4), c,
                    torch.where((sector == 2) | (sector == 5), x, torch.zeros_like(c)))

    r, g, b = r + m, g + m, b + m
    return r, g, b


def benchmark():
    device = "cpu"  # Test on CPU where Python overhead is most visible and where torch.where shines
    if torch.cuda.is_available():
        # Uncomment to test on CUDA if needed, but CPU improvements usually translate to CUDA for these ops
        # device = "cuda"
        pass

    print(f"Benchmarking on {device}...")

    B, C, H, W = 4, 3, 512, 512
    tensor = torch.rand((B, C, H, W), device=device)
    warp_tensor = torch.rand((B, C, H, W), device=device)
    blend_factor = 0.5

    # Warmup
    for _ in range(10):
        _ = overlay_blend_masked(tensor, warp_tensor, blend_factor)
        _ = overlay_blend_where(tensor, warp_tensor, blend_factor)

    # Benchmark Overlay
    start = time.time()
    iterations = 100
    for _ in range(iterations):
        _ = overlay_blend_masked(tensor, warp_tensor, blend_factor)
    t_masked = time.time() - start

    start = time.time()
    for _ in range(iterations):
        _ = overlay_blend_where(tensor, warp_tensor, blend_factor)
    t_where = time.time() - start

    print(f"Overlay Masked: {t_masked:.4f}s")
    print(f"Overlay Where:  {t_where:.4f}s")
    print(f"Speedup: {t_masked/t_where:.2f}x")

    # Verify correctness
    res_masked = overlay_blend_masked(tensor, warp_tensor, blend_factor)
    res_where = overlay_blend_where(tensor, warp_tensor, blend_factor)
    diff = torch.abs(res_masked - res_where).max().item()
    print(f"Overlay Max Diff: {diff:.8f}")
    assert diff < 1e-5, "Overlay implementations differ!"

    # Benchmark HSV
    h = torch.rand((B, 1, H, W), device=device)
    s = torch.rand((B, 1, H, W), device=device)
    v = torch.rand((B, 1, H, W), device=device)

    # Warmup
    for _ in range(10):
        _ = hsv_to_rgb_masked(h, s, v)
        _ = hsv_to_rgb_where(h, s, v)

    start = time.time()
    for _ in range(iterations):
        _ = hsv_to_rgb_masked(h, s, v)
    t_hsv_masked = time.time() - start

    start = time.time()
    for _ in range(iterations):
        _ = hsv_to_rgb_where(h, s, v)
    t_hsv_where = time.time() - start

    print(f"HSV Masked: {t_hsv_masked:.4f}s")
    print(f"HSV Where:  {t_hsv_where:.4f}s")
    print(f"HSV Speedup: {t_hsv_masked/t_hsv_where:.2f}x")

    # Verify HSV correctness
    r1, g1, b1 = hsv_to_rgb_masked(h, s, v)
    r2, g2, b2 = hsv_to_rgb_where(h, s, v)
    diff_hsv = max(
        torch.abs(r1 - r2).max().item(),
        torch.abs(g1 - g2).max().item(),
        torch.abs(b1 - b2).max().item()
    )
    print(f"HSV Max Diff: {diff_hsv:.8f}")
    assert diff_hsv < 1e-5, "HSV implementations differ!"

if __name__ == "__main__":
    benchmark()
