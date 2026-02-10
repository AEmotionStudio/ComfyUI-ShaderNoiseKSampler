
import torch
import torch.nn.functional as F
import math
import time
import re

# Legacy Implementation for Baseline
def legacy_grad3d(ix, iy, iz, gx, gy, gz, seed):
    h = (ix * 1619 + iy * 31337 + iz * 6971 + seed * 2459)
    h = torch.fmod(h * h * h, 1013)
    h_int = h.long() % 12

    u = torch.where(h_int < 8, gx, gy)
    v = torch.where(h_int < 4, gy, torch.where((h_int == 12) | (h_int == 14), gx, gz))
    return torch.where(h_int % 2 == 0, u, -u) + torch.where((h_int // 2) % 2 == 0, v, -v)

def legacy_simplex_3d(coords, seed=0):
    if isinstance(seed, torch.Tensor):
        seed = seed.item()
    seed = int(seed)

    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    F3 = 1.0 / 3.0
    G3 = 1.0 / 6.0

    s = (x + y + z) * F3
    i = torch.floor(x + s)
    j = torch.floor(y + s)
    k = torch.floor(z + s)

    t = (i + j + k) * G3
    x0 = x - (i - t)
    y0 = y - (j - t)
    z0 = z - (k - t)

    x_ge_y = (x0 >= y0).float()
    y_ge_z = (y0 >= z0).float()
    x_ge_z = (x0 >= z0).float()

    i1 = x_ge_y * x_ge_z
    j1 = (1 - x_ge_y) * y_ge_z
    k1 = (1 - x_ge_z) * (1 - y_ge_z)

    i2 = x_ge_y + (1 - x_ge_y) * x_ge_z
    j2 = x_ge_y * (1 - x_ge_z) + (1 - x_ge_y)
    k2 = (1 - x_ge_z) + x_ge_z * (1 - x_ge_y)

    noise = torch.zeros_like(x0)

    def compute_contribution(ix, iy, iz, dx, dy, dz):
        t = 0.6 - dx*dx - dy*dy - dz*dz
        mask = (t >= 0).float()
        t_sq = t * t
        t_quad = t_sq * t_sq
        g = legacy_grad3d(ix, iy, iz, dx, dy, dz, seed)
        return mask * t_quad * g

    noise = noise + compute_contribution(i, j, k, x0, y0, z0)

    x1 = x0 - i1 + G3
    y1 = y0 - j1 + G3
    z1 = z0 - k1 + G3
    noise = noise + compute_contribution(i + i1, j + j1, k + k1, x1, y1, z1)

    x2 = x0 - i2 + 2.0 * G3
    y2 = y0 - j2 + 2.0 * G3
    z2 = z0 - k2 + 2.0 * G3
    noise = noise + compute_contribution(i + i2, j + j2, k + k2, x2, y2, z2)

    x3 = x0 - 1.0 + 3.0 * G3
    y3 = y0 - 1.0 + 3.0 * G3
    z3 = z0 - 1.0 + 3.0 * G3
    noise = noise + compute_contribution(i + 1, j + 1, k + 1, x3, y3, z3)

    result = noise * 32.0
    return result.unsqueeze(-1)


# Read and prepare the codebase implementation
with open("shaders/temporal_coherent_noise.py", "r") as f:
    code = f.read()

# Patch imports
code = re.sub(r"from \.base import .*", "class BaseNoiseGenerator: pass", code)
code = re.sub(r"from \.registry import .*", "def shader_generator(*args, **kwargs): return lambda cls: cls", code)
code = re.sub(r"from \.\.utils.*", "", code)
code = re.sub(r"from \.\.core.*", "", code)
code = re.sub(r"logger = .*", "logger = None", code)

# Execute in a separate namespace
namespace = {
    "torch": torch,
    "math": math,
    "F": F,
    "DEFAULT_CHANNELS": 4,
    "logging": type("MockLogging", (), {"getLogger": lambda x: None}),
    "ShaderParams": type("ShaderParams", (), {}), # Mock ShaderParams
    "get_param_value": lambda x: x
}

try:
    exec(code, namespace)
    TemporalCoherentNoiseGenerator = namespace["TemporalCoherentNoiseGenerator"]
    print("✅ Successfully loaded TemporalCoherentNoiseGenerator from source")
except Exception as e:
    print(f"❌ Failed to load source: {e}")
    exit(1)


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    B, H, W = 1, 512, 512
    coords = torch.randn(B, H, W, 3, device=device)

    # Check optimized implementation from codebase
    optimized_simplex_3d = TemporalCoherentNoiseGenerator._simplex_3d

    # Verify correctness
    out_orig = legacy_simplex_3d(coords)
    out_opt = optimized_simplex_3d(coords)

    diff = (out_orig - out_opt).abs().max()
    print(f"Max difference: {diff.item()}")

    # We expect minor float differences due to operation order/optimization
    if diff > 1e-5:
        print("❌ Outputs do not match significantly!")
        print("Orig sample:", out_orig[0,0,0,0].item())
        print("Opt sample:", out_opt[0,0,0,0].item())
    else:
        print("✅ Outputs match within tolerance!")

    # Benchmark
    iterations = 50

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    for _ in range(iterations):
        legacy_simplex_3d(coords)
    torch.cuda.synchronize() if device.type == "cuda" else None
    dur_legacy = time.time() - start

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    for _ in range(iterations):
        optimized_simplex_3d(coords)
    torch.cuda.synchronize() if device.type == "cuda" else None
    dur_opt = time.time() - start

    print(f"Legacy: {dur_legacy:.4f}s")
    print(f"Optimized (Codebase): {dur_opt:.4f}s")
    speedup = dur_legacy / dur_opt if dur_opt > 0 else 0
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    run_benchmark()
