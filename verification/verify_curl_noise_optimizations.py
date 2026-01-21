
import sys
import os
import torch
import time

# Add project root
sys.path.append(os.getcwd())

try:
    from shaders.curl_noise import CurlNoiseGenerator
except ImportError:
    print("Could not import shaders.curl_noise. Make sure you are in the project root.")
    sys.exit(1)

def verify_random_val():
    print("--- Verifying random_val in module ---")
    # Access private helper by creating a mock or extracting it if possible
    # But random_val is inside get_curl_noise, so we can't access it directly.
    # We can run get_curl_noise with shape_type="spots" which uses random_val

    batch_size = 1
    height = 64
    width = 64
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    params = {
        "shape_type": "spots",
        "shapemaskstrength": 1.0,
        "base_seed": 12345
    }

    # Run once
    t0 = time.time()
    res1 = CurlNoiseGenerator.get_curl_noise(batch_size, height, width, params, device=device)
    t1 = time.time()

    # Run again
    res2 = CurlNoiseGenerator.get_curl_noise(batch_size, height, width, params, device=device)
    t2 = time.time()

    print(f"First run: {t1-t0:.4f}s")
    print(f"Second run: {t2-t1:.4f}s")

    diff = torch.abs(res1 - res2).max().item()
    print(f"Deterministic check (spots): max diff = {diff}")
    assert diff < 1e-6, "Output not deterministic!"
    print("✅ random_val (via spots) works and is deterministic")

def verify_interpolate_colors():
    print("\n--- Verifying _interpolate_colors in module ---")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    stops = [
        (0.0, (0.0, 0.0, 0.0)),
        (1.0, (1.0, 1.0, 1.0))
    ]
    t = torch.rand((1, 1, 64, 64), device=device)

    # First call - should cache
    t0 = time.time()
    r1, g1, b1 = CurlNoiseGenerator._interpolate_colors(stops, t)
    t1 = time.time()

    # Second call - should use cache
    r2, g2, b2 = CurlNoiseGenerator._interpolate_colors(stops, t)
    t2 = time.time()

    print(f"First call: {t1-t0:.4f}s")
    print(f"Second call: {t2-t1:.4f}s")

    diff = torch.abs(r1 - r2).max().item()
    assert diff < 1e-6
    print("✅ _interpolate_colors works and is deterministic")

if __name__ == "__main__":
    verify_random_val()
    verify_interpolate_colors()
