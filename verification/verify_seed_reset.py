import torch
import sys
import os

# Add parent directory to path to import shader_params_reader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shader_params_reader import ShaderParamsReader

def verify_seed_reset():
    print("--- Verifying Global Seed Reset Vulnerability ---")

    # 1. Set a known global seed
    initial_seed = 12345
    torch.manual_seed(initial_seed)

    # Generate a random number to establish state
    r1 = torch.rand(1).item()
    print(f"Random value 1 (seed={initial_seed}): {r1}")

    # 2. Call ShaderParamsReader.random_val
    # This simulates the vulnerable call
    coords = torch.zeros((1, 10, 10, 2))
    base_seed = 999
    seed_offset = 1

    print(f"Calling ShaderParamsReader.random_val with base_seed={base_seed}...")
    _ = ShaderParamsReader.random_val(coords, base_seed, seed_offset)

    # 3. Generate another random number
    # If the seed was NOT reset, this should be the next number in the sequence for seed 12345
    # If the seed WAS reset (to 1000), this will be the first number for seed 1000
    r2 = torch.rand(1).item()
    print(f"Random value 2 (after function call): {r2}")

    # 4. Check against expected behavior
    # Reset to initial seed and get 2nd number
    torch.manual_seed(initial_seed)
    torch.rand(1) # skip first
    expected_r2_if_no_reset = torch.rand(1).item()

    # Reset to what random_val sets (999+1 = 1000)
    torch.manual_seed(base_seed + seed_offset)
    expected_r2_if_reset = torch.rand(1).item()

    print(f"Expected r2 if NO reset: {expected_r2_if_no_reset}")
    print(f"Expected r2 if reset: {expected_r2_if_reset}")

    if abs(r2 - expected_r2_if_reset) < 1e-6:
        print("\n[VULNERABILITY CONFIRMED] Global seed WAS reset by the function call.")
        print("This confirms that random_val() modifies global state unpredictably.")
        return True
    elif abs(r2 - expected_r2_if_no_reset) < 1e-6:
        print("\n[SAFE] Global seed was NOT reset.")
        return False
    else:
        print("\n[UNKNOWN] Something else happened.")
        return False

def verify_determinism():
    print("\n--- Verifying Determinism of random_val ---")
    # Verify that removing manual_seed doesn't affect output
    # We can't easily modify the code in memory, but we can check if output varies with global seed

    coords = torch.randn((1, 10, 10, 2))
    base_seed = 555
    seed_offset = 10

    # Set global seed to A
    torch.manual_seed(100)
    out1 = ShaderParamsReader.random_val(coords, base_seed, seed_offset)

    # Set global seed to B
    torch.manual_seed(200)
    out2 = ShaderParamsReader.random_val(coords, base_seed, seed_offset)

    # random_val sets the seed internally, so we expect out1 == out2 regardless of external seed
    # But wait, the function implementation is:
    # torch.manual_seed(...)
    # return ... (using deterministic math)

    # So the output SHOULD be deterministic based on inputs.
    # The vulnerability is the SIDE EFFECT.

    if torch.allclose(out1, out2):
         print("Output is consistent (as expected).")
    else:
         print("Output varies? That's unexpected.")

if __name__ == "__main__":
    if verify_seed_reset():
        verify_determinism()
        sys.exit(1) # Fail if vulnerable
    else:
        sys.exit(0) # Pass if safe
