import math
import sys

def test_nan_clamp():
    val = float('nan')
    clamped = max(-1000.0, min(val, 1000.0))
    print(f"NaN clamped: {clamped}")
    if math.isnan(clamped):
        print("NaN passed through clamping (FAIL)")
    else:
        print("NaN successfully clamped (PASS)")

def test_inf_to_int():
    try:
        val = float('inf')
        int(val)
        print("int(inf) succeeded (FAIL)")
    except OverflowError:
        print("Caught OverflowError for int(inf) (PASS)")
    except Exception as e:
        print(f"Caught unexpected exception for int(inf): {e}")

if __name__ == "__main__":
    test_nan_clamp()
    test_inf_to_int()
