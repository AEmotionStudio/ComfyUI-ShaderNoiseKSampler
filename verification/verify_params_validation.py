
import sys
import os
import unittest

# Ensure we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from shader_params_reader import ShaderParamsReader
except ImportError:
    # If torch is not installed or import fails for other reasons, we might need to mock it.
    # But since I installed torch, let's try to import directly first.
    print("Failed to import ShaderParamsReader. Checking torch installation...")
    try:
        import torch
        print("Torch is installed.")
    except ImportError:
        print("Torch is NOT installed. Mocking torch...")
        from unittest.mock import MagicMock
        sys.modules["torch"] = MagicMock()
        from shader_params_reader import ShaderParamsReader

class TestShaderParamsValidation(unittest.TestCase):
    def test_invalid_string_parameters_are_sanitized(self):
        """
        Verify that invalid string parameters ARE sanitized to safe defaults.
        """
        invalid_params = {
            "shader_type": "malicious_script_injection_attempt_or_just_garbage_text_that_could_cause_dos",
            "shape_type": "another_invalid_type_string_that_should_be_rejected",
            "colorScheme": "invalid_color_scheme_name",
            "octaves": 5,
            "scale": 1.5
        }

        # Act
        sanitized = ShaderParamsReader.validate_and_sanitize_params(invalid_params)

        # Assert - SECURE BEHAVIOR
        self.assertEqual(sanitized["shader_type"], "tensor_field")
        self.assertEqual(sanitized["shape_type"], "none")
        self.assertEqual(sanitized["colorScheme"], "none")

        # Assert valid parameters are kept
        self.assertEqual(sanitized["octaves"], 5)
        self.assertEqual(sanitized["scale"], 1.5)

        print("\n[VERIFICATION] SUCCESS: Invalid strings were sanitized to defaults.")

    def test_valid_string_parameters_are_kept(self):
        """
        Verify that valid string parameters are preserved.
        """
        valid_params = {
            "shader_type": "curl_noise",
            "shape_type": "spiral",
            "colorScheme": "plasma",
        }

        # Act
        sanitized = ShaderParamsReader.validate_and_sanitize_params(valid_params)

        # Assert
        self.assertEqual(sanitized["shader_type"], "curl_noise")
        self.assertEqual(sanitized["shape_type"], "spiral")
        self.assertEqual(sanitized["colorScheme"], "plasma")

        print("\n[VERIFICATION] SUCCESS: Valid strings were preserved.")

    def test_aliases(self):
        """
        Verify that aliases are correctly mapped.
        """
        alias_params = {
            "shader_type": "tensorfield" # Should map to tensor_field
        }
        sanitized = ShaderParamsReader.validate_and_sanitize_params(alias_params)
        self.assertEqual(sanitized["shader_type"], "tensor_field")
        print("\n[VERIFICATION] SUCCESS: Aliases were handled correctly.")

if __name__ == "__main__":
    unittest.main()
