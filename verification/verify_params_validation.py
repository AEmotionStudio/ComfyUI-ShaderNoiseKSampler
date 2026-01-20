
import sys
import os
import unittest

# Ensure we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Handle import even if torch is missing in dev environment
try:
    from shader_params_reader import ShaderParamsReader
except ImportError:
    # If torch is not installed or import fails for other reasons, we might need to mock it.
    print("Failed to import ShaderParamsReader. Checking torch installation...")
    try:
        import torch
        print("Torch is installed.")
        # If torch is installed but import still failed, try importing again to see the real error
        from shader_params_reader import ShaderParamsReader
    except ImportError:
        print("Torch is NOT installed. Mocking torch...")
        from unittest.mock import MagicMock
        sys.modules["torch"] = MagicMock()
        try:
            from shader_params_reader import ShaderParamsReader
        except ImportError:
            print("Still failed to import ShaderParamsReader even with mock.")
            raise

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

    def test_legacy_integer_shape_types(self):
        """
        Verify that legacy integer shape types are correctly mapped.
        """
        # Test valid integer mapping
        legacy_params = {
            "shape_type": 1 # Should map to "circle"
        }
        sanitized = ShaderParamsReader.validate_and_sanitize_params(legacy_params)
        self.assertEqual(sanitized["shape_type"], "circle")

        # Test valid string-integer mapping
        legacy_str_params = {
            "shape_type": "2" # Should map to "square"
        }
        sanitized_str = ShaderParamsReader.validate_and_sanitize_params(legacy_str_params)
        self.assertEqual(sanitized_str["shape_type"], "square")

        print("\n[VERIFICATION] SUCCESS: Legacy integer shape types handled correctly.")

    def test_invalid_integer_shape_types(self):
        """
        Verify that INVALID integer shape types are rejected (default to none).
        """
        invalid_int_params = {
            "shape_type": 999
        }
        sanitized = ShaderParamsReader.validate_and_sanitize_params(invalid_int_params)
        self.assertEqual(sanitized["shape_type"], "none")

        invalid_str_int_params = {
            "shape_type": "999"
        }
        sanitized_str = ShaderParamsReader.validate_and_sanitize_params(invalid_str_int_params)
        self.assertEqual(sanitized_str["shape_type"], "none")

        print("\n[VERIFICATION] SUCCESS: Invalid integer shape types rejected.")

if __name__ == "__main__":
    unittest.main()
