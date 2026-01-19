import sys
import os
import unittest
import json
from unittest.mock import MagicMock, patch

# Add the current directory to sys.path
sys.path.insert(0, os.getcwd())

# Mock torch
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()

# Mock ComfyUI modules BEFORE importing the package
sys.modules["comfy"] = MagicMock()
sys.modules["comfy.sample"] = MagicMock()
sys.modules["comfy.samplers"] = MagicMock()
sys.modules["comfy.model_sampling"] = MagicMock()
sys.modules["comfy.model_base"] = MagicMock()
sys.modules["comfy.latent_formats"] = MagicMock()
sys.modules["nodes"] = MagicMock()
sys.modules["nodes.common_ksampler"] = MagicMock()

# Define test class
class TestDirectSamplerSecurity(unittest.TestCase):
    def setUp(self):
        # Import inside test to ensure mocks are active
        try:
            import temp_pkg.shader_noise_ksampler as snk
            import temp_pkg.direct_shader_ksampler as dsk
            self.dsk = dsk
        except ImportError as e:
            self.fail(f"Failed to import from temp_pkg: {e}")

        self.sampler = self.dsk.DirectShaderNoiseKSampler()

        # Create dummy inputs
        self.model = MagicMock()
        self.model.model.model_name = "test_model" # Avoid attribute error in logging

        self.latent_image = {"samples": MagicMock()}
        self.latent_image["samples"].device = "cpu"
        self.latent_image["samples"].shape = [1, 4, 64, 64] # B, C, H, W

    def test_octaves_clamping_basic(self):
        # Patch ShaderNoiseKSampler.sample
        with patch('temp_pkg.shader_noise_ksampler.ShaderNoiseKSampler.sample') as mock_parent_sample:
            # Call sample with octaves=100 (HIGH VALUE)
            self.sampler.sample(
                model=self.model,
                seed=123,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                positive=MagicMock(),
                negative=MagicMock(),
                latent_image=self.latent_image,
                octaves=100.0 # <--- The malicious input
            )

            # Verify that parent sample was called
            self.assertTrue(mock_parent_sample.called, "Parent sample method was not called")

            # Get the arguments passed to parent sample
            call_args = mock_parent_sample.call_args
            kwargs = call_args.kwargs
            shader_params = kwargs.get('shader_params_override')

            print(f"DEBUG (Basic): Passed octaves: {shader_params.get('octaves')}")

            self.assertEqual(shader_params.get('octaves'), 20, "octaves should be clamped to 20")

    def test_mapper_bypass(self):
        # Patch ShaderNoiseKSampler.sample
        with patch('temp_pkg.shader_noise_ksampler.ShaderNoiseKSampler.sample') as mock_parent_sample:
            # Craft a payload that uses the mapper to bypass
            # The simplified mapper has: elif attr == "smooth": adjustments["octaves"] = max(current_params["octaves"] - 1, 1)
            # If we pass octaves=1000000, and target_attribute_changes='{"smooth": 1.0}'
            # It calculates max(999999, 1) -> 999999 and writes it back.

            target_changes = json.dumps({"smooth": 1.0})

            self.sampler.sample(
                model=self.model,
                seed=123,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                positive=MagicMock(),
                negative=MagicMock(),
                latent_image=self.latent_image,
                octaves=1000000.0, # <--- Malicious input
                target_attribute_changes=target_changes
            )

            # Verify results
            self.assertTrue(mock_parent_sample.called)
            call_args = mock_parent_sample.call_args
            kwargs = call_args.kwargs
            shader_params = kwargs.get('shader_params_override')

            print(f"DEBUG (Mapper): Passed octaves: {shader_params.get('octaves')}")

            # Should be clamped to 20. If bypass works, it will be huge.
            self.assertLessEqual(shader_params.get('octaves'), 20, "octaves should be clamped even when using mapper")

if __name__ == '__main__':
    unittest.main()
