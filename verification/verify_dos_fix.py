import sys
import os
import unittest
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
    def test_octaves_clamping(self):
        # Import inside test to ensure mocks are active
        try:
            import temp_pkg.shader_noise_ksampler as snk
            import temp_pkg.direct_shader_ksampler as dsk
        except ImportError as e:
            self.fail(f"Failed to import from temp_pkg: {e}")

        # Patch ShaderNoiseKSampler.sample
        with patch('temp_pkg.shader_noise_ksampler.ShaderNoiseKSampler.sample') as mock_parent_sample:
            # Instantiate the sampler
            sampler = dsk.DirectShaderNoiseKSampler()

            # Create dummy inputs
            model = MagicMock()
            model.model.model_name = "test_model" # Avoid attribute error in logging

            latent_image = {"samples": MagicMock()}
            latent_image["samples"].device = "cpu"
            latent_image["samples"].shape = [1, 4, 64, 64] # B, C, H, W

            # Call sample with octaves=100 (HIGH VALUE)
            sampler.sample(
                model=model,
                seed=123,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                positive=MagicMock(),
                negative=MagicMock(),
                latent_image=latent_image,
                octaves=100.0 # <--- The malicious input
            )

            # Verify that parent sample was called
            self.assertTrue(mock_parent_sample.called, "Parent sample method was not called")

            # Get the arguments passed to parent sample
            call_args = mock_parent_sample.call_args
            kwargs = call_args.kwargs

            # Check shader_params_override
            shader_params = kwargs.get('shader_params_override')
            self.assertIsNotNone(shader_params, "shader_params_override was not passed")

            print(f"DEBUG: Passed octaves: {shader_params.get('octaves')}")
            print(f"DEBUG: Passed shaderOctaves: {shader_params.get('shaderOctaves')}")

            self.assertEqual(shader_params.get('octaves'), 20, "octaves should be clamped to 20")
            # We also want shaderOctaves to be clamped if it exists
            if 'shaderOctaves' in shader_params:
                self.assertEqual(shader_params.get('shaderOctaves'), 20, "shaderOctaves should be clamped to 20")

if __name__ == '__main__':
    unittest.main()
