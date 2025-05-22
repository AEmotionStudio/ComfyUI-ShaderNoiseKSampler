import json
import os
import torch
import math

class ShaderParamsReader:
    """
    Class for reading and applying shader parameters from JSON file
    Implements the Lt=Sα(N)∘Kβ(t) pattern where shader transforms are applied to noise before sampling
    """
    
    @staticmethod
    def smoothstep(edge0, edge1, x):
        """
        GLSL-style smoothstep.
        """
        # Ensure edges are tensors for broadcasting with x
        if not isinstance(edge0, torch.Tensor):
            edge0 = torch.full_like(x, float(edge0), device=x.device, dtype=x.dtype)
        if not isinstance(edge1, torch.Tensor):
            edge1 = torch.full_like(x, float(edge1), device=x.device, dtype=x.dtype)

        # Calculate t, handling potential edge0 >= edge1 cases by clamping
        delta = edge1 - edge0
        # Avoid division by zero or near-zero, maintain sign for correct 1-smoothstep
        safe_delta = torch.where(torch.abs(delta) < 1e-8, torch.sign(delta) * 1e-8 + 1e-8*(1-torch.abs(torch.sign(delta))), delta)
        t = torch.clamp((x - edge0) / safe_delta, 0.0, 1.0)

        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def random_val(coords, base_seed, seed_offset):
        """
        Generate a random-like value based on coordinates and seed.
        Matches the random_val helper in CurlNoiseGenerator.
        """
        # Ensure base_seed and seed_offset are appropriate for torch.manual_seed
        # torch.manual_seed expects an integer.
        current_seed = int(base_seed) + int(seed_offset)
        torch.manual_seed(current_seed)
        
        # Use a simple hash-like function based on coordinates
        # Ensuring coords are float for calculations
        coords_float = coords.float()
        hash_val = torch.sin(coords_float[:, :, :, 0] * (12.9898 + seed_offset) + coords_float[:, :, :, 1] * (78.233 + seed_offset)) * 43758.5453
        return torch.frac(hash_val)

    @staticmethod
    def get_shader_params(custom_path=None):
        """
        Utility function to read shader parameters from file.
        Returns a dictionary of shader parameters.
        
        Args:
            custom_path: Optional path to a custom JSON file
            
        Returns:
            Dictionary of shader parameters
        """
        # Default values in case file doesn't exist or is invalid
        default_params = {
            "shader_type": "tensor_field",
            "visualization_type": 3,  # ellipses
            "scale": 1.0,
            "phase_shift": 0.0,
            "warp_strength": 0.5,
            "time": 0.0,
            "octaves": 3.0,
            "intensity": 0.8,  # influence/strength of the shader
            "shapemaskstrength": 1.0,  # strength of the shape mask
            "shape_type": "none"  # type of shape mask
        }
        
        # Get the extension directory (where this file is located)
        EXTENSION_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the shader_params.json file (default or custom)
        if custom_path:
            params_file = custom_path
        else:
            # Try to find params in root directory first
            params_file = os.path.join(EXTENSION_DIR, "shader_params.json")
            
            # If not found, try the data folder
            if not os.path.exists(params_file):
                params_file = os.path.join(EXTENSION_DIR, "data", "shader_params.json")
        
        # Try to read the shader parameters from the file
        try:
            if os.path.exists(params_file):
                # print(f"Found parameters file: {params_file}")
                with open(params_file, 'r') as f:
                    loaded_params = json.load(f)
                    
                    # Map between different parameter naming conventions
                    param_mapping = {
                        "shaderType": "shader_type",
                        "shaderScale": "scale",
                        "shaderOctaves": "octaves",
                        "shaderWarpStrength": "warp_strength",
                        "shaderPhaseShift": "phase_shift",
                        "shapeMaskStrength": "shapemaskstrength",
                        "shaderShapeStrength": "shapemaskstrength",
                        "shaderShapeType": "shape_type"
                    }
                    
                    # Output raw loaded params for debugging
                    # print(f"Raw JSON params: {loaded_params}")
                    
                    # Convert parameter names if needed
                    params = {}
                    for key, value in loaded_params.items():
                        if key in param_mapping:
                            params[param_mapping[key]] = value
                        else:
                            params[key] = value
                            
                    # Special handling for shaderColorIntensity to maintain both versions
                    if "shaderColorIntensity" in loaded_params:
                        # Keep the original key
                        params["shaderColorIntensity"] = loaded_params["shaderColorIntensity"]
                        # Also provide as intensity for backward compatibility
                        params["intensity"] = loaded_params["shaderColorIntensity"]
                    
                    # Handle specific shader type mapping
                    if "shader_type" in params:
                        shader_type = params["shader_type"]
                        # Convert string values to standardized format
                        if shader_type.lower() == "tensor_field" or shader_type.lower() == "tensorfield":
                            params["shader_type"] = "tensor_field"
                        elif shader_type.lower() == "heterogeneous_fbm" or shader_type.lower() == "heterogeneousfbm":
                            params["shader_type"] = "heterogeneous_fbm"
                        elif shader_type.lower() == "projection_3d" or shader_type.lower() == "projection3d":
                            params["shader_type"] = "projection_3d"
                        elif shader_type.lower() == "cellular":
                            params["shader_type"] = "cellular"
                        elif shader_type.lower() == "fractal":
                            params["shader_type"] = "fractal"
                        elif shader_type.lower() == "perlin":
                            params["shader_type"] = "perlin"
                        elif shader_type.lower() == "waves":
                            params["shader_type"] = "waves"
                        elif shader_type.lower() == "gaussian":
                            params["shader_type"] = "gaussian"
                        elif shader_type.lower() == "domain_warp":
                            params["shader_type"] = "domain_warp"
                        elif shader_type.lower() == "interference" or shader_type.lower() == "interference_patterns":
                            params["shader_type"] = "interference_patterns"
                            print(f"Mapped shader type '{shader_type}' to 'interference_patterns'")
                        elif shader_type.lower() == "spectral" or shader_type.lower() == "spectral_noise":
                            params["shader_type"] = "spectral"
                            print(f"Mapped shader type '{shader_type}' to 'spectral'")
                        elif shader_type.lower() == "projection" or shader_type.lower() == "projection_3d" or shader_type.lower() == "3d_projection":
                            params["shader_type"] = "projection_3d"
                            print(f"Mapped shaderType '{shader_type}' to 'projection_3d'")
                        elif shader_type.lower() == "curl" or shader_type.lower() == "curl_noise":
                            params["shader_type"] = "curl_noise"
                            # print(f"Mapped shaderType '{shader_type}' to 'curl_noise'") # Mapped to curl_noise
                    
                    # Also check if shader type is in the shaderType field (alternate field name)
                    if "shaderType" in loaded_params and "shader_type" not in params:
                        shader_type = loaded_params["shaderType"]
                        if isinstance(shader_type, str):
                            if shader_type.lower() == "tensor_field" or shader_type.lower() == "tensorfield":
                                params["shader_type"] = "tensor_field"
                            elif shader_type.lower() == "heterogeneous_fbm" or shader_type.lower() == "heterogeneousfbm":
                                params["shader_type"] = "heterogeneous_fbm"
                            elif shader_type.lower() == "projection_3d" or shader_type.lower() == "projection3d":
                                params["shader_type"] = "projection_3d"
                            elif shader_type.lower() == "cellular":
                                params["shader_type"] = "cellular"
                            elif shader_type.lower() == "fractal":
                                params["shader_type"] = "fractal"
                            elif shader_type.lower() == "perlin":
                                params["shader_type"] = "perlin"
                            elif shader_type.lower() == "waves":
                                params["shader_type"] = "waves"
                            elif shader_type.lower() == "gaussian":
                                params["shader_type"] = "gaussian"
                            elif shader_type.lower() == "domain_warp":
                                params["shader_type"] = "domain_warp"
                            elif shader_type.lower() == "interference" or shader_type.lower() == "interference_patterns":
                                params["shader_type"] = "interference_patterns"
                                print(f"Mapped shaderType '{shader_type}' to 'interference_patterns'")
                            elif shader_type.lower() == "spectral" or shader_type.lower() == "spectral_noise":
                                params["shader_type"] = "spectral"
                                print(f"Mapped shaderType '{shader_type}' to 'spectral'")
                    
                    # Fill in any missing parameters with defaults
                    for key, value in default_params.items():
                        if key not in params:
                            params[key] = value
                    
                    # print(f"Successfully loaded shader parameters: {params}")
                    return params
            else:
                print(f"Parameters file not found at: {params_file}")
        except Exception as e:
            print(f"Error loading shader parameters: {e}")
        
        print(f"Using default shader parameters")
        return default_params
    
    @staticmethod
    def apply_shader_to_noise(noise, shader_params=None, influence=None):
        """
        Apply shader effects to the initial noise before sampling
        Implements the Sα(N) part of Lt=Sα(N)∘Kβ(t)
        
        Args:
            noise: Initial noise tensor [batch, channels, height, width]
            shader_params: Dictionary of shader parameters (or None to load from file)
            influence: How much to blend shader noise (0.0-1.0, None uses value from params)
            
        Returns:
            Modified noise tensor with same shape as input
        """
        if shader_params is None:
            shader_params = ShaderParamsReader.get_shader_params()
        
        # Extract basic parameters
        batch, channels, height, width = noise.shape
        device = noise.device
        
        # Use provided influence or get from parameters
        if influence is None:
            influence = shader_params.get("intensity", 0.8)
        
        # Ensure influence is a float
        influence = float(influence)
        
        # Skip if no influence
        if influence <= 0.0:
            return noise
        
        # Extract shader parameters
        shader_type = shader_params.get("shader_type", "tensor_field")
        viz_type = shader_params.get("visualization_type", 3)  # default to ellipses
        scale = shader_params.get("scale", 1.0)
        phase_shift = shader_params.get("phase_shift", 0.0)
        warp_strength = shader_params.get("warp_strength", 0.5)
        time = shader_params.get("time", 0.0)
        octaves = shader_params.get("octaves", 3.0)
        seed = shader_params.get("seed", 0)
        
        # Extract shape mask parameters
        shape_type = shader_params.get("shape_type", "none")
        shape_mask_strength = shader_params.get("shapemaskstrength", 1.0)
        
        # Debug print for shape mask parameters
        print(f"Shape mask parameters: type={shape_type}, strength={shape_mask_strength}")
        
        # Create coordinate grid (normalized to [-1, 1])
        y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=device),
                             torch.linspace(-1, 1, width, device=device),
                             indexing='ij')
        
        # Combine into coordinate tensor
        p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
        
        # Generate different shader patterns
        if False: # Placeholder for any future shader types to be handled here
            pass # Generate shader_noise for other types if needed
        else:
            # If the shader type was one of the removed ones or is not handled,
            # print a message and return the original noise unchanged.
            print(f"Shader type '{shader_type}' is not handled by apply_shader_to_noise or its simple implementation was removed. Returning original noise.")
            return noise
            
        # -- REMOVED Unreachable code: permutation, normalization, expansion, blending --
    
    @staticmethod
    def _lerp(a, b, t):
        """Helper for linear interpolation."""
        return a + (b - a) * t

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        """
        Convert HSV to RGB.
        h, s, v are expected in [0,1] range and shape [B, 1, H, W].
        Returns R, G, B components, each as [B, 1, H, W] in [0,1] range.
        """
        # Ensure inputs are correctly shaped for broadcasting if they are single values
        if not isinstance(h, torch.Tensor): h = torch.full_like(s if isinstance(s, torch.Tensor) else v, float(h)) # Fallback for s or v if h is scalar
        if not isinstance(s, torch.Tensor): s = torch.full_like(h, float(s))
        if not isinstance(v, torch.Tensor): v = torch.full_like(h, float(v))
        
        c = v * s
        h_prime = h * 6.0  # h is [0,1]
        
        # Ensure h_prime is a tensor for fmod
        if not isinstance(h_prime, torch.Tensor):
            h_prime = torch.full_like(c, float(h_prime))

        x = c * (1.0 - torch.abs(torch.fmod(h_prime, 2.0) - 1.0))
        m = v - c
        
        r, g, b = torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h)

        # Masks for hue ranges
        mask0 = (h_prime < 1.0)
        mask1 = (h_prime >= 1.0) & (h_prime < 2.0)
        mask2 = (h_prime >= 2.0) & (h_prime < 3.0)
        mask3 = (h_prime >= 3.0) & (h_prime < 4.0)
        mask4 = (h_prime >= 4.0) & (h_prime < 5.0)
        mask5 = (h_prime >= 5.0) # covers up to 6.0

        # Assign R, G, B based on hue
        r[mask0], g[mask0], b[mask0] = c[mask0], x[mask0], torch.zeros_like(x)[mask0]
        r[mask1], g[mask1], b[mask1] = x[mask1], c[mask1], torch.zeros_like(x)[mask1]
        r[mask2], g[mask2], b[mask2] = torch.zeros_like(x)[mask2], c[mask2], x[mask2]
        r[mask3], g[mask3], b[mask3] = torch.zeros_like(x)[mask3], x[mask3], c[mask3]
        r[mask4], g[mask4], b[mask4] = x[mask4], torch.zeros_like(x)[mask4], c[mask4]
        r[mask5], g[mask5], b[mask5] = c[mask5], torch.zeros_like(x)[mask5], x[mask5]
        
        r, g, b = r + m, g + m, b + m
        return r, g, b

    @staticmethod
    def _interpolate_colors(stops, t):
        """
        Interpolate colors based on stops.
        t is a normalized value tensor [B, 1, H, W] in [0,1] range.
        stops: list of [value, color_tuple_or_tensor e.g. (R,G,B) or [1,3,1,1] tensor].
        Returns R, G, B components, each as [B, 1, H, W] in [0,1] range (if stops are [0,1]).
        """
        device = t.device
        dtype = t.dtype
        
        processed_stops = []
        for val, color_val in stops:
            if isinstance(color_val, (list, tuple)):
                c_tensor = torch.tensor(color_val, device=device, dtype=dtype).view(1, 3, 1, 1)
            else: # assume it's already a tensor of shape [1,3,1,1] or broadcastable
                c_tensor = color_val.to(device=device, dtype=dtype)
                if len(c_tensor.shape) == 1 and c_tensor.shape[0] == 3: # e.g. torch.tensor([r,g,b])
                    c_tensor = c_tensor.view(1, 3, 1, 1)
            processed_stops.append((float(val), c_tensor))

        # Initialize final_color based on t's shape [B, 1, H, W] -> output [B, 3, H, W]
        final_color = torch.zeros((t.shape[0], 3, t.shape[2], t.shape[3]), device=device, dtype=dtype)
        
        for i in range(len(processed_stops) - 1):
            t0, c0 = processed_stops[i]    # c0 is [1, 3, 1, 1]
            t1, c1 = processed_stops[i+1]  # c1 is [1, 3, 1, 1]
            
            # Mask for pixels in the current segment
            if i == len(processed_stops) - 2: # Last segment includes t1
                segment_mask = (t >= t0) & (t <= t1)
            else: # Other segments are [t0, t1)
                segment_mask = (t >= t0) & (t < t1)

            # Normalize t within the segment [t0, t1] -> [0, 1]
            denominator = (t1 - t0)
            # Avoid division by zero if t0 == t1
            safe_denominator = torch.where(torch.abs(denominator) < 1e-8, torch.sign(denominator) * 1e-8 + 1e-8 * (1.0 - torch.abs(torch.sign(denominator))), denominator)
            
            local_t = (t - t0) / safe_denominator
            local_t_clamped = torch.clamp(local_t, 0.0, 1.0) # Shape [B, 1, H, W]
            
            # Lerp colors. c0, c1 are [1,3,1,1], local_t_clamped is [B,1,H,W]
            # Resulting interp_color_segment will be [B,3,H,W]
            interp_color_segment = ShaderParamsReader._lerp(c0, c1, local_t_clamped)
            
            # Apply using segment_mask (expanded to [B,3,H,W])
            final_color = torch.where(segment_mask.expand_as(interp_color_segment), interp_color_segment, final_color)

        # Handle cases where t is outside the defined stops range
        mask_below_first = t < processed_stops[0][0]
        first_color_expanded = processed_stops[0][1].expand_as(final_color)
        final_color = torch.where(mask_below_first.expand_as(final_color), first_color_expanded, final_color)
        
        mask_above_last = t > processed_stops[-1][0]
        last_color_expanded = processed_stops[-1][1].expand_as(final_color)
        final_color = torch.where(mask_above_last.expand_as(final_color), last_color_expanded, final_color)

        return final_color[:, 0:1], final_color[:, 1:2], final_color[:, 2:3]

    @staticmethod
    def apply_color_scheme(noise_tensor, shader_params=None):
        """
        Apply color scheme to a shader noise tensor based on shader_params
        
        Args:
            noise_tensor: Input noise tensor of shape [batch, channels, height, width]
            shader_params: Dictionary of shader parameters (or None to load from file)
            
        Returns:
            Modified noise tensor with color scheme applied
        """
        if shader_params is None:
            shader_params = ShaderParamsReader.get_shader_params()
        
        # Get color scheme and intensity parameters
        color_scheme = shader_params.get("colorScheme", "none")
        
        # Try to get the color intensity with priority for shaderColorIntensity
        color_intensity = shader_params.get("shaderColorIntensity", 
                           shader_params.get("intensity", 0.8))
        
        # Skip if no color scheme or zero intensity
        if color_scheme == "none" or color_intensity <= 0.0:
            print(f"Skipping color scheme application: scheme={color_scheme}, intensity={color_intensity}")
            return noise_tensor
            
        print(f"APPLYING COLOR SCHEME: {color_scheme} with intensity {color_intensity}")
        
        # Extract dimensions
        batch, channels, height, width = noise_tensor.shape
        device = noise_tensor.device
        
        # Create empty color tensor that we'll fill based on the scheme
        color_tensor = torch.zeros_like(noise_tensor)
        
        # Make sure to preserve the 4th channel if it exists
        if channels > 3:
            color_tensor[:, 3:] = noise_tensor[:, 3:]
        
        # Helper function to normalize the noise to 0-1 range for colormaps
        def normalize_to_01(tensor):
            return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        
        # Map base noise to 0-1 for color mapping
        base_noise = normalize_to_01(noise_tensor[:, 0])
        
        # Create a [B, 1, H, W] version of base_noise for helpers
        t_color = base_noise.unsqueeze(1)
        
        # Handle different color schemes
        if color_scheme == "rgb":
            # RGB color scheme: create three distinct channels
            if channels >= 3:
                # R channel - emphasize details in first latent dimension
                color_tensor[:, 0] = noise_tensor[:, 0] * 1.5
                # G channel - use second latent dimension with slight phase shift
                color_tensor[:, 1] = noise_tensor[:, 1] * 1.3
                # B channel - use third latent dimension with different scaling
                color_tensor[:, 2] = noise_tensor[:, 2] * 0.8
                
        elif color_scheme == "complementary":
            # Complementary colors: create opposing patterns in different channels
            if channels >= 3:
                # First channel - original
                color_tensor[:, 0] = noise_tensor[:, 0] * 1.5
                # Second channel - inverted phase from channel 0
                color_tensor[:, 1] = -noise_tensor[:, 0] * 0.8
                # Third channel - different frequency 
                color_tensor[:, 2] = noise_tensor[:, 2] * 1.2
                
        elif color_scheme == "monochrome":
            # Monochrome: apply the same pattern to all channels with slight variations
            if channels > 1:
                base_channel = noise_tensor[:, 0:1].clone()
                # Expand to all channels with slight variations in scaling
                scales = torch.tensor([1.0, 0.95, 0.9, 0.85][:channels], device=device).view(1, -1, 1, 1)
                color_tensor = base_channel * scales
                
        elif color_scheme == "gradient":
            # Gradient: create a position-based color gradient
            if channels >= 3:
                # Create coordinate grid for gradient
                y_norm = torch.linspace(0, 1, height, device=device).view(1, 1, -1, 1).expand(batch, 1, -1, width)
                x_norm = torch.linspace(0, 1, width, device=device).view(1, 1, 1, -1).expand(batch, 1, height, -1)
                
                # R channel - horizontal gradient + noise
                color_tensor[:, 0:1] = x_norm + noise_tensor[:, 0:1] * 0.4
                # G channel - vertical gradient + noise
                color_tensor[:, 1:2] = y_norm + noise_tensor[:, 1:2] * 0.4
                # B channel - diagonal gradient + noise
                color_tensor[:, 2:3] = (x_norm + y_norm) / 2 + noise_tensor[:, 2:3] * 0.4
                
        elif color_scheme == "blue_red":
            if channels >= 3:
                # Blue to red gradient (cold to hot) using lerp
                c0 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=t_color.dtype).view(1, 3, 1, 1) # Blue
                c1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=t_color.dtype).view(1, 3, 1, 1) # Red
                # _lerp expects t_color to be broadcastable with c0, c1.
                # t_color is [B,1,H,W], c0/c1 are [1,3,1,1]. Result is [B,3,H,W]
                interpolated_color = ShaderParamsReader._lerp(c0, c1, t_color)
                color_tensor[:, 0:1] = interpolated_color[:, 0:1] # Red
                color_tensor[:, 1:2] = interpolated_color[:, 1:2] # Green
                color_tensor[:, 2:3] = interpolated_color[:, 2:3] # Blue

        elif color_scheme == "viridis":
            if channels >= 3:
                stops = [
                    (0.0, (0.267, 0.005, 0.329)),  # #440154
                    (0.33, (0.188, 0.407, 0.553)), # #30678D
                    (0.66, (0.208, 0.718, 0.471)), # #35B778
                    (1.0, (0.992, 0.906, 0.143))   # #FDE724
                ]
                r, g, b = ShaderParamsReader._interpolate_colors(stops, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
                
        elif color_scheme == "plasma":
            if channels >= 3:
                # Use robust color stops for plasma, matching curl_noise.py
                stops = [
                    (0.0, (0.05, 0.03, 0.53)),
                    (0.25, (0.40, 0.00, 0.66)),
                    (0.5, (0.70, 0.18, 0.53)),
                    (0.75, (0.94, 0.46, 0.25)),
                    (1.0, (0.98, 0.80, 0.08))
                ]
                r, g, b = ShaderParamsReader._interpolate_colors(stops, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
        
        elif color_scheme == "inferno":
            if channels >= 3:
                stops = [
                    (0.0, (0.001, 0.001, 0.016)),
                    (0.25, (0.259, 0.039, 0.408)),
                    (0.5, (0.576, 0.149, 0.404)),
                    (0.75, (0.867, 0.318, 0.227)),
                    (0.85, (0.988, 0.647, 0.039)),
                    (1.0, (0.988, 1.000, 0.643))
                ]
                r, g, b = ShaderParamsReader._interpolate_colors(stops, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
        
        elif color_scheme == "magma":
            if channels >= 3:
                stops = [
                    (0.0, (0.001, 0.001, 0.016)),
                    (0.25, (0.231, 0.059, 0.439)),
                    (0.5, (0.549, 0.161, 0.506)),
                    (0.75, (0.871, 0.288, 0.408)),
                    (0.85, (0.996, 0.624, 0.427)),
                    (1.0, (0.988, 0.992, 0.749))
                ]
                r, g, b = ShaderParamsReader._interpolate_colors(stops, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
        
        elif color_scheme == "turbo":
            if channels >= 3:
                stops = [
                    (0.0, (0.188, 0.071, 0.235)),
                    (0.25, (0.275, 0.408, 0.859)),
                    (0.5, (0.149, 0.749, 0.549)),
                    (0.65, (0.831, 1.000, 0.314)),
                    (0.85, (0.980, 0.718, 0.298)),
                    (1.0, (0.729, 0.004, 0.000))
                ]
                r, g, b = ShaderParamsReader._interpolate_colors(stops, t_color)
                # Turbo often benefits from a slight boost/rescale
                r, g, b = r * 1.2 - 0.1, g * 1.2 - 0.1, b * 1.2 - 0.1
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = torch.clamp(r, 0, 1), torch.clamp(g, 0, 1), torch.clamp(b, 0, 1)

        elif color_scheme == "jet":
            if channels >= 3:
                stops = [
                    (0.0, (0.000, 0.000, 0.5)),    # Dark Blue
                    (0.125, (0.000, 0.000, 1.000)),# Blue
                    (0.375, (0.000, 1.000, 1.000)),# Cyan
                    (0.625, (1.000, 1.000, 0.000)),# Yellow
                    (0.875, (1.000, 0.000, 0.000)),# Red
                    (1.0, (0.500, 0.000, 0.000))   # Dark Red
                ]
                r, g, b = ShaderParamsReader._interpolate_colors(stops, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
                
        elif color_scheme == "rainbow":
            if channels >= 3:
                # Use HSV to RGB for rainbow: hue from t_color, constant saturation and value
                hue = t_color # base_noise is already [B,1,H,W] and [0,1]
                saturation = torch.ones_like(hue) * 0.9 # High saturation
                value = torch.ones_like(hue) * 0.9      # Bright value
                r, g, b = ShaderParamsReader._hsv_to_rgb(hue, saturation, value)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
                
        elif color_scheme == "cool":
            if channels >= 3:
                c0 = torch.tensor([0.0, 1.0, 1.0], device=device, dtype=t_color.dtype).view(1, 3, 1, 1) # Cyan
                c1 = torch.tensor([1.0, 0.0, 1.0], device=device, dtype=t_color.dtype).view(1, 3, 1, 1) # Magenta
                interpolated_color = ShaderParamsReader._lerp(c0, c1, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = interpolated_color[:,0:1], interpolated_color[:,1:2], interpolated_color[:,2:3]
                
        elif color_scheme == "hot":
            if channels >= 3:
                stops = [
                    (0.0, (0.0, 0.0, 0.0)),      # Black
                    (0.375, (1.0, 0.0, 0.0)),    # Red
                    (0.75, (1.0, 1.0, 0.0)),     # Yellow
                    (1.0, (1.0, 1.0, 1.0))       # White
                ]
                r, g, b = ShaderParamsReader._interpolate_colors(stops, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
                
        elif color_scheme == "parula":
            if channels >= 3:
                stops = [
                    (0.0, (0.208, 0.165, 0.529)), # #352a87
                    (0.25, (0.059, 0.361, 0.867)), # #0f5cdd
                    (0.5, (0.000, 0.710, 0.651)), # #00b5a6
                    (0.75, (1.000, 0.765, 0.216)), # #ffc337
                    (1.0, (0.988, 0.996, 0.643))  # #fcfea4
                ]
                r, g, b = ShaderParamsReader._interpolate_colors(stops, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
                
        elif color_scheme == "hsv":
            if channels >= 3:
                hue = t_color 
                saturation = torch.ones_like(hue) * 0.95 # Full saturation
                value = torch.ones_like(hue) * 0.95      # Full value
                r, g, b = ShaderParamsReader._hsv_to_rgb(hue, saturation, value)
                # Original SPR HSV scaled output to [-1,1]. We keep [0,1] from _hsv_to_rgb for consistency with other interpolated.
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
                
        elif color_scheme == "autumn":
            if channels >= 3:
                c0 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=t_color.dtype).view(1, 3, 1, 1) # Red
                c1 = torch.tensor([1.0, 1.0, 0.0], device=device, dtype=t_color.dtype).view(1, 3, 1, 1) # Yellow
                interpolated_color = ShaderParamsReader._lerp(c0, c1, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = interpolated_color[:,0:1], interpolated_color[:,1:2], interpolated_color[:,2:3]

        elif color_scheme == "winter":
            if channels >= 3:
                c0 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=t_color.dtype).view(1, 3, 1, 1) # Blue
                c1 = torch.tensor([0.0, 1.0, 0.5], device=device, dtype=t_color.dtype).view(1, 3, 1, 1) # Greenish-Cyan
                interpolated_color = ShaderParamsReader._lerp(c0, c1, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = interpolated_color[:,0:1], interpolated_color[:,1:2], interpolated_color[:,2:3]

        elif color_scheme == "spring":
            if channels >= 3:
                c0 = torch.tensor([1.0, 0.0, 1.0], device=device, dtype=t_color.dtype).view(1, 3, 1, 1) # Magenta
                c1 = torch.tensor([1.0, 1.0, 0.0], device=device, dtype=t_color.dtype).view(1, 3, 1, 1) # Yellow
                interpolated_color = ShaderParamsReader._lerp(c0, c1, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = interpolated_color[:,0:1], interpolated_color[:,1:2], interpolated_color[:,2:3]

        elif color_scheme == "summer":
            if channels >= 3:
                c0 = torch.tensor([0.0, 0.5, 0.4], device=device, dtype=t_color.dtype).view(1, 3, 1, 1) # Dark Green
                c1 = torch.tensor([1.0, 1.0, 0.4], device=device, dtype=t_color.dtype).view(1, 3, 1, 1) # Yellow
                interpolated_color = ShaderParamsReader._lerp(c0, c1, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = interpolated_color[:,0:1], interpolated_color[:,1:2], interpolated_color[:,2:3]
                
        elif color_scheme == "copper":
            if channels >= 3:
                c0 = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=t_color.dtype).view(1, 3, 1, 1) # Black
                c1 = torch.tensor([1.0, 0.6235, 0.3922], device=device, dtype=t_color.dtype).view(1, 3, 1, 1) # Copper color approx (255,159,100)
                interpolated_color = ShaderParamsReader._lerp(c0, c1, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = interpolated_color[:,0:1], interpolated_color[:,1:2], interpolated_color[:,2:3]
                
        elif color_scheme == "pink":
            if channels >= 3:
                stops = [
                    (0.0, (0.05, 0.05, 0.05)), # Dark gray
                    (0.5, (1.0, 0.41, 0.71)),   # Hot Pink approx
                    (1.0, (1.0, 0.75, 0.80))    # Light Pink
                ]
                r, g, b = ShaderParamsReader._interpolate_colors(stops, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
                
        elif color_scheme == "bone":
            if channels >= 3:
                stops = [ # Standard bone colormap
                    (0.0, (0.0, 0.0, 0.0)),
                    (0.375, (0.3294, 0.3294, 0.4549)), # (84, 84, 116)
                    (0.75, (0.6275, 0.7569, 0.7569)),  # (160, 193, 193)
                    (1.0, (1.0, 1.0, 1.0))
                ]
                r, g, b = ShaderParamsReader._interpolate_colors(stops, t_color)
                # Original shader_params_reader 'bone' scaled to [-1,1]. Let's keep [0,1] for consistency.
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
                
        elif color_scheme == "ocean":
            if channels >= 3:
                 stops = [ # Based on matplotlib's ocean
                     (0.0, (0.0, 0.0, 0.0)),      # Black
                     (0.33, (0.0, 0.0, 0.5)),     # Dark Blue
                     (0.66, (0.0, 0.5, 1.0)),     # Light Blue
                     (1.0, (0.7, 1.0, 1.0))       # Very Light Cyan/White
                 ]
                 r, g, b = ShaderParamsReader._interpolate_colors(stops, t_color)
                 color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
                
        elif color_scheme == "terrain":
            if channels >= 3:
                stops = [ # Standard terrain colormap
                    (0.0, (0.2, 0.2, 0.6)),      # Deep water blue
                    (0.15, (0.0, 0.5, 0.0)),     # Dark Green (low land)
                    (0.33, (0.0, 0.8, 0.4)),     # Green (land)
                    (0.5, (0.87, 0.87, 0.4)),    # Yellowish (hills)
                    (0.75, (0.6, 0.4, 0.2)),     # Brown (mountains)
                    (1.0, (1.0, 1.0, 1.0))       # White (snow peaks)
                ]
                r, g, b = ShaderParamsReader._interpolate_colors(stops, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
                
        elif color_scheme == "neon":
            if channels >= 3:
                # Using a multi-stop lerp for vibrant neon effect
                stops = [
                    (0.0, (1.0, 0.0, 0.5)),   # Magenta
                    (0.33, (0.0, 1.0, 1.0)),  # Cyan
                    (0.66, (1.0, 1.0, 0.0)),  # Yellow
                    (1.0, (0.5, 0.0, 1.0))    # Purple
                ]
                r, g, b = ShaderParamsReader._interpolate_colors(stops, t_color)
                color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
                
        elif color_scheme == "fire":
            if channels >= 3:
                 stops = [ # Standard fire colormap
                     (0.0, (0.0, 0.0, 0.0)),      # Black
                     (0.25, (1.0, 0.0, 0.0)),     # Red
                     (0.6, (1.0, 1.0, 0.0)),      # Yellow
                     (1.0, (1.0, 1.0, 1.0))       # White
                 ]
                 r, g, b = ShaderParamsReader._interpolate_colors(stops, t_color)
                 color_tensor[:, 0:1], color_tensor[:, 1:2], color_tensor[:, 2:3] = r, g, b
                 
        elif color_scheme == "fantasy":
            # Fantasy colors: magical and otherworldly - keeping original SPR logic
            if channels >= 3:
                # Create swirling color pattern
                angle = torch.atan2(noise_tensor[:, 1], noise_tensor[:, 0])
                radius = torch.sqrt(noise_tensor[:, 0]**2 + noise_tensor[:, 1]**2)
                
                # Purple/pink base
                color_tensor[:, 0] = torch.sin(angle * 2.0 + radius * 3.0) * 0.5 + 0.5
                # Teal/blue variations
                color_tensor[:, 1] = torch.sin(angle * 3.0 - radius * 2.0) * 0.5 + 0.5
                # Golden highlights
                color_tensor[:, 2] = torch.sin(radius * 5.0) * 0.5 + 0.5
                
                # Normalize to maintain proper distribution
                color_tensor = (color_tensor - 0.5) * 2.0
        else:
            # Default case - return original noise if color scheme not implemented or recognized
            print(f"WARNING: Color scheme '{color_scheme}' not recognized, using original noise")
            return noise_tensor
                
        # Blend with original based on intensity
        # Ensure color_tensor values are appropriately scaled if necessary before blending.
        # For now, assuming [0,1] range from most new schemes is acceptable for blending.
        result = noise_tensor * (1.0 - color_intensity) + color_tensor * color_intensity
        print(f"Applied {color_scheme} color scheme - result shape: {result.shape}")
        return result

    @staticmethod
    def apply_shape_mask(coords_normalized_01, shape_type, time=0.0, base_seed=0, use_temporal_coherence=False):
        """
        Apply shape mask to coordinates.
        Coordinates are expected to be in the [0, 1] range.
        
        Args:
            coords_normalized_01: Coordinate tensor [batch, height, width, 2] in [0, 1] range.
            shape_type: Type of shape to apply (integer or string).
            time: Animation time.
            base_seed: Base seed for randomness if shapes require it.
            use_temporal_coherence: Flag for temporal coherence.
            
        Returns:
            Shape mask tensor [batch, height, width, 1]
        """
        batch, height, width, _ = coords_normalized_01.shape
        device = coords_normalized_01.device

        # For shapes that assume coordinates centered at (0,0) and range approx [-0.5, 0.5] or [-1,1]
        # we create centered coordinates from the [0,1] input.
        centered_coords = coords_normalized_01 - 0.5 # Now in [-0.5, 0.5] range
        
        # Distance from center for centered_coords
        center_dist = torch.sqrt(centered_coords[:, :, :, 0]**2 + centered_coords[:, :, :, 1]**2) # Max dist ~0.707
        
        # Angle from center for centered_coords
        angle = torch.atan2(centered_coords[:, :, :, 1], centered_coords[:, :, :, 0])
        
        # Default mask
        mask_output = torch.ones((batch, height, width), device=device)
        
        # Convert string shape_type to standardized string format
        if isinstance(shape_type, str):
            shape_type = shape_type.lower()
        
        # Handle both numeric and string shape types
        # Note: Shapes from original apply_shape_mask are adapted to the new coordinate system.
        # The radius/size parameters might need adjustment if they were tuned for [-1,1] p.
        
        if shape_type == 1 or shape_type == "circle": # Original "radial" was also circle
            # Circle - centered_coords range from approx -0.5 to 0.5. center_dist max ~0.707
            # To make a circle that fills most of the [0,1] original space, radius should be ~0.5
            # CN version: 1.0 - torch.clamp(dist * 2, 0, 1) where dist is from center of [0,1] grid
            # For coords_normalized_01, dist from center (0.5,0.5) is `center_dist_01`
            center_x_01, center_y_01 = 0.5, 0.5
            y_diff_01 = coords_normalized_01[:, :, :, 1] - center_y_01
            x_diff_01 = coords_normalized_01[:, :, :, 0] - center_x_01
            dist_01 = torch.sqrt(x_diff_01**2 + y_diff_01**2)
            mask_output = 1.0 - torch.clamp(dist_01 * 2.0, 0.0, 1.0) # Match CN circle
            
        elif shape_type == 2 or shape_type == "square":
            # Square - centered_coords values are in [-0.5, 0.5]
            # CN version: x_mask = torch.abs(coords_bhwc[:, :, :, 0] - 0.5) * 2
            #             y_mask = torch.abs(coords_bhwc[:, :, :, 1] - 0.5) * 2
            #             dist = torch.max(x_mask, y_mask)
            #             shape_mask = 1.0 - torch.clamp(dist, 0, 1)
            # This uses coords_normalized_01 (same as coords_bhwc in CN)
            x_mask_sq = torch.abs(coords_normalized_01[:, :, :, 0] - 0.5) * 2.0
            y_mask_sq = torch.abs(coords_normalized_01[:, :, :, 1] - 0.5) * 2.0
            dist_sq = torch.max(x_mask_sq, y_mask_sq)
            mask_output = 1.0 - torch.clamp(dist_sq, 0.0, 1.0)

        elif shape_type == "radial": # Use the same logic as radial_animated (curl_noise.py radial)
            # Uses coords_normalized_01 (range [0,1])
            time_tensor = torch.tensor(time, device=device, dtype=coords_normalized_01.dtype)
            center_x = 0.5 + 0.2 * torch.cos(time_tensor)
            center_y = 0.5 + 0.2 * torch.sin(time_tensor)
            # Calculate distance from the animated center using coords_normalized_01
            y_diff = coords_normalized_01[:, :, :, 1] - center_y
            x_diff = coords_normalized_01[:, :, :, 0] - center_x
            dist_from_anim_center = torch.sqrt(x_diff**2 + y_diff**2) * 2.0 # Multiplied by 2 like in curl_noise
            mask_output = torch.clamp(1.0 - dist_from_anim_center, 0.0, 1.0)
            
        elif shape_type == 3 or shape_type == "star": # SPR original "star"
            # Star-like shape - using centered_coords
            points = 5.0
            star_radius = 0.25 + 0.125 * torch.cos(angle * points) 
            mask_output = (center_dist < star_radius).float()
            
        elif shape_type == "linear": # Ported from curl_noise.py
            # Uses original [0,1] coordinates (coords_normalized_01)
            # Convert time calculation to tensor for torch.fmod
            time_tensor_02 = torch.tensor(time * 0.2, device=device, dtype=coords_normalized_01.dtype)
            x_offset = torch.fmod(time_tensor_02, 1.0) * 2.0 # Match JS fract animation
            shifted_x = torch.fmod(coords_normalized_01[:, :, :, 0] + x_offset, 1.0)
            mask_output = shifted_x # Mask values will be [0,1]
            
        elif shape_type == "radial_animated": # Ported from curl_noise.py (its "radial" shape)
            # Uses coords_normalized_01 (range [0,1])
            time_tensor = torch.tensor(time, device=device, dtype=coords_normalized_01.dtype)
            center_x = 0.5 + 0.2 * torch.cos(time_tensor)
            center_y = 0.5 + 0.2 * torch.sin(time_tensor)
            # Calculate distance from the animated center using coords_normalized_01
            y_diff = coords_normalized_01[:, :, :, 1] - center_y
            x_diff = coords_normalized_01[:, :, :, 0] - center_x
            dist_from_anim_center = torch.sqrt(x_diff**2 + y_diff**2) * 2.0 # Multiplied by 2 like in curl_noise
            mask_output = torch.clamp(1.0 - dist_from_anim_center, 0.0, 1.0)
            
        elif shape_type == "spiral": # Ported and enhanced from curl_noise.py, uses centered_coords
            # centered_coords are in [-0.5, 0.5]
            # theta and r are calculated from centered_coords
            # angle = atan2(centered_coords_y, centered_coords_x) - already available as 'angle'
            # r = norm(centered_coords) * 2.0 - center_dist is norm(centered_coords), so r = center_dist * 2.0
            r_spiral = center_dist * 2.0 # center_dist is norm of coords in [-0.5,0.5], max ~0.707. So r_spiral max ~1.414
            
            time_tensor = torch.tensor(time, device=device, dtype=coords_normalized_01.dtype)
            theta_animated = angle + time_tensor # angle is already calculated from centered_coords
            
            mask_output = torch.fmod((theta_animated / (2.0 * math.pi) + r_spiral), 1.0)
            
        elif shape_type == "checkerboard": # Ported from curl_noise.py
            # Uses coords_normalized_01 (range [0,1])
            grid_size = 8.0 # From curl_noise.py
            # Convert time expressions to tensors
            time_tensor_gs_02 = torch.tensor(time * grid_size * 0.2, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_gs_01 = torch.tensor(time * grid_size * 0.1, device=device, dtype=coords_normalized_01.dtype)
            
            x_offset = time_tensor_gs_02
            y_offset = time_tensor_gs_01
            
            # Apply offset and scale for grid calculation
            x_grid_val = (coords_normalized_01[:, :, :, 0] + x_offset / grid_size) * grid_size
            y_grid_val = (coords_normalized_01[:, :, :, 1] + y_offset / grid_size) * grid_size
            
            # Floor and compute pattern. The *0.5 and fmod 1.0 results in 0 or 0.5 values, then combined for 0, 0.5, 1.0.
            # To get a binary mask (0 or 1), we can fmod the sum by 2 and then check if it's < 1, or directly use fmod 1.0 from curl_noise.
            x_grid_processed = torch.floor(x_grid_val) * 0.5
            y_grid_processed = torch.floor(y_grid_val) * 0.5
            mask_output = torch.fmod(x_grid_processed + y_grid_processed, 1.0)
            
        elif shape_type == "spots": # Ported from curl_noise.py (more complex version)
            # Uses coords_normalized_01 (same as coords_bhwc in CN)
            mask_spots_cn = torch.zeros_like(coords_normalized_01[:, :, :, 0])
            num_spots_cn = 10
            time_tensor_cn_spots = torch.tensor(time, device=device, dtype=coords_normalized_01.dtype)
            
            for i in range(num_spots_cn):
                # Use ShaderParamsReader.random_val
                rand_x_cn = ShaderParamsReader.random_val(coords_normalized_01, base_seed, i * 78)
                rand_y_cn = ShaderParamsReader.random_val(coords_normalized_01, base_seed, i * 12)
                size_cn_base = (ShaderParamsReader.random_val(coords_normalized_01, base_seed, i * 93) * 0.3 + 0.1)
                
                angle_float_cn = time + float(i) # time is already a float or tensor
                angle_tensor_cn = torch.tensor(angle_float_cn, device=device, dtype=coords_normalized_01.dtype)
                spot_pos_x_cn = 0.5 + torch.cos(angle_tensor_cn) * 0.4 * rand_x_cn
                spot_pos_y_cn = 0.5 + torch.sin(angle_tensor_cn) * 0.4 * rand_y_cn
                
                size_anim_angle_cn = torch.tensor(time * 2.0 + float(i), device=device, dtype=coords_normalized_01.dtype)
                size_cn_final = size_cn_base * (1.0 + 0.2 * torch.sin(size_anim_angle_cn))
                
                dist_cn_spots = torch.sqrt((coords_normalized_01[:, :, :, 0] - spot_pos_x_cn)**2 + (coords_normalized_01[:, :, :, 1] - spot_pos_y_cn)**2)
                # Avoid division by zero or very small size
                spot_mask_cn_indiv = torch.clamp(1.0 - dist_cn_spots / (size_cn_final + 1e-8), 0.0, 1.0)
                mask_spots_cn = torch.maximum(mask_spots_cn, spot_mask_cn_indiv)
            mask_output = mask_spots_cn
            
        elif shape_type == "hexgrid": # Adapted from original, using centered_coords
            # This was complex. Let's simplify for [0,1] input.
            # Using coords_normalized_01 directly for hexgrid based on curl_noise's hexgrid logic
            hex_uv = coords_normalized_01 * 6.0 # Scale for hex grid density

            # Convert time expressions to tensors before sin/cos
            time_tensor_05 = torch.tensor(time * 0.5, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_03 = torch.tensor(time * 0.3, device=device, dtype=coords_normalized_01.dtype)
            time_tensor = torch.tensor(time, device=device, dtype=coords_normalized_01.dtype)

            hex_uv_anim = hex_uv.clone() # Avoid in-place modification if hex_uv is reused
            hex_uv_anim[:, :, :, 0] += torch.sin(time_tensor_05) * 0.5
            hex_uv_anim[:, :, :, 1] += torch.cos(time_tensor_03) * 0.5
            
            r_vec = torch.tensor([1.0, 1.73], device=device, dtype=coords_normalized_01.dtype).reshape(1, 1, 1, 2)
            h_vec = r_vec * 0.5
            a_vec = torch.fmod(hex_uv_anim, r_vec) - h_vec
            b_vec = torch.fmod(hex_uv_anim + h_vec, r_vec) - h_vec
            
            dist_hex = torch.minimum(torch.norm(a_vec, dim=-1), torch.norm(b_vec, dim=-1))
            cell_size = 0.3 + 0.1 * torch.sin(time_tensor)
            # Use ShaderParamsReader.smoothstep
            mask_output = ShaderParamsReader.smoothstep(cell_size + 0.05, cell_size - 0.05, dist_hex)

        elif shape_type == "stripes": # Ported from curl_noise.py
            # Uses coords_normalized_01 (range [0,1])
            freq = 10.0
            # Convert time expressions to tensors
            time_tensor = torch.tensor(time, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_02 = torch.tensor(time * 0.2, device=device, dtype=coords_normalized_01.dtype)
            
            angle_anim = 0.5 * torch.sin(time_tensor_02)
            cos_a = torch.cos(angle_anim)
            sin_a = torch.sin(angle_anim)
            
            # Rotate coordinates directly using coords_normalized_01 to match curl_noise.py
            # This rotates around the (0,0) origin of the [0,1] coordinate system.
            rotated_x = coords_normalized_01[:, :, :, 0] * cos_a - coords_normalized_01[:, :, :, 1] * sin_a
            # rotated_y is not strictly needed for this pattern as stripes are based on rotated_x
            
            stripes_val = torch.sin(rotated_x * freq + time_tensor) 
            # Apply smoothstep to create distinct stripes
            mask_output = ShaderParamsReader.smoothstep(0.0, 0.1, stripes_val) * ShaderParamsReader.smoothstep(0.0, -0.1, -stripes_val)
            
        elif shape_type == "radial_gradient_static": # Renamed from "gradient"
            # This is the original radial gradient from shader_params_reader.py
            # center_dist max ~0.707 for centered_coords (which are coords_normalized_01 - 0.5).
            # To have gradient from center to edge of original [0,1] box, we need to normalize center_dist.
            # Max distance from center of a [0,1] box is sqrt(0.5^2+0.5^2) = ~0.707.
            # So center_dist / 0.707 normalizes it roughly to [0,1] for points within the box.
            # Or simpler, just use center_dist directly, it gives a gradient from 0 to ~0.7
            mask_output = 1.0 - torch.clamp(center_dist / 0.5, 0.0, 1.0) # Soft radial gradient, 0.5 radius

        elif shape_type == "gradient": # Ported from curl_noise.py (animated directional gradient)
            # Uses coords_normalized_01 (equivalent to coords_bhwc in curl_noise.py)
            time_tensor_02_grad = torch.tensor(time * 0.2, device=device, dtype=coords_normalized_01.dtype)
            angle_grad = time_tensor_02_grad
            dir_x_grad = torch.cos(angle_grad)
            dir_y_grad = torch.sin(angle_grad)
            # Project centered coordinates onto the direction vector
            # coords_normalized_01 are [0,1], so (coords_normalized_01 - 0.5) makes them [-0.5, 0.5]
            proj_grad = (coords_normalized_01[:, :, :, 0] - 0.5) * dir_x_grad + \
                        (coords_normalized_01[:, :, :, 1] - 0.5) * dir_y_grad + 0.5
            mask_output = proj_grad # Result is roughly in [0,1] range

        elif shape_type == "vignette": # Ported from curl_noise.py
            # Uses coords_normalized_01 (range [0,1])
            # Convert time expressions to tensors
            time_tensor_03 = torch.tensor(time * 0.3, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_04 = torch.tensor(time * 0.4, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_05 = torch.tensor(time * 0.5, device=device, dtype=coords_normalized_01.dtype)

            center_x_anim = 0.5 + 0.2 * torch.sin(time_tensor_03)
            center_y_anim = 0.5 + 0.2 * torch.cos(time_tensor_04)
            
            # Calculate distance from animated center using coords_normalized_01
            dist_from_anim_center_x = coords_normalized_01[:, :, :, 0] - center_x_anim
            dist_from_anim_center_y = coords_normalized_01[:, :, :, 1] - center_y_anim
            dist_vignette = torch.sqrt(dist_from_anim_center_x**2 + dist_from_anim_center_y**2)
            
            radius_anim = 0.6 + 0.2 * torch.sin(time_tensor_05)
            smoothness = 0.3 # As in curl_noise.py
            
            mask_output = 1.0 - ShaderParamsReader.smoothstep(radius_anim - smoothness, radius_anim, dist_vignette)
            
        elif shape_type == "cross": # Ported from curl_noise.py
            # Uses coords_normalized_01 (range [0,1]), but calculations are around center (0.5,0.5)
            # Convert time expressions to tensors
            time_tensor = torch.tensor(time, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_02 = torch.tensor(time * 0.2, device=device, dtype=coords_normalized_01.dtype)
            
            thickness_anim = 0.1 + 0.05 * torch.sin(time_tensor)
            rotation_anim = time_tensor_02
            cos_r = torch.cos(rotation_anim)
            sin_r = torch.sin(rotation_anim)
            
            # Use centered_coords for rotation calculation
            # centered_coords are already coords_normalized_01 - 0.5
            rotated_x = centered_coords[:, :, :, 0] * cos_r - centered_coords[:, :, :, 1] * sin_r 
            rotated_y = centered_coords[:, :, :, 0] * sin_r + centered_coords[:, :, :, 1] * cos_r
            
            # Shift back to [0,1]-like domain for comparison if needed, or compare in [-0.5,0.5] domain
            # The original curl_noise compared rotated_x/y against 0.5 after adding 0.5.
            # Here, rotated_x/y are already centered around 0. So, we compare against 0.
            
            # Horizontal bar (rotated)
            h_bar = ShaderParamsReader.smoothstep(0.0 - thickness_anim, 0.0 - thickness_anim + 0.02, rotated_y) * \
                    ShaderParamsReader.smoothstep(0.0 + thickness_anim, 0.0 + thickness_anim - 0.02, rotated_y)
            # Vertical bar (rotated)
            v_bar = ShaderParamsReader.smoothstep(0.0 - thickness_anim, 0.0 - thickness_anim + 0.02, rotated_x) * \
                    ShaderParamsReader.smoothstep(0.0 + thickness_anim, 0.0 + thickness_anim - 0.02, rotated_x)
            mask_output = torch.maximum(h_bar, v_bar)
            
        elif shape_type == "triangles": # Adapted from original, using centered_coords
            # Re-evaluate scaling for centered_coords.
            # Let's use coords_normalized_01 for a direct port attempt of a triangle grid like curl_noise
            t_tensor = torch.tensor(time * 0.2, device=device, dtype=coords_normalized_01.dtype)
            t_sin_arg = t_tensor
            t_cos_arg = torch.tensor(time * 0.7, device=device, dtype=coords_normalized_01.dtype)
            t_border_arg = torch.tensor(time * 1.5, device=device, dtype=coords_normalized_01.dtype)
            scale_factor = 5.0
            uv_tri = coords_normalized_01 * scale_factor
            uv_tri_anim = uv_tri.clone()
            uv_tri_anim[:, :, :, 0] += torch.sin(t_sin_arg) * 0.5
            uv_tri_anim[:, :, :, 1] += torch.cos(t_cos_arg) * 0.5
            
            gv = torch.fmod(uv_tri_anim, 1.0) - 0.5 # gv is now in [-0.5, 0.5]
            
            d1 = torch.abs(gv[:, :, :, 0] + gv[:, :, :, 1])
            d2 = torch.abs(gv[:, :, :, 0] - gv[:, :, :, 1])
            d3 = torch.abs(gv[:, :, :, 0]) * 0.866 + torch.abs(gv[:, :, :, 1]) * 0.5 # Approx dist for equilateral
            
            d_tri = torch.minimum(torch.minimum(d1, d2), d3) * 0.7 
            
            border_width = 0.05 + 0.03 * torch.sin(t_border_arg)
            mask_output = ShaderParamsReader.smoothstep(border_width, border_width - 0.02, d_tri)

        elif shape_type == "concentric": # Ported from curl_noise.py
            # Uses coords_normalized_01 (range [0,1]) for calculating distance from an animated center.
            # Convert time expressions to tensors
            time_tensor_03 = torch.tensor(time * 0.3, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_04 = torch.tensor(time * 0.4, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_01 = torch.tensor(time * 0.1, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_05 = torch.tensor(time * 0.5, device=device, dtype=coords_normalized_01.dtype)

            center_x_anim = 0.5 + 0.2 * torch.sin(time_tensor_03)
            center_y_anim = 0.5 + 0.2 * torch.cos(time_tensor_04)
            
            dist_from_center = torch.sqrt((coords_normalized_01[:, :, :, 0] - center_x_anim)**2 + 
                                          (coords_normalized_01[:, :, :, 1] - center_y_anim)**2)
            
            freq_anim = 10.0 + 5.0 * torch.sin(time_tensor_01)
            phase_anim = time_tensor_05
            rings_val = torch.sin(dist_from_center * freq_anim + phase_anim)
            mask_output = ShaderParamsReader.smoothstep(0.0, 0.1, rings_val) * ShaderParamsReader.smoothstep(0.0, -0.1, -rings_val)

        elif shape_type == "rays": # Ported from curl_noise.py
            # Uses coords_normalized_01 (range [0,1]) for calculating angle and dist from an animated center.
            # Convert time expressions to tensors
            time_tensor_03 = torch.tensor(time * 0.3, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_04 = torch.tensor(time * 0.4, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_05 = torch.tensor(time * 0.5, device=device, dtype=coords_normalized_01.dtype)

            center_x_anim = 0.5 + 0.1 * torch.sin(time_tensor_03)
            center_y_anim = 0.5 + 0.1 * torch.cos(time_tensor_04)
            
            to_center_x = coords_normalized_01[:, :, :, 0] - center_x_anim
            to_center_y = coords_normalized_01[:, :, :, 1] - center_y_anim
            
            angle_rays = torch.atan2(to_center_y, to_center_x)
            freq_rays = 8.0
            phase_rays = time_tensor_05
            rays_val = torch.sin(angle_rays * freq_rays + phase_rays)
            
            dist_rays = torch.sqrt(to_center_x**2 + to_center_y**2)
            falloff = 1.0 - ShaderParamsReader.smoothstep(0.0, 0.8, dist_rays)
            mask_output = ShaderParamsReader.smoothstep(0.0, 0.3, rays_val) * falloff
            
        elif shape_type == "zigzag": # Ported from curl_noise.py
            # Uses coords_normalized_01 (range [0,1]) and centers for rotation.
            freq_zigzag = 10.0
            # Convert time expressions to tensors
            time_tensor = torch.tensor(time, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_02 = torch.tensor(time * 0.2, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_05 = torch.tensor(time * 0.5, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_03 = torch.tensor(time * 0.3, device=device, dtype=coords_normalized_01.dtype)

            angle_zigzag = 0.5 * torch.sin(time_tensor_02)
            cos_a = torch.cos(angle_zigzag)
            sin_a = torch.sin(angle_zigzag)
            
            # Rotate coords_normalized_01 directly to match curl_noise.py (rotation around 0,0 of the [0,1] grid)
            rotated_x_norm = coords_normalized_01[:, :, :, 0] * cos_a - coords_normalized_01[:, :, :, 1] * sin_a
            rotated_y_norm = coords_normalized_01[:, :, :, 0] * sin_a + coords_normalized_01[:, :, :, 1] * cos_a
            
            # The original curl_noise performed fmod(rotated_coord * freq - time_offset, 1.0).
            # Then 2.0 * fmod_result - 1.0 to bring to [-1,1], then abs for [0,1].
            # Since rotated_x_norm and rotated_y_norm are in a range determined by the rotation of [0,1] coordinates,
            # multiplying by freq_zigzag will expand this range before fmod.
            zigzag1 = torch.abs(2.0 * torch.fmod(rotated_x_norm * freq_zigzag - time_tensor_05, 1.0) - 1.0)
            zigzag2 = torch.abs(2.0 * torch.fmod(rotated_y_norm * freq_zigzag + time_tensor_03, 1.0) - 1.0)
            
            zigzag_combined = torch.minimum(zigzag1, zigzag2)
            thickness_anim = 0.3 + 0.1 * torch.sin(time_tensor)
            # torch.heaviside(input, values) outputs values where input > 0, and 0 where input < 0.
            # For input == 0, it outputs values[0] if it's a tensor, or just values if scalar.
            # A common way to get a step is (input > threshold).float()
            # curl_noise's step(edge, x) is (x >= edge).float()
            # So, heaviside(zigzag - thickness, torch.tensor(0.5)) is similar to (zigzag - thickness >= 0).float()
            # which is (zigzag >= thickness).float()
            mask_output = (zigzag_combined >= thickness_anim).float() # More direct step function

        elif shape_type == "gradient_x": # Ported from curl_noise.py
            # Uses coords_normalized_01 (range [0,1])
            mask_output = coords_normalized_01[:, :, :, 0]

        elif shape_type == "gradient_y": # Ported from curl_noise.py
            # Uses coords_normalized_01 (range [0,1])
            mask_output = coords_normalized_01[:, :, :, 1]

        elif shape_type == "stars": # Ported from curl_noise.py
            # Uses coords_normalized_01 (same as coords_bhwc in CN)
            mask_stars_cn = torch.zeros_like(coords_normalized_01[:, :, :, 0])
            num_stars_cn = 20
            time_tensor_cn_stars = torch.tensor(time, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_01_cn = torch.tensor(time * 0.1, device=device, dtype=coords_normalized_01.dtype)
            time_tensor_015_cn = torch.tensor(time * 0.15, device=device, dtype=coords_normalized_01.dtype)

            for i in range(num_stars_cn):
                # Use ShaderParamsReader.random_val
                rand_x_stars = ShaderParamsReader.random_val(coords_normalized_01, base_seed, i * 78 + 10)
                rand_y_stars = ShaderParamsReader.random_val(coords_normalized_01, base_seed, i * 12 + 20)
                
                time_sin_arg_stars = torch.tensor(float(i), device=device, dtype=coords_normalized_01.dtype) + time_tensor_01_cn
                time_cos_arg_stars = torch.tensor(float(i) * 1.5, device=device, dtype=coords_normalized_01.dtype) + time_tensor_015_cn
                
                star_pos_x_cn = torch.fmod(rand_x_stars + 0.05 * torch.sin(time_sin_arg_stars), 1.0)
                star_pos_y_cn = torch.fmod(rand_y_stars + 0.05 * torch.cos(time_cos_arg_stars), 1.0)

                brightness_arg_stars = torch.tensor(float(i), device=device, dtype=coords_normalized_01.dtype) + time_tensor_cn_stars * (0.5 + rand_x_stars * 0.5)
                brightness_cn = 0.5 + 0.5 * torch.sin(brightness_arg_stars)
                size_stars_cn = 0.01 + 0.015 * rand_y_stars * brightness_cn
                
                dist_stars_cn = torch.sqrt((coords_normalized_01[:, :, :, 0] - star_pos_x_cn)**2 + (coords_normalized_01[:, :, :, 1] - star_pos_y_cn)**2)
                # Use ShaderParamsReader.smoothstep for soft stars
                star_mask_cn_indiv = ShaderParamsReader.smoothstep(size_stars_cn, size_stars_cn * 0.5, dist_stars_cn) * brightness_cn
                mask_stars_cn = torch.maximum(mask_stars_cn, star_mask_cn_indiv)
            mask_output = mask_stars_cn
            
        else: # Default for unknown or "none"
            if shape_type not in ["none", "0", 0]: # Only print warning for actual unknown types
                 print(f"Unknown shape type: {shape_type}, using default (full mask)")
            # Default is full mask (all ones)
            mask_output = torch.ones((batch, height, width), device=device) 
        
        # Ensure mask_output is [B, H, W, 1]
        if len(mask_output.shape) == 3: # If it's [B, H, W]
            mask_output = mask_output.unsqueeze(-1)
        elif len(mask_output.shape) == 4 and mask_output.shape[-1] != 1: # If it's [B,H,W,C] C!=1
            print(f"Warning: Shape mask generated with {mask_output.shape[-1]} channels. Taking first channel.")
            mask_output = mask_output[..., 0:1]

        return mask_output

# Legacy functions for backward compatibility
def get_shader_params():
    """Legacy function that calls the new class method"""
    return ShaderParamsReader.get_shader_params()

def test_params():
    """Test function to check if parameters are loading correctly"""
    params = ShaderParamsReader.get_shader_params()
    print(f"TEST: Current shader parameters: {params}")
    return params

def generate_noise_tensor(shader_params, height, width, batch_size=1, device="cuda", seed=0, target_channels=None):
    """
    Legacy function that uses the new class to generate noise
    
    Args:
        shader_params: Dictionary containing shader parameters
        height: Height of the tensor
        width: Width of the tensor
        batch_size: Number of images in the batch
        device: Device to create tensor on
        seed: Random seed for deterministic noise generation
        target_channels: Number of output channels (optional, default is 4)
        
    Returns:
        Noise tensor with shape [batch_size, channels, height, width]
        where channels is determined by target_channels (default: 4)
    """
    # Import ShaderToTensor for direct shader noise generation
    from .shader_to_tensor import ShaderToTensor
    
    # Make a copy of shader_params to avoid modifying the original
    shader_params = shader_params.copy()
    
    # Add target_channels to shader_params if provided
    if target_channels is not None:
        shader_params["target_channels"] = target_channels
        print(f"Using target_channels={target_channels} in default generator")
    
    # Check if we're using specialized shader types which are handled separately in the sampler
    shader_type = shader_params.get("shader_type", "tensor_field")
    if shader_type == "cellular":
        print("Note: Cellular shader type detected in generate_noise_tensor, but this will be handled by the specialized cellular noise generator")
    elif shader_type == "fractal":
        print("Note: Fractal shader type detected in generate_noise_tensor, but this will be handled by the specialized fractal noise generator")
    elif shader_type == "perlin":
        print("Note: Perlin shader type detected in generate_noise_tensor, but this will be handled by the specialized perlin noise generator")
    elif shader_type == "waves":
        print("Note: Waves shader type detected in generate_noise_tensor, but this will be handled by the specialized waves noise generator")
    elif shader_type == "gaussian":
        print("Note: Gaussian shader type detected in generate_noise_tensor, but this will be handled by the specialized gaussian noise generator")
    elif shader_type == "tensor_field":
        print("Note: Tensor field shader type detected in generate_noise_tensor, but this will be handled by the specialized tensor field generator")
    elif shader_type == "heterogeneous_fbm":
        print("Note: Heterogeneous FBM shader type detected in generate_noise_tensor, but this will be handled by the specialized generator")
    elif shader_type == "interference_patterns":
        print("Note: Interference patterns shader type detected in generate_noise_tensor, but this will be handled by the specialized generator")
    
    # Use the seed for deterministic generation
    torch.manual_seed(seed)
    
    # Extract parameters from shader_params
    viz_type = shader_params.get("visualization_type", 3)
    scale = shader_params.get("scale", 1.0)
    warp_strength = shader_params.get("warp_strength", 0.5)
    phase_shift = shader_params.get("phase_shift", 0.0)
    time = shader_params.get("time", 0.0)
    octaves = shader_params.get("octaves", 3.0)
    shape_type = shader_params.get("shape_type", "none")
    shape_mask_strength = shader_params.get("shapemaskstrength", 1.0)
    
    # Generate shader noise directly using ShaderToTensor class
    # This avoids using random noise as a starting point
    shader_noise = ShaderToTensor.shader_noise_to_tensor(
        batch_size=batch_size,
        height=height,
        width=width,
        shader_type=shader_type,
        visualization_type=viz_type,
        scale=scale,
        phase_shift=phase_shift,
        warp_strength=warp_strength,
        time=time,
        device=device,
        seed=seed,
        octaves=octaves,
        shape_type=shape_type,
        shape_mask_strength=shape_mask_strength,
        shader_params=shader_params
    )
    
    # Reset random seed state
    torch.manual_seed(torch.seed())
    
    # Apply color scheme transformation
    color_scheme = shader_params.get("colorScheme", "none")
    color_intensity = shader_params.get("shaderColorIntensity", 0.8)
    
    if color_scheme != "none" and color_intensity > 0:
        print(f"Applying color scheme: {color_scheme} with intensity: {color_intensity}")
        
        # Track channel stats before applying color
        num_channels = shader_noise.shape[1]
        channel_means_before = [shader_noise[:, i].mean().item() for i in range(num_channels)]
        channel_stds_before = [shader_noise[:, i].std().item() for i in range(num_channels)]
        print(f"Channel means before color: {[f'{m:.4f}' for m in channel_means_before]}")
        
        # Apply color scheme
        colored_noise = ShaderParamsReader.apply_color_scheme(shader_noise, shader_params)
        
        # Track channel stats after applying color
        channel_means_after = [colored_noise[:, i].mean().item() for i in range(num_channels)]
        channel_stds_after = [colored_noise[:, i].std().item() for i in range(num_channels)]
        print(f"Channel means after color: {[f'{m:.4f}' for m in channel_means_after]}")
        
        # Normalize each channel separately while preserving mean differences
        # This ensures the color impact remains visible
        # We only normalize the standard deviation to keep it at ~1.0
        normalized_colored_noise = torch.zeros_like(colored_noise)
        for i in range(num_channels):
            # Only normalize the standard deviation while keeping the mean offset
            channel = colored_noise[:, i:i+1]
            normalized_colored_noise[:, i:i+1] = (channel - channel.mean()) / (channel.std() + 1e-8) + channel_means_after[i]
        
        # Verify the normalization preserved color differences
        final_means = [normalized_colored_noise[:, i].mean().item() for i in range(num_channels)]
        final_stds = [normalized_colored_noise[:, i].std().item() for i in range(num_channels)]
        print(f"Final means after normalization: {[f'{m:.4f}' for m in final_means]}")
        print(f"Final stds after normalization: {[f'{s:.4f}' for s in final_stds]}")
        
        return normalized_colored_noise
    
    return shader_noise

# Example of use in sampling process:
#
# 1. Load shader parameters
# shader_params = ShaderParamsReader.get_shader_params()
#
# 2. Generate initial noise for sampling
# noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
#
# 3. Apply shader transformation to noise (implements Sα(N))
# modified_noise = ShaderParamsReader.apply_shader_to_noise(noise, shader_params)
#
# 4. Use modified noise in sampling (implements Kβ(t))
# samples = comfy.sample.sample(
#     model=model,
#     noise=modified_noise,
#     # ... other parameters
#     disable_noise=True,  # Using our pre-modified noise
# )