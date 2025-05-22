import torch
import math
# Import ShaderParamsReader for apply_shape_mask functionality
from ..shader_params_reader import ShaderParamsReader

class DomainWarpGenerator:
    """
    PyTorch implementation of domain warp animation that closely matches 
    the WebGL domain warp shader from shader_renderer.js
    
    This class generates domain warping noise, where one noise function distorts
    the input coordinates for another noise function, creating complex, swirling 
    patterns used to influence the sampling process in image generation.

    Now enhanced with true temporal coherence for smooth animations.
    """
    
    @staticmethod
    def get_domain_warp(batch_size, height, width, shader_params, device="cuda", seed=0):
        """
        Generate domain warp noise tensor that matches the WebGL implementation
        
        Args:
            batch_size: Number of images in batch
            height: Height of tensor
            width: Width of tensor
            shader_params: Dictionary containing parameters from shader_params.json
            device: Device to create tensor on
            seed: Random seed value to ensure different patterns with different seeds
            
        Returns:
            Tensor with shape [batch_size, channels, height, width]
        """
        # Get debugger instance
        # debugger = get_debugger()
        
        # Extract parameters from shader_params - check both original and converted names
        # This ensures compatibility with both raw JSON params and converted params
        scale = shader_params.get("shaderScale", shader_params.get("scale", 1.0))
        warp_strength = shader_params.get("shaderWarpStrength", shader_params.get("warp_strength", 0.5))
        phase_shift = shader_params.get("shaderPhaseShift", shader_params.get("phase_shift", 0.5))
        octaves = shader_params.get("shaderOctaves", shader_params.get("octaves", 1))
        time = shader_params.get("time", 0.0)  # This would ideally come from the animation state
        
        # Extract temporal coherence parameters - CHANGED: Default to False 
        base_seed = shader_params.get("base_seed", seed)  # Use provided seed as default base_seed
        use_temporal_coherence = shader_params.get("temporal_coherence", False)  # CHANGED: Default to False
        
        # Extract target channel count if provided
        target_channels = shader_params.get("target_channels", 9)  # Default to 9 if not specified
        
        # --- Model Specific Channel Overrides ---
        # Check if the model is CosmosVideo and override the channel count if needed
        model_class = shader_params.get("model_class", "")
        inner_model_class = shader_params.get("inner_model_class", "")

        if inner_model_class == "CosmosVideo":
            target_channels = 16
            # Update shader_params as well for consistency downstream
            shader_params["target_channels"] = 16 
        elif model_class == "CosmosVideo": # Fallback check on outer model class
             target_channels = 16
             shader_params["target_channels"] = 16
        elif inner_model_class == "ACEStep":
            target_channels = 8
            shader_params["target_channels"] = 8  
        
        # Apply seed variation if provided
        seed_variation = shader_params.get("seed_variation", 0)
        if seed_variation > 0:
            seed = seed + seed_variation
            # if debugger.enabled and debugger.debug_level >= 2:
            #     print(f"Using seed variation {seed_variation}, adjusted seed to {seed}")
        
        # FIXED: If not using temporal coherence, use the current seed value not base_seed
        # This ensures different noise patterns with different seeds
        current_seed = base_seed if use_temporal_coherence else seed
        
        # Extract shape mask parameters
        shape_type = shader_params.get("shaderShapeType", shader_params.get("shape_type", "none"))
        shape_mask_strength = shader_params.get("shaderShapeStrength", shader_params.get("shapemaskstrength", 1.0))
        
        # Extract color scheme parameters
        color_scheme = shader_params.get("colorScheme", "none")
        color_intensity = shader_params.get("shaderColorIntensity", shader_params.get("intensity", 0.8))
        
        # Print params being used for debugging
        # print(f"DomainWarp: Using shader_type: {shader_params.get('shader_type', 'domain_warp')}")
        # print(f"DomainWarp: Scale={scale}, Warp Strength={warp_strength}, Phase Shift={phase_shift}, Octaves={octaves}")
        # print(f"DomainWarp: Temporal settings: time={time}, base_seed={base_seed}, coherence={use_temporal_coherence}")
        # print(f"DomainWarp: Shape Mask: type={shape_type}, strength={shape_mask_strength}")
        # print(f"DomainWarp: Color Scheme: scheme={color_scheme}, intensity={color_intensity}")
        # print(f"DomainWarp: Target Channels: {target_channels}")
        
        # Log shader parameters for debugging
        # if debugger.enabled and debugger.debug_level >= 1:
        #     debugger.log_generation_operation(
        #         "domain_warp_start", 
        #         {"shader_params": shader_params}, 
        #         None, 
        #         {
        #             "scale": scale,
        #             "warp_strength": warp_strength,
        #             "phase_shift": phase_shift,
        #             "octaves": octaves,
        #             "time": time,
        #             "seed": seed,
        #             "target_channels": target_channels
        #         }
        #     )
        
        # Create coordinate grid (normalized to [0, 1] for domain warp shader)
        y, x = torch.meshgrid(
            torch.linspace(0, 1, height, device=device),
            torch.linspace(0, 1, width, device=device),
            indexing='ij'
        )
        
        # Combine into coordinate tensor [batch, height, width, 2]
        p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Track the coordinate tensor
        # if debugger.enabled and debugger.debug_level >= 2:
        #     debugger.track_tensor_shape_history(p, "coordinates", "initial_creation")
        
        # Generate domain warp noise based on octaves value
        warp_type = int(octaves % 4)  # Same as in shader: int(mod(u_octaves, 4.0))
        
        # Helper for lerp
        def lerp(a, b, t):
            return a + (b - a) * t

        # Helper for HSV to RGB conversion (from curl_noise.py)
        def hsv_to_rgb(h, s, v, device):
            # h, s, v expected shapes [B, 1, H, W]
            # Ensure all inputs are on the correct device
            h = h.to(device)
            s = s.to(device)
            v = v.to(device)

            c = v * s
            h_prime = h * 6.0
            x = c * (1.0 - torch.abs(torch.fmod(h_prime, 2.0) - 1.0))
            m = v - c
            
            r, g, b = torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h)

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

        # Helper for color stops interpolation
        def interpolate_colors(stops, t, device):
            # stops: list of [value, color_tensor]
            # t: normalized value tensor [B, 1, H, W]
            
            # Find the two stops t falls between
            idx = torch.zeros_like(t, dtype=torch.long)
            for i in range(len(stops) - 1):
                idx = torch.where((t >= stops[i][0]) & (t < stops[i+1][0]), torch.full_like(idx, i), idx)
            idx = torch.where(t >= stops[-1][0], torch.full_like(idx, len(stops) - 2), idx) # Handle >= last stop value

            # Gather the start and end stops based on idx
            # Initialize with target spatial shape, ensure first stop color tensor is on the correct device
            final_color_shape = (t.shape[0], 3, t.shape[2], t.shape[3]) # B, C, H, W
            final_color = torch.zeros(final_color_shape, device=device)
            
            for i in range(len(stops) - 1):
                mask = (idx == i) # Shape [B, 1, H, W]
                t0_val, c0_list = stops[i]
                t1_val, c1_list = stops[i+1]

                c0 = torch.tensor(c0_list, device=device).view(1, 3, 1, 1) # Shape [1, 3, 1, 1]
                c1 = torch.tensor(c1_list, device=device).view(1, 3, 1, 1) # Shape [1, 3, 1, 1]
                
                # Normalize t within the segment [t0, t1] -> [0, 1] for all pixels
                denominator = (t1_val - t0_val + 1e-8) # Avoid division by zero
                local_t_all = (t - t0_val) / denominator
                local_t_clamped = torch.clamp(local_t_all, 0.0, 1.0) # Shape [B, 1, H, W]

                # Lerp the colors for this segment - c0/c1 will broadcast
                interp_color = lerp(c0, c1, local_t_clamped) # Shape [B, 3, H, W]
                
                # Apply the interpolated color where the mask is true
                final_color = torch.where(mask.expand_as(interp_color), interp_color, final_color)

            return final_color[:, 0:1], final_color[:, 1:2], final_color[:, 2:3] # R, G, B
        
        # Generate warped noise - pass phase_shift to the domain_warp function
        try:
            # with debugger.time_operation("domain_warp_generation") if debugger.enabled else contextlib.nullcontext():
            result = DomainWarpGenerator._domain_warp_with_phase(
                p, device, octaves, seed, 0, warp_type, scale, warp_strength, phase_shift, time
            )
                
            # Track the initial result tensor
            # if debugger.enabled and debugger.debug_level >= 2:
            #     debugger.track_tensor_shape_history(result, "domain_warp_result", "after_generation")
            #     debugger.analyze_tensor(result, "domain_warp_raw")
        except Exception as e:
            # if debugger.enabled:
            #     debugger.add_warning(f"Error generating domain warp: {str(e)}", category="generation_error")
            # print(f"❌ Error generating domain warp: {str(e)}")
            # Fallback to random noise if generation fails
            result = torch.randn((batch_size, height, width, 1), device=device)
        
        # Apply contrast adjustment in addition to the internal phase shift effects
        contrast = 1.0 + phase_shift * 0.5  # Reduced effect since we're now using phase in the patterns
        result *= contrast
        
        # Apply shape mask if requested
        if shape_type not in ["none", "0"] and shape_mask_strength > 0:
            try:
                # Create shape mask using ShaderParamsReader
                # Transform p from [0,1] to [-1,1] for shape mask --- This is no longer needed for the updated reader
                # p_normalized = p * 2.0 - 1.0 
                
                # Use same base_seed for shape mask to ensure temporal coherence
                # The seed for shape_mask itself is handled internally by apply_shape_mask if needed,
                # but we pass base_seed and use_temporal_coherence for its own logic.
                current_mask_seed = base_seed if use_temporal_coherence else seed
                # torch.manual_seed(current_mask_seed + 500) # Seeding is now internal to reader if necessary for its own ops
                
                shape_mask = ShaderParamsReader.apply_shape_mask(
                    p, # p is already in [0,1] range
                    shape_type,
                    time=time, 
                    base_seed=base_seed, 
                    use_temporal_coherence=use_temporal_coherence
                )
                
                # Track shape mask tensor
                # if debugger.enabled and debugger.debug_level >= 2:
                #     debugger.track_tensor_shape_history(shape_mask, "shape_mask", "after_creation")
                
                # Apply shape mask to the noise
                result = torch.lerp(result, result * shape_mask, shape_mask_strength)
            except Exception as e:
                # if debugger.enabled:
                #     debugger.add_warning(f"Error applying shape mask: {str(e)}", category="shape_mask_error")
                pass
        # Ensure output is in valid [-1, 1] range
        result = torch.clamp(result, -1.0, 1.0)
        
        # Convert from [batch, height, width, 1] to [batch, 1, height, width]
        # Permute dimensions to match expected format
        result = result.permute(0, 3, 1, 2)  # [batch, 1, height, width]
        
        # Track tensor after permutation
        # if debugger.enabled and debugger.debug_level >= 2:
        #     debugger.track_tensor_shape_history(result, "result_permuted", "after_permutation")
        
        # Create channel variations for color
        if color_scheme != "none" and color_intensity > 0:
            try:
                # with debugger.time_operation("color_variations") if debugger.enabled else contextlib.nullcontext():
                    # Initialize base channel (red)
                    r_channel = result.clone()
                    
                    # Scale intensity factor based on color_intensity
                    intensity_factor = 0.5 + color_intensity * 0.5  # Maps 0-1 to 0.5-1.0
                    
                    # Apply color scheme specific variations
                    if color_scheme == "viridis":
                        # Viridis color scheme
                        # Ensure result (base noise for coloring) is normalized to [0,1] for interpolate_colors
                        # The 'result' tensor here is the single channel noise before colorization
                        normalized_value = (result - result.min()) / (result.max() - result.min() + 1e-8)
                        
                        stops = [
                            (0.0, [0.267, 0.005, 0.329]), # #440154
                            (0.33, [0.188, 0.407, 0.553]), # #30678D
                            (0.66, [0.208, 0.718, 0.471]), # #35B778
                            (1.0, [0.992, 0.906, 0.143])  # #FDE724
                        ]
                        r_channel, g_channel, b_channel = interpolate_colors(stops, normalized_value, device)

                    elif color_scheme == "plasma":
                        # Plasma color scheme from curl_noise.py
                        # 1. Generate proxy for x and y components (vx_proxy, vy_proxy)
                        #    similar to how g_channel and b_channel are made for other vibrant schemes.
                        p_vx_proxy = p * (scale * 0.92) # Slightly different params for variety
                        p_vy_proxy = p * (scale * 1.08)

                        vx_proxy_bhwc = DomainWarpGenerator._domain_warp_with_phase(
                            p_vx_proxy, device, octaves, seed + 10, 0, warp_type, scale * 0.92, 
                            warp_strength * (1.0 + 0.15 * intensity_factor), 
                            phase_shift + 0.20 * intensity_factor, time
                        ) # [B, H, W, 1]
                        vy_proxy_bhwc = DomainWarpGenerator._domain_warp_with_phase(
                            p_vy_proxy, device, octaves, seed + 20, 0, warp_type, scale * 1.08, 
                            warp_strength * (1.0 - 0.15 * intensity_factor), 
                            phase_shift - 0.20 * intensity_factor, time
                        ) # [B, H, W, 1]

                        # No need for contrast application or permute yet for these proxies
                        # Ensure they are in [0,1] range if _domain_warp_with_phase output is not guaranteed
                        vx_proxy_bhwc = torch.clamp(vx_proxy_bhwc, 0.0, 1.0)
                        vy_proxy_bhwc = torch.clamp(vy_proxy_bhwc, 0.0, 1.0)
                        
                        # Convert to [B, 1, H, W]
                        vx_proxy = vx_proxy_bhwc.permute(0, 3, 1, 2)
                        vy_proxy = vy_proxy_bhwc.permute(0, 3, 1, 2)

                        # 2. Calculate vangle
                        vangle = torch.atan2(vy_proxy, vx_proxy) # Output is [-pi, pi]
                        vangle = (vangle + math.pi) / (2 * math.pi) # Normalize to [0, 1]

                        # 3. Normalize base result for normalized_value
                        # 'result' is [B, 1, H, W] and in [-1, 1], normalize to [0, 1]
                        normalized_value = (result + 1.0) / 2.0

                        # 4. Prepare time_broadcast
                        time_tensor = torch.tensor(time, device=device, dtype=result.dtype)
                        time_broadcast = time_tensor.view(-1, 1, 1, 1) if time_tensor.numel() > 1 else time_tensor
                        if time_broadcast.dim() == 0: # Ensure it's at least 1D for view(-1,1,1,1)
                            time_broadcast = time_broadcast.unsqueeze(0)
                        time_broadcast = time_broadcast.view(-1, 1, 1, 1) # Ensure [B_or_1, 1, 1, 1]
                        if time_broadcast.shape[0] != r_channel.shape[0] and time_broadcast.shape[0] == 1:
                           time_broadcast = time_broadcast.expand(r_channel.shape[0], -1, -1, -1)

                        # 5. Apply plasma calculations
                        r_channel_plasma = 0.5 + 0.5 * torch.sin(vangle * 6.28318 + time_broadcast)
                        g_channel_plasma = 0.5 + 0.5 * torch.sin(vangle * 6.28318 + normalized_value * 3.14159 + time_broadcast * 2.0)
                        b_channel_plasma = 0.5 + 0.5 * torch.cos(vangle * 3.14159 + normalized_value * 6.28318 + time_broadcast * 3.0)

                        r_channel = r_channel_plasma
                        g_channel = g_channel_plasma
                        b_channel = b_channel_plasma

                    elif color_scheme == "inferno":
                        # Inferno color scheme from curl_noise.py
                        # Ensure result (base noise for coloring) is normalized to [0,1] for interpolate_colors
                        # The 'result' tensor here is the single channel noise before colorization [B, 1, H, W]
                        # It's already in [-1, 1] range from clamp, so normalize to [0, 1]
                        normalized_value = (result + 1.0) / 2.0 
                        
                        stops = [
                            (0.0, [0.001, 0.001, 0.016]), 
                            (0.25, [0.259, 0.039, 0.408]),
                            (0.5, [0.576, 0.149, 0.404]), 
                            (0.75, [0.867, 0.318, 0.227]), 
                            (0.85, [0.988, 0.647, 0.039]), 
                            (1.0, [0.988, 1.000, 0.643])  
                        ]
                        r_channel, g_channel, b_channel = interpolate_colors(stops, normalized_value, device)

                    elif color_scheme == "magma":
                        # Magma color scheme from curl_noise.py
                        # Ensure result (base noise for coloring) is normalized to [0,1] for interpolate_colors
                        # The 'result' tensor here is the single channel noise before colorization [B, 1, H, W]
                        # It's already in [-1, 1] range from clamp, so normalize to [0, 1]
                        normalized_value = (result + 1.0) / 2.0 
                        
                        stops = [
                            (0.0, [0.001, 0.001, 0.016]),
                            (0.25, [0.231, 0.059, 0.439]),
                            (0.5, [0.549, 0.161, 0.506]),
                            (0.75, [0.871, 0.288, 0.408]),
                            (0.85, [0.996, 0.624, 0.427]),
                            (1.0, [0.988, 0.992, 0.749])
                        ]
                        r_channel, g_channel, b_channel = interpolate_colors(stops, normalized_value, device)

                    elif color_scheme == "turbo":
                        # Turbo color scheme from curl_noise.py
                        # Ensure result (base noise for coloring) is normalized to [0,1] for interpolate_colors
                        # The 'result' tensor here is the single channel noise before colorization [B, 1, H, W]
                        # It's already in [-1, 1] range from clamp, so normalize to [0, 1]
                        normalized_value = (result + 1.0) / 2.0

                        stops = [
                            (0.0, [0.188, 0.071, 0.235]), # #30123b
                            (0.25, [0.275, 0.408, 0.859]), # #4669db
                            (0.5, [0.149, 0.749, 0.549]), # #26bf8c
                            (0.65, [0.831, 1.000, 0.314]), # #d4ff50
                            (0.85, [0.980, 0.718, 0.298]), # #fab74c
                            (1.0, [0.729, 0.004, 0.000])  # #ba0100
                        ]
                        r_channel, g_channel, b_channel = interpolate_colors(stops, normalized_value, device)

                    elif color_scheme == "jet":
                        # Jet color scheme from curl_noise.py
                        # Ensure result (base noise for coloring) is normalized to [0,1] for interpolate_colors
                        # The 'result' tensor here is the single channel noise before colorization [B, 1, H, W]
                        # It's already in [-1, 1] range from clamp, so normalize to [0, 1]
                        normalized_value = (result + 1.0) / 2.0

                        stops = [
                            (0.0, [0.000, 0.000, 0.498]), # #00007f
                            (0.125, [0.000, 0.000, 1.000]), # #0000ff blue
                            (0.375, [0.000, 1.000, 1.000]), # #00ffff cyan
                            (0.625, [1.000, 1.000, 0.000]), # #ffff00 yellow
                            (0.875, [1.000, 0.000, 0.000]), # #ff0000 red
                            (1.0, [0.498, 0.000, 0.000])  # #7f0000 dark red
                        ]
                        r_channel, g_channel, b_channel = interpolate_colors(stops, normalized_value, device)

                    elif color_scheme == "rainbow":
                        # Rainbow color scheme using HSV conversion, similar to curl_noise.py
                        # The 'result' tensor is [B, 1, H, W] and in [-1, 1]
                        
                        # Normalize base result for HUE component [0, 1]
                        hue_component = (result + 1.0) / 2.0 
                        
                        # Create a phase-shifted version for VALUE component [0, 1]
                        # We can use a rolled version of the result or a slightly modified noise
                        # For simplicity, let's use a rolled version for now.
                        # Roll factor can be a fraction of width or height.
                        roll_factor_h = int(height * 0.1 * intensity_factor)
                        roll_factor_w = int(width * 0.1 * intensity_factor)
                        rolled_result = torch.roll(result, shifts=(roll_factor_h, roll_factor_w), dims=(2, 3))
                        value_component = (rolled_result + 1.0) / 2.0
                        
                        # Fixed saturation
                        saturation_component = torch.ones_like(hue_component) * 0.8
                        
                        # Convert HSV to RGB
                        # hsv_to_rgb expects [B, 1, H, W] inputs
                        r_channel, g_channel, b_channel = hsv_to_rgb(hue_component, saturation_component, value_component, device)

                    elif color_scheme == "magma": # Keep magma separate from rainbow now
                        # Magma color scheme from curl_noise.py
                        # Ensure result (base noise for coloring) is normalized to [0,1] for interpolate_colors
                        # The 'result' tensor here is the single channel noise before colorization [B, 1, H, W]
                        # It's already in [-1, 1] range from clamp, so normalize to [0, 1]
                        normalized_value = (result + 1.0) / 2.0 
                        
                        stops = [
                            (0.0, [0.001, 0.001, 0.016]),
                            (0.25, [0.231, 0.059, 0.439]),
                            (0.5, [0.549, 0.161, 0.506]),
                            (0.75, [0.871, 0.288, 0.408]),
                            (0.85, [0.996, 0.624, 0.427]),
                            (1.0, [0.988, 0.992, 0.749])
                        ]
                        r_channel, g_channel, b_channel = interpolate_colors(stops, normalized_value, device)

                    elif color_scheme == "cool":
                        # Cool color scheme from curl_noise.py
                        # Interpolates between Cyan and Magenta
                        # 'result' is [B, 1, H, W] and in [-1, 1]
                        normalized_value = (result + 1.0) / 2.0 # Normalize to [0, 1]

                        c0 = torch.tensor([0.0, 1.0, 1.0], device=device).view(1, 3, 1, 1) # Cyan
                        c1 = torch.tensor([1.0, 0.0, 1.0], device=device).view(1, 3, 1, 1) # Magenta
                        
                        color_lerped = lerp(c0, c1, normalized_value)
                        r_channel, g_channel, b_channel = color_lerped[:, 0:1], color_lerped[:, 1:2], color_lerped[:, 2:3]

                    elif color_scheme == "hot":
                        # Hot color scheme from curl_noise.py
                        normalized_value = (result + 1.0) / 2.0
                        stops = [
                            (0.0, [0.0, 0.0, 0.0]), # Black
                            (0.375, [1.0, 0.0, 0.0]), # Red
                            (0.75, [1.0, 1.0, 0.0]), # Yellow
                            (1.0, [1.0, 1.0, 1.0])  # White
                        ]
                        r_channel, g_channel, b_channel = interpolate_colors(stops, normalized_value, device)

                    elif color_scheme == "parula":
                        normalized_value = (result + 1.0) / 2.0
                        stops = [
                            (0.0, [0.208, 0.165, 0.529]),
                            (0.25, [0.059, 0.361, 0.867]),
                            (0.5, [0.000, 0.710, 0.651]),
                            (0.75, [1.000, 0.765, 0.216]),
                            (1.0, [0.988, 0.996, 0.643])
                        ]
                        r_channel, g_channel, b_channel = interpolate_colors(stops, normalized_value, device)

                    elif color_scheme == "pink":
                        normalized_value = (result + 1.0) / 2.0
                        stops = [
                            (0.0, [0.05, 0.05, 0.05]), 
                            (0.5, [1.0, 0.0, 1.0]), 
                            (1.0, [1.0, 1.0, 1.0])
                        ]
                        r_channel, g_channel, b_channel = interpolate_colors(stops, normalized_value, device)

                    elif color_scheme == "bone":
                        normalized_value = (result + 1.0) / 2.0
                        stops = [
                            (0.0, [0.0, 0.0, 0.0]),
                            (0.375, [0.329, 0.329, 0.455]),
                            (0.75, [0.627, 0.757, 0.757]),
                            (1.0, [1.0, 1.0, 1.0])
                        ]
                        r_channel, g_channel, b_channel = interpolate_colors(stops, normalized_value, device)

                    elif color_scheme == "ocean":
                        normalized_value = (result + 1.0) / 2.0
                        stops = [
                            (0.0, [0.0, 0.0, 0.0]),
                            (0.33, [0.0, 0.0, 0.6]),
                            (0.66, [0.0, 0.6, 1.0]),
                            (1.0, [0.6, 1.0, 1.0])
                        ]
                        r_channel, g_channel, b_channel = interpolate_colors(stops, normalized_value, device)

                    elif color_scheme == "terrain":
                        normalized_value = (result + 1.0) / 2.0
                        stops = [
                            (0.0, [0.2, 0.2, 0.6]),
                            (0.33, [0.0, 0.8, 0.4]),
                            (0.66, [1.0, 0.8, 0.0]),
                            (1.0, [1.0, 1.0, 1.0])
                        ]
                        r_channel, g_channel, b_channel = interpolate_colors(stops, normalized_value, device)

                    elif color_scheme == "neon":
                        normalized_value = (result + 1.0) / 2.0
                        stops = [
                            (0.0, [1.0, 0.0, 1.0]),
                            (0.5, [0.0, 1.0, 1.0]),
                            (1.0, [1.0, 1.0, 0.0])
                        ]
                        r_channel, g_channel, b_channel = interpolate_colors(stops, normalized_value, device)

                    elif color_scheme == "fire":
                        normalized_value = (result + 1.0) / 2.0
                        stops = [
                            (0.0, [0.0, 0.0, 0.0]),
                            (0.25, [1.0, 0.0, 0.0]),
                            (0.6, [1.0, 1.0, 0.0]),
                            (1.0, [1.0, 1.0, 1.0])
                        ]
                        r_channel, g_channel, b_channel = interpolate_colors(stops, normalized_value, device)

                    elif color_scheme == "blue_red":
                        normalized_value = (result + 1.0) / 2.0
                        c0 = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3, 1, 1) # Blue
                        c1 = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3, 1, 1) # Red
                        color_lerped = lerp(c0, c1, normalized_value)
                        r_channel, g_channel, b_channel = color_lerped[:, 0:1], color_lerped[:, 1:2], color_lerped[:, 2:3]

                    elif color_scheme == "autumn":
                        normalized_value = (result + 1.0) / 2.0
                        c0 = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3, 1, 1) # Red
                        c1 = torch.tensor([1.0, 1.0, 0.0], device=device).view(1, 3, 1, 1) # Yellow
                        color_lerped = lerp(c0, c1, normalized_value)
                        r_channel, g_channel, b_channel = color_lerped[:, 0:1], color_lerped[:, 1:2], color_lerped[:, 2:3]

                    elif color_scheme == "winter":
                        normalized_value = (result + 1.0) / 2.0
                        c0 = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3, 1, 1) # Blue
                        c1 = torch.tensor([0.0, 1.0, 0.5], device=device).view(1, 3, 1, 1) # Greenish-Cyan
                        color_lerped = lerp(c0, c1, normalized_value)
                        r_channel, g_channel, b_channel = color_lerped[:, 0:1], color_lerped[:, 1:2], color_lerped[:, 2:3]

                    elif color_scheme == "spring":
                        normalized_value = (result + 1.0) / 2.0
                        c0 = torch.tensor([1.0, 0.0, 1.0], device=device).view(1, 3, 1, 1) # Magenta
                        c1 = torch.tensor([1.0, 1.0, 0.0], device=device).view(1, 3, 1, 1) # Yellow
                        color_lerped = lerp(c0, c1, normalized_value)
                        r_channel, g_channel, b_channel = color_lerped[:, 0:1], color_lerped[:, 1:2], color_lerped[:, 2:3]

                    elif color_scheme == "summer":
                        normalized_value = (result + 1.0) / 2.0
                        c0 = torch.tensor([0.0, 0.5, 0.4], device=device).view(1, 3, 1, 1) # Dark Green
                        c1 = torch.tensor([1.0, 1.0, 0.4], device=device).view(1, 3, 1, 1) # Yellow
                        color_lerped = lerp(c0, c1, normalized_value)
                        r_channel, g_channel, b_channel = color_lerped[:, 0:1], color_lerped[:, 1:2], color_lerped[:, 2:3]

                    elif color_scheme == "copper":
                        normalized_value = (result + 1.0) / 2.0
                        c0 = torch.tensor([0.0, 0.0, 0.0], device=device).view(1, 3, 1, 1) # Black
                        c1 = torch.tensor([1.0, 0.6, 0.4], device=device).view(1, 3, 1, 1) # Copper
                        color_lerped = lerp(c0, c1, normalized_value)
                        r_channel, g_channel, b_channel = color_lerped[:, 0:1], color_lerped[:, 1:2], color_lerped[:, 2:3]
                        
                    elif color_scheme in ["complementary"]:
                        # For contrasting schemes, create more opposing patterns
                        # Use negated and inverted variations of the original pattern
                        
                        # Create offset and modified coordinates
                        p_g = torch.roll(p, shifts=int(width * 0.1 * intensity_factor), dims=2)
                        p_b = torch.roll(p, shifts=int(height * 0.1 * intensity_factor), dims=1)
                        
                        # Generate warped patterns with contrasting parameters
                        g_warp_result = DomainWarpGenerator._domain_warp_with_phase(
                            p_g, device, octaves, seed, (warp_type + 1) % 4, 
                            scale * 1.1, 
                            warp_strength * (1.0 - 0.3 * intensity_factor), 
                            phase_shift + 0.5, time
                        )
                        
                        b_warp_result = DomainWarpGenerator._domain_warp_with_phase(
                            p_b, device, octaves, seed, (warp_type + 2) % 4, 
                            scale * 0.9, 
                            warp_strength * (1.0 + 0.3 * intensity_factor), 
                            phase_shift + 0.5, time
                        )
                        
                        # Apply contrast and permute
                        g_warp_result *= contrast * 0.9  # Slightly reduced contrast
                        b_warp_result *= contrast * 1.1  # Slightly increased contrast
                        g_warp_result = torch.clamp(g_warp_result, -1.0, 1.0).permute(0, 3, 1, 2)
                        b_warp_result = torch.clamp(b_warp_result, -1.0, 1.0).permute(0, 3, 1, 2)
                        
                        # Use these as green and blue channels
                        g_channel = g_warp_result
                        b_channel = b_warp_result
                        
                    else:
                        # For other schemes, use more subtle variations with phase shifts
                        # Original coordinates with slight offsets
                        
                        # Create subtly modified versions of the pattern
                        g_channel = torch.roll(r_channel, shifts=int(height * 0.05 * intensity_factor), dims=2)
                        b_channel = torch.roll(r_channel, shifts=int(width * 0.05 * intensity_factor), dims=3)
                        
                        # Apply subtle transformations
                        if intensity_factor > 0.6:
                            # For higher intensity, add more pronounced color variations
                            g_channel = g_channel * (1.0 - 0.2 * intensity_factor) + 0.1 * intensity_factor
                            b_channel = b_channel * (1.0 - 0.3 * intensity_factor) - 0.1 * intensity_factor
                        
            except Exception as e:
                # if debugger.enabled:
                #     debugger.add_warning(f"Error creating color variations: {str(e)}", category="color_error")
                # print(f"⚠️ Error creating color variations: {str(e)}")
                # Fallback to simple replication
                r_channel = result
                g_channel = result
                b_channel = result
        else:
            # Without color scheme, duplicate the channel
            r_channel = result
            g_channel = result
            b_channel = result
            
        # Alpha channel remains the original noise
        a_channel = result
        
        # Combine all channels
        result = torch.cat([r_channel, g_channel, b_channel, a_channel], dim=1)
        
        # Track results after RGB channel combination
        # if debugger.enabled and debugger.debug_level >= 2:
        #     debugger.track_tensor_shape_history(result, "result_rgba", "after_color_channels")
        #     debugger.analyze_tensor(result, "domain_warp_rgba")
        
        # Expand to target_channels if specified
        try:
            # with debugger.time_operation("channel_expansion") if debugger.enabled else contextlib.nullcontext():
                if result.shape[1] == 4: # This is the RGBA tensor from color variations or replication
                    if target_channels == 4:
                        # Keep as is - just 4 channels
                        # if debugger.enabled and debugger.debug_level >= 1:
                        #     print(f"Using 4 channels for output (RGBA from color scheme or replication)")
                        #     # No change to 'result' needed here.
                        pass
                    
                    elif target_channels > 4:
                        num_base_channels = result.shape[1] # Should be 4
                        additional_channels_needed = target_channels - num_base_channels
                        additional_channels_list = []
                        
                        # Parameters from the main get_domain_warp function scope:
                        # p, device, octaves (original), seed (original from get_domain_warp call), 
                        # scale, warp_strength, phase_shift, time (original from get_domain_warp call)

                        base_channels_tensor = result # This is the initial RGBA

                        for i in range(additional_channels_needed):
                            current_channel_gen_index = i # for variation uniqueness: 0, 1, ... up to additional_channels_needed-1
                            variation_seed = seed + 700 + (current_channel_gen_index * 120) # Original seed from get_domain_warp
                            torch.manual_seed(variation_seed)

                            p_perturbed = p.clone()
                            pert_scale = 0.01 + (current_channel_gen_index * 0.005) # Slightly increasing perturbation
                            p_perturbed += (torch.randn_like(p) * pert_scale)
                            p_perturbed = torch.clamp(p_perturbed, 0.0, 1.0) # Keep coordinates in [0,1] range

                            time_offset = current_channel_gen_index * 0.03

                            # Vary parameters for the _domain_warp_with_phase call
                            varied_octaves_for_call = int(octaves + (current_channel_gen_index % 3) - 1)
                            varied_octaves_for_call = max(1, varied_octaves_for_call) # Ensure octaves >= 1
                            
                            # These are the noise_type and warp_type args for _domain_warp_with_phase's internal logic
                            call_noise_type = current_channel_gen_index % 3 # Cycles 0,1,2 (FBM, Ridged, Billow for base)
                            call_warp_type  = current_channel_gen_index % 4 # Cycles 0,1,2,3 (None, Offset, Dir, Swirl for warp)

                            varied_scale_factor = 1.0 + ((current_channel_gen_index % 5 - 2) * 0.05) # Approx +/- 10%
                            varied_warp_strength_factor = 1.0 + ((current_channel_gen_index % 7 - 3) * 0.05) # Approx +/- 15%
                            varied_phase_shift_val = phase_shift * (1.0 + ((current_channel_gen_index % 9 - 4) * 0.05)) # Approx +/- 20%
                            # Ensure phase_shift stays in a reasonable range (e.g., [0,1] if that's its typical input domain)
                            varied_phase_shift_clamped = torch.clamp(torch.tensor(varied_phase_shift_val), 0.0, 1.0).item()
                            
                            new_channel_bhwc = DomainWarpGenerator._domain_warp_with_phase(
                                p_perturbed, device, varied_octaves_for_call, variation_seed,
                                call_noise_type, call_warp_type,
                                scale * varied_scale_factor,
                                warp_strength * varied_warp_strength_factor,
                                varied_phase_shift_clamped,
                                time + time_offset
                            ) # Output is [B,H,W,1] and in [0,1] range from _domain_warp_with_phase

                            new_channel_bchw = new_channel_bhwc.permute(0, 3, 1, 2) # [B, 1, H, W]
                            additional_channels_list.append(new_channel_bchw)
                        
                        if additional_channels_list:
                            all_additional = torch.cat(additional_channels_list, dim=1)
                            result = torch.cat([base_channels_tensor, all_additional], dim=1)
                        # If no additional channels were actually generated (e.g. if additional_channels_needed was 0),
                        # result remains base_channels_tensor, which is correct.
                        
                        # if debugger.enabled and debugger.debug_level >= 1:
                        #     print(f"Expanded to {result.shape[1]} structured channels.")
                        #     debugger.track_tensor_shape_history(result, f"result_{result.shape[1]}ch_structured", f"channel_expansion_structured_{result.shape[1]}")

                    elif target_channels < 4: # Target is 1, 2, or 3
                        # if debugger.enabled and debugger.debug_level >= 1:
                        #     print(f"Reducing channels from 4 to {target_channels}")
                        result = result[:, :target_channels, :, :] # Slice from the RGBA result
                        # if debugger.enabled: # Add tracking for sliced result
                        #     debugger.track_tensor_shape_history(result, f"result_{target_channels}ch_sliced", f"channel_slicing_{target_channels}")
                    # If target_channels == 4, result is already the correct 4-channel tensor.
                
                # else: This case implies result.shape[1] was not 4 initially.
                # This shouldn't happen if the preceding logic correctly produces a 4-channel 'result'
                # (from color variations or replication of a single channel noise).
                # If it does, the original code didn't explicitly handle it either beyond the try-except for channel expansion.

        except Exception as e:
            # if debugger.enabled:
            #     debugger.add_warning(f"Error expanding channels: {str(e)}", category="channel_expansion_error")
            # print(f"⚠️ Error expanding channels: {str(e)}")
            # Ensure we have at least the base channels
            if result.shape[1] != target_channels:
                correct_shape = (batch_size, target_channels, height, width)
                correct_result = torch.zeros(correct_shape, device=device)
                # Copy available channels
                channels_to_copy = min(result.shape[1], target_channels)
                correct_result[:, :channels_to_copy] = result[:, :channels_to_copy]
                # Fill remaining channels with noise if needed
                if channels_to_copy < target_channels:
                    for c in range(channels_to_copy, target_channels):
                        correct_result[:, c] = torch.randn_like(result[:, 0])
                result = correct_result
                # print(f"✅ Created fallback tensor with {target_channels} channels") # Removed this line
        
        # Final tracking and analysis of the output tensor
        # if debugger.enabled:
        #     debugger.track_tensor_shape_history(result, "final_result", "final_output")
        #     debugger.analyze_tensor(result, "domain_warp_final")
        #     if debugger.debug_level >= 1:
        #         print(f"✅ Domain warp generated with shape {result.shape}")
        #         # Log tensor stats
        #         with torch.no_grad():
        #         print(f"📊 Tensor stats: min={result.min().item():.4f}, max={result.max().item():.4f}, "
        #               f"mean={result.mean().item():.4f}, std={result.std().item():.4f}")
        
        return result
    
    @staticmethod
    def _domain_warp_with_phase(p, device, octaves, seed, noise_type, warp_type, scale, warp_strength, phase, time=0.0):
        """
        Generate domain warped noise with phase modulation
        
        Args:
            p: Input coordinates tensor [batch, height, width, 2]
            device: Device to create tensor on
            octaves: Number of octaves for FBM noise  
            seed: Random seed value
            noise_type: Type of base noise to use
            warp_type: Type of domain warping to apply
            scale: Scale factor for input coordinates
            warp_strength: Strength of the domain warping
            phase: Value between 0 and 1 to blend between base and warped
            time: Animation time value
            
        Returns:
            Warped noise tensor [batch, height, width, 1]
        """
        # Convert octaves to integer
        octaves = int(octaves)
        
        # For very small phases, return pure base noise
        if phase < 0.01:
            # Generate base noise
            if noise_type == 0:  # FBM
                torch.manual_seed(seed)
                # FIXED: Update the call to match the new function signature, passing time as t parameter
                return DomainWarpGenerator.fbm_base(p * scale, octaves, seed, False, 0.8, 2.0, 0.5, time)
            elif noise_type == 1:  # Ridged
                torch.manual_seed(seed)
                return DomainWarpGenerator.ridged_multi(p * scale, octaves, device, seed, time)
            else:  # Billow
                torch.manual_seed(seed)
                return DomainWarpGenerator.billow(p * scale, octaves, device, seed, time)
        
        # For higher phases, we blend between base noise and warped noise
        # Generate base noise for later blending
        if noise_type == 0:  # FBM
            torch.manual_seed(seed)
            base_noise = DomainWarpGenerator.fbm_base(p * scale, octaves, seed, False, 0.8, 2.0, 0.5, time)
        elif noise_type == 1:  # Ridged
            torch.manual_seed(seed)
            base_noise = DomainWarpGenerator.ridged_multi(p * scale, octaves, device, seed, time)
        else:  # Billow
            torch.manual_seed(seed)
            base_noise = DomainWarpGenerator.billow(p * scale, octaves, device, seed, time)

        # Generate domain warped noise
        # Increase the warp strength for more visible effect
        enhanced_warp_strength = warp_strength * (1.0 + phase * 2.0)
        
        # Use a larger scale for warped noise to create more differentiation
        warped_scale = scale * (1.0 + phase * 0.5)
        
        # Use a different seed for the warped noise to ensure pattern difference
        warp_seed = seed + 42
        
        # Create a warped version with enhanced parameters
        warped_noise = DomainWarpGenerator.domain_warp(
            p, warp_type, warped_scale, enhanced_warp_strength, time, 
            octaves, device, warp_seed, False
        )
        
        # Blend between base and warped with enhanced contrast
        phase_adjusted = phase ** 0.8  # Make the phase response more linear
        
        # Apply contrast enhancement to warped noise for more difference
        warped_enhanced = (warped_noise - 0.5) * (1.0 + phase * 0.5) + 0.5
        
        # Blend with a method that preserves more detail
        blended = base_noise * (1.0 - phase_adjusted) + warped_enhanced * phase_adjusted
        
        # Add subtle detail enhancement based on the difference between base and warped
        difference = torch.abs(warped_enhanced - base_noise) 
        blended = blended + difference * phase_adjusted * 0.2
        
        # Normalize the result to ensure it stays in appropriate range
        blended = torch.clamp(blended, 0.0, 1.0)
        
        # Check if blended result is too similar to base noise (for debugging)
        mean_diff = torch.mean(torch.abs(blended - base_noise)).item()
        
        return blended
    
    @staticmethod
    def fbm_base(p, num_octaves, seed, temporal=False, h=0.8, lacunarity=2.0, gain=0.5, t=0.0):
        """
        Generate fractal Brownian motion base noise with enhanced structure
        
        Args:
            p: Input coordinates tensor [batch, height, width, dim]
            num_octaves: Number of noise layers to combine (minimum 3 for good structure)
            seed: Random seed for noise generation
            temporal: Whether to include temporal dimension
            h: Fractal dimension factor controlling roughness
            lacunarity: Frequency multiplier between octaves
            gain: Amplitude multiplier between octaves
            t: Time value for temporal noise
            
        Returns:
            FBM noise tensor [batch, height, width, 1]
        """
        batch, height, width, dim = p.shape
        
        # Ensure minimum octaves for good structure and convert to integer
        num_octaves = max(3, int(num_octaves))
        
        # FIXED: Make sure seed is an integer, not a device
        # If seed is a device object (which could happen due to parameter ordering issues),
        # use a default seed value
        if not isinstance(seed, (int, float, torch.Tensor)) or isinstance(seed, torch.device):
            # print(f"WARNING: Invalid seed type ({type(seed)}), using default seed") # Removed this line
            seed = 12345  # Use a default seed value
        
        # Convert seed to integer if it's a tensor or float
        if isinstance(seed, torch.Tensor):
            seed = seed.item()
        seed = int(seed)
        
        # Add variation based on seed
        seed_hash = seed % 10000
        
        # Create seed-specific parameter variations
        frequency_variation = 1.0 + (seed_hash % 100) / 200.0  # 0.5% - 1.5% variation
        lacunarity_variation = lacunarity * (1.0 + (seed_hash % 73) / 500.0)  # Slight variation
        gain_variation = gain * (1.0 + (seed_hash % 83) / 400.0)  # Slight variation
        
        # Initialize result with slight bias for more distinctive patterns
        result = torch.zeros(batch, height, width, 1, device=p.device)
        bias = (seed_hash % 100) / 1000.0 - 0.05  # Small initial bias: -0.05 to 0.05
        result += bias
        
        amplitude = 1.0
        frequency = 1.0 * frequency_variation
        
        # Apply initial domain warping for improved structure
        warp_strength = 0.5 + (seed_hash % 50) / 100.0  # 0.5 to 1.0
        warp_frequency = 1.2 + (seed_hash % 80) / 100.0  # 1.2 to 2.0
        
        # Generate initial domain warping
        warp = DomainWarpGenerator.simplex_noise(p * warp_frequency, seed + 1234) * warp_strength
        
        # Save original coordinates
        p_original = p.clone()
        
        # Different seeds for each octave to avoid pattern repetition
        octave_seeds = [(seed + i * 1337) % 100000 for i in range(num_octaves)]
        
        for i in range(num_octaves):
            # Apply progressive domain warping for better structure
            if i > 0:
                warp_amt = 0.1 * (i / num_octaves) * warp
                p = p_original + warp_amt
            
            # Generate noise at current frequency
            if temporal:
                noise = DomainWarpGenerator.simplex_noise_3d(p * frequency, octave_seeds[i], t * frequency)
            else:
                noise = DomainWarpGenerator.simplex_noise(p * frequency, octave_seeds[i])
            
            # Apply non-linear transformations for more structure
            if i == 0:
                # First octave - add ridges for backbone structure
                noise = torch.abs(noise) * 2.0 - 1.0
            elif i == 1:
                # Second octave - enhance details with slight distortion
                noise = noise * (1.0 + torch.sin(p[..., 0:1] * 3.0 + seed_hash) * 0.1)
            elif i >= num_octaves - 2:
                # Last octaves - add fine details
                noise = noise * (1.0 + torch.cos(p[..., 1:2] * 2.0 + seed_hash + i) * 0.1)
            
            # Add weighted noise to result
            result += amplitude * noise
            
            # Update parameters for next octave
            frequency *= lacunarity_variation
            amplitude *= gain_variation * (1.0 - 0.1 * (i / max(1, num_octaves - 1)))
        
        # Normalize result to [-1, 1] range with enhanced contrast
        result_min = torch.min(result)
        result_max = torch.max(result)
        result = 2.0 * (result - result_min) / (result_max - result_min) - 1.0
        
        # Apply final non-linear transform for better visual structure
        result = torch.tanh(result * 1.2) * 0.9
        
        return result
    
    @staticmethod
    def simplex_noise(p, seed):
        """
        Generate simplex noise for input coordinates
        
        Args:
            p: Coordinate tensor [batch, height, width, dim] or other shape that can be reshaped
            seed: Random seed for noise generation
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        # Handle different input shapes
        original_shape = p.shape
        
        # Check input tensor dimensions and reshape if necessary
        if len(original_shape) == 4:
            batch, height, width, dim = original_shape
        elif len(original_shape) == 3:
            # Assume [height, width, dim] format - add batch dimension
            height, width, dim = original_shape
            batch = 1
            p = p.unsqueeze(0)  # Add batch dimension
        elif len(original_shape) == 5:
            # Handle 5D tensor (e.g., from 3D noise with time dimension)
            batch, height, width, depth, dim = original_shape
            # Reshape to 4D by flattening height and width
            p = p.reshape(batch, height * width, depth, dim)
            height, width = height * width, depth
            dim = original_shape[-1]
        else:
            # For any other shape, try to adapt
            total_elements = p.numel()
            dim = original_shape[-1] if len(original_shape) > 1 else 1
            
            if dim < 1:
                dim = 1
                
            # Calculate remaining dimensions
            remaining = total_elements // dim
            
            # Default to square-ish dimensions if we need to reshape
            width = int(math.sqrt(remaining))
            height = remaining // width
            batch = 1
            
            # Reshape to 4D
            p = p.reshape(batch, height, width, dim)
        
        # Use seed to create pattern variations
        seed_hash = seed % 10000
        scale_variation = 1.0 + (seed_hash % 100) / 100.0 * 0.5  # Scale between 1.0 and 1.5
        rotation_angle = (seed_hash % 628) / 100.0  # Angle between 0 and 6.28 (2π)
        
        # Simplex noise constants
        F2 = 0.5 * (math.sqrt(3.0) - 1.0)
        G2 = (3.0 - math.sqrt(3.0)) / 6.0
        
        # Apply seed-based rotation and scaling to coordinates
        # Create rotation matrix
        cos_theta = torch.cos(torch.tensor(rotation_angle))
        sin_theta = torch.sin(torch.tensor(rotation_angle))
        
        # Make sure p has the right dimensions for coordinate operations
        if p.shape[-1] < 2:
            # If dim < 2, expand to at least 2 dimensions for x,y coordinates
            if p.shape[-1] == 1:
                # Duplicate the single dimension
                p = torch.cat([p, p], dim=-1)
            else:
                # Add a zero dimension
                zeros = torch.zeros_like(p[..., 0:1])
                p = torch.cat([p, zeros], dim=-1)
        
        # Apply rotation and scaling to input coordinates for more variation
        x = p[..., 0:1].clone()
        y = p[..., 1:2].clone()
        p_rotated = torch.cat([
            (x * cos_theta - y * sin_theta) * scale_variation,
            (x * sin_theta + y * cos_theta) * scale_variation
        ], dim=-1)
        
        # Extract coordinates
        x = p_rotated[..., 0:1]
        y = p_rotated[..., 1:2]
        
        # Skew coordinates to simplex space
        s = (x + y) * F2
        i = torch.floor(x + s)
        j = torch.floor(y + s)
        
        # Unskew back to regular space
        t = (i + j) * G2
        X0 = i - t
        Y0 = j - t
        
        # Calculate relative coordinates from cell origin
        x0 = x - X0
        y0 = y - Y0
        
        # Determine simplex (triangle) we're in and corresponding vertices
        # i1, j1 are the second closest corner
        # i2, j2 are the farthest corner (always (1,1) relative to (i0,j0))
        i1 = torch.zeros_like(x0)
        j1 = torch.zeros_like(y0)
        
        i1[x0 > y0] = 1.0
        j1[x0 <= y0] = 1.0
        
        # Coordinates of the other corners
        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2
        
        # Hash the coordinates based on seed for consistent gradients
        # This ensures different seeds produce different patterns
        def hash_coords(ix, iy, seed):
            # Better hashing function that uses seed
            # Ensure all operations are tensor-compatible
            h = ix * 1619 + iy * 31337 + seed * 2459
            h = torch.fmod(h * h * h, 1013)
            h = torch.fmod(h * h, 1013)
            h = torch.fmod(h * h * h, 1013)
            return h.type(torch.int64)
        
        # Calculate contribution from each corner
        def gradient(hash_val, x, y):
            # Improved gradient selection for more structured patterns
            # Use tensor operations instead of conditional statements
            h = hash_val & 7
            
            # Create all possible gradient vectors based on h value
            grad_x = torch.zeros_like(x)
            grad_y = torch.zeros_like(y)
            
            # h == 0: return x + y
            mask_0 = (h == 0)
            grad_x = torch.where(mask_0, x + grad_x, grad_x)
            grad_y = torch.where(mask_0, y + grad_y, grad_y)
            
            # h == 1: return x - y
            mask_1 = (h == 1)
            grad_x = torch.where(mask_1, x + grad_x, grad_x)
            grad_y = torch.where(mask_1, -y + grad_y, grad_y)
            
            # h == 2: return -x + y
            mask_2 = (h == 2)
            grad_x = torch.where(mask_2, -x + grad_x, grad_x)
            grad_y = torch.where(mask_2, y + grad_y, grad_y)
            
            # h == 3: return -x - y
            mask_3 = (h == 3)
            grad_x = torch.where(mask_3, -x + grad_x, grad_x)
            grad_y = torch.where(mask_3, -y + grad_y, grad_y)
            
            # h == 4: return x + x
            mask_4 = (h == 4)
            grad_x = torch.where(mask_4, x * 2.0 + grad_x, grad_x)
            grad_y = torch.where(mask_4, grad_y, grad_y)
            
            # h == 5: return -x - x
            mask_5 = (h == 5)
            grad_x = torch.where(mask_5, -x * 2.0 + grad_x, grad_x)
            grad_y = torch.where(mask_5, grad_y, grad_y)
            
            # h == 6: return y + y
            mask_6 = (h == 6)
            grad_x = torch.where(mask_6, grad_x, grad_x)
            grad_y = torch.where(mask_6, y * 2.0 + grad_y, grad_y)
            
            # h == 7: return -y - y
            mask_7 = (h == 7)
            grad_x = torch.where(mask_7, grad_x, grad_x)
            grad_y = torch.where(mask_7, -y * 2.0 + grad_y, grad_y)
            
            return grad_x + grad_y
        
        def corner_contribution(ix, iy, x, y):
            hash_val = hash_coords(ix, iy, seed_hash)
            grad = gradient(hash_val, x, y)
            
            # Calculate falloff function (improved)
            t = 0.5 - x*x - y*y
            # Enhanced falloff for better structure
            t = torch.maximum(t, torch.zeros_like(t))
            
            # Apply steeper falloff for more defined features
            t4 = t * t * t * t * t  # t^5 for sharper features
            return t4 * grad
        
        # Calculate noise contribution from each corner
        i0 = i.long()
        j0 = j.long()
        i1 = (i0 + i1.long()) % 1024  # Prevent overflow
        j1 = (j0 + j1.long()) % 1024
        i2 = (i0 + 1) % 1024
        j2 = (j0 + 1) % 1024
        
        # Compute contributions from three corners
        n0 = corner_contribution(i0, j0, x0, y0)
        n1 = corner_contribution(i1, j1, x1, y1)
        n2 = corner_contribution(i2, j2, x2, y2)
        
        # Sum up and scale to [-1,1]
        # Apply non-linear transformation for better structure
        noise = 70.0 * (n0 + n1 + n2)
        
        # Apply contrast enhancement for more structured appearance
        noise = torch.tanh(noise * 1.2) * 0.85
        
        # Reshape back to original batch, height, width dimensions with single channel
        if len(original_shape) == 3:
            # Remove the batch dimension we added
            noise = noise.squeeze(0)
        elif len(original_shape) == 5:
            # Reshape back to original 5D shape with single output channel
            batch, height, width, depth, _ = original_shape
            noise = noise.reshape(batch, height, width, depth, 1)
        
        return noise.unsqueeze(-1) if noise.shape[-1] != 1 else noise

    @staticmethod
    def fbm_base_temporal(p, octaves, time, device, seed, use_temporal_coherence=True):
        """
        Generate FBM (Fractal Brownian Motion) noise with temporal coherence
        
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            octaves: Number of octaves
            time: Animation time value
            device: Device to create tensor on
            seed: Base seed value (consistent across frames)
            use_temporal_coherence: Whether to use true temporal coherence
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        batch, height, width, _ = p.shape
        
        # Convert octaves to integer
        octaves = int(octaves)
        
        # FIXED: Make sure seed is an integer, not a device
        # If seed is a device object (which could happen due to parameter ordering issues),
        # use a default seed value
        if not isinstance(seed, (int, float, torch.Tensor)) or isinstance(seed, torch.device):
            # print(f"WARNING: Invalid seed type ({type(seed)}), using default seed") # Removed this line
            seed = 12345  # Use a default seed value
            
        # Convert seed to integer if it's a tensor or float
        if isinstance(seed, torch.Tensor):
            seed = seed.item()
        seed = int(seed)
        
        # Standard FBM using user-controlled octaves
        result = torch.zeros(batch, height, width, 1, device=p.device)  # Use p.device instead of device parameter
        amp = 1.0
        freq = 1.0
        
        for i in range(min(octaves, 8)):
            # Generate simplex noise at current frequency
            current_p = p * freq
            
            # Add time as a third dimension for temporal coherence
            time_offset = time * (0.2 + i * 0.05)  # Different rate per octave
            
            # Create 3D coordinates with time
            current_p_temporal = torch.cat([
                current_p,
                torch.ones_like(current_p[:, :, :, :1]) * time_offset
            ], dim=-1)
            
            # Generate 3D simplex noise with temporal coherence
            noise = DomainWarpGenerator.simplex_noise_3d(current_p_temporal, seed + i)
            
            # Add to result with current amplitude
            result += amp * noise
            
            # Prepare for next octave
            freq *= 2.0
            amp *= 0.5
        
        return result
    
    @staticmethod
    def simplex_noise_3d(coords, seed=0):
        """
        3D simplex noise implementation in PyTorch for temporal coherence
        This provides coherent noise values when the third dimension is time
        
        Args:
            coords: Coordinate tensor [batch, height, width, 3] or other compatible shape
            seed: Random seed value
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        # FIXED: Make sure seed is an integer, not a device
        if not isinstance(seed, (int, float, torch.Tensor)) or isinstance(seed, torch.device):
            # print(f"WARNING: Invalid seed type ({type(seed)}), using default seed") # Removed this line
            seed = 12345  # Use a default seed value
            
        # Convert seed to integer if it's a tensor or float
        if isinstance(seed, torch.Tensor):
            seed = seed.item()
        seed = int(seed)
        
        # Handle different input shapes
        original_shape = coords.shape
        
        # Check input tensor dimensions and reshape if necessary
        if len(original_shape) == 4 and original_shape[-1] >= 3:
            batch, height, width, _ = original_shape
        elif len(original_shape) == 3:
            # Assume [height, width, dim] format - add batch dimension
            height, width, _ = original_shape
            batch = 1
            coords = coords.unsqueeze(0)  # Add batch dimension
        else:
            # For any other shape, try to adapt
            total_elements = coords.numel()
            dim = original_shape[-1] if len(original_shape) > 1 else 3
            
            if dim < 3:
                # If less than 3 dims, pad with zeros
                if len(original_shape) > 1:
                    padding_needed = 3 - original_shape[-1]
                    zeros = torch.zeros((*original_shape[:-1], padding_needed), device=coords.device)
                    coords = torch.cat([coords, zeros], dim=-1)
                else:
                    # Handle 1D tensor
                    coords = coords.reshape(-1, 1)
                    zeros = torch.zeros(coords.shape[0], 2, device=coords.device)
                    coords = torch.cat([coords, zeros], dim=-1)
                
            # Calculate remaining dimensions
            total_elements = coords.numel()
            remaining = total_elements // 3
            
            # Default to square-ish dimensions if we need to reshape
            width = int(math.sqrt(remaining))
            height = remaining // width
            batch = 1
            
            # Reshape to 4D
            coords = coords.reshape(batch, height, width, 3)
        
        device = coords.device
        
        # Set deterministic seed
        torch.manual_seed(seed)
        
        # Extract x, y, z (time) components
        x, y, z = coords[:, :, :, 0], coords[:, :, :, 1], coords[:, :, :, 2]
        
        # 3D Simplex noise constants
        F3 = 1.0 / 3.0
        G3 = 1.0 / 6.0
        
        # Skew input space to determine which simplex cell contains point
        s = (x + y + z) * F3
        i = torch.floor(x + s)
        j = torch.floor(y + s)
        k = torch.floor(z + s)
        
        t = (i + j + k) * G3
        X0 = i - t
        Y0 = j - t
        Z0 = k - t
        
        # Unskewed distances from cell origin
        x0 = x - X0
        y0 = y - Y0
        z0 = z - Z0
        
        # Determine simplex (tetrahedron) based on which of x0, y0, z0 is largest
        # We need to rank the coordinates to determine which simplex we're in
        # This is a simplified approach that works for most cases
        x_ge_y = (x0 >= y0).float()
        y_ge_z = (y0 >= z0).float()
        x_ge_z = (x0 >= z0).float()
        
        # Calculate offsets for second corner
        i1 = (x_ge_y * x_ge_z).float()
        j1 = ((1.0 - x_ge_y) * y_ge_z).float()
        k1 = ((1.0 - x_ge_z) * (1.0 - y_ge_z)).float()
        
        # Calculate offsets for third corner
        i2 = (x_ge_y + (1.0 - x_ge_y) * x_ge_z).float()
        j2 = (x_ge_y * (1.0 - x_ge_z) + (1.0 - x_ge_y)).float()
        k2 = ((1.0 - x_ge_z) + x_ge_z * (1.0 - x_ge_y)).float()
        
        # Calculate corner positions (in unskewed coordinates)
        x1 = x0 - i1 + G3
        y1 = y0 - j1 + G3
        z1 = z0 - k1 + G3
        
        x2 = x0 - i2 + 2.0 * G3
        y2 = y0 - j2 + 2.0 * G3
        z2 = z0 - k2 + 2.0 * G3
        
        x3 = x0 - 1.0 + 3.0 * G3
        y3 = y0 - 1.0 + 3.0 * G3
        z3 = z0 - 1.0 + 3.0 * G3
        
        # Hash coordinates of the corners to get random gradients
        # This is a simplified hash function that works for visualization
        def hash_coord(a, b, c, s):
            h = 1664525 * a + 1013904223
            h += 1664525 * b + 1013904223
            h += 1664525 * c + 1013904223
            h += s  # Add seed for variation
            h = torch.fmod(h, 3777.7777)
            
            # Map hash to a unit vector (gradient)
            theta = h * 3.883222  # ~pi*1.236 to cover the unit sphere
            phi = h * 0.67899  # ~pi/5 for variation
            
            # Create 3D gradient direction
            gx = torch.cos(theta) * torch.cos(phi)
            gy = torch.sin(theta) * torch.cos(phi)
            gz = torch.sin(phi)
            return torch.stack([gx, gy, gz], dim=-1)
        
        # Get gradient vectors for each corner
        g0 = hash_coord(i, j, k, seed)
        g1 = hash_coord(i + i1, j + j1, k + k1, seed)
        g2 = hash_coord(i + i2, j + j2, k + k2, seed)
        g3 = hash_coord(i + 1.0, j + 1.0, k + 1.0, seed)
        
        # Calculate noise contribution from each corner
        def corner_influence(grad, x, y, z):
            # Distance vector from corner to point
            t = 0.6 - x*x - y*y - z*z
            t = torch.maximum(t, torch.zeros_like(t))
            t4 = t*t*t*t
            
            # Gradient dot product with vertex->point vector
            n = grad[:,:,:,0]*x + grad[:,:,:,1]*y + grad[:,:,:,2]*z
            return t4 * n
        
        # Compute contributions from the four corners
        n0 = corner_influence(g0, x0, y0, z0)
        n1 = corner_influence(g1, x1, y1, z1)
        n2 = corner_influence(g2, x2, y2, z2)
        n3 = corner_influence(g3, x3, y3, z3)
        
        # Sum contributions and scale to match 2D output range
        noise = (n0 + n1 + n2 + n3) * 30.0
        
        # Reshape back to original dimensions if needed
        if len(original_shape) == 3:
            # Remove the batch dimension we added
            noise = noise.squeeze(0)

        return noise.unsqueeze(-1) if noise.shape[-1] != 1 else noise

    @staticmethod
    def domain_warp(p, warp_type=0, scale=1.0, warp_strength=0.5, time=0.0, octaves=3, device=None, seed=0, use_temporal_coherence=False):
        """
        Generate domain warped noise
    
        Args:
            p: Input coordinates tensor [batch, height, width, 2]
            warp_type: Type of domain warping (0-3)
            scale: Scale factor for input coordinates
            warp_strength: Strength of the warping effect
            time: Animation time parameter
            octaves: Number of octaves for fractal noise
            device: Device to create tensor on
            seed: Random seed for noise generation
            use_temporal_coherence: Whether to use temporal coherence for animation
        
        Returns:
            Warped noise tensor [batch, height, width, 1]
        """
        # Ensure device is set
        if device is None:
            device = p.device

        # Set torch seed to ensure consistent noise generation
        torch.manual_seed(seed)

        # Convert warp_type to int if needed
        warp_type = int(warp_type)
        
        # Clamp to valid range
        warp_type = max(0, min(warp_type, 3))
        
        # Convert octaves to integer
        octaves = int(octaves)
        
        # For warp_type 0 (no warping), return base FBM noise
        if warp_type == 0:
            # Updated to use the new function signature
            return DomainWarpGenerator.fbm_base(p * scale, octaves, seed, use_temporal_coherence, 0.8, 2.0, 0.5, time)
        
        # For other warp types, generate base noise first
        torch.manual_seed(seed)
        base_noise = DomainWarpGenerator.fbm_base(p * scale, octaves, seed, use_temporal_coherence, 0.8, 2.0, 0.5, time)
        
        # Create warped coordinates
        warp_seed = seed + 42
        torch.manual_seed(warp_seed)
        
        # Generate warp vector field
        if warp_type == 1:
            # Simple offset warping
            warp_coords = p * scale * 0.5
            warp_noise = DomainWarpGenerator.simplex_noise(warp_coords, warp_seed) * warp_strength
        elif warp_type == 2:
            # Directional warping
            warp_coords_x = p * scale * 0.4 + torch.tensor([[[[0.0, 0.5]]]], device=device)
            warp_coords_y = p * scale * 0.4 + torch.tensor([[[[0.5, 0.0]]]], device=device)
            warp_noise_x = DomainWarpGenerator.simplex_noise(warp_coords_x, warp_seed) * warp_strength
            warp_noise_y = DomainWarpGenerator.simplex_noise(warp_coords_y, warp_seed + 17) * warp_strength
            warp_noise = torch.cat([warp_noise_x, warp_noise_y], dim=-1)
        else:  # warp_type == 3, swirl warping
            # Calculate distance from center and angle for swirl
            center = torch.tensor([[[[0.0, 0.0]]]], device=device)
            offset = p - center
            dist = torch.sqrt(torch.sum(offset * offset, dim=-1, keepdim=True))
            angle = torch.atan2(offset[..., 1:2], offset[..., 0:1])
            
            # Create swirl effect
            swirl = angle + dist * warp_strength * 5.0
            warp_x = torch.cos(swirl) * dist
            warp_y = torch.sin(swirl) * dist
            warp_noise = torch.cat([warp_x, warp_y], dim=-1) - offset
        
        # Apply warp to get new coordinates
        p_warped = p + warp_noise
        
        # Generate noise with warped coordinates
        warped_noise = DomainWarpGenerator.fbm_base(
            p_warped * scale, octaves, warp_seed, use_temporal_coherence, 0.8, 2.0, 0.5, time
        )
        
        # Blend base and warped noise
        blended = (base_noise + warped_noise) * 0.5
        
        # Add detail enhancement
        detail = torch.abs(warped_noise - base_noise) * 0.5
        result = blended + detail
        
        return torch.clamp(result, -1.0, 1.0)

    @staticmethod
    def ridged_multi(p, num_octaves, device, seed, t=0.0):
        """
        Generate ridged multifractal noise with enhanced structure
        
        Args:
            p: Input coordinates tensor [batch, height, width, dim]
            num_octaves: Number of noise layers to combine (minimum 3 for good structure)
            device: Device for computation (not used directly, kept for API compatibility)
            seed: Random seed for noise generation
            t: Time value for temporal noise
            
        Returns:
            Ridged multifractal noise tensor [batch, height, width, 1]
        """
        batch, height, width, dim = p.shape
        
        # Ensure minimum octaves for good structure and convert to integer
        num_octaves = max(3, int(num_octaves))
        
        # Handle seed type issues
        if not isinstance(seed, (int, float, torch.Tensor)) or isinstance(seed, torch.device):
            # print(f"WARNING: Invalid seed type ({type(seed)}), using default seed") # Removed this line
            seed = 12345
        
        # Convert seed to integer
        if isinstance(seed, torch.Tensor):
            seed = seed.item()
        seed = int(seed)
        
        # Add variation based on seed
        seed_hash = seed % 10000
        
        # Create seed-specific parameter variations
        frequency_variation = 1.0 + (seed_hash % 100) / 200.0
        lacunarity = 2.0 + (seed_hash % 73) / 500.0
        gain = 0.5 * (1.0 + (seed_hash % 83) / 400.0)
        
        # Initialize result with slight bias
        result = torch.zeros(batch, height, width, 1, device=p.device)
        bias = (seed_hash % 100) / 1000.0 - 0.05
        result += bias
        
        # Initial frequency and amplitude
        frequency = 1.0 * frequency_variation
        amplitude = 1.0
        
        # Apply initial domain warping for improved structure
        warp_strength = 0.5 + (seed_hash % 50) / 100.0
        warp_frequency = 1.2 + (seed_hash % 80) / 100.0
        warp = DomainWarpGenerator.simplex_noise(p * warp_frequency, seed + 1234) * warp_strength
        
        # Save original coordinates
        p_original = p.clone()
        
        # Different seeds for each octave
        octave_seeds = [(seed + i * 1337) % 100000 for i in range(num_octaves)]
        
        # Weight values sum
        sum_weights = 0.0
        
        for i in range(num_octaves):
            # Apply progressive domain warping for better structure
            if i > 0:
                warp_amt = 0.1 * (i / num_octaves) * warp
                p = p_original + warp_amt
            
            # Generate noise at current frequency
            noise = DomainWarpGenerator.simplex_noise(p * frequency, octave_seeds[i])
            
            # Ridged multifractal: abs(noise) * 2 - 1
            noise = torch.abs(noise) * 2.0 - 1.0
            
            # Invert and raise to a power to create ridges
            noise = 1.0 - torch.abs(noise)
            noise = torch.pow(noise, 2.5 + (i * 0.1))
            
            # Apply special enhancements for certain octaves
            if i == 0:
                # First octave - enhance ridge sharpness
                noise = noise * 1.25
            elif i == 1:
                # Second octave - add turbulence
                noise = noise * (1.0 + torch.sin(p[..., 0:1] * 3.0 + seed_hash) * 0.1)
            elif i >= num_octaves - 2:
                # Last octaves - add fine ridge details
                noise = noise * (1.0 + torch.cos(p[..., 1:2] * 4.0 + seed_hash + i) * 0.15)
            
            # Add weighted noise to result
            weight = amplitude
            result += weight * noise
            sum_weights += weight
            
            # Update parameters for next octave
            frequency *= lacunarity
            amplitude *= gain * (1.0 - 0.15 * (i / max(1, num_octaves - 1)))
        
        # Normalize by sum of weights
        if sum_weights > 0:
            result /= sum_weights
        
        # Apply final non-linear transform for better visual structure
        result = torch.tanh(result * 1.5) * 0.95
        
        return result
    
    @staticmethod
    def billow(p, num_octaves, device, seed, t=0.0):
        """
        Generate billow noise with enhanced structure
        
        Args:
            p: Input coordinates tensor [batch, height, width, dim]
            num_octaves: Number of noise layers to combine (minimum 3 for good structure)
            device: Device for computation (not used directly, kept for API compatibility)
            seed: Random seed for noise generation
            t: Time value for temporal noise
            
        Returns:
            Billow noise tensor [batch, height, width, 1]
        """
        batch, height, width, dim = p.shape
        
        # Ensure minimum octaves for good structure and convert to integer
        num_octaves = max(3, int(num_octaves))
        
        # Handle seed type issues
        if not isinstance(seed, (int, float, torch.Tensor)) or isinstance(seed, torch.device):
            # print(f"WARNING: Invalid seed type ({type(seed)}), using default seed") # Removed this line
            seed = 12345
        
        # Convert seed to integer
        if isinstance(seed, torch.Tensor):
            seed = seed.item()
        seed = int(seed)
        
        # Add variation based on seed
        seed_hash = seed % 10000
        
        # Create seed-specific parameter variations
        frequency_variation = 1.0 + (seed_hash % 100) / 200.0
        lacunarity = 2.0 + (seed_hash % 73) / 500.0
        gain = 0.5 * (1.0 + (seed_hash % 83) / 400.0)
        
        # Initialize result with slight bias
        result = torch.zeros(batch, height, width, 1, device=p.device)
        bias = (seed_hash % 100) / 1000.0 - 0.05
        result += bias
        
        # Initial frequency and amplitude
        frequency = 1.0 * frequency_variation
        amplitude = 1.0
        
        # Apply initial domain warping for improved structure
        warp_strength = 0.5 + (seed_hash % 50) / 100.0
        warp_frequency = 1.2 + (seed_hash % 80) / 100.0
        warp = DomainWarpGenerator.simplex_noise(p * warp_frequency, seed + 1234) * warp_strength
        
        # Save original coordinates
        p_original = p.clone()
        
        # Different seeds for each octave
        octave_seeds = [(seed + i * 1337) % 100000 for i in range(num_octaves)]
        
        # Weight values sum
        sum_weights = 0.0
        
        for i in range(num_octaves):
            # Apply progressive domain warping for better structure
            if i > 0:
                warp_amt = 0.1 * (i / num_octaves) * warp
                p = p_original + warp_amt
            
            # Generate noise at current frequency
            noise = DomainWarpGenerator.simplex_noise(p * frequency, octave_seeds[i])
            
            # Billow noise: 2 * |noise| - 1
            noise = 2.0 * torch.abs(noise) - 1.0
            
            # Apply special enhancements for certain octaves
            if i == 0:
                # First octave - enhance main structure
                noise = noise * 1.2
            elif i == 1:
                # Second octave - add variation
                noise = noise * (1.0 + torch.sin(p[..., 0:1] * 2.5 + seed_hash) * 0.15)
            elif i >= num_octaves - 2:
                # Last octaves - add fine details
                detail_factor = 0.2 + 0.1 * (i - (num_octaves - 2)) / 2.0
                noise = noise * (1.0 + torch.cos(p[..., 1:2] * 3.5 + seed_hash + i) * detail_factor)
            
            # Add weighted noise to result
            weight = amplitude
            result += weight * noise
            sum_weights += weight
            
            # Update parameters for next octave
            frequency *= lacunarity
            amplitude *= gain * (1.0 - 0.1 * (i / max(1, num_octaves - 1)))
        
        # Normalize by sum of weights
        if sum_weights > 0:
            result /= sum_weights
        
        # Apply final non-linear transform for better visual structure
        result = torch.tanh(result * 1.3) * 0.9
        
        return result

def enhance_domain_warp_for_blending(shader_params):
    """
    Enhance domain warp parameters for more effective blending
    
    Args:
        shader_params: The shader parameters dictionary
    
    Returns:
        Enhanced shader parameters
    """
    # Do not modify user-specified parameters; return a copy.
    return shader_params.copy() if shader_params else {}

def get_domain_warp(self, batch_size, height, width, shader_params=None, device="cpu", seed=None, use_temporal_coherence=False):
    """
    Get a domain warp noise tensor
    
    Args:
        batch_size: Batch size
        height: Height of tensor
        width: Width of tensor
        shader_params: Dictionary of shader parameters
        device: Device to create tensor on
        seed: Random seed
        use_temporal_coherence: Whether to use temporal coherence
    
    Returns:
        Domain warp noise tensor
    """
    # Extract and validate parameters
    params = shader_params.copy() if shader_params else {}
    
    # Extract seed variation if provided
    seed_variation = params.get('seed_variation', 0)
    if seed_variation > 0:
        # Apply seed variation to make patterns more distinctive
        if seed is not None:
            seed = seed + seed_variation
            # print(f"Using seed variation: {seed_variation} for more distinctive patterns") # Removed this line
    
    # Ensure minimum effective parameters for visual impact
    warp_strength = max(params.get('warp_strength', 0.5), 0.8)
    scale = max(params.get('scale', 1.0), 1.5)
    octaves = max(params.get('octaves', 1), 3)
    time = params.get('time', 0.0)
    phase_shift = params.get('phase_shift', 0.5)
    
    # Determine current seed based on temporal coherence
    current_seed = seed if not use_temporal_coherence else None
    
    # Generate the domain warp tensor
    warp_tensor = generate_domain_warp_tensor(
        batch_size, height, width, params, 
        scale, warp_strength, phase_shift, time, 
        device, current_seed, octaves, 
        use_temporal_coherence
    )
    
    return warp_tensor

def generate_domain_warp_tensor(batch_size, height, width, shader_params=None, scale=1.0, warp_strength=0.5, 
                               phase_shift=0.5, time=0.0, device="cuda", seed=0, octaves=4,
                               use_temporal_coherence=False, target_channels=None):
    """
    Directly generate a domain warp tensor without using ShaderToTensor
    
    Args:
        batch_size: Number of images in batch
        height: Height of tensor
        width: Width of tensor
        shader_params: Dictionary of shader parameters (overrides individual parameters if provided)
        scale: Scale factor for coordinates
        warp_strength: Strength of domain warping
        phase_shift: Phase shift parameter
        time: Animation time value
        device: Device to create tensor on
        seed: Random seed value
        octaves: Number of octaves for FBM noise
        use_temporal_coherence: Whether to use temporal coherence
        target_channels: Number of output channels (4, 6, or 9) - helpful for different models
        
    Returns:
        Tensor with shape [batch_size, channels, height, width]
    """
    # Get debugger instance
    # debugger = get_debugger()   
    
    # Safety check: ensure shader_params is a dictionary or create a default one
    if shader_params is None or not isinstance(shader_params, dict):
        if not isinstance(shader_params, dict) and shader_params is not None:
            print(f"Warning: shader_params is not a dictionary ({type(shader_params)}), using defaults")
        shader_params = {
            "scale": scale,
            "warp_strength": warp_strength,
            "phase_shift": phase_shift,
            "octaves": octaves,
            "time": time,
            "temporal_coherence": use_temporal_coherence,
            "base_seed": seed if use_temporal_coherence else None,
        }
    
    # Add target_channels to shader_params if provided
    if target_channels is not None:
        shader_params["target_channels"] = target_channels
        # if debugger.enabled and debugger.debug_level >= 1:
        #     print(f"Setting target_channels={target_channels} in shader_params")
    
    # Log generation parameters
    # if debugger.enabled and debugger.debug_level >= 1:
        # debugger.log_generation_operation(
        #     "generate_domain_warp_tensor_start",
        #     {},  # empty dict for input_tensors since we don't have actual tensors yet
        #     torch.zeros((1, 1, 1, 1), device=device),  # dummy tensor for output
        #     {
        #         "batch_size": batch_size,
        #         "height": height,
        #         "width": width,
        #         "scale": scale,
        #         "warp_strength": warp_strength,
        #         "phase_shift": phase_shift,
        #         "seed": seed,
        #         "octaves": octaves,
        #         "target_channels": target_channels,
        #         "shader_params": shader_params
        #     }
        # )
    
    # Add colorScheme if not present to make output more visually distinct
    if "colorScheme" not in shader_params:
        shader_params["colorScheme"] = "viridis"
        shader_params["shaderColorIntensity"] = 0.9
        # print("Adding color scheme for more visually distinct output") # Removed this line
    
    # Ensure seed creates distinctive patterns (add some prime number offsets)
    try:
        # with debugger.time_operation("seed_transformation") if debugger.enabled else contextlib.nullcontext():
            if shader_params.get("temporal_coherence", use_temporal_coherence):
                # For temporal coherence, keep base_seed consistent
                base_seed = shader_params.get("base_seed", seed)
                shader_params["base_seed"] = base_seed
                
                # Ensure each frame has a distinctive enough seed variation
                shader_params["seed_variation"] = (seed % 100) * 73 + 17
                # print(f"Using temporal coherence with base_seed={base_seed}, variation={shader_params['seed_variation']}") # Removed this line
            else:
                pass # Add pass to ensure the else block is not empty
                # print(f"Using seed={seed} directly for non-temporal coherence (for pattern consistency).") # Removed this line
                # The `seed` variable that is passed to DomainWarpGenerator.get_domain_warp
                # will be the original incoming seed from the arguments of this function.
    except Exception as e:
        # if debugger.enabled:
        #     debugger.add_warning(f"Error in seed transformation: {str(e)}", category="seed_error")
        pass
    
    # Generate domain warp noise
    try:
        result = DomainWarpGenerator.get_domain_warp(
            batch_size, height, width, shader_params, device, seed
        )
    except Exception as e:
        # print(f"❌ Error generating domain warp tensor: {str(e)}") # Removed this line
        # Fallback to random noise with correct shape and channel count
        target_ch = shader_params.get("target_channels", 9)
        result = torch.randn((batch_size, target_ch, height, width), device=device)
        # print(f"✅ Created fallback tensor with {target_ch} channels and shape {result.shape}") # Removed this line
    
    # Final analysis of the output tensor
    # if debugger.enabled:
    #     debugger.track_tensor_shape_history(result, "final_result", "final_output")
    #     debugger.analyze_tensor(result, "domain_warp_final_tensor")
    # if debugger.debug_level >= 1:
    #     print(f"✅ Domain warp tensor generated with shape {result.shape}")
    #     # Log tensor stats
    #     with torch.no_grad():
    #     print(f"📊 Tensor stats: min={result.min().item():.4f}, max={result.max().item():.4f}, "
    #           f"mean={result.mean().item():.4f}, std={result.std().item():.4f}")
    
    return result

def get_rgb_noise(p, device, octaves, seed, noise_type, warp_type, scale, warp_strength, intensity, time):
    """
    Generate RGB noise with domain warping for colored textures
    
    Args:
        p: Input coordinates tensor [batch, height, width, 2]
        device: Device to use for tensor operations
        octaves: Number of octaves for noise
        seed: Random seed
        noise_type: Type of base noise (0: FBM, 1: Ridged, 2: Billow)
        warp_type: Type of domain warping (0: None, 1-3: Different warp types)
        scale: Scale factor for input coordinates
        warp_strength: Strength of the domain warping
        intensity: Intensity of the color variation
        time: Animation time parameter
        
    Returns:
        RGB noise tensor [batch, height, width, 3]
    """
    # Create color channel variations
    batch, h, w, _ = p.shape
    
    # Intensity factor controls variation between channels
    intensity_factor = intensity * 0.5
    
    # Different coordinate sets for each channel
    p_r = p.clone()
    p_g = p.clone() + torch.tensor([[[[12.9898, 78.233]]]], device=device) * 0.0001
    p_b = p.clone() + torch.tensor([[[[39.346, 11.135]]]], device=device) * 0.0001
    
    # Generate base red channel
    r_warp_result = DomainWarpGenerator._domain_warp_with_phase(
        p_r, device, octaves, seed, noise_type, warp_type, 
        scale, 
        warp_strength, 
        1.0, time
    )
    
    # Generate warped patterns with contrasting parameters
    g_warp_result = DomainWarpGenerator._domain_warp_with_phase(
        p_g, device, octaves, seed, noise_type, (warp_type + 1) % 4, 
        scale * 1.1, 
        warp_strength * (1.0 - 0.3 * intensity_factor), 
        1.0, time + 0.1
    )
    
    b_warp_result = DomainWarpGenerator._domain_warp_with_phase(
        p_b, device, octaves, seed, noise_type, (warp_type + 2) % 4, 
        scale * 0.9, 
        warp_strength * (1.0 + 0.3 * intensity_factor), 
        1.0, time + 0.2
    )
    
    # Combine results into RGB noise tensor
    rgb_noise = torch.stack([r_warp_result, g_warp_result, b_warp_result], dim=1)
    
    return rgb_noise 

def add_domain_warp_to_tensor(tensor, shader_params=None, scale=1.0, warp_strength=0.5, 
                             phase_shift=0.5, time=0.0, seed=0, octaves=4, 
                             blend_mode="multiply", blend_factor=0.8, device=None):
    """
    Add domain warp noise to an existing tensor for texture enhancement
    
    Args:
        tensor: Input tensor to enhance [batch, channels, height, width]
        shader_params: Dictionary of shader parameters (overrides individual parameters if provided)
        scale: Scale factor for coordinates
        warp_strength: Strength of domain warping
        phase_shift: Phase shift parameter
        time: Animation time value
        seed: Random seed value
        octaves: Number of octaves for noise
        blend_mode: How to blend the noise with the original tensor ("add", "multiply", "overlay", "alpha")
        blend_factor: How much of the effect to apply (0.0 to 1.0)
        device: Device to create noise on (defaults to tensor's device)
        
    Returns:
        Enhanced tensor with domain warp applied
    """
    # Check if tensor is a class type (for registration with ShaderToTensor)
    if isinstance(tensor, type):
        # Register method with the class
        @classmethod
        def domain_warp(cls, p, warp_type=0, scale=1.0, warp_strength=0.5, time=0.0, octaves=3, device=None, seed=0, use_temporal_coherence=False):
            """Apply domain warping to coordinates"""
            if device is None:
                device = p.device
                
            # Get tensor dimensions
            batch, height, width, _ = p.shape
            
            # Generate domain warped noise
            return DomainWarpGenerator.domain_warp(
                p, warp_type, scale, warp_strength, time, 
                octaves, device, seed, use_temporal_coherence
            )
        
        # Add domain_warp method to the class
        setattr(tensor, "domain_warp", domain_warp)
        
        # Add apply_domain_warp method for blending with existing tensor
        @classmethod
        def apply_domain_warp(cls, tensor, shader_params=None, scale=1.0, warp_strength=0.5, 
                             phase_shift=0.5, time=0.0, seed=0, octaves=4, 
                             blend_mode="multiply", blend_factor=0.8, device=None):
            """Add domain warp noise to a tensor"""
            if device is None:
                device = tensor.device
                
            # Get tensor dimensions
            batch, channels, height, width = tensor.shape
            
            # Generate domain warp tensor
            warp_tensor = generate_domain_warp_tensor(
                batch, height, width, shader_params,
                scale, warp_strength, phase_shift, time,
                device, seed, octaves,
                target_channels=channels  # Pass the target channels
            )
            
            # Ensure compatible dimensions
            if warp_tensor.shape[1] == 4 and channels != 4:
                # Use only the first 'channels' dimensions of the warp_tensor
                warp_tensor = warp_tensor[:, :channels]
            elif warp_tensor.shape[1] < channels:
                # Duplicate the last channel to match dimensions
                last_channel = warp_tensor[:, -1:].repeat(1, channels - warp_tensor.shape[1], 1, 1)
                warp_tensor = torch.cat([warp_tensor, last_channel], dim=1)
            
            # Apply blend mode
            if blend_mode == "add":
                result = tensor + warp_tensor * blend_factor
            elif blend_mode == "multiply":
                normalized_warp = warp_tensor * 0.5
                result = tensor * (1.0 + normalized_warp * blend_factor)
            elif blend_mode == "overlay":
                tensor_norm = tensor * 2.0 - 1.0
                warp_norm = warp_tensor * 2.0 - 1.0
                
                dark_mask = (tensor_norm < 0)
                light_mask = (tensor_norm >= 0)
                
                result = torch.zeros_like(tensor)
                result[dark_mask] = (tensor_norm[dark_mask] * (1.0 + warp_norm[dark_mask] * blend_factor))
                result[light_mask] = tensor_norm[light_mask] + warp_norm[light_mask] * blend_factor * (1.0 - tensor_norm[light_mask])
                
                result = (result + 1.0) * 0.5
            elif blend_mode == "alpha":
                result = tensor * (1.0 - blend_factor) + warp_tensor * blend_factor
            else:
                result = tensor * (1.0 - blend_factor) + warp_tensor * blend_factor
            
            # Ensure result is in valid range
            result = torch.clamp(result, 0.0, 1.0)
            
            return result
        
        # Add the method to the class
        setattr(tensor, "apply_domain_warp", apply_domain_warp)
        # print("=== Added domain_warp functionality to ShaderToTensor ===") # Removed this line
        return tensor
    
    # Regular tensor processing
    if device is None:
        device = tensor.device
        
    # Get tensor dimensions
    batch, channels, height, width = tensor.shape
    
    # Generate domain warp tensor
    warp_tensor = generate_domain_warp_tensor(
        batch, height, width, shader_params,
        scale, warp_strength, phase_shift, time,
        device, seed, octaves,
        target_channels=channels  # Pass the target channels
    )
    
    # Ensure compatible dimensions
    if warp_tensor.shape[1] == 4 and channels != 4:
        # Use only the first 'channels' dimensions of the warp_tensor
        warp_tensor = warp_tensor[:, :channels]
    elif warp_tensor.shape[1] < channels:
        # Duplicate the last channel to match dimensions
        last_channel = warp_tensor[:, -1:].repeat(1, channels - warp_tensor.shape[1], 1, 1)
        warp_tensor = torch.cat([warp_tensor, last_channel], dim=1)
    
    # Apply blend mode
    if blend_mode == "add":
        # Add mode: tensor + warp_tensor * blend_factor
        result = tensor + warp_tensor * blend_factor
    elif blend_mode == "multiply":
        # Multiply mode: tensor * (1 + warp_tensor * blend_factor)
        # Normalize warp_tensor to [-0.5, 0.5] for better multiplication effect
        normalized_warp = warp_tensor * 0.5
        result = tensor * (1.0 + normalized_warp * blend_factor)
    elif blend_mode == "overlay":
        # Overlay mode: blend between multiply and screen modes
        # For dark areas in original, multiply; for light areas, screen
        tensor_norm = tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        warp_norm = warp_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        
        # Overlay formula: if base < 0.5, 2 * base * blend; else 1 - 2 * (1 - base) * (1 - blend)
        dark_mask = (tensor_norm < 0)
        light_mask = (tensor_norm >= 0)
        
        # Apply overlay formula
        result = torch.zeros_like(tensor)
        result[dark_mask] = (tensor_norm[dark_mask] * (1.0 + warp_norm[dark_mask] * blend_factor))
        result[light_mask] = tensor_norm[light_mask] + warp_norm[light_mask] * blend_factor * (1.0 - tensor_norm[light_mask])
        
        # Normalize back to [0, 1]
        result = (result + 1.0) * 0.5
    elif blend_mode == "alpha":
        # Alpha blend: lerp(tensor, warp_tensor, blend_factor)
        result = tensor * (1.0 - blend_factor) + warp_tensor * blend_factor
    else:
        # Default to alpha blend if unknown mode
        result = tensor * (1.0 - blend_factor) + warp_tensor * blend_factor
    
    # Ensure result is in valid range
    result = torch.clamp(result, 0.0, 1.0)
    
    return result 