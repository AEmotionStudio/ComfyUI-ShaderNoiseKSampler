import torch
import math
# Import ShaderParamsReader for apply_shape_mask functionality
from ..shader_params_reader import ShaderParamsReader

class TensorFieldGenerator:
    """
    PyTorch implementation of tensor field animation that closely matches
    the WebGL tensor field shader from shader_renderer.js
    
    This class generates tensor field patterns that can be used to influence the
    sampling process in image generation.
    Used to influence the sampling process in image generation.
    
    Now enhanced with temporal coherence for smooth animations.
    """
    
    @staticmethod
    def get_tensor_field(batch_size, height, width, shader_params, device="cuda", seed=0):
        """
        Generate tensor field noise tensor that matches the WebGL implementation
        
        Args:
            batch_size: Number of images in batch
            height: Height of tensor
            width: Width of tensor
            shader_params: Dictionary containing parameters from shader_params.json
            device: Device to create tensor on
            seed: Random seed value to ensure different patterns with different seeds
            
        Returns:
            Tensor with shape [batch_size, target_channels, height, width]
        """
        # Get debugger instance
        # debugger = get_debugger()

        # Extract parameters from shader_params - check both original and converted names
        scale = shader_params.get("shaderScale", shader_params.get("scale", 1.0))
        warp_strength = shader_params.get("shaderWarpStrength", shader_params.get("warp_strength", 0.5))
        phase_shift = shader_params.get("shaderPhaseShift", shader_params.get("phase_shift", 0.5))
        octaves = shader_params.get("shaderOctaves", shader_params.get("octaves", 1))
        time = shader_params.get("time", 0.0)
        
        base_seed = shader_params.get("base_seed", seed)
        use_temporal_coherence = shader_params.get("temporal_coherence", shader_params.get("useTemporalCoherence", False))
        
        shape_type = shader_params.get("shaderShapeType", shader_params.get("shape_type", "none"))
        shape_mask_strength = shader_params.get("shaderShapeStrength", shader_params.get("shapemaskstrength", 1.0))
        
        color_scheme = shader_params.get("colorScheme", shader_params.get("color_scheme", "none"))
        color_intensity = shader_params.get("shaderColorIntensity", shader_params.get("color_intensity", 0.8))

        target_channels = shader_params.get("target_channels", 4)
        model_class = shader_params.get("model_class", "")
        inner_model_class = shader_params.get("inner_model_class", "")

        if inner_model_class == "CosmosVideo" or model_class == "CosmosVideo":
            target_channels = 16
            shader_params["target_channels"] = 16
        elif inner_model_class == "ACEStep":
            target_channels = 8
            shader_params["target_channels"] = 8
        elif inner_model_class == "WAN21" or model_class == "WAN21":
            target_channels = 16
            shader_params["target_channels"] = 16
        
        # Only track tensor shapes, not dictionaries
        if isinstance(shader_params, torch.Tensor):
            #debugger.track_tensor_shape_history(shader_params, "shader_params_input", "initial_parameters")
            pass

        y, x = torch.meshgrid(
            torch.linspace(0, 1, height, device=device),
            torch.linspace(0, 1, width, device=device),
            indexing='ij'
        )
        p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        #debugger.track_tensor_shape_history(p, "p_coord_grid", "coordinate_grid_creation")
        pass
        
        all_generated_channels_list = []
        loop_seed_master = base_seed if use_temporal_coherence else seed

        # Define color channel specific variations (g_time_offset, b_time_offset etc.)
        # This block is from the original color path, made available for the loop.
        g_time_offset, b_time_offset = 0.0, 0.0
        g_vis_type, b_vis_type = int(octaves % 4), int(octaves % 4)
        g_scale, b_scale = scale, scale
        g_warp, b_warp = warp_strength, warp_strength
        
        if color_scheme != "none" and color_intensity > 0:
            intensity_factor = 0.5 + color_intensity * 0.5
            if color_scheme.lower() == "rainbow": # Removed "magma"
                g_time_offset = 0.33 * intensity_factor
                b_time_offset = 0.66 * intensity_factor
                g_vis_type = (int(octaves % 4) + 1) % 4 if intensity_factor > 0.7 else int(octaves % 4)
                b_vis_type = (int(octaves % 4) + 2) % 4 if intensity_factor > 0.7 else int(octaves % 4)
                g_scale = scale * (1.0 + 0.1 * intensity_factor)
                b_scale = scale * (1.0 - 0.1 * intensity_factor)
                g_warp = warp_strength * (1.0 - 0.2 * intensity_factor)
                b_warp = warp_strength * (1.0 + 0.2 * intensity_factor)
            elif color_scheme.lower() == "magma":
                g_time_offset = 0.40 * intensity_factor 
                b_time_offset = 0.70 * intensity_factor 
                g_vis_type = (int(octaves % 4) + 2) % 4 # Different viz type for G
                b_vis_type = (int(octaves % 4) + 0) % 4 # B can use same as R, but other params differ
                g_scale = scale * (1.0 + 0.12 * intensity_factor) 
                b_scale = scale * (1.0 - 0.18 * intensity_factor) 
                g_warp = warp_strength * (1.0 - 0.12 * intensity_factor) 
                b_warp = warp_strength * (1.0 + 0.18 * intensity_factor)
            elif color_scheme.lower() == "inferno":
                g_time_offset = 0.15 * intensity_factor # Smaller time offset for G
                b_time_offset = 0.35 * intensity_factor # Smaller time offset for B
                g_vis_type = (int(octaves % 4) + 3) % 4 # Different viz type for G
                b_vis_type = (int(octaves % 4) + 1) % 4 # Different viz type for B, distinct from G
                g_scale = scale * (1.0 - 0.05 * intensity_factor) # Slight scale decrease for G
                b_scale = scale * (1.0 + 0.05 * intensity_factor) # Slight scale increase for B
                g_warp = warp_strength * (1.0 + 0.05 * intensity_factor) # Slight warp increase for G
                b_warp = warp_strength * (1.0 - 0.1 * intensity_factor) # Slight warp decrease for B
            elif color_scheme.lower() == "turbo":
                g_time_offset = 0.10 * intensity_factor
                b_time_offset = 0.20 * intensity_factor 
                g_vis_type = (int(octaves % 4) + 0) % 4 # G can use same viz as R initially
                b_vis_type = (int(octaves % 4) + 2) % 4 # B uses a shifted viz type
                g_scale = scale * (1.0 + 0.20 * intensity_factor) # More significant scale change for G
                b_scale = scale * (1.0 - 0.20 * intensity_factor) # More significant scale change for B
                g_warp = warp_strength * (1.0 - 0.08 * intensity_factor)
                b_warp = warp_strength * (1.0 + 0.22 * intensity_factor)
            elif color_scheme.lower() == "plasma":
                g_time_offset = 0.25 * intensity_factor
                b_time_offset = 0.55 * intensity_factor
                g_vis_type = (int(octaves % 4) + 2) % 4 # Shift visualization type for G
                b_vis_type = int(octaves % 4) # B can use same viz type as R but with different params
                g_scale = scale * (1.0 - 0.15 * intensity_factor) # Slightly different scale for G
                b_scale = scale * (1.0 + 0.15 * intensity_factor) # Slightly different scale for B
                g_warp = warp_strength * (1.0 + 0.1 * intensity_factor) # Different warp for G
                b_warp = warp_strength * (1.0 - 0.25 * intensity_factor) # Different warp for B
            elif color_scheme.lower() == "jet":
                g_time_offset = 0.30 * intensity_factor 
                b_time_offset = 0.60 * intensity_factor 
                g_vis_type = (int(octaves % 4) + 1) % 4 
                b_vis_type = (int(octaves % 4) + 3) % 4 
                g_scale = scale * (1.0 - 0.07 * intensity_factor)
                b_scale = scale * (1.0 + 0.11 * intensity_factor) 
                g_warp = warp_strength * (1.0 + 0.13 * intensity_factor) 
                b_warp = warp_strength * (1.0 - 0.09 * intensity_factor)
            elif color_scheme.lower() == "viridis":
                g_time_offset = 0.20 * intensity_factor
                b_time_offset = 0.40 * intensity_factor # Corrected from 0.50 from previous bad edit
                g_vis_type = (int(octaves % 4) + 1) % 4
                b_vis_type = (int(octaves % 4) + 3) % 4 # Using +3 for a different variation pattern
                g_scale = scale * (1.0 + 0.08 * intensity_factor) # Corrected from -0.07
                b_scale = scale * (1.0 - 0.08 * intensity_factor) # Corrected from +0.11
                g_warp = warp_strength * (1.0 - 0.15 * intensity_factor) # Corrected from +0.13
                b_warp = warp_strength * (1.0 + 0.15 * intensity_factor) # Corrected from -0.09
            elif color_scheme.lower() == "blue_red":
                # R channel (i=0) will use base parameters by default.
                # G channel (i=1) - Aim for raw value of -1.0.
                # Parameters set for minimal G, will be overwritten later if color_intensity > 0.
                g_time_offset = 0.0  # No specific time shift for G from R
                g_vis_type = 0     # Eigenvalue magnitude
                g_scale = scale * 0.01 # Very small scale for G
                g_warp = 0.0       # No warp for G

                # B channel (i=2) - Aim for inverted R.
                r_viz_type = int(octaves % 4)
                if r_viz_type == 2: # If R's viz_type is hyperstreamlines
                    b_time_offset = math.pi
                    b_vis_type = 2
                else:
                    # Fallback: make B different from R
                    b_time_offset = 0.5 * intensity_factor 
                    b_vis_type = (r_viz_type + 1) % 4 
                b_scale = scale # Same scale as R
                b_warp = warp_strength # Same warp as R
            elif color_scheme.lower() == "hot":
                g_time_offset = 0.1 * intensity_factor
                b_time_offset = 0.7 * intensity_factor
                g_vis_type = (int(octaves % 4) + 1) % 4
                b_vis_type = (int(octaves % 4) + 3) % 4
                g_scale = scale * (1.0 + 0.15 * intensity_factor)
                b_scale = scale * (1.0 - 0.15 * intensity_factor)
                g_warp = warp_strength * (1.0 - 0.1 * intensity_factor)
                b_warp = warp_strength * (1.0 + 0.1 * intensity_factor)
            elif color_scheme.lower() == "cool":
                g_time_offset = 0.6 * intensity_factor
                b_time_offset = 0.2 * intensity_factor
                g_vis_type = (int(octaves % 4) + 0) % 4
                b_vis_type = (int(octaves % 4) + 2) % 4
                g_scale = scale * (1.0 - 0.1 * intensity_factor)
                b_scale = scale * (1.0 + 0.1 * intensity_factor)
                g_warp = warp_strength * (1.0 + 0.2 * intensity_factor)
                b_warp = warp_strength * (1.0 - 0.2 * intensity_factor)
            elif color_scheme.lower() == "parula":
                g_time_offset = 0.22 * intensity_factor
                b_time_offset = 0.48 * intensity_factor
                g_vis_type = (int(octaves % 4) + 2) % 4
                b_vis_type = (int(octaves % 4) + 0) % 4
                g_scale = scale * (1.0 + 0.03 * intensity_factor)
                b_scale = scale * (1.0 - 0.06 * intensity_factor)
                g_warp = warp_strength * (1.0 - 0.04 * intensity_factor)
                b_warp = warp_strength * (1.0 + 0.07 * intensity_factor)
            elif color_scheme.lower() == "hsv": # HSV in curl_noise uses hue from angle, value from magnitude
                g_time_offset = 0.5 * intensity_factor # For G, let's try a phase shift
                b_time_offset = 0.0 # For B, let's try no major time shift from R for magnitude-like effect
                g_vis_type = (int(octaves % 4) + 1) % 4 # G will have a different pattern
                b_vis_type = 0 # B will try to represent magnitude (eigenvalue viz type)
                g_scale = scale * (1.0 + 0.05 * intensity_factor)
                b_scale = scale * 0.8 # B scale slightly reduced
                g_warp = warp_strength
                b_warp = warp_strength * 0.5 # B warp reduced
            elif color_scheme.lower() == "autumn":
                g_time_offset = 0.18 * intensity_factor
                b_time_offset = 0.38 * intensity_factor
                g_vis_type = (int(octaves % 4) + 3) % 4
                b_vis_type = (int(octaves % 4) + 1) % 4
                g_scale = scale * (1.0 - 0.02 * intensity_factor)
                b_scale = scale * (1.0 + 0.04 * intensity_factor)
                g_warp = warp_strength * (1.0 + 0.06 * intensity_factor)
                b_warp = warp_strength * (1.0 - 0.08 * intensity_factor)
            elif color_scheme.lower() == "winter":
                g_time_offset = 0.65 * intensity_factor
                b_time_offset = 0.25 * intensity_factor
                g_vis_type = (int(octaves % 4) + 0) % 4
                b_vis_type = (int(octaves % 4) + 2) % 4
                g_scale = scale * (1.0 - 0.12 * intensity_factor)
                b_scale = scale * (1.0 + 0.08 * intensity_factor)
                g_warp = warp_strength * (1.0 + 0.18 * intensity_factor)
                b_warp = warp_strength * (1.0 - 0.22 * intensity_factor)
            elif color_scheme.lower() == "spring":
                g_time_offset = 0.35 * intensity_factor
                b_time_offset = 0.75 * intensity_factor
                g_vis_type = (int(octaves % 4) + 1) % 4
                b_vis_type = (int(octaves % 4) + 3) % 4
                g_scale = scale * (1.0 + 0.10 * intensity_factor)
                b_scale = scale * (1.0 - 0.10 * intensity_factor)
                g_warp = warp_strength * (1.0 - 0.15 * intensity_factor)
                b_warp = warp_strength * (1.0 + 0.15 * intensity_factor)
            elif color_scheme.lower() == "summer":
                g_time_offset = 0.45 * intensity_factor
                b_time_offset = 0.12 * intensity_factor
                g_vis_type = (int(octaves % 4) + 2) % 4
                b_vis_type = (int(octaves % 4) + 0) % 4
                g_scale = scale * (1.0 + 0.07 * intensity_factor)
                b_scale = scale * (1.0 - 0.13 * intensity_factor)
                g_warp = warp_strength * (1.0 + 0.03 * intensity_factor)
                b_warp = warp_strength * (1.0 - 0.07 * intensity_factor)
            elif color_scheme.lower() == "copper":
                g_time_offset = 0.05 * intensity_factor
                b_time_offset = 0.10 * intensity_factor
                g_vis_type = (int(octaves % 4) + 0) % 4
                b_vis_type = (int(octaves % 4) + 0) % 4
                g_scale = scale * (1.0 - 0.20 * intensity_factor) # G less prominent
                b_scale = scale * (1.0 - 0.30 * intensity_factor) # B even less prominent
                g_warp = warp_strength * 0.7
                b_warp = warp_strength * 0.5
            elif color_scheme.lower() == "pink":
                g_time_offset = 0.55 * intensity_factor
                b_time_offset = 0.80 * intensity_factor
                g_vis_type = (int(octaves % 4) + 1) % 4
                b_vis_type = (int(octaves % 4) + 2) % 4
                g_scale = scale * (1.0 + 0.02 * intensity_factor)
                b_scale = scale * (1.0 + 0.06 * intensity_factor) # B slightly more prominent scale
                g_warp = warp_strength * (1.0 - 0.03 * intensity_factor)
                b_warp = warp_strength * (1.0 + 0.03 * intensity_factor)
            elif color_scheme.lower() == "bone":
                g_time_offset = 0.30 * intensity_factor
                b_time_offset = 0.30 * intensity_factor # G and B similar time shifts
                g_vis_type = (int(octaves % 4) + 0) % 4
                b_vis_type = (int(octaves % 4) + 0) % 4 # G and B similar viz
                g_scale = scale * 0.9
                b_scale = scale * 0.8
                g_warp = warp_strength * 0.9
                b_warp = warp_strength * 0.8
            elif color_scheme.lower() == "ocean":
                g_time_offset = 0.25 * intensity_factor
                b_time_offset = 0.70 * intensity_factor
                g_vis_type = (int(octaves % 4) + 2) % 4
                b_vis_type = (int(octaves % 4) + 0) % 4
                g_scale = scale * (1.0 - 0.05 * intensity_factor)
                b_scale = scale * (1.0 + 0.15 * intensity_factor)
                g_warp = warp_strength * (1.0 + 0.10 * intensity_factor)
                b_warp = warp_strength * (1.0 - 0.10 * intensity_factor)
            elif color_scheme.lower() == "terrain":
                g_time_offset = 0.10 * intensity_factor
                b_time_offset = 0.50 * intensity_factor
                g_vis_type = (int(octaves % 4) + 3) % 4
                b_vis_type = (int(octaves % 4) + 1) % 4
                g_scale = scale * (1.0 + 0.18 * intensity_factor)
                b_scale = scale * (1.0 - 0.04 * intensity_factor)
                g_warp = warp_strength * (1.0 - 0.20 * intensity_factor)
                b_warp = warp_strength * (1.0 + 0.05 * intensity_factor)
            elif color_scheme.lower() == "neon":
                g_time_offset = 0.60 * intensity_factor
                b_time_offset = 0.90 * intensity_factor # Large shift for B
                g_vis_type = (int(octaves % 4) + 1) % 4
                b_vis_type = (int(octaves % 4) + 2) % 4
                g_scale = scale # G same scale as R
                b_scale = scale # B same scale as R
                g_warp = warp_strength * (1.0 + 0.25 * intensity_factor) # Stronger G warp
                b_warp = warp_strength * (1.0 - 0.25 * intensity_factor) # Inverse B warp
            elif color_scheme.lower() == "fire":
                g_time_offset = 0.08 * intensity_factor
                b_time_offset = 0.16 * intensity_factor # B also low offset
                g_vis_type = (int(octaves % 4) + 0) % 4
                b_vis_type = (int(octaves % 4) + 0) % 4 # G and B similar to R viz
                g_scale = scale * (1.0 + 0.25 * intensity_factor) # G stronger scale
                b_scale = scale * (1.0 - 0.25 * intensity_factor) # B weaker scale
                g_warp = warp_strength * (1.0 + 0.10 * intensity_factor)
                b_warp = warp_strength * 0.3 # B very low warp
            elif color_scheme.lower() in ["complementary"]: # "hot", "cool" removed
                g_time_offset = 0.5 * intensity_factor
                b_time_offset = 1.0 * intensity_factor
                g_vis_type = (int(octaves % 4) + 2) % 4 if intensity_factor > 0.6 else int(octaves % 4)
                b_vis_type = (int(octaves % 4) + 1) % 4 if intensity_factor > 0.6 else int(octaves % 4)
            else: # Default variations for other color schemes
                g_time_offset = 0.2 * intensity_factor
                b_time_offset = 0.4 * intensity_factor
                g_vis_type = int(octaves % 4)
                b_vis_type = int(octaves % 4)
                g_scale = scale * (1.0 + 0.05 * intensity_factor)
                b_scale = scale * (1.0 - 0.05 * intensity_factor)
                g_warp = warp_strength * (1.0 - 0.1 * intensity_factor)
                b_warp = warp_strength * (1.0 + 0.1 * intensity_factor)

        for i in range(target_channels):
            current_channel_gen_index = i
            
            # Initialize current parameters to base/default for this channel
            current_p = p.clone() # Start with original coordinates
            current_time = time
            current_viz_type = int(octaves % 4)
            current_scale = scale
            current_warp = warp_strength
            # Base seed for this channel, will be adjusted for R,G,B if color active
            current_seed = loop_seed_master + 700 + (current_channel_gen_index * 130)
            
            torch.manual_seed(current_seed)

            is_rgb_color_channel = color_scheme != "none" and color_intensity > 0 and i < 3
            
            if is_rgb_color_channel:
                if i == 0: # Red channel
                    current_seed = base_seed if use_temporal_coherence else seed # Use main seed
                    # Parameters are already set to base (scale, warp_strength, time, octaves%4)
                elif i == 1: # Green channel
                    current_time = time + g_time_offset
                    current_viz_type = g_vis_type
                    current_scale = g_scale
                    current_warp = g_warp
                    current_seed = (base_seed + 42) if use_temporal_coherence else (seed + 42)
                elif i == 2: # Blue channel
                    current_time = time + b_time_offset
                    current_viz_type = b_vis_type
                    current_scale = b_scale
                    current_warp = b_warp
                    current_seed = (base_seed + 123) if use_temporal_coherence else (seed + 123)
            else: # Monochrome path OR additional channels (i >= 3) in color mode
                # Perturb parameters for structured variation
                pert_scale_val = 0.005 + (current_channel_gen_index * 0.001) # Small perturbation
                current_p += (torch.randn_like(p) * pert_scale_val)
                current_p = torch.clamp(current_p, 0.0, 1.0)
                
                current_time = time + (current_channel_gen_index * 0.02) # Small time offset
                current_viz_type = (int(octaves % 4) + current_channel_gen_index) % 4 # Vary viz type
                
                current_scale_factor = 1.0 + ((current_channel_gen_index % 5 - 2) * 0.03) # Vary scale slightly
                current_scale = scale * current_scale_factor
                
                current_warp_factor = 1.0 + ((current_channel_gen_index % 7 - 3) * 0.03) # Vary warp slightly
                current_warp = warp_strength * current_warp_factor
                # current_seed is already set by the loop using current_channel_gen_index

            # Generate the single channel noise
            single_channel_bhw1 = TensorFieldGenerator.tensor_field(
                current_p, current_viz_type, current_scale, current_warp, 
                current_time, device, current_seed, use_temporal_coherence
            )
            
            # Ensure single_channel_bhw1 is [B, H, W, 1]
            if len(single_channel_bhw1.shape) == 3: # If tensor_field returned [B,H,W]
                single_channel_bhw1 = single_channel_bhw1.unsqueeze(-1)

            # Post-process the generated channel
            current_contrast = 1.0 + phase_shift
            single_channel_bhw1 *= current_contrast
            single_channel_bhw1 = torch.clamp(single_channel_bhw1, -1.0, 1.0)
            
            # Specific overwrite for G channel in "blue_red" scheme
            if is_rgb_color_channel and i == 1 and color_scheme.lower() == "blue_red" and color_intensity > 0:
                # Force G channel to be -1.0 (raw) to achieve normalized 0 later
                single_channel_bhw1 = torch.full_like(single_channel_bhw1, -1.0)
            
            if shape_type not in ["none", "0"] and shape_mask_strength > 0:
                # Use original p for a consistent mask across channels, or current_p for varied mask
                p_normalized_for_mask = p * 2.0 - 1.0 
                # p_normalized_for_mask = current_p * 2.0 - 1.0 # Alternative: varied mask
                
                shape_mask_for_channel = ShaderParamsReader.apply_shape_mask(p_normalized_for_mask, shape_type)
                
                # Ensure shape_mask_for_channel is [B,H,W,1] for lerp
                if len(shape_mask_for_channel.shape) == 3: # Is [B,H,W]
                    shape_mask_for_channel = shape_mask_for_channel.unsqueeze(-1)
                # Expand batch dim if necessary
                if shape_mask_for_channel.shape[0] != single_channel_bhw1.shape[0]:
                    shape_mask_for_channel = shape_mask_for_channel.expand(single_channel_bhw1.shape[0], -1, -1, -1)
                
                #debugger.track_tensor_shape_history(shape_mask_for_channel, f"shape_mask_for_channel_ch{i}", "shape_mask_application")
                #debugger.track_tensor_shape_history(single_channel_bhw1, f"single_channel_bhw1_ch{i}", "shape_mask_application")
                single_channel_bhw1 = torch.lerp(single_channel_bhw1, single_channel_bhw1 * shape_mask_for_channel, shape_mask_strength)
                single_channel_bhw1 = torch.clamp(single_channel_bhw1, -1.0, 1.0)

            single_channel_b1hw = single_channel_bhw1.permute(0, 3, 1, 2)
            #debugger.track_tensor_shape_history(single_channel_b1hw, f"single_channel_b1hw_final_ch{i}", "individual_channel_permutation")
            all_generated_channels_list.append(single_channel_b1hw)
            
        final_result_bchw = torch.cat(all_generated_channels_list, dim=1)
        #debugger.track_tensor_shape_history(final_result_bchw, "final_result_bchw_pre_debug_report", "all_channels_concatenated")
        # print(f"TensorField: Generated {final_result_bchw.shape[1]} structured channels directly.") # Removed this line

        # Channel count mismatch warning (like curl_noise.py)
        if final_result_bchw.shape[1] != target_channels:
            # print(f"⚠️ Channel mismatch: Generated {final_result_bchw.shape[1]} but need {target_channels}") # Removed this line
            #debugger.add_warning(
            #    f"Channel count mismatch: got {final_result_bchw.shape[1]} but expected {target_channels}",
            #    category="channel_count"
            #)
            # Optionally, pad or trim channels to match target_channels
            corrected = torch.zeros((batch_size, target_channels, height, width), device=device)
            min_channels = min(final_result_bchw.shape[1], target_channels)
            corrected[:, :min_channels] = final_result_bchw[:, :min_channels]
            if final_result_bchw.shape[1] < target_channels:
                std = final_result_bchw.std()
                mean = final_result_bchw.mean()
                for c in range(min_channels, target_channels):
                    phase = (c / target_channels) * 2 * 3.14159
                    corrected[:, c] = torch.sin(p[..., 0].permute(0, 3, 1, 2) * 5 + phase) * std + mean
            final_result_bchw = corrected
            #debugger.track_tensor_shape_history(final_result_bchw, "final_result_bchw_corrected", "channel_correction")
            # print(f"✅ Adjusted tensor field to {target_channels} channels for compatibility") # Removed this line

        # Generate a debug report for this tensor generation process
        #debugger.visualize_tensor_shapes_report()
        return final_result_bchw
    
    @staticmethod
    def compute_tensor_properties(p, scale, warp_strength, time, device, seed, use_temporal_coherence=False):
        """
        Compute tensor field properties (eigenvalues and eigenvectors)
        Matches the computeTensorProperties function in WebGL shader
        
        Args:
            p: Coordinate tensor [batch, height, width, 2] in [0, 1] range
            scale: Scale factor
            warp_strength: Amount of warping
            time: Animation time
            device: Device to use for computation
            seed: Random seed value
            use_temporal_coherence: Whether to use temporal coherence for animations
            
        Returns:
            Tuple of (lambda1, lambda2, v1, v2) where lambda are eigenvalues and v are eigenvectors
        """
        batch, height, width, _ = p.shape
        
        # Offset based on time
        offset = torch.tensor([[[[time * 0.05, 0.0]]]], device=device)
        p1 = p * scale + offset
        
        # Apply warp to coordinates if warp strength is non-zero
        if warp_strength > 0.0:
            # Generate warp field based on noise
            if use_temporal_coherence:
                # Use 3D noise for temporal coherence
                warp_noise1 = TensorFieldGenerator.simplex_noise_3d(
                    p1 * 0.3, 
                    seed,
                    time_offset=time * 0.2
                )
                warp_noise2 = TensorFieldGenerator.simplex_noise_3d(
                    p1 * 0.3, 
                    seed + 1,
                    time_offset=time * 0.2 + 3.33
                )
            else:
                # Original implementation
                warp_noise1 = TensorFieldGenerator.simplex_noise(p1 * 0.3 + torch.tensor([[[[0.0, 1.0]]]], device=device), seed)
                warp_noise2 = TensorFieldGenerator.simplex_noise(p1 * 0.3 + torch.tensor([[[[1.0, 0.0]]]], device=device), seed + 1)
            
            # Apply warp to coordinates
            p1 = p1 + torch.cat([warp_noise1, warp_noise2], dim=-1) * warp_strength
        
        # Use noise derivatives to generate tensor field
        eps = 0.01
        
        # Compute approximate derivatives of noise field
        if use_temporal_coherence:
            # Use 3D noise with time for smoother transitions
            n00 = TensorFieldGenerator.simplex_noise_3d(p1, seed + 2, time_offset=time * 0.1)
            n10 = TensorFieldGenerator.simplex_noise_3d(p1 + torch.tensor([[[[eps, 0.0]]]], device=device), seed + 2, time_offset=time * 0.1)
            n01 = TensorFieldGenerator.simplex_noise_3d(p1 + torch.tensor([[[[0.0, eps]]]], device=device), seed + 2, time_offset=time * 0.1)
            n11 = TensorFieldGenerator.simplex_noise_3d(p1 + torch.tensor([[[[eps, eps]]]], device=device), seed + 2, time_offset=time * 0.1)
            n_minus_x = TensorFieldGenerator.simplex_noise_3d(p1 - torch.tensor([[[[eps, 0.0]]]], device=device), seed + 2, time_offset=time * 0.1)
            n_minus_y = TensorFieldGenerator.simplex_noise_3d(p1 - torch.tensor([[[[0.0, eps]]]], device=device), seed + 2, time_offset=time * 0.1)
        else:
            # Original implementation
            n00 = TensorFieldGenerator.simplex_noise(p1, seed + 2)
            n10 = TensorFieldGenerator.simplex_noise(p1 + torch.tensor([[[[eps, 0.0]]]], device=device), seed + 2)
            n01 = TensorFieldGenerator.simplex_noise(p1 + torch.tensor([[[[0.0, eps]]]], device=device), seed + 2)
            n11 = TensorFieldGenerator.simplex_noise(p1 + torch.tensor([[[[eps, eps]]]], device=device), seed + 2)
            n_minus_x = TensorFieldGenerator.simplex_noise(p1 - torch.tensor([[[[eps, 0.0]]]], device=device), seed + 2)
            n_minus_y = TensorFieldGenerator.simplex_noise(p1 - torch.tensor([[[[0.0, eps]]]], device=device), seed + 2)
        
        # Calculate derivatives (gradient components)
        dx = (n10 - n00) / eps
        dy = (n01 - n00) / eps
        
        # Second order derivatives for tensor components
        dxx = (n10 - 2.0 * n00 + n_minus_x) / (eps * eps)
        dyy = (n01 - 2.0 * n00 + n_minus_y) / (eps * eps)
        dxy = (n11 - n10 - n01 + n00) / (eps * eps)
        
        # Construct tensor matrix components
        T00 = dxx
        T01 = dxy
        T10 = dxy
        T11 = dyy
        
        # Calculate eigenvalues
        trace = T00 + T11
        det = T00 * T11 - T01 * T10
        # Use safe sqrt to avoid NaN
        discriminant = torch.sqrt(torch.clamp(trace * trace - 4.0 * det, min=1e-8))
        
        # Two eigenvalues
        lambda1 = (trace + discriminant) * 0.5
        lambda2 = (trace - discriminant) * 0.5
        
        # Initialize eigenvectors
        v1 = torch.zeros_like(p)
        v2 = torch.zeros_like(p)
        
        # Expand mask dimensions for proper broadcasting
        mask_T01 = (torch.abs(T01) > 0.0001).expand(-1, -1, -1, 2)
        mask_T10 = (torch.abs(T10) > 0.0001).expand(-1, -1, -1, 2)
        
        # Calculate first eigenvector - handling each case separately
        # Where T01 is significant, use (T01, lambda1 - T00) direction
        dir1_T01 = torch.cat([T01, lambda1 - T00], dim=-1)
        norm1_T01 = torch.norm(dir1_T01, dim=-1, keepdim=True).expand(-1, -1, -1, 2)
        norm1_T01 = torch.where(norm1_T01 > 1e-8, norm1_T01, torch.ones_like(norm1_T01))
        v1_T01 = dir1_T01 / norm1_T01
        
        # Where T10 is significant (and T01 is not), use (lambda1 - T11, T10) direction
        dir1_T10 = torch.cat([lambda1 - T11, T10], dim=-1)
        norm1_T10 = torch.norm(dir1_T10, dim=-1, keepdim=True).expand(-1, -1, -1, 2)
        norm1_T10 = torch.where(norm1_T10 > 1e-8, norm1_T10, torch.ones_like(norm1_T10))
        v1_T10 = dir1_T10 / norm1_T10
        
        # Apply the result based on the masks using torch.where instead of indexing
        mask_T10_not_T01 = mask_T10 & (~mask_T01)
        v1 = torch.where(mask_T01, v1_T01, v1)
        v1 = torch.where(mask_T10_not_T01, v1_T10, v1)
        
        # Diagonal case - where neither T01 nor T10 is significant
        mask_diag = ~(mask_T01 | mask_T10)
        v1 = torch.where(mask_diag, torch.tensor([1.0, 0.0], device=device), v1)
        
        # Second eigenvector is perpendicular to first
        v2[..., 0] = -v1[..., 1]
        v2[..., 1] = v1[..., 0]
        
        # Debugger calls for compute_tensor_properties
        # It's tricky to place these without knowing the structure of the caller,
        # but we can track the inputs and outputs of this static method.
        # The debugger instance would ideally be passed in or accessed globally.
        # For now, we'll assume a global debugger or skip detailed tracking inside staticmethods
        # unless it's explicitly passed.
        # If this method is only called by get_tensor_field, debugger is in scope.

        return lambda1, lambda2, v1, v2
    
    @staticmethod
    def tensor_field(p, visualization_type, scale, warp_strength, time, device, seed, use_temporal_coherence=False):
        """
        Generate tensor field based on visualization type
        Matches the tensorField function in WebGL shader
        
        Args:
            p: Coordinate tensor [batch, height, width, 2] in [0, 1] range
            visualization_type: Type of visualization (0-3)
            scale: Scale factor
            warp_strength: Amount of warping
            time: Animation time
            device: Device to use for computation
            seed: Random seed value
            use_temporal_coherence: Whether to use temporal coherence for animations
            
        Returns:
            Tensor field values [batch, height, width, 1]
        """

        batch, height, width, _ = p.shape
        
        # Calculate tensor field components
        lambda1, lambda2, v1, v2 = TensorFieldGenerator.compute_tensor_properties(
            p, scale, warp_strength, time, device, seed, use_temporal_coherence
        )
        
        
        # Different visualization modes based on type
        result_tensor = None
        if visualization_type == 0:
            # Eigenvalue visualization - shows magnitude of deformation
            max_eig = torch.maximum(torch.abs(lambda1), torch.abs(lambda2))
            result_tensor = torch.clamp(max_eig, -1.0, 1.0)
            
        elif visualization_type == 1:
            # Eigenvector streamlines - shows direction of principal stress
            st = p
            line_width = 0.08
            
            # Calculate distance to streamline along first eigenvector
            # Use deterministic flow phase calculation with time for temporal coherence
            if use_temporal_coherence:
                flow_phase = torch.tensor(time * 0.2, device=device)
            else:
                flow_phase = time * 0.2
                
            t = st[..., 0] * v1[..., 0] + st[..., 1] * v1[..., 1]
            streamline1 = torch.abs(torch.frac(t * 5.0 + flow_phase) - 0.5) * 2.0
            
            # Calculate distance to streamline along second eigenvector
            t = st[..., 0] * v2[..., 0] + st[..., 1] * v2[..., 1]
            streamline2 = torch.abs(torch.frac(t * 5.0 - flow_phase) - 0.5) * 2.0
            
            # Combine streamlines
            pattern = torch.minimum(streamline1, streamline2)
            smoothed = 1.0 - torch.clamp(pattern / line_width, 0.0, 1.0) * 2.0
            
            result_tensor = smoothed.unsqueeze(-1)
            
        elif visualization_type == 2:
            # Hyperstreamlines - thickness varies with eigenvalue magnitude
            # Ensure lambda1 and lambda2 have the right shape
            lambda1_reshaped = lambda1.clone()
            lambda2_reshaped = lambda2.clone()
            
            # Make sure lambda1/lambda2 have the right shape for broadcasting
            if len(lambda1.shape) == 3:
                lambda1_reshaped = lambda1_reshaped.unsqueeze(-1)
            if len(lambda2.shape) == 3:
                lambda2_reshaped = lambda2_reshaped.unsqueeze(-1)
            
            # Create direction vectors with correct sign
            lambda_sign = torch.sign(lambda1_reshaped)
            dir1 = v1 * lambda_sign
            
            lambda2_sign = torch.sign(lambda2_reshaped)
            dir2 = v2 * lambda2_sign
            
            # Weights based on eigenvalue magnitudes - ensure they have compatible dimensions
            abs_lambda1 = torch.abs(lambda1_reshaped)
            abs_lambda2 = torch.abs(lambda2_reshaped)
            
            # Ensure weight1 has the right shape for broadcasting
            denominator = abs_lambda1 + abs_lambda2 + 0.001
            weight1 = abs_lambda1 / denominator
            weight2 = 1.0 - weight1
            
            # Make sure weights have shape [batch, height, width]
            if len(weight1.shape) == 4 and weight1.shape[-1] == 1:
                weight1 = weight1.squeeze(-1)
            if len(weight2.shape) == 4 and weight2.shape[-1] == 1:
                weight2 = weight2.squeeze(-1)
            
            # Calculate t values along each direction with more stable time implementation for coherence
            if use_temporal_coherence:
                t_phase = torch.tensor(time, device=device)
                t1 = torch.cos(5.0 * (p[..., 0] * dir1[..., 0] + p[..., 1] * dir1[..., 1]) + t_phase)
                t2 = torch.cos(5.0 * (p[..., 0] * dir2[..., 0] + p[..., 1] * dir2[..., 1]) - t_phase)
            else:
                t1 = torch.cos(5.0 * (p[..., 0] * dir1[..., 0] + p[..., 1] * dir1[..., 1]) + time)
                t2 = torch.cos(5.0 * (p[..., 0] * dir2[..., 0] + p[..., 1] * dir2[..., 1]) - time)
            
            # Combine with weights - ensure dimensions are compatible for element-wise multiplication
            result = (t1 * weight1 + t2 * weight2) * 0.5
            
            result_tensor = result.unsqueeze(-1)
            
        else:  # visualization_type == 3
            # Tensor ellipses
            
            # Create an elliptical pattern aligned with eigenvectors and scaled by eigenvalues
            centered = p - torch.floor(p * 4.0 + 0.5) / 4.0  # Create grid
            
            # Transform point to eigenvector basis
            x = torch.sum(centered * v1, dim=-1)
            y = torch.sum(centered * v2, dim=-1)
            
            # Ensure lambda1 and lambda2 have shapes compatible with x and y
            lambda1_tensor = lambda1.clone()
            lambda2_tensor = lambda2.clone()
            
            # Make sure lambda1 and lambda2 have the right shape for broadcasting with x and y
            if len(lambda1.shape) == 4 and lambda1.shape[-1] == 1:
                lambda1_tensor = lambda1_tensor.squeeze(-1)
            elif len(lambda1.shape) == 3:
                # This is already the right shape
                pass
            else:
                # Reshape to match batch, height, width
                lambda1_tensor = lambda1_tensor.view(batch, height, width)
            
            if len(lambda2.shape) == 4 and lambda2.shape[-1] == 1:
                lambda2_tensor = lambda2_tensor.squeeze(-1)
            elif len(lambda2.shape) == 3:
                # This is already the right shape
                pass
            else:
                # Reshape to match batch, height, width
                lambda2_tensor = lambda2_tensor.view(batch, height, width)
            
            # Scale by eigenvalues (normalized to prevent distortion)
            # Make sure all tensors have compatible shapes
            abs_lambda1 = torch.abs(lambda1_tensor)
            abs_lambda2 = torch.abs(lambda2_tensor)
            max_eig = torch.maximum(abs_lambda1, abs_lambda2) + 0.1
            
            # Now all tensors should have shape [batch, height, width]
            scaled_x = x * abs_lambda1 / max_eig
            scaled_y = y * abs_lambda2 / max_eig
            
            # Create ellipse
            ellipse = torch.sqrt(scaled_x**2 + scaled_y**2)
            radius = 0.05
            
            # Animate pulsing ellipses with more deterministic calculation for temporal coherence
            if use_temporal_coherence:
                radius *= 1.0 + 0.3 * torch.sin(torch.tensor(time * 2.0, device=device))
            else:
                radius *= 1.0 + 0.3 * torch.sin(torch.tensor(time * 2.0, device=device))
            
            # Return elliptical pattern
            smoothed = 1.0 - torch.clamp((ellipse - radius) / 0.01, 0.0, 1.0) * 2.0
            
            result_tensor = smoothed.unsqueeze(-1)
        
        return result_tensor
            
    @staticmethod
    def simplex_noise_3d(coords, seed=0, time_offset=0.0):
        """
        3D Simplex noise implementation in PyTorch
        This is used for temporal coherence, treating time as a third dimension
        
        Args:
            coords: Coordinate tensor [batch, height, width, 2]
            seed: Random seed value
            time_offset: Time offset to use as the third dimension
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        batch, height, width, _ = coords.shape
        device = coords.device
        
        # Create 3D coordinates by adding time as the third dimension
        coords_3d = torch.cat([
            coords,
            torch.ones(batch, height, width, 1, device=device) * time_offset
        ], dim=-1)
        
        # Set deterministic seed
        torch.manual_seed(seed)
        
        # Constants for skewing and unskewing the grid
        F3 = 1.0/3.0
        G3 = 1.0/6.0
        
        # Skew the input space to determine which simplex cell we're in
        s = (coords_3d[:, :, :, 0] + coords_3d[:, :, :, 1] + coords_3d[:, :, :, 2]) * F3
        s = s.unsqueeze(-1)
        
        i = torch.floor(coords_3d[:, :, :, 0] + s[:, :, :, 0])
        j = torch.floor(coords_3d[:, :, :, 1] + s[:, :, :, 0])
        k = torch.floor(coords_3d[:, :, :, 2] + s[:, :, :, 0])
        
        i = i.unsqueeze(-1)
        j = j.unsqueeze(-1)
        k = k.unsqueeze(-1)
        
        # Unskew the cell origin back to (x,y,z) space
        t = (i + j + k) * G3
        X0 = i - t
        Y0 = j - t
        Z0 = k - t
        
        # The x,y,z distances from the cell origin
        x0 = coords_3d[:, :, :, 0:1] - X0
        y0 = coords_3d[:, :, :, 1:2] - Y0
        z0 = coords_3d[:, :, :, 2:3] - Z0
        
        # For the 3D case, the simplex shape is a tetrahedron
        # Determine which simplex we're in
        g1 = torch.zeros_like(x0)
        g2 = torch.zeros_like(x0)
        g3 = torch.zeros_like(x0)
        
        # Find out which of the six possible tetrahedra we're in
        g1 = (x0 >= y0).float() * (x0 >= z0).float()
        g2 = ((x0 >= y0).float() * (x0 < z0).float() + 
              (x0 < y0).float() * (y0 >= z0).float())
        g3 = (x0 < y0).float() * (y0 < z0).float()
        
        c1 = (g1 > 0.0).float()
        c2 = (g2 > 0.0).float()
        c3 = (g3 > 0.0).float()
        
        # Offsets for corners
        i1 = c1
        j1 = c2
        k1 = c3
        
        i2 = 1.0 - c3
        j2 = 1.0 - c1
        k2 = 1.0 - c2
        
        i3 = torch.ones_like(i1)
        j3 = torch.ones_like(j1)
        k3 = torch.ones_like(k1)
        
        # Work out the hashed gradient indices of the four simplex corners
        def hash(i, j, k, seed_val=seed):
            return ((i * 1664525 + j * 1013904223 + k * 60493) % 16384) / 16384.0 + seed_val/10.0
        
        # Get random gradients at corners
        def get_gradient(hash_val):
            # Convert hash to spherical coordinates 
            theta = hash_val * math.pi * 2.0
            phi = hash_val * 13.0  # Add variation to phi
            
            # Convert to cartesian coordinates on unit sphere
            x = torch.cos(theta) * torch.cos(phi)
            y = torch.sin(theta) * torch.cos(phi)
            z = torch.sin(phi)
            
            return torch.cat([x, y, z], dim=-1)
        
        # Generate gradients at corners
        h000 = hash(i, j, k)
        h100 = hash(i+1, j, k)
        h010 = hash(i, j+1, k)
        h110 = hash(i+1, j+1, k)
        h001 = hash(i, j, k+1)
        h101 = hash(i+1, j, k+1)
        h011 = hash(i, j+1, k+1)
        h111 = hash(i+1, j+1, k+1)
        
        # Get gradient vectors
        g000 = get_gradient(h000)
        g100 = get_gradient(h100)
        g010 = get_gradient(h010)
        g110 = get_gradient(h110)
        g001 = get_gradient(h001)
        g101 = get_gradient(h101)
        g011 = get_gradient(h011)
        g111 = get_gradient(h111)
        
        # Calculate contribution from each corner
        # Corner 0
        t0 = 0.6 - x0*x0 - y0*y0 - z0*z0
        t0 = torch.maximum(torch.zeros_like(t0), t0)
        n0 = t0 * t0 * t0 * t0 * torch.sum(
            torch.cat([x0, y0, z0], dim=-1) * g000, dim=-1, keepdim=True
        )
        
        # Corner 1
        x1 = x0 - i1 + G3
        y1 = y0 - j1 + G3
        z1 = z0 - k1 + G3
        t1 = 0.6 - x1*x1 - y1*y1 - z1*z1
        t1 = torch.maximum(torch.zeros_like(t1), t1)
        n1 = t1 * t1 * t1 * t1 * (
            c1 * (g100[:, :, :, 0:1] * x1 + g100[:, :, :, 1:2] * y1 + g100[:, :, :, 2:3] * z1) +
            c2 * (g010[:, :, :, 0:1] * x1 + g010[:, :, :, 1:2] * y1 + g010[:, :, :, 2:3] * z1) +
            c3 * (g001[:, :, :, 0:1] * x1 + g001[:, :, :, 1:2] * y1 + g001[:, :, :, 2:3] * z1)
        )
        
        # Corner 2
        x2 = x0 - i2 + 2.0*G3
        y2 = y0 - j2 + 2.0*G3
        z2 = z0 - k2 + 2.0*G3
        t2 = 0.6 - x2*x2 - y2*y2 - z2*z2
        t2 = torch.maximum(torch.zeros_like(t2), t2)
        n2 = t2 * t2 * t2 * t2 * (
            (1.0-c3) * (g110[:, :, :, 0:1] * x2 + g110[:, :, :, 1:2] * y2 + g110[:, :, :, 2:3] * z2) +
            (1.0-c1) * (g011[:, :, :, 0:1] * x2 + g011[:, :, :, 1:2] * y2 + g011[:, :, :, 2:3] * z2) +
            (1.0-c2) * (g101[:, :, :, 0:1] * x2 + g101[:, :, :, 1:2] * y2 + g101[:, :, :, 2:3] * z2)
        )
        
        # Corner 3
        x3 = x0 - i3 + 3.0*G3
        y3 = y0 - j3 + 3.0*G3
        z3 = z0 - k3 + 3.0*G3
        t3 = 0.6 - x3*x3 - y3*y3 - z3*z3
        t3 = torch.maximum(torch.zeros_like(t3), t3)
        n3 = t3 * t3 * t3 * t3 * torch.sum(
            torch.cat([x3, y3, z3], dim=-1) * g111, dim=-1, keepdim=True
        )
        
        # Sum up and scale the result to be in [-1,1]
        noise = 32.0 * (n0 + n1 + n2 + n3)
        
        return noise
    
    @staticmethod
    def simplex_noise(coords, seed=0):
        """
        Simplex noise implementation in PyTorch
        This provides noise values similar to snoise in GLSL
        
        Args:
            coords: Coordinate tensor [batch, height, width, 2]
            seed: Random seed value
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        batch, height, width, _ = coords.shape
        device = coords.device
        
        # Set deterministic seed
        torch.manual_seed(seed)
        
        # Integer and fractional parts
        i0 = torch.floor(coords)
        i1 = i0 + 1.0
        f0 = coords - i0
        
        # Dot products with random gradients
        def random_gradient(p, seed_val=seed):
            # Simple hash function
            h = p[:, :, :, 0] * 15.0 + p[:, :, :, 1] * 37.0 + seed_val
            h = torch.sin(h) * 43758.5453
            angle = h * 2.0 * math.pi
            return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        
        # Get gradients at corners
        g00 = random_gradient(i0)
        g10 = random_gradient(torch.cat([i1[:, :, :, 0:1], i0[:, :, :, 1:2]], dim=-1))
        g01 = random_gradient(torch.cat([i0[:, :, :, 0:1], i1[:, :, :, 1:2]], dim=-1))
        g11 = random_gradient(i1)
        
        # Calculate contributions from each corner
        def contribution(grad, pos, point):
            # Calculate falloff
            t = 0.5 - torch.sum((point - pos) ** 2, dim=-1, keepdim=True)
            t = torch.maximum(t, torch.zeros_like(t))
            t = t * t * t * t  # Quintic interpolation
            
            # Dot product of gradient and offset
            offset = point - pos
            return t * torch.sum(grad * offset, dim=-1, keepdim=True)
        
        # Get contributions
        n00 = contribution(g00, i0, coords)
        n10 = contribution(g10, torch.cat([i1[:, :, :, 0:1], i0[:, :, :, 1:2]], dim=-1), coords)
        n01 = contribution(g01, torch.cat([i0[:, :, :, 0:1], i1[:, :, :, 1:2]], dim=-1), coords)
        n11 = contribution(g11, i1, coords)
        
        # Smoothstep for interpolation
        t = f0 * f0 * f0 * (f0 * (f0 * 6.0 - 15.0) + 10.0)
        
        # Bilinear interpolation with smoothstep
        n_x0 = n00 * (1.0 - t[:, :, :, 0:1]) + n10 * t[:, :, :, 0:1]
        n_x1 = n01 * (1.0 - t[:, :, :, 0:1]) + n11 * t[:, :, :, 0:1]
        noise = n_x0 * (1.0 - t[:, :, :, 1:2]) + n_x1 * t[:, :, :, 1:2]
        
        # Scale to match WebGL shader output range
        final_noise = 2.8 * noise
        return final_noise

def generate_tensor_field_tensor(shader_params, height, width, batch_size=1, device="cuda", seed=0, target_channels=None):
    """
    Generate tensor field noise tensor directly using shader_params
    
    Args:
        shader_params: Dictionary containing parameters from shader_params.json
        height: Height of tensor
        width: Width of tensor
        batch_size: Number of images in batch
        device: Device to create tensor on
        seed: Random seed to ensure consistent noise with same seed
        target_channels: Optional target number of channels. If provided, will be set in shader_params.
        
    Returns:
        Noise tensor with shape [batch_size, final_channels, height, width]
    """
    # Get debugger instance
    # debugger = get_debugger()

    # Extract temporal coherence parameters if present
    time = shader_params.get("time", 0.0)
    base_seed = shader_params.get("base_seed", seed)
    use_temporal_coherence = shader_params.get("useTemporalCoherence", shader_params.get("temporal_coherence", False))

    # If target_channels is provided to this function, add/update it in shader_params
    # so get_tensor_field can use it.
    # This function itself doesn't take target_channels as a direct arg in this version,
    # assuming it's already in shader_params if needed.
    # For consistency with domain_warp, let's add it as an optional parameter.
    # The edit below will be to the function signature and its body for this.

    # Track coordinate grid tensor for debugging (like curl_noise.py)
    y, x = torch.meshgrid(
        torch.linspace(0, 1, height, device=device),
        torch.linspace(0, 1, width, device=device),
        indexing='ij'
    )
    p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    # debugger.track_tensor_shape_history(p, "p_coord_grid", "coordinate_grid_creation")

    # Set deterministic seed for this operation
    torch.manual_seed(base_seed if use_temporal_coherence else seed)
    
    # Generate tensor field noise using the TensorFieldGenerator
    noise_bhwc = TensorFieldGenerator.get_tensor_field(
        batch_size=batch_size,
        height=height,
        width=width,
        shader_params=shader_params, # shader_params now contains target_channels if provided
        device=device,
        seed=base_seed if use_temporal_coherence else seed # get_tensor_field handles seed internally based on coherence
    )
    
    # Reset random seed state to not affect other operations
    torch.manual_seed(torch.seed())
    
    # get_tensor_field is now responsible for the correct number of channels.
    # The fixed expansion to 4 channels here is removed.
    
    # Normalize the noise to match typical noise distribution
    # This should be done carefully if the output from get_tensor_field is already in a specific range (e.g. -1 to 1)
    # Assuming get_tensor_field output is [-1, 1], normalization might alter this.
    # For now, keeping the normalization as it was in the original file.
    noise_bhwc = (noise_bhwc - noise_bhwc.mean()) / (noise_bhwc.std() + 1e-8)
    
    # debugger.track_tensor_shape_history(noise_bhwc, "noise_bhwc_normalized_output", "generate_tensor_field_tensor_output")
    return noise_bhwc

def add_temporal_coherent_tensor_field_to_shader_params(shader_params, animation_frame, base_seed, fps=24, duration=1.0, phase_shift=0.0, use_temporal_coherence=True, color_scheme="none", color_intensity=0.8):
    """
    Helper function to add temporal coherence parameters to shader_params for tensor field
    
    Args:
        shader_params: Dictionary containing shader parameters
        animation_frame: Current frame in the animation
        base_seed: Base seed to use for all frames
        fps: Frames per second of the animation (default: 24)
        duration: Total duration of the animation in seconds (default: 1.0)
        phase_shift: Phase shift to apply to time calculation (default: 0.0)
        use_temporal_coherence: Whether to use temporal coherence (default: True)
        color_scheme: Color scheme to use (default: "none")
        color_intensity: Intensity of the color effect (default: 0.8)
        
    Returns:
        Updated shader_params with temporal coherence parameters
    """
    # Calculate total frames
    animation_length = int(fps * duration)
    
    # Calculate normalized time value between 0.0 and 1.0
    time = animation_frame / max(animation_length - 1, 1)
    
    # Apply phase shift (loops from 0 to 1)
    time = (time + phase_shift) % 1.0
    
    # Add parameters to shader_params
    shader_params["time"] = time
    shader_params["base_seed"] = base_seed
    shader_params["temporal_coherence"] = use_temporal_coherence
    
    # Add color scheme parameters if specified
    if color_scheme != "none":
        shader_params["colorScheme"] = color_scheme
        shader_params["shaderColorIntensity"] = color_intensity
        # print(f"Added color scheme: {color_scheme} with intensity {color_intensity}") # Removed this line
    
    # print(f"Tensor field temporal settings: frame={animation_frame}/{animation_length}, time={time:.3f}, base_seed={base_seed}, coherence={use_temporal_coherence}") # Removed this line
    
    # debugger = get_debugger()
    # debugger.track_tensor_shape_history(shader_params, "shader_params_with_temporal", "add_temporal_coherent_params_output")
    return shader_params

def generate_tensor_field_animation_frames(output_dir, width, height, frame_count, 
                                     scale=1.0, warp_strength=0.5, phase_shift=0.0, 
                                     visualization_type=3, 
                                     shape_type="none", shape_mask_strength=1.0,
                                     fps=24, duration=1.0, device="cuda", seed=42,
                                     color_scheme="none", color_intensity=0.8):
    """
    Generate a sequence of tensor field animation frames
    
    Args:
        output_dir: Directory to save frames to
        width: Width of frames
        height: Height of frames
        frame_count: Number of frames to generate
        scale: Scale parameter
        warp_strength: Warp strength parameter
        phase_shift: Phase shift parameter
        visualization_type: Type of tensor field visualization (0-3)
        shape_type: Shape mask type
        shape_mask_strength: Shape mask strength
        fps: Frames per second
        duration: Animation duration in seconds
        device: Device to use for computation
        seed: Random seed for deterministic results
        color_scheme: Color scheme to use
        color_intensity: Color intensity
        
    Returns:
        List of paths to generated frames
    """
    import os
    import torch
    from PIL import Image
    import numpy as np
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Frame paths
    frame_paths = []
    
    # Set octaves to control visualization type (same as in shader)
    octaves = visualization_type % 4
    
    # Base shader parameters
    shader_params = {
        "shaderScale": scale,
        "shaderWarpStrength": warp_strength,
        "shaderPhaseShift": phase_shift,
        "shaderOctaves": octaves,
        "shaderShapeType": shape_type,
        "shaderShapeStrength": shape_mask_strength
    }
    
    # debugger = get_debugger()
    # debugger.track_tensor_shape_history(shader_params, "base_shader_params_animation_frames", "animation_frame_generation_start")
    
    # print(f"Generating {frame_count} tensor field animation frames with:") # Removed this line
    # print(f"Parameters: scale={scale}, warp={warp_strength}, phase={phase_shift}") # Removed this line
    # print(f"Visualization type: {visualization_type}") # Removed this line
    # print(f"Color settings: scheme={color_scheme}, intensity={color_intensity}") # Removed this line
    # print(f"Output size: {width}x{height}") # Removed this line
    
    # Generate frames
    for frame in range(frame_count):
        # print(f"Generating frame {frame+1}/{frame_count}...") # Removed this line
        
        # Add temporal parameters to shader_params
        frame_params = add_temporal_coherent_tensor_field_to_shader_params(
            shader_params.copy(), frame, seed, fps, duration, 0.0, True,
            color_scheme, color_intensity
        )
        # debugger.track_tensor_shape_history(frame_params, f"frame_params_animation_frame_{frame}", "animation_frame_params_update")
        
        # Generate noise
        tensor_field = TensorFieldGenerator.get_tensor_field(
            batch_size=1,
            height=height,
            width=width,
            shader_params=frame_params,
            device=device,
            seed=seed
        )
        
        # debugger.track_tensor_shape_history(tensor_field, f"tensor_field_generated_animation_frame_{frame}", "animation_frame_tensor_generation")
        
        # Convert to image
        # Scale from [-1, 1] to [0, 1]
        tensor_field = (tensor_field + 1.0) * 0.5
        
        # Convert to numpy array and then to PIL Image
        img_np = tensor_field[0].permute(1, 2, 0).cpu().numpy()
        
        # Convert to uint8 for saving as image
        img_np = (img_np * 255).astype(np.uint8)
        
        # Create PIL image (use RGB channels only)
        img = Image.fromarray(img_np[:, :, :3])
        
        # Save the image
        frame_path = os.path.join(output_dir, f"tensor_field_frame_{frame:04d}.png")
        img.save(frame_path)
        frame_paths.append(frame_path)
        
    # print(f"Generated {len(frame_paths)} frames in {output_dir}") # Removed this line
    return frame_paths

# Integration with ShaderToTensor class
def add_tensor_field_to_tensor(tensor_class):
    """
    Add tensor field implementation to ShaderToTensor class
    Call this function to monkey patch the ShaderToTensor class
    
    Args:
        tensor_class: The ShaderToTensor class to modify
    """
    def tensor_field(cls, p, time, scale, warp_strength, phase_shift, octaves=3.0, seed=0, shape_type="none", shape_mask_strength=1.0):
        """
        Tensor field implementation for ShaderToTensor
        
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            time: Animation time
            scale: Scale factor
            warp_strength: Warp amount
            phase_shift: Contrast adjustment
            octaves: Controls visualization type
            seed: Random seed
            shape_type: Type of shape mask to apply ("none" for no mask)
            shape_mask_strength: Strength of shape mask application
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        # Create parameter dictionary from individual params
        shader_params = {
            "shaderScale": scale,
            "shaderWarpStrength": warp_strength,
            "shaderPhaseShift": phase_shift,
            "shaderOctaves": octaves,
            "time": time,
            "shaderShapeType": shape_type,
            "shaderShapeStrength": shape_mask_strength
        }
        
        # Get shape from p
        batch, height, width, _ = p.shape
        
        # Use the TensorFieldGenerator directly
        tensor_noise = TensorFieldGenerator.get_tensor_field(
            batch_size=batch,
            height=height,
            width=width,
            shader_params=shader_params,
            device=p.device,
            seed=seed  # Pass the seed to the generator
        )
        
        # Convert from [B, 4, H, W] to [B, H, W, 1] for consistency with other functions
        return tensor_noise.permute(0, 2, 3, 1)[:, :, :, 0:1]
    
    # Add the method to the class
    setattr(tensor_class, 'tensor_field', classmethod(tensor_field))
    
    def tensor_field_with_params(cls, batch_size, height, width, shader_params, time, device, seed):
        """
        New method that directly takes the full shader_params dictionary
        
        Args:
            batch_size: Number of images in batch
            height: Height of tensor
            width: Width of tensor
            shader_params: Full parameter dictionary from JSON
            time: Animation time
            device: Device to create tensor on
            seed: Random seed
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        # print(f">>> tensor_field_with_params called with full params dictionary") # Removed
        # print(f">>> Raw params: {shader_params}") # Removed
        
        # Add time to shader params if not already present
        if "time" not in shader_params:
            shader_params["time"] = time
        
        # Ensure shape parameters are properly extracted and included
        shape_type = shader_params.get("shaderShapeType", shader_params.get("shape_type", "none"))
        shape_mask_strength = shader_params.get("shaderShapeStrength", shader_params.get("shapemaskstrength", 1.0))
        
        # Make sure the correct parameter names are set for consistency
        shader_params["shaderShapeType"] = shape_type
        shader_params["shaderShapeStrength"] = shape_mask_strength
        
        # print(f">>> Shape mask params: type={shape_type}, strength={shape_mask_strength}") # Removed
        
        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(0, 1, height, device=device),
            torch.linspace(0, 1, width, device=device),
            indexing='ij'
        )
        
        # Combine into coordinate tensor [batch, height, width, 2]
        p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Set deterministic seed to ensure consistent results
        torch.manual_seed(seed)
        
        # Use the TensorFieldGenerator directly with the full params
        tensor_noise = TensorFieldGenerator.get_tensor_field(
            batch_size=batch_size,
            height=height,
            width=width,
            shader_params=shader_params,  # Pass the full parameter dictionary
            device=device,
            seed=seed
        )
        
        # Convert to BHWC format to match expected output
        result = tensor_noise.permute(0, 2, 3, 1)[:, :, :, 0:1]
        # print(f">>> tensor_field_with_params produced tensor shape: {result.shape}") # Removed
        
        # debugger.track_tensor_shape_history(result, "result_tf_with_params_output", "tensor_field_with_params_output")
        
        return result
    
    # Add the new method to the class
    setattr(tensor_class, 'tensor_field_with_params', classmethod(tensor_field_with_params))
    
    # print("=== Added tensor_field functionality to ShaderToTensor ===") # Removed this line

    # Update shader_noise_to_tensor to handle tensor_field type
    original_shader_noise_to_tensor = tensor_class.shader_noise_to_tensor
    
    @classmethod
    def new_shader_noise_to_tensor(cls, batch_size=1, height=64, width=64, shader_type="tensor_field", 
                               visualization_type=3, scale=1.0, phase_shift=0.0, 
                               warp_strength=0.0, time=0.0, device="cuda", seed=0, octaves=3.0,
                               shader_params=None, shape_type="none", shape_mask_strength=1.0):
        """Updated shader_noise_to_tensor with tensor_field support"""
        # print(f"Generating {shader_type} noise with scale={scale}, octaves={octaves}, warp={warp_strength}, phase={phase_shift}") # Removed
        # print(f"Shape params: type={shape_type}, strength={shape_mask_strength}") # Removed
        
        # Check if we have a shader_params dictionary passed
        has_shader_params = shader_params is not None
        # if has_shader_params: # Removed
            # print(f"Full shader_params provided: {shader_params}") # Removed
        
        # Handle tensor_field shader type explicitly
        if shader_type == "tensor_field":
            # print("EXECUTING TENSOR_FIELD BRANCH") # Removed
            
            # If we have the full params dict and the new method, use it
            if has_shader_params and hasattr(cls, 'tensor_field_with_params'):
                # Ensure shape parameters are in the shader_params
                if "shaderShapeType" not in shader_params and "shape_type" not in shader_params:
                    shader_params["shaderShapeType"] = shape_type
                if "shaderShapeStrength" not in shader_params and "shapemaskstrength" not in shader_params:
                    shader_params["shaderShapeStrength"] = shape_mask_strength
                
                return cls.tensor_field_with_params(
                    batch_size, height, width, shader_params, time, device, seed
                )
            
            # Otherwise use the original implementation
            # Create coordinate grid
            y, x = torch.meshgrid(
                torch.linspace(0, 1, height, device=device),
                torch.linspace(0, 1, width, device=device),
                indexing='ij'
            )
            
            # Combine into coordinate tensor [batch, height, width, 2]
            p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            
            # Set deterministic seed to ensure consistent results with same seed
            torch.manual_seed(seed)
            
            # Generate tensor field noise
            result = cls.tensor_field(
                p, time, scale, warp_strength, phase_shift, octaves, seed, shape_type, shape_mask_strength
            )
            
            # print(f"Tensor field noise raw result shape: {result.shape}") # Removed
            
            # Ensure result has the right shape [batch, height, width, 1]
            if len(result.shape) == 3:  # Missing dimension
                result = result.unsqueeze(-1)
                
            # print(f"Final tensor field noise shape: {result.shape}") # Removed
            
            # debugger.track_tensor_shape_history(result, "result_new_shader_noise_to_tensor_tf_branch", "new_shader_noise_to_tensor_tf_output")
            
            return result
        else:
            # Call original method for other shader types
            return original_shader_noise_to_tensor(
                batch_size, height, width, shader_type, visualization_type, 
                scale, phase_shift, warp_strength, time, device, seed, octaves,
                shader_params, shape_type, shape_mask_strength
            )
    
    # Replace the original method only if it hasn't been replaced by other noise integrations already
    if not hasattr(tensor_class, 'shader_noise_to_tensor'):
        setattr(tensor_class, 'shader_noise_to_tensor', new_shader_noise_to_tensor)
    else:
        # If already replaced, we need to update the existing method to handle tensor_field
        existing_method = tensor_class.shader_noise_to_tensor
        
        @classmethod
        def updated_shader_noise_to_tensor(cls, batch_size=1, height=64, width=64, shader_type="tensor_field", 
                                   visualization_type=3, scale=1.0, phase_shift=0.0, 
                                   warp_strength=0.0, time=0.0, device="cuda", seed=0, octaves=3.0,
                                   shader_params=None, shape_type="none", shape_mask_strength=1.0):
            # Handle tensor_field shader type explicitly
            if shader_type == "tensor_field":
                # print("EXECUTING TENSOR_FIELD BRANCH") # Removed
                
                # If we have the full params dict and the new method, use it
                has_shader_params = shader_params is not None
                if has_shader_params:
                    # Ensure shape parameters are in the shader_params
                    if "shaderShapeType" not in shader_params and "shape_type" not in shader_params:
                        shader_params["shaderShapeType"] = shape_type
                    if "shaderShapeStrength" not in shader_params and "shapemaskstrength" not in shader_params:
                        shader_params["shaderShapeStrength"] = shape_mask_strength
                        
                    return cls.tensor_field_with_params(
                        batch_size, height, width, shader_params, time, device, seed
                    )
                
                # Otherwise use the original implementation
                # Create coordinate grid
                y, x = torch.meshgrid(
                    torch.linspace(0, 1, height, device=device),
                    torch.linspace(0, 1, width, device=device),
                    indexing='ij'
                )
                
                # Combine into coordinate tensor [batch, height, width, 2]
                p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
                
                # Set deterministic seed
                torch.manual_seed(seed)
                
                # Generate tensor field noise
                result = cls.tensor_field(
                    p, time, scale, warp_strength, phase_shift, octaves, seed, shape_type, shape_mask_strength
                )
                
                # Ensure result has the right shape
                if len(result.shape) == 3:
                    result = result.unsqueeze(-1)
                
                # print(f"Final tensor field noise shape: {result.shape}") # Removed
                
                # debugger.track_tensor_shape_history(result, "result_updated_shader_noise_to_tensor_tf_branch", "updated_shader_noise_to_tensor_tf_output")
                return result
            else:
                # Call existing method for other shader types
                return existing_method(
                    batch_size, height, width, shader_type, visualization_type, 
                    scale, phase_shift, warp_strength, time, device, seed, octaves,
                    shader_params, shape_type, shape_mask_strength
                )
        
        setattr(tensor_class, 'shader_noise_to_tensor', updated_shader_noise_to_tensor)

def integrate_tensor_field():
    """
    Integrate tensor field implementation into existing code
    This should be called in __init__.py to ensure tensor field is available
    """
    # Import the necessary classes
    from ..shader_to_tensor import ShaderToTensor
    
    # Add the tensor field implementation to ShaderToTensor
    add_tensor_field_to_tensor(ShaderToTensor)
    
    # print("Tensor field implementation integrated successfully") # Removed this line