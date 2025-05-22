import torch
import torch.nn.functional as F
import math

class CurlNoiseGenerator:
    """
    PyTorch implementation of Curl Noise that closely matches
    the WebGL Curl Noise shader from shader_renderer.js
    
    This class generates fluid-like curl noise by computing
    divergence-free vector fields and advecting properties along them. 
    Used to influence the sampling process in image generation.
    
    Now enhanced with true temporal coherence for smooth animations.
    """
    
    @staticmethod
    def get_curl_noise(batch_size, height, width, shader_params, device="cuda", seed=0, target_channels=4):
        """
        Generate curl noise tensor that matches the WebGL implementation
        
        Args:
            batch_size: Number of images in batch
            height: Height of tensor
            width: Width of tensor
            shader_params: Dictionary containing parameters from shader_params.json. 
                           Should include 'target_channels' if determined upstream.
            device: Device to create tensor on
            seed: Random seed for deterministic results
            target_channels: Fallback number of channels if not in shader_params (default: 4)
            
        Returns:
            Tensor with shape [batch_size, target_channels, height, width]
        """
        # Get debugger instance
        # debugger = get_debugger() 
        
        # --- Helper function for smoothstep ---
        def smoothstep(edge0, edge1, x):
            # Ensure edges are tensors for broadcasting with x
            if not isinstance(edge0, torch.Tensor):
                # Use x's properties for the new tensor
                edge0 = torch.full_like(x, float(edge0), device=x.device, dtype=x.dtype)
            if not isinstance(edge1, torch.Tensor):
                edge1 = torch.full_like(x, float(edge1), device=x.device, dtype=x.dtype)

            # Calculate t, handling potential edge0 >= edge1 cases by clamping
            delta = edge1 - edge0
            # Avoid division by zero or near-zero, maintain sign for correct 1-smoothstep
            safe_delta = torch.where(torch.abs(delta) < 1e-8, torch.sign(delta) * 1e-8 + 1e-8*(1-torch.abs(torch.sign(delta))), delta)
            t = torch.clamp((x - edge0) / safe_delta, 0.0, 1.0)

            return t * t * (3.0 - 2.0 * t)
        # -------------------------------------
        
        # --- Parameter Extraction --- Corrected Keys ---
        scale = shader_params.get("scale", 1.0)
        # Check for both 'warp_strength' and 'warp' keys
        warp_strength = shader_params.get("warp_strength", shader_params.get("warp", 0.5))
        # Check for both 'phase_shift' and 'phase' keys
        phase_shift = shader_params.get("phase_shift", shader_params.get("phase", 0.5))
        octaves = int(shader_params.get("octaves", 3))
        time = shader_params.get("time", 0.0)
        
        # Extract temporal coherence parameters (base_seed is used across frames)
        base_seed = shader_params.get("base_seed", seed)
        use_temporal_coherence = shader_params.get("useTemporalCoherence", False)
        
        # Extract shape mask parameters
        shape_type = shader_params.get("shape_type", "none")
        # Check for both 'shapemaskstrength' and 'shape_strength' keys
        shape_mask_strength = shader_params.get("shapemaskstrength", shader_params.get("shape_strength", 1.0))
        
        # Extract color scheme parameters - Check both JS and Python style keys
        color_scheme = shader_params.get("colorScheme", shader_params.get("color_scheme", "none"))
        color_intensity = shader_params.get("shaderColorIntensity", shader_params.get("color_intensity", 0.8))

        # --- Channel Detection Logic ---        
        # Prioritize target_channels from shader_params if available (set upstream)
        if "target_channels" in shader_params:
            detected_channels = int(shader_params["target_channels"])
            if detected_channels != target_channels: # Log if different from function arg
                print(f"CurlNoise: Channel mismatch - function arg {target_channels}, shader_params {detected_channels}. Using shader_params.") # Added print
            target_channels = detected_channels
        else:
            # Fallback to function argument if not in shader_params
            print(f"CurlNoise: Using target_channels {target_channels} from function argument (not in shader_params).") # Added print
            # Optional: Add back minimal name-based detection here as a last resort?
            # For now, just rely on the default passed to the function.

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

        # Print params being used for debugging
        # print(f"CurlNoise: Using shader_type: {shader_params.get('shader_type', 'curl_noise')}") # Added print
        # print(f"CurlNoise: Scale={scale}, Warp Strength={warp_strength}, Phase Shift={phase_shift}, Octaves={octaves}") # Modified print
        # print(f"CurlNoise: Temporal settings: time={time}, base_seed={base_seed}, coherence={use_temporal_coherence}") # Modified print
        # print(f"CurlNoise: Shape Mask: type={shape_type}, strength={shape_mask_strength}") # Modified print
        # print(f"CurlNoise: Color Scheme: scheme={color_scheme}, intensity={color_intensity}") # Modified print
        # print(f"CurlNoise: Target Channels: {target_channels}") # Modified print
        
        # Create coordinate grid (normalized to [0, 1])
        y_coords = torch.linspace(0, 1, height, device=device).view(1, 1, height, 1).expand(batch_size, 1, height, width)
        x_coords = torch.linspace(0, 1, width, device=device).view(1, 1, 1, width).expand(batch_size, 1, height, width)
        coords = torch.cat((x_coords, y_coords), dim=1)
        
        # Track the initial coordinate grid tensor
        # debugger.track_tensor_shape_history(coords, "coords_grid", "initial_creation")
        
        # Apply curl noise with FBM to get velocity field
        coords_bhwc = coords.permute(0, 2, 3, 1)  # Convert to [batch, height, width, 2]
        
        # Set random seed for this operation to ensure deterministic results
        torch.manual_seed(base_seed)
        
        # Store coords_bhwc as p for later use in additional channel generation
        p = coords_bhwc
        
        # --- Refactored Velocity and Advection ---
        # 1. Get the base velocity field using scaled coordinates
        base_velocity = CurlNoiseGenerator.get_velocity_field(
            p * scale, # Scale coordinates here for velocity generation
            time,
            octaves,
            device,
            base_seed, # Use base_seed for consistent velocity field across frames if temporal
            use_temporal_coherence
        )
        # debugger.track_tensor_shape_history(base_velocity, "base_velocity", "after_get_velocity_field")
        
        # 2. Apply warp intensity using warp_strength
        warped_velocity = CurlNoiseGenerator.apply_warp_intensity(base_velocity, warp_strength)
        # debugger.track_tensor_shape_history(warped_velocity, "warped_velocity", "after_apply_warp_intensity")
        
        # The 'velocity' variable will now refer to the *warped* velocity for subsequent steps
        velocity = warped_velocity # BHWC format [B, H, W, 2]
        
        # --- Apply shape mask (to the warped velocity) ---
        shape_mask = None
        shape_factor = 1.0 # shape_factor is not used correctly later, lerp is better
        if shape_type not in ["none", "0"] and shape_mask_strength > 0:
            # Create the mask based on coordinates
            shape_mask = None
            
            # Helper function for random values based on coords and seed
            def random_val(coords, seed_offset):
                torch.manual_seed(base_seed + seed_offset)
                # Use a simple hash-like function based on coordinates
                hash_val = torch.sin(coords[:, :, :, 0] * (12.9898 + seed_offset) + coords[:, :, :, 1] * (78.233 + seed_offset)) * 43758.5453
                return torch.frac(hash_val)

            if shape_type == "circle": # Existing 'radial' in JS is actually circle
                center_x, center_y = 0.5, 0.5
                y_diff = coords_bhwc[:, :, :, 1] - center_y
                x_diff = coords_bhwc[:, :, :, 0] - center_x
                dist = torch.sqrt(x_diff**2 + y_diff**2)
                shape_mask = 1.0 - torch.clamp(dist * 2, 0, 1) # Match GLSL clamp
            elif shape_type == "square": # Not in JS, but kept for potential use
                x_mask = torch.abs(coords_bhwc[:, :, :, 0] - 0.5) * 2
                y_mask = torch.abs(coords_bhwc[:, :, :, 1] - 0.5) * 2
                dist = torch.max(x_mask, y_mask)
                shape_mask = 1.0 - torch.clamp(dist, 0, 1)
            elif shape_type == "radial": # Actual radial gradient
                # Convert time to tensor
                time_tensor = torch.tensor(time, device=device)
                center_x, center_y = 0.5, 0.5
                center_x += 0.2 * torch.cos(time_tensor)
                center_y += 0.2 * torch.sin(time_tensor)
                y_diff = coords_bhwc[:, :, :, 1] - center_y
                x_diff = coords_bhwc[:, :, :, 0] - center_x
                dist = torch.sqrt(x_diff**2 + y_diff**2) * 2.0
                shape_mask = torch.clamp(1.0 - dist, 0.0, 1.0)
            elif shape_type == "linear":
                # Convert time calculation to tensor for torch.fmod
                time_tensor_02 = torch.tensor(time * 0.2, device=device)
                x_offset = torch.fmod(time_tensor_02, 1.0) * 2.0 # Match JS fract animation
                shifted_x = torch.fmod(coords_bhwc[:, :, :, 0] + x_offset, 1.0)
                shape_mask = shifted_x
            elif shape_type == "spiral":
                centered = coords_bhwc - 0.5
                theta = torch.atan2(centered[:, :, :, 1], centered[:, :, :, 0])
                r = torch.norm(centered, dim=-1) * 2.0
                # Convert time to tensor
                time_tensor = torch.tensor(time, device=device)
                theta += time_tensor
                shape_mask = torch.fmod((theta / (2.0 * math.pi) + r), 1.0)
            elif shape_type == "checkerboard":
                grid_size = 8.0
                # Convert time expressions to tensors
                time_tensor_gs_02 = torch.tensor(time * grid_size * 0.2, device=device)
                time_tensor_gs_01 = torch.tensor(time * grid_size * 0.1, device=device)
                x_offset = time_tensor_gs_02
                y_offset = time_tensor_gs_01
                x_grid = torch.floor((coords_bhwc[:, :, :, 0] + x_offset / grid_size) * grid_size) * 0.5
                y_grid = torch.floor((coords_bhwc[:, :, :, 1] + y_offset / grid_size) * grid_size) * 0.5
                shape_mask = torch.fmod(x_grid + y_grid, 1.0)
            elif shape_type == "spots":
                mask = torch.zeros_like(coords_bhwc[:, :, :, 0])
                num_spots = 10
                # Convert time to tensor once outside the loop
                time_tensor = torch.tensor(time, device=device)
                for i in range(num_spots):
                    rand_x = random_val(coords_bhwc, i * 78)
                    rand_y = random_val(coords_bhwc, i * 12)
                    size = (random_val(coords_bhwc, i * 93) * 0.3 + 0.1) # Base size
                    
                    angle_float = time + float(i)
                    angle_tensor = torch.tensor(angle_float, device=device) # Convert angle to tensor
                    spot_pos_x = 0.5 + torch.cos(angle_tensor) * 0.4 * rand_x
                    spot_pos_y = 0.5 + torch.sin(angle_tensor) * 0.4 * rand_y
                    
                    # Convert time expression to tensor for size calculation
                    size_angle_tensor = torch.tensor(time * 2.0 + float(i), device=device)
                    size *= 1.0 + 0.2 * torch.sin(size_angle_tensor) # Pulsing size
                    
                    dist = torch.sqrt((coords_bhwc[:, :, :, 0] - spot_pos_x)**2 + (coords_bhwc[:, :, :, 1] - spot_pos_y)**2)
                    star_mask = torch.clamp(1.0 - dist / size, 0.0, 1.0)
                    mask = torch.maximum(mask, star_mask)
                shape_mask = mask
            elif shape_type == "hexgrid":
                hex_uv = coords_bhwc * 6.0
                # Convert time expressions to tensors before sin/cos
                time_tensor_05 = torch.tensor(time * 0.5, device=device)
                time_tensor_03 = torch.tensor(time * 0.3, device=device)
                time_tensor = torch.tensor(time, device=device)

                hex_uv[:, :, :, 0] += torch.sin(time_tensor_05) * 0.5
                hex_uv[:, :, :, 1] += torch.cos(time_tensor_03) * 0.5
                
                r = torch.tensor([1.0, 1.73], device=device).reshape(1, 1, 1, 2)
                h = r * 0.5
                a = torch.fmod(hex_uv, r) - h
                b = torch.fmod(hex_uv + h, r) - h
                
                dist = torch.minimum(torch.norm(a, dim=-1), torch.norm(b, dim=-1))
                cell_size = 0.3 + 0.1 * torch.sin(time_tensor)
                shape_mask = smoothstep(cell_size + 0.05, cell_size - 0.05, dist)
            elif shape_type == "stripes":
                freq = 10.0
                # Convert time expressions to tensors
                time_tensor = torch.tensor(time, device=device)
                time_tensor_02 = torch.tensor(time * 0.2, device=device)
                angle = 0.5 * torch.sin(time_tensor_02)
                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)
                rotated_x = coords_bhwc[:, :, :, 0] * cos_a - coords_bhwc[:, :, :, 1] * sin_a
                stripes = torch.sin(rotated_x * freq + time_tensor) # Use time_tensor here
                shape_mask = smoothstep(0.0, 0.1, stripes) * smoothstep(0.0, -0.1, -stripes)
            elif shape_type == "gradient": # Renamed from JS gradient_x/y to just gradient
                # Convert time expression to tensor
                time_tensor_02 = torch.tensor(time * 0.2, device=device)
                angle = time_tensor_02
                dir_x = torch.cos(angle)
                dir_y = torch.sin(angle)
                proj = (coords_bhwc[:, :, :, 0] - 0.5) * dir_x + (coords_bhwc[:, :, :, 1] - 0.5) * dir_y + 0.5
                shape_mask = proj
            elif shape_type == "vignette":
                # Convert time expressions to tensors
                time_tensor_03 = torch.tensor(time * 0.3, device=device)
                time_tensor_04 = torch.tensor(time * 0.4, device=device)
                time_tensor_05 = torch.tensor(time * 0.5, device=device)
                center_x = 0.5 + 0.2 * torch.sin(time_tensor_03)
                center_y = 0.5 + 0.2 * torch.cos(time_tensor_04)
                dist = torch.sqrt((coords_bhwc[:, :, :, 0] - center_x)**2 + (coords_bhwc[:, :, :, 1] - center_y)**2)
                radius = 0.6 + 0.2 * torch.sin(time_tensor_05)
                smoothness = 0.3
                shape_mask = 1.0 - smoothstep(radius - smoothness, radius, dist)
            elif shape_type == "cross":
                # Convert time expressions to tensors
                time_tensor = torch.tensor(time, device=device)
                time_tensor_02 = torch.tensor(time * 0.2, device=device)
                thickness = 0.1 + 0.05 * torch.sin(time_tensor)
                rotation = time_tensor_02
                cos_r = torch.cos(rotation)
                sin_r = torch.sin(rotation)
                centered_x = coords_bhwc[:, :, :, 0] - 0.5
                centered_y = coords_bhwc[:, :, :, 1] - 0.5
                rotated_x = centered_x * cos_r - centered_y * sin_r + 0.5
                rotated_y = centered_x * sin_r + centered_y * cos_r + 0.5
                
                h_bar = smoothstep(0.5 - thickness, 0.5 - thickness + 0.02, rotated_y) * \
                        smoothstep(0.5 + thickness, 0.5 + thickness - 0.02, rotated_y)
                v_bar = smoothstep(0.5 - thickness, 0.5 - thickness + 0.02, rotated_x) * \
                        smoothstep(0.5 + thickness, 0.5 + thickness - 0.02, rotated_x)
                shape_mask = torch.maximum(h_bar, v_bar)
            elif shape_type == "stars":
                mask = torch.zeros_like(coords_bhwc[:, :, :, 0])
                num_stars = 20
                # Convert time to tensor once outside the loop
                time_tensor = torch.tensor(time, device=device)
                time_tensor_01 = torch.tensor(time * 0.1, device=device)
                time_tensor_015 = torch.tensor(time * 0.15, device=device)
                for i in range(num_stars):
                    rand_x = random_val(coords_bhwc, i * 78 + 10)
                    rand_y = random_val(coords_bhwc, i * 12 + 20)
                    # Convert time expressions to tensors
                    time_sin_arg = torch.tensor(float(i), device=device) + time_tensor_01
                    time_cos_arg = torch.tensor(float(i) * 1.5, device=device) + time_tensor_015
                    star_pos_x = torch.fmod(rand_x + 0.05 * torch.sin(time_sin_arg), 1.0)
                    star_pos_y = torch.fmod(rand_y + 0.05 * torch.cos(time_cos_arg), 1.0)

                    # Convert time expression for brightness to tensor
                    brightness_arg = torch.tensor(float(i), device=device) + time_tensor * (0.5 + rand_x * 0.5)
                    brightness = 0.5 + 0.5 * torch.sin(brightness_arg)
                    size = 0.01 + 0.015 * rand_y * brightness
                    
                    dist = torch.sqrt((coords_bhwc[:, :, :, 0] - star_pos_x)**2 + (coords_bhwc[:, :, :, 1] - star_pos_y)**2)
                    star_mask = smoothstep(size, size * 0.5, dist) * brightness
                    mask = torch.maximum(mask, star_mask)
                shape_mask = mask
            elif shape_type == "triangles":
                # Convert time expressions to tensors
                t_tensor = torch.tensor(time * 0.2, device=device)
                t_sin_arg = t_tensor
                t_cos_arg = torch.tensor(time * 0.7, device=device) # Keep original time * 0.7 for cos
                t_border_arg = torch.tensor(time * 1.5, device=device) # For border width calculation
                scale_factor = 5.0
                uv = coords_bhwc * scale_factor
                uv[:, :, :, 0] += torch.sin(t_sin_arg) * 0.5
                uv[:, :, :, 1] += torch.cos(t_cos_arg) * 0.5
                
                gv = torch.fmod(uv, 1.0) - 0.5
                
                # Simplified distance to triangle edges - not exact but visually similar
                d1 = torch.abs(gv[:, :, :, 0] + gv[:, :, :, 1])
                d2 = torch.abs(gv[:, :, :, 0] - gv[:, :, :, 1])
                d3 = torch.abs(gv[:, :, :, 0]) * 0.866 + torch.abs(gv[:, :, :, 1]) * 0.5
                
                d = torch.minimum(torch.minimum(d1, d2), d3) * 0.7 # Approximate distance
                
                border_width = 0.05 + 0.03 * torch.sin(t_border_arg)
                shape_mask = smoothstep(border_width, border_width - 0.02, d)
            elif shape_type == "concentric":
                # Convert time expressions to tensors
                time_tensor_03 = torch.tensor(time * 0.3, device=device)
                time_tensor_04 = torch.tensor(time * 0.4, device=device)
                time_tensor_01 = torch.tensor(time * 0.1, device=device)
                time_tensor_05 = torch.tensor(time * 0.5, device=device)

                center_x = 0.5 + 0.2 * torch.sin(time_tensor_03)
                center_y = 0.5 + 0.2 * torch.cos(time_tensor_04)
                dist = torch.sqrt((coords_bhwc[:, :, :, 0] - center_x)**2 + (coords_bhwc[:, :, :, 1] - center_y)**2)
                freq = 10.0 + 5.0 * torch.sin(time_tensor_01)
                phase = time_tensor_05
                rings = torch.sin(dist * freq + phase)
                shape_mask = smoothstep(0.0, 0.1, rings) * smoothstep(0.0, -0.1, -rings)
            elif shape_type == "rays":
                 # Convert time expressions to tensors
                time_tensor_03 = torch.tensor(time * 0.3, device=device)
                time_tensor_04 = torch.tensor(time * 0.4, device=device)
                time_tensor_05 = torch.tensor(time * 0.5, device=device)

                center_x = 0.5 + 0.1 * torch.sin(time_tensor_03)
                center_y = 0.5 + 0.1 * torch.cos(time_tensor_04)
                to_center_x = coords_bhwc[:, :, :, 0] - center_x
                to_center_y = coords_bhwc[:, :, :, 1] - center_y
                angle = torch.atan2(to_center_y, to_center_x)
                freq = 8.0
                phase = time_tensor_05
                rays_val = torch.sin(angle * freq + phase)
                dist = torch.sqrt(to_center_x**2 + to_center_y**2)
                falloff = 1.0 - smoothstep(0.0, 0.8, dist)
                shape_mask = smoothstep(0.0, 0.3, rays_val) * falloff
            elif shape_type == "zigzag":
                freq = 10.0
                # Convert time expressions to tensors
                time_tensor = torch.tensor(time, device=device)
                time_tensor_02 = torch.tensor(time * 0.2, device=device)
                time_tensor_05 = torch.tensor(time * 0.5, device=device)
                time_tensor_03 = torch.tensor(time * 0.3, device=device)

                angle = 0.5 * torch.sin(time_tensor_02)
                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)
                rotated_x = coords_bhwc[:, :, :, 0] * cos_a - coords_bhwc[:, :, :, 1] * sin_a
                rotated_y = coords_bhwc[:, :, :, 0] * sin_a + coords_bhwc[:, :, :, 1] * cos_a
                
                zigzag1 = torch.abs(2.0 * torch.fmod(rotated_x * freq - time_tensor_05, 1.0) - 1.0)
                zigzag2 = torch.abs(2.0 * torch.fmod(rotated_y * freq + time_tensor_03, 1.0) - 1.0)
                
                zigzag = torch.minimum(zigzag1, zigzag2)
                thickness = 0.3 + 0.1 * torch.sin(time_tensor)
                shape_mask = torch.heaviside(zigzag - thickness, torch.tensor(0.5, device=device)) # step function in torch
            elif shape_type == "gradient_x": # Keep original implementations if needed
                shape_mask = coords_bhwc[:, :, :, 0]
            elif shape_type == "gradient_y":
                shape_mask = coords_bhwc[:, :, :, 1]


            if shape_mask is not None:
                # Ensure mask is [B, H, W, 1] for broadcasting
                if len(shape_mask.shape) == 2: # Shape is [H, W]
                    shape_mask = shape_mask.unsqueeze(0).unsqueeze(-1) # Add batch and channel -> [1, H, W, 1]
                    if batch_size > 1:
                        shape_mask = shape_mask.expand(batch_size, height, width, 1) # Expand batch dim
                elif len(shape_mask.shape) == 3: # Shape is [B, H, W]
                    shape_mask = shape_mask.unsqueeze(-1) # Add channel -> [B, H, W, 1]
                elif len(shape_mask.shape) == 4 and shape_mask.shape[-1] != 1: # Shape is [B, H, W, C] where C!=1
                    # This case shouldn't happen based on current mask logic, but handle defensively
                    print(f"Warning: Unexpected shape mask shape {shape_mask.shape}. Taking first channel.")
                    shape_mask = shape_mask[..., 0:1] # Take first channel -> [B, H, W, 1]
                # If shape is already [B, H, W, 1], do nothing

                # Track shape mask tensor (now guaranteed to be 4D)
                # debugger.track_tensor_shape_history(shape_mask, "shape_mask", "after_creation_and_reshape")

                # Apply shape mask using lerp to the *warped* velocity
                velocity = torch.lerp(velocity, velocity * shape_mask, shape_mask_strength)
                # debugger.track_tensor_shape_history(velocity, "velocity_masked", "after_shape_mask_lerp")


            # print(f"Applied shape mask '{shape_type}' with strength {shape_mask_strength}")
        
        # --- Prepare Warped Velocity for Channel Generation ---
        # Convert *warped* velocity from [B,H,W,2] to [B,2,H,W] for PyTorch processing
        velocity_bchw = velocity.permute(0, 3, 1, 2)
        # debugger.track_tensor_shape_history(velocity_bchw, "velocity_bchw", "after_permutation_of_warped")
        
        # Check if velocity field has the correct shape with 2 channels (X and Y components)
        velocity_shape = velocity_bchw.shape
        # print(f"Warped velocity field shape (BCHW): {velocity_shape}")
        
        # If velocity only has 1 channel instead of 2, fix it
        if velocity_shape[1] == 1:
            # print("WARNING: Warped velocity field has only 1 channel, expanding to 2 channels")
            # Duplicate the channel to create an X and Y component
            velocity_bchw = velocity_bchw.repeat(1, 2, 1, 1)
            # Apply a small rotation to the second channel to make it different
            velocity_bchw[:, 1] = velocity_bchw[:, 0] * 0.8
            # debugger.track_tensor_shape_history(velocity_bchw, "velocity_bchw", "after_shape_correction")
        
        # Extract velocity components (vx, vy) from the *warped* velocity field
        vx = velocity_bchw[:, 0:1]  # X component with shape [batch, 1, height, width]
        vy = velocity_bchw[:, 1:2]  # Y component with shape [batch, 1, height, width]
        # debugger.track_tensor_shape_history(vx, "vx_warped", "warped_velocity_component")
        # debugger.track_tensor_shape_history(vy, "vy_warped", "warped_velocity_component")

        # --- Generate Base Noise Pattern via Advection ---
        # Calculate dt for advection based on phase_shift
        dt = 0.2 + phase_shift * 1.8

        # Perform advection using the *warped* velocity (BHWC format) and scaled coordinates (BHWC format)
        # The advect function expects velocity in BHWC format [B, H, W, 2]
        advected_noise_pattern = CurlNoiseGenerator.advect(
            p * scale, # Use scaled coordinates for advection sampling
            velocity, # Use the warped velocity (BHWC format)
            time,
            dt,
            octaves,
            scale, # Pass scale for noise sampling inside advect
            device,
            base_seed, # Use base_seed for consistent advection pattern across frames if temporal
            use_temporal_coherence
        )
        # advected_noise_pattern is expected to be [B, H, W, 1]
        # debugger.track_tensor_shape_history(advected_noise_pattern, "advected_noise_pattern", "after_advect")

        # Convert advected pattern to B,C,H,W format [B, 1, H, W]
        advected_noise_bchw = advected_noise_pattern.permute(0, 3, 1, 2)
        # debugger.track_tensor_shape_history(advected_noise_bchw, "advected_noise_bchw", "after_advect_permutation")
        
        # Determine how many channels to generate
        channels_to_generate = target_channels  # Use all the channels the model requires
        
        # --- Channel Generation Logic ---
        # Apply color scheme if specified, using warped velocity components (vx, vy)
        if color_scheme not in ["none", "0"] and color_intensity > 0 and channels_to_generate >= 3:
            # Apply color transformations based on the *warped* velocity field's magnitude and direction
            vmag = torch.sqrt(vx**2 + vy**2) # Magnitude of warped velocity
            
            # Use a more robust normalization approach for vmag
            flat_vmag = vmag.view(vmag.shape[0], -1)  # [batch, all_other_dims]
            max_vals = flat_vmag.max(dim=1, keepdim=True)[0]  # Get max per batch
            vmag = vmag / (max_vals.view(-1, 1, 1) + 1e-8)  # Reshape to broadcast correctly
            
            # --- FIX START ---
            # Ensure vmag used for color interpolation has only 1 channel
            # This addresses the potential shape mismatch [B, C, ...] vs [1, 3, ...] in lerp
            # where C might be > 1 (e.g., 16) due to batch size or other factors.
            vmag_interp = vmag
            if vmag.shape[1] != 1:
                # print(f"WARNING: vmag has shape {vmag.shape} before color interpolation. Taking first channel.")
                vmag_interp = vmag[:, 0:1] # Shape [B, 1, H, W]
            # --- FIX END ---

            # Track component tensors
            # debugger.track_tensor_shape_history(vx, "vx_component", "velocity_component")
            # debugger.track_tensor_shape_history(vy, "vy_component", "velocity_component")
            # Log the original vmag shape and the one used for interpolation
            # debugger.track_tensor_shape_history(vmag, "vmag_original", "velocity_component_original")
            # debugger.track_tensor_shape_history(vmag_interp, "vmag_interp", "velocity_component_for_interp")

            # Compute angle from X axis (normalized to [0,1])
            vangle = torch.atan2(vy, vx)
            vangle = (vangle + math.pi) / (2 * math.pi) # Normalize to [0, 1]
            
            # Ensure angle is [B, 1, H, W] for consistency
            if len(vangle.shape) == 3: # [B, H, W]
                vangle = vangle.unsqueeze(1)
            elif len(vangle.shape) == 4 and vangle.shape[1] != 1: # [B, C, H, W] with C!=1
                vangle = vangle[:, 0:1, :, :] # Take first channel
            # Already [B, 1, H, W] requires no change
            
            # Ensure time is a tensor for calculations
            time_tensor = torch.tensor(time, device=device, dtype=vx.dtype)
            
            # Create basic RGB mapping of velocity
            # Ensure channels have 4 dimensions [B, 1, H, W] before use
            r_channel = vx # Already [B, 1, H, W]
            g_channel = vy # Already [B, 1, H, W]
            b_channel = vmag_interp # Use the single-channel version for direct blue mapping
            
            # Helper for HSV to RGB
            def hsv_to_rgb(h, s, v):
                # h, s, v expected shapes [B, 1, H, W]
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

            # Helper for lerp
            def lerp(a, b, t):
                return a + (b - a) * t

            # Helper for color stops interpolation
            def interpolate_colors(stops, t):
                # stops: list of [value, color_tensor]
                # t: normalized value tensor [B, 1, H, W]
                
                # Find the two stops t falls between
                idx = torch.zeros_like(t, dtype=torch.long)
                for i in range(len(stops) - 1):
                    idx = torch.where((t >= stops[i][0]) & (t < stops[i+1][0]), torch.full_like(idx, i), idx)
                idx = torch.where(t >= stops[-1][0], torch.full_like(idx, len(stops) - 2), idx) # Handle >= last stop value

                # Gather the start and end stops based on idx
                # Need to manually index based on the calculated 'idx'
                # This is complex with tensors, so we'll use masks for each segment
                final_color = torch.zeros_like(stops[0][1].expand(-1, -1, t.shape[2], t.shape[3])) # Initialize with target spatial shape
                
                for i in range(len(stops) - 1):
                    mask = (idx == i) # Shape [B, 1, H, W]
                    t0, c0 = stops[i] # c0 shape [1, 3, 1, 1]
                    t1, c1 = stops[i+1] # c1 shape [1, 3, 1, 1]
                    
                    # Normalize t within the segment [t0, t1] -> [0, 1] for all pixels
                    # Avoid division by zero for constant segments
                    denominator = (t1 - t0 + 1e-8)
                    local_t_all = (t - t0) / denominator
                    local_t_clamped = torch.clamp(local_t_all, 0.0, 1.0) # Shape [B, 1, H, W]

                    # Lerp the colors for this segment - c0/c1 will broadcast
                    interp_color = lerp(c0, c1, local_t_clamped) # Shape [B, 3, H, W]
                    
                    # Apply the interpolated color where the mask is true
                    # Expand mask to match color channels
                    final_color = torch.where(mask.expand_as(interp_color), interp_color, final_color)

                return final_color[:, 0:1], final_color[:, 1:2], final_color[:, 2:3] # R, G, B

            # Ensure input is normalized [0, 1] for color stops
            normalized_value = vmag_interp # Use the fixed vmag for all color calculations

            # Different coloring options
            if color_scheme == "rainbow":
                # Rainbow coloring based on angle
                h = vangle # Hue from angle [0, 1]
                s = torch.ones_like(h) * 0.8 # Fixed saturation
                v = normalized_value # Value from magnitude [0, 1] # Use fixed value
                result_r, result_g, result_b = hsv_to_rgb(h, s, v)
                
            elif color_scheme == "heatmap":
                # Heatmap coloring based on magnitude
                result_r = torch.pow(normalized_value, 0.5) # Use fixed value
                result_g = torch.pow(normalized_value, 1.5) # Use fixed value
                result_b = torch.pow(normalized_value, 3.0) # Use fixed value
                
            elif color_scheme == "plasma":
                # Plasma-like coloring based on angle and magnitude
                 # Ensure time_tensor has compatible shape [B, 1, 1, 1] for broadcasting
                 time_broadcast = time_tensor.view(-1, 1, 1, 1) if time_tensor.numel() > 1 else time_tensor
                
                 result_r = 0.5 + 0.5 * torch.sin(vangle * 6.28318 + time_broadcast)
                 result_g = 0.5 + 0.5 * torch.sin(vangle * 6.28318 + normalized_value * 3.14159 + time_broadcast * 2.0) # Use fixed value
                 result_b = 0.5 + 0.5 * torch.cos(vangle * 3.14159 + normalized_value * 6.28318 + time_broadcast * 3.0) # Use fixed value
            elif color_scheme == "vorticity":
                # Vorticity visualization - more accurate curl representation
                # Calculate numerical curl: dVx/dy - dVy/dx
                # We need gradient computation here. Using a simpler approximation for now:
                # curl_approx = (vx shifted up - vx shifted down) - (vy shifted right - vy shifted left)
                # This requires padding or careful indexing.
                # Alternative: Use vx - vy as a proxy for rotational direction
                
                curl_magnitude = torch.abs(vx - vy)
                
                # Normalize for visualization - using robust approach
                flat_curl = curl_magnitude.view(curl_magnitude.shape[0], -1)
                max_vals = flat_curl.max(dim=1, keepdim=True)[0]
                curl_norm = curl_magnitude / (max_vals.view(-1, 1, 1, 1) + 1e-8)

                # Use a blue-white-red color scheme to show positive/negative vorticity proxy
                positive_curl_mask = (vx > vy).float()
                negative_curl_mask = (vx <= vy).float()
                
                result_r = positive_curl_mask * curl_norm # Red for positive curl proxy
                result_g = (positive_curl_mask + negative_curl_mask) * (1.0 - curl_norm) # White in middle
                result_b = negative_curl_mask * curl_norm # Blue for negative curl proxy

            # --- Add other color schemes from JS ---
            elif color_scheme == "blue_red":
                 c0 = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3, 1, 1)
                 c1 = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3, 1, 1)
                 color = lerp(c0, c1, normalized_value) # Use fixed value
                 result_r, result_g, result_b = color[:, 0:1], color[:, 1:2], color[:, 2:3]
            elif color_scheme == "viridis":
                 stops = [
                     (0.0, torch.tensor([0.267, 0.005, 0.329], device=device).view(1, 3, 1, 1)), # #440154
                     (0.33, torch.tensor([0.188, 0.407, 0.553], device=device).view(1, 3, 1, 1)), # #30678D
                     (0.66, torch.tensor([0.208, 0.718, 0.471], device=device).view(1, 3, 1, 1)), # #35B778
                     (1.0, torch.tensor([0.992, 0.906, 0.143], device=device).view(1, 3, 1, 1))  # #FDE724
                 ]
                 result_r, result_g, result_b = interpolate_colors(stops, normalized_value) # Use fixed value
            # Plasma already implemented above
            elif color_scheme == "inferno":
                 stops = [
                     (0.0, torch.tensor([0.001, 0.001, 0.016], device=device).view(1, 3, 1, 1)), # #000004
                     (0.25, torch.tensor([0.259, 0.039, 0.408], device=device).view(1, 3, 1, 1)), # #420A68
                     (0.5, torch.tensor([0.576, 0.149, 0.404], device=device).view(1, 3, 1, 1)), # #932667
                     (0.75, torch.tensor([0.867, 0.318, 0.227], device=device).view(1, 3, 1, 1)), # #DD513A
                     (0.85, torch.tensor([0.988, 0.647, 0.039], device=device).view(1, 3, 1, 1)), # #FCA50A
                     (1.0, torch.tensor([0.988, 1.000, 0.643], device=device).view(1, 3, 1, 1))  # #FCFFA4
                 ]
                 result_r, result_g, result_b = interpolate_colors(stops, normalized_value) # Use fixed value
            elif color_scheme == "magma":
                 stops = [
                     (0.0, torch.tensor([0.001, 0.001, 0.016], device=device).view(1, 3, 1, 1)), # #000004
                     (0.25, torch.tensor([0.231, 0.059, 0.439], device=device).view(1, 3, 1, 1)), # #3B0F70
                     (0.5, torch.tensor([0.549, 0.161, 0.506], device=device).view(1, 3, 1, 1)), # #8C2981
                     (0.75, torch.tensor([0.871, 0.288, 0.408], device=device).view(1, 3, 1, 1)), # #DE4968
                     (0.85, torch.tensor([0.996, 0.624, 0.427], device=device).view(1, 3, 1, 1)), # #FE9F6D
                     (1.0, torch.tensor([0.988, 0.992, 0.749], device=device).view(1, 3, 1, 1))  # #FCFDBF
                 ]
                 result_r, result_g, result_b = interpolate_colors(stops, normalized_value) # Use fixed value
            elif color_scheme == "turbo":
                stops = [
                    (0.0, torch.tensor([0.188, 0.071, 0.235], device=device).view(1, 3, 1, 1)), # #30123b
                    (0.25, torch.tensor([0.275, 0.408, 0.859], device=device).view(1, 3, 1, 1)), # #4669db
                    (0.5, torch.tensor([0.149, 0.749, 0.549], device=device).view(1, 3, 1, 1)), # #26bf8c
                    (0.65, torch.tensor([0.831, 1.000, 0.314], device=device).view(1, 3, 1, 1)), # #d4ff50
                    (0.85, torch.tensor([0.980, 0.718, 0.298], device=device).view(1, 3, 1, 1)), # #fab74c
                    (1.0, torch.tensor([0.729, 0.004, 0.000], device=device).view(1, 3, 1, 1))  # #ba0100
                ]
                result_r, result_g, result_b = interpolate_colors(stops, normalized_value) # Use fixed value
            elif color_scheme == "jet":
                stops = [
                    (0.0, torch.tensor([0.000, 0.000, 0.498], device=device).view(1, 3, 1, 1)), # #00007f
                    (0.125, torch.tensor([0.000, 0.000, 1.000], device=device).view(1, 3, 1, 1)), # #0000ff blue
                    (0.375, torch.tensor([0.000, 1.000, 1.000], device=device).view(1, 3, 1, 1)), # #00ffff cyan
                    (0.625, torch.tensor([1.000, 1.000, 0.000], device=device).view(1, 3, 1, 1)), # #ffff00 yellow
                    (0.875, torch.tensor([1.000, 0.000, 0.000], device=device).view(1, 3, 1, 1)), # #ff0000 red
                    (1.0, torch.tensor([0.498, 0.000, 0.000], device=device).view(1, 3, 1, 1))  # #7f0000 dark red
                ]
                result_r, result_g, result_b = interpolate_colors(stops, normalized_value) # Use fixed value
            # Rainbow already implemented above
            elif color_scheme == "cool":
                 c0 = torch.tensor([0.0, 1.0, 1.0], device=device).view(1, 3, 1, 1) # Cyan
                 c1 = torch.tensor([1.0, 0.0, 1.0], device=device).view(1, 3, 1, 1) # Magenta
                 color = lerp(c0, c1, normalized_value) # Use fixed value
                 result_r, result_g, result_b = color[:, 0:1], color[:, 1:2], color[:, 2:3]
            elif color_scheme == "hot":
                 stops = [
                     (0.0, torch.tensor([0.0, 0.0, 0.0], device=device).view(1, 3, 1, 1)), # Black
                     (0.375, torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3, 1, 1)), # Red
                     (0.75, torch.tensor([1.0, 1.0, 0.0], device=device).view(1, 3, 1, 1)), # Yellow
                     (1.0, torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 3, 1, 1))  # White
                 ]
                 result_r, result_g, result_b = interpolate_colors(stops, normalized_value) # Use fixed value
            elif color_scheme == "parula":
                 stops = [
                     (0.0, torch.tensor([0.208, 0.165, 0.529], device=device).view(1, 3, 1, 1)), # #352a87
                     (0.25, torch.tensor([0.059, 0.361, 0.867], device=device).view(1, 3, 1, 1)), # #0f5cdd
                     (0.5, torch.tensor([0.000, 0.710, 0.651], device=device).view(1, 3, 1, 1)), # #00b5a6
                     (0.75, torch.tensor([1.000, 0.765, 0.216], device=device).view(1, 3, 1, 1)), # #ffc337
                     (1.0, torch.tensor([0.988, 0.996, 0.643], device=device).view(1, 3, 1, 1))  # #fcfea4
                 ]
                 result_r, result_g, result_b = interpolate_colors(stops, normalized_value) # Use fixed value
            elif color_scheme == "hsv":
                 # Use angle for hue, magnitude for value
                 h = vangle
                 s = torch.ones_like(h) * 0.9 # High saturation
                 v = normalized_value # Use fixed value
                 result_r, result_g, result_b = hsv_to_rgb(h, s, v)
            elif color_scheme == "autumn":
                 c0 = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3, 1, 1) # Red
                 c1 = torch.tensor([1.0, 1.0, 0.0], device=device).view(1, 3, 1, 1) # Yellow
                 color = lerp(c0, c1, normalized_value) # Use fixed value
                 result_r, result_g, result_b = color[:, 0:1], color[:, 1:2], color[:, 2:3]
            elif color_scheme == "winter":
                 c0 = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3, 1, 1) # Blue
                 c1 = torch.tensor([0.0, 1.0, 0.5], device=device).view(1, 3, 1, 1) # Greenish-Cyan
                 color = lerp(c0, c1, normalized_value) # Use fixed value
                 result_r, result_g, result_b = color[:, 0:1], color[:, 1:2], color[:, 2:3]
            elif color_scheme == "spring":
                 c0 = torch.tensor([1.0, 0.0, 1.0], device=device).view(1, 3, 1, 1) # Magenta
                 c1 = torch.tensor([1.0, 1.0, 0.0], device=device).view(1, 3, 1, 1) # Yellow
                 color = lerp(c0, c1, normalized_value) # Use fixed value
                 result_r, result_g, result_b = color[:, 0:1], color[:, 1:2], color[:, 2:3]
            elif color_scheme == "summer":
                 c0 = torch.tensor([0.0, 0.5, 0.4], device=device).view(1, 3, 1, 1) # Dark Green
                 c1 = torch.tensor([1.0, 1.0, 0.4], device=device).view(1, 3, 1, 1) # Yellow
                 color = lerp(c0, c1, normalized_value) # Use fixed value
                 result_r, result_g, result_b = color[:, 0:1], color[:, 1:2], color[:, 2:3]
            elif color_scheme == "copper":
                 c0 = torch.tensor([0.0, 0.0, 0.0], device=device).view(1, 3, 1, 1) # Black
                 c1 = torch.tensor([1.0, 0.6, 0.4], device=device).view(1, 3, 1, 1) # Copper
                 color = lerp(c0, c1, normalized_value) # Use fixed value
                 result_r, result_g, result_b = color[:, 0:1], color[:, 1:2], color[:, 2:3]
            elif color_scheme == "pink":
                 stops = [
                     (0.0, torch.tensor([0.05, 0.05, 0.05], device=device).view(1, 3, 1, 1)), # Dark gray
                     (0.5, torch.tensor([1.0, 0.0, 1.0], device=device).view(1, 3, 1, 1)), # Magenta
                     (1.0, torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 3, 1, 1))  # White
                 ]
                 result_r, result_g, result_b = interpolate_colors(stops, normalized_value) # Use fixed value
            elif color_scheme == "bone":
                 stops = [
                     (0.0, torch.tensor([0.0, 0.0, 0.0], device=device).view(1, 3, 1, 1)), # Black
                     (0.375, torch.tensor([0.329, 0.329, 0.455], device=device).view(1, 3, 1, 1)), # Dark blueish-gray
                     (0.75, torch.tensor([0.627, 0.757, 0.757], device=device).view(1, 3, 1, 1)), # Light blueish-gray
                     (1.0, torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 3, 1, 1))  # White
                 ]
                 result_r, result_g, result_b = interpolate_colors(stops, normalized_value) # Use fixed value
            elif color_scheme == "ocean":
                 stops = [
                     (0.0, torch.tensor([0.0, 0.0, 0.0], device=device).view(1, 3, 1, 1)), # Black
                     (0.33, torch.tensor([0.0, 0.0, 0.6], device=device).view(1, 3, 1, 1)), # Dark Blue
                     (0.66, torch.tensor([0.0, 0.6, 1.0], device=device).view(1, 3, 1, 1)), # Light Blue
                     (1.0, torch.tensor([0.6, 1.0, 1.0], device=device).view(1, 3, 1, 1))  # Cyan
                 ]
                 result_r, result_g, result_b = interpolate_colors(stops, normalized_value) # Use fixed value
            elif color_scheme == "terrain":
                 stops = [
                     (0.0, torch.tensor([0.2, 0.2, 0.6], device=device).view(1, 3, 1, 1)), # Blue
                     (0.33, torch.tensor([0.0, 0.8, 0.4], device=device).view(1, 3, 1, 1)), # Green
                     (0.66, torch.tensor([1.0, 0.8, 0.0], device=device).view(1, 3, 1, 1)), # Yellow
                     (1.0, torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 3, 1, 1))  # White
                 ]
                 result_r, result_g, result_b = interpolate_colors(stops, normalized_value) # Use fixed value
            elif color_scheme == "neon":
                 stops = [
                     (0.0, torch.tensor([1.0, 0.0, 1.0], device=device).view(1, 3, 1, 1)), # Magenta
                     (0.5, torch.tensor([0.0, 1.0, 1.0], device=device).view(1, 3, 1, 1)), # Cyan
                     (1.0, torch.tensor([1.0, 1.0, 0.0], device=device).view(1, 3, 1, 1))  # Yellow
                 ]
                 result_r, result_g, result_b = interpolate_colors(stops, normalized_value) # Use fixed value
            elif color_scheme == "fire":
                 stops = [
                     (0.0, torch.tensor([0.0, 0.0, 0.0], device=device).view(1, 3, 1, 1)), # Black
                     (0.25, torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3, 1, 1)), # Red
                     (0.6, torch.tensor([1.0, 1.0, 0.0], device=device).view(1, 3, 1, 1)), # Yellow
                     (1.0, torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 3, 1, 1))  # White
                 ]
                 result_r, result_g, result_b = interpolate_colors(stops, normalized_value) # Use fixed value
            # --- End of added color schemes ---
            else:
                # Default color scheme based on velocity components if unknown
                # print(f"Warning: Unknown color scheme '{color_scheme}', using default.")
                result_r = r_channel
                result_g = g_channel
                result_b = b_channel
            
            # Apply color intensity parameter
            if color_intensity < 1.0:
                # Interpolate between grayscale and colored
                grayscale = (result_r + result_g + result_b) / 3.0
                result_r = lerp(grayscale, result_r, color_intensity)
                result_g = lerp(grayscale, result_g, color_intensity)
                result_b = lerp(grayscale, result_b, color_intensity)
            
            # Collect first 3 channels based on color scheme
            channels = []
            if channels_to_generate >= 1:
                channels.append(result_r)
            if channels_to_generate >= 2:
                channels.append(result_g)
            if channels_to_generate >= 3:
                channels.append(result_b)
            
            # Initial result with color channels
            result = torch.cat(channels, dim=1)
            
            # If we need more than 3 channels, generate structured variations
            if target_channels > 3:
                # print(f"Generating all {target_channels} channels with color scheme '{color_scheme}'")
                # Generate additional structured channels
                additional_channels = []
                
                for c in range(3, target_channels):
                    # Create a controlled variation seed
                    variation_seed = base_seed + 600 + (c * 100)
                    torch.manual_seed(variation_seed)
                    
                    # Use slight parameter variations for controlled diversity
                    time_offset = c * 0.1  # Larger offset for more variation with color
                    
                    # Create perturbed coordinates with controlled variations
                    c_p = p.clone()
                    phase_angle = torch.tensor([c * 0.2, c * 0.3], device=device).reshape(1, 1, 1, 2)
                    c_p += torch.sin(phase_angle) * 0.1
                    
                    # Generate a new velocity field with varied parameters
                    chan_velocity = CurlNoiseGenerator.get_velocity_field(
                        c_p, 
                        time + time_offset,
                        octaves * (1.0 + 0.05 * (c-2)),  # Slightly different octaves
                        device, 
                        variation_seed,
                        use_temporal_coherence
                    )
                    
                    # Apply same shape mask if applicable
                    if shape_type not in ["none", "0"] and shape_mask_strength > 0 and shape_mask is not None:
                        # chan_velocity = chan_velocity * (1.0 - shape_factor) + (chan_velocity * shape_mask) * shape_factor # Original line with error
                        # Use lerp for safer application, shape_mask should be [B, H, W, 1]
                        # Ensure chan_velocity is also [B, H, W, 2] before lerp
                        if chan_velocity.shape[-1] == shape_mask.shape[-1]: # Both likely [B,H,W,1]
                             chan_velocity = torch.lerp(chan_velocity, chan_velocity * shape_mask, shape_mask_strength)
                        elif chan_velocity.shape[-1] == 2 and shape_mask.shape[-1] == 1: # Velocity [B,H,W,2], Mask [B,H,W,1]
                             chan_velocity = torch.lerp(chan_velocity, chan_velocity * shape_mask, shape_mask_strength)
                        else:
                             # print(f"Warning: Skipping shape mask application for channel {c} due to incompatible shapes: velocity {chan_velocity.shape}, mask {shape_mask.shape}")
                             pass # Removed print
                    
                    # Create color-aware variation based on channel number
                    if c == 3:
                        # Alpha channel - use average of velocity magnitude
                        velo_mag = torch.sqrt(chan_velocity[:,:,:,0]**2 + chan_velocity[:,:,:,1]**2)
                        extra_channel = velo_mag.unsqueeze(-1).permute(0, 3, 1, 2)
                    else:
                        # Extra channels - mix components with controlled weights
                        component1 = chan_velocity[:,:,:,0].unsqueeze(-1)
                        component2 = chan_velocity[:,:,:,1].unsqueeze(-1)
                        
                        # Create a unique mix based on channel number
                        mix_ratio = (c * 0.2) % 1.0
                        mixed = component1 * mix_ratio + component2 * (1.0 - mix_ratio)
                        extra_channel = mixed.permute(0, 3, 1, 2)
                    
                    # Normalize for consistency
                    extra_channel = (extra_channel - extra_channel.mean()) / (extra_channel.std() + 1e-8)
                    
                    additional_channels.append(extra_channel)
                
                # Combine with original color channels
                if additional_channels:
                    extra_tensor = torch.cat(additional_channels, dim=1)
                    result = torch.cat([result, extra_tensor], dim=1)
                
                # print(f"Generated all {result.shape[1]} channels with color scheme '{color_scheme}'")
        else:
            # If no color scheme, apply structured variations for all channels
            vx = velocity_bchw[:, 0:1]  # X component [batch, 1, height, width]
            vy = velocity_bchw[:, 1:2]  # Y component [batch, 1, height, width]
            
            # Compute magnitude for third channel
            vmag = torch.sqrt(vx**2 + vy**2)
            
            # Safe normalization that handles empty dimensions
            try:
                # Try the direct cascaded max approach first
                max_val = vmag.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
                vmag = vmag / (max_val + 1e-8)
            except IndexError:
                # Fall back to a safer approach if dimensions don't match expectations
                # print(f"Warning: Using fallback normalization for vmag with shape {vmag.shape}")
                if vmag.numel() > 0:  # If tensor has any elements
                    flat_vmag = vmag.reshape(vmag.shape[0], -1)  # Flatten all dimensions except batch
                    max_val = flat_vmag.max(dim=1, keepdim=True)[0].reshape(-1, 1, 1, 1)  # Proper reshaping
                    vmag = vmag / (max_val + 1e-8)
                else:
                    # Handle completely empty tensor
                    vmag = torch.ones_like(vmag)  # Use 1.0 as safe default
                
            # Get the fast mode flag
            fast_high_channel_noise = shader_params.get("fast_high_channel_noise", False)
            HIGH_CHANNEL_THRESHOLD = 16 # Define threshold for fast mode

            if fast_high_channel_noise and target_channels > HIGH_CHANNEL_THRESHOLD:
                # print(f"🚀 Using FAST mode for {target_channels} channels.")
                # Generate only the first 4 base channels
                base_channels_list = []
                if target_channels >= 1:
                    base_channels_list.append(vx)
                    # debugger.track_tensor_shape_history(base_channels_list[-1], "vx_channel", "channel_creation_fast")
                if target_channels >= 2:
                    base_channels_list.append(vy)
                    # debugger.track_tensor_shape_history(base_channels_list[-1], "vy_channel", "channel_creation_fast")
                if target_channels >= 3:
                    base_channels_list.append(vmag)
                    # debugger.track_tensor_shape_history(base_channels_list[-1], "vmag_channel", "channel_creation_fast")
                if target_channels >= 4:
                    # Generate just one extra channel for variation (like the original c=3 logic)
                    # print("Generating structured variation for channel 3 (fast mode base)")
                    variation_seed = base_seed + 500 + (3 * 100)
                    torch.manual_seed(variation_seed)
                    time_offset = 3 * 0.05
                    c_p = p.clone() + torch.sin(torch.tensor([3 * 0.1, 3 * 0.2], device=device).reshape(1, 1, 1, 2))
                    c_velocity = CurlNoiseGenerator.get_velocity_field(c_p, time + time_offset, octaves + (3 * 0.1), device, variation_seed, use_temporal_coherence)

                    # Basic safety check for c_velocity shape
                    if c_velocity.shape[-1] < 1:
                        # print(f"⚠️ Fast mode velocity (c=3) has incorrect shape {c_velocity.shape}, correcting")
                        c_velocity = torch.zeros((*c_velocity.shape[:-1], 1), device=device) # Add at least one component

                    if shape_type not in ["none", "0"] and shape_mask_strength > 0 and shape_mask is not None:
                         c_velocity = c_velocity * (1.0 - shape_factor) + (c_velocity * shape_mask) * shape_factor

                    component = c_velocity[..., 0:1] # Use X component
                    extra_channel_4 = component.permute(0, 3, 1, 2)
                    extra_channel_4 = (extra_channel_4 - extra_channel_4.mean()) / (extra_channel_4.std() + 1e-8)
                    # debugger.track_tensor_shape_history(extra_channel_4, f"extra_channel_3", "channel_structured_variation_fast")
                    base_channels_list.append(extra_channel_4)

                if not base_channels_list: # Handle cases where target_channels < 1
                     # print("⚠️ Target channels < 1, cannot generate base channels for fast mode.")
                     result = torch.zeros((batch_size, target_channels, height, width), device=device)
                else:
                    base_channels = torch.cat(base_channels_list, dim=1)
                    num_base = base_channels.shape[1]
                    if num_base == 0: # Avoid division by zero if no channels were added
                        # print("⚠️ No base channels generated, creating zero tensor.")
                        result = torch.zeros((batch_size, target_channels, height, width), device=device)
                    else:
                        num_repeats = math.ceil(target_channels / num_base)
                        # print(f"   Repeating {num_base} base channels {num_repeats} times.")
                        # Ensure num_repeats is at least 1
                        num_repeats = max(1, num_repeats)
                        result = torch.cat([base_channels] * num_repeats, dim=1)[:, :target_channels] # Tile and slice
                        # debugger.track_tensor_shape_history(result, "result_fast_mode", "fast_channel_tiling")

            else: # Original slow loop for normal mode or low channel counts
                 # Start with a list of structured channels
                 channels = []
                 # print(f"Generating structured variations for all {target_channels} channels directly (Normal Mode)")

                 # First, add the base channels: Advected noise, Warped X, Warped Y, Warped Mag
                 if target_channels >= 1:
                     channels.append(advected_noise_bchw) # Use the computed noise pattern
                     # debugger.track_tensor_shape_history(channels[-1], "advected_noise_channel", "channel_creation")
                 if target_channels >= 2:
                     channels.append(vx)  # Warped X velocity component
                     # debugger.track_tensor_shape_history(channels[-1], "vx_channel", "channel_creation")
                 if target_channels >= 3:
                     channels.append(vy)  # Warped Y velocity component
                     # debugger.track_tensor_shape_history(channels[-1], "vy_channel", "channel_creation")
                 if target_channels >= 4:
                     channels.append(vmag) # Warped velocity magnitude
                     # debugger.track_tensor_shape_history(channels[-1], "vmag_channel", "channel_creation")


                 # Then generate the remaining channels up to target_channels (starting from index 4)
                 for c in range(4, target_channels):
                     # print(f"Generating structured variation for channel {c}")
                     # Create a controlled variation seed derived from base_seed
                     variation_seed = base_seed + 500 + (c * 100)
                     torch.manual_seed(variation_seed)

                     # Generate a unique velocity field with controlled parameter variations
                     # Use a slight offset in time, coordinates, and other parameters
                     time_offset = c * 0.05  # Small time offset per channel

                     # Create slightly perturbed coordinates for this channel
                     c_p = p.clone()
                     # Add small, controlled perturbation (not random)
                     c_p += torch.sin(torch.tensor([c * 0.1, c * 0.2], device=device).reshape(1, 1, 1, 2))

                     # Generate a new velocity field with varied parameters
                     c_velocity = CurlNoiseGenerator.get_velocity_field(
                         c_p,
                         time + time_offset,  # Slight time offset
                         octaves + (c * 0.1),  # Slight octave variation
                         device,
                         variation_seed,  # Controlled seed variation
                         use_temporal_coherence
                     )

                     # Ensure the velocity field has the correct shape (2 components)
                     if c_velocity.shape[-1] < 2:
                         # print(f"⚠️ Channel {c} velocity has incorrect shape {c_velocity.shape}, correcting")
                         # Create a properly shaped velocity field
                         c_proper_velocity = torch.zeros((*c_velocity.shape[:-1], 2), device=device)
                         # Copy the existing components
                         c_proper_velocity[..., :c_velocity.shape[-1]] = c_velocity
                         # Add variation for missing components
                         if c_velocity.shape[-1] == 1:
                             c_proper_velocity[..., 1] = c_velocity[..., 0] * 0.7 + torch.sin(torch.tensor(c * 0.3, device=device)) * 0.3
                         c_velocity = c_proper_velocity

                     # Apply same shape mask if used
                     if shape_type not in ["none", "0"] and shape_mask_strength > 0 and shape_mask is not None:
                         # Apply shape mask using lerp
                         # Ensure mask is broadcastable: [B, H, W, 1]
                         # Ensure c_velocity is [B, H, W, 2]
                         if c_velocity.shape[-1] == 2 and shape_mask.shape[-1] == 1:
                              c_velocity = torch.lerp(c_velocity, c_velocity * shape_mask, shape_mask_strength)
                         else:
                              # print(f"Warning: Skipping shape mask for channel {c} due to shapes: vel={c_velocity.shape}, mask={shape_mask.shape}")
                              pass # Removed print


                     # Extract a component based on channel number for variety
                     # Use index c-1 because we already added first 4 channels
                     if c < 6:  # First few channels use different components or combinations
                         if c == 3:
                             # For variety, use X component
                             component = c_velocity[..., 0:1]
                         elif c == 4:
                             # For variety, use Y component
                             component = c_velocity[..., 1:2]
                         elif c == 5:  # c == 5
                             # For variety, use a mix of X and Y
                             mix_ratio = 0.7
                             component = c_velocity[..., 0:1] * mix_ratio + c_velocity[..., 1:2] * (1.0 - mix_ratio)
                     else:
                         # For later channels, alternate between components with unique transformations
                         component_idx = c % 2  # Alternate between x and y components
                         component = c_velocity[..., component_idx:component_idx+1]

                     # Apply a unique transformation for variety
                     if c % 3 == 0:
                         # Apply sine transformation
                         component = torch.sin(component * 3.14159)
                     elif c % 3 == 1:
                         # Apply absolute value transformation
                         component = torch.abs(component) * 2.0 - 1.0

                     # Convert from [batch, height, width, 1] to [batch, 1, height, width]
                     extra_channel = component.permute(0, 3, 1, 2)

                     # Normalize if needed for consistency
                     extra_channel = (extra_channel - extra_channel.mean()) / (extra_channel.std() + 1e-8)

                     # Track the extra channel
                     # debugger.track_tensor_shape_history(extra_channel, f"extra_channel_{c}", "channel_structured_variation")

                     # Add to our channel list
                     channels.append(extra_channel)

                 # Combine all channels into a single tensor
                 if not channels: # Handle case where target_channels < 1
                     # print("⚠️ Target channels < 1, cannot generate channels for normal mode.")
                     result = torch.zeros((batch_size, target_channels, height, width), device=device)
                 else:
                     result = torch.cat(channels, dim=1)
                     # debugger.track_tensor_shape_history(result, "result_all_channels", "all_channels_combined")
                     # print(f"Generated curl noise with all {result.shape[1]} structured channels directly")

        # If we need additional channels beyond what was already generated
        if target_channels > result.shape[1]:
            # print(f"⚠️ Generating {target_channels - result.shape[1]} additional channels for target_channels={target_channels}")
            
            # Create additional channels
            additional_channels = []
            
            # Log operation start
            # debugger.log_generation_operation("start_additional_channels", 
            #                                 {"result": result}, 
            #                                 result, 
            #                                 {"target_channels": target_channels, 
            #                                 "current_channels": result.shape[1]})
            
            # Only generate needed channels to avoid too many
            for c in range(result.shape[1], target_channels):
                # Use a custom random seed derived from the base seed
                c_seed = seed + 100 * c
                torch.manual_seed(c_seed)
                
                # Offset the coordinates slightly for variation
                c_offset = 0.1 * (c - result.shape[1])  # Small offset for each channel
                c_p = p.clone()
                c_p += torch.randn_like(c_p) * 0.05  # Add small random offsets
                
                # Track perturbed coordinates
                # debugger.track_tensor_shape_history(c_p, f"perturbed_coords_{c}", "channel_generation")
                
                # Generate a unique velocity field for this channel
                chan_velocity = CurlNoiseGenerator.get_velocity_field(
                    c_p, time + c_offset, octaves, device, c_seed, use_temporal_coherence
                )
                
                # Track channel velocity field
                # debugger.track_tensor_shape_history(chan_velocity, f"chan_velocity_{c}", "channel_generation")
                
                # Check shape of velocity field - critical for ensuring correct channels
                # print(f"Chan velocity shape before extraction: {chan_velocity.shape}")
                
                # Handle the case where velocity has a shape that would lead to 64 channels
                # The typical shape should be [batch, height, width, 2] or similar
                if chan_velocity.shape[1] == 64 and chan_velocity.shape[2] == 64:
                    # print(f"⚠️ Detected potential problematic velocity shape: {chan_velocity.shape}")
                    # We might need to fix the velocity field directly
                    if chan_velocity.shape[3] > 1:
                        # Just extract one component directly for simplicity
                        component = 0  # Use first component (x velocity)
                        chan_simple = chan_velocity[:, :, :, component:component+1]
                        # Skip further processing, go straight to properly formatted tensor
                        chan_result = torch.zeros((batch_size, 1, height, width), device=device)
                        # Fill with the component we extracted
                        for h in range(height):
                            for w in range(width):
                                chan_result[0, 0, h, w] = chan_simple[0, h, w, 0]
                        
                        # Skip to tracking step
                        # print(f"✅ Fixed velocity field, created tensor with shape: {chan_result.shape}")
                        # debugger.track_tensor_shape_history(chan_result, f"chan_result_pre_permute_{c}", "channel_extraction_fixed")
                        # debugger.track_tensor_shape_history(chan_result, f"chan_result_post_permute_{c}", "channel_permutation_fixed")
                        additional_channels.append(chan_result)
                        continue  # Skip the rest of this loop iteration
                
                # Apply the same shape mask if applicable
                if shape_type not in ["none", "0"] and shape_mask_strength > 0 and shape_mask is not None:
                    chan_velocity = chan_velocity * (1.0 - shape_factor) + (chan_velocity * shape_mask) * shape_factor
                
                # Extract a single component to use as the channel - safely check dimension size
                if chan_velocity.shape[-1] > 1:
                    # If we have at least 2 dimensions, use modulo to alternate
                    chan_component = c % chan_velocity.shape[-1]
                    chan_result = chan_velocity[:, :, :, chan_component:chan_component+1]
                    # print(f"Extracting component {chan_component} from velocity with shape {chan_velocity.shape}")
                else:
                    # If only one dimension, use it directly
                    chan_result = chan_velocity[:, :, :, 0:1]
                    # Add a small random variation if it's not the first channel
                    torch.manual_seed(seed + c * 100)
                    chan_result = chan_result + torch.randn_like(chan_result) * 0.1
                
                # Track pre-permute tensor
                # debugger.track_tensor_shape_history(chan_result, f"chan_result_pre_permute_{c}", "channel_extraction")
                
                # Convert from [batch, height, width, 1] to [batch, 1, height, width]
                if chan_result.shape[1] == height and chan_result.shape[2] == width:
                    # Shape is [batch, height, width, 1] - standard case
                    chan_result = chan_result.permute(0, 3, 1, 2)
                    # print(f"Standard permutation: {chan_result.shape}")
                else:
                    # Non-standard shape, handle more carefully
                    # print(f"Non-standard shape before permutation: {chan_result.shape}")
                    
                    # Make sure we have 4 dimensions
                    if len(chan_result.shape) < 4:
                        chan_result = chan_result.unsqueeze(-1)
                    
                    # Create a properly shaped tensor directly
                    proper_chan = torch.zeros((batch_size, 1, height, width), device=device)
                    
                    # Fill with resized content from chan_result
                    if chan_result.shape[1] == 1 and chan_result.shape[2] == 1:
                        # Single value case
                        proper_chan.fill_(chan_result[0, 0, 0, 0].item())
                    elif chan_result.shape[1] == 1:
                        # Expand height dimension
                        for h in range(height):
                            proper_chan[0, 0, h, :] = F.interpolate(
                                chan_result[0, 0:1, 0:1, :].permute(0, 3, 1, 2),
                                size=(1, width),
                                mode='nearest'
                            )[0, 0]
                    elif chan_result.shape[2] == 1:
                        # Expand width dimension
                        for w in range(width):
                            proper_chan[0, 0, :, w] = F.interpolate(
                                chan_result[0, :, 0:1, 0:1].permute(0, 3, 1, 2),
                                size=(height, 1),
                                mode='nearest'
                            )[0, 0, :, 0]
                    else:
                        # Use interpolation for general resizing
                        resized = F.interpolate(
                            chan_result.permute(0, 3, 1, 2),
                            size=(height, width),
                            mode='bilinear',
                            align_corners=False
                        )
                        proper_chan = resized
                    
                    chan_result = proper_chan
                    # print(f"Reshaped to: {chan_result.shape}")
                
                # Track post-permute tensor
                # debugger.track_tensor_shape_history(chan_result, f"chan_result_post_permute_{c}", "channel_permutation")
                
                # Add to our additional channels
                additional_channels.append(chan_result)
        
            # Combine original and additional channels if we have any new ones
            if additional_channels:
                # Track result pre-concatenation
                # debugger.track_tensor_shape_history(result, "result_pre_concat", "pre_concatenation")
                
                # Combine tensors
                extra_tensor = torch.cat(additional_channels, dim=1)
                result = torch.cat([result, extra_tensor], dim=1)
                
                # Track result post-concatenation
                # debugger.track_tensor_shape_history(result, "result_post_concat", "post_concatenation")
                
                # Log operation
                # debugger.log_generation_operation("concatenate_channels", 
                #                                 {"base": result, "additional": extra_tensor}, 
                #                                 result)
                
                # print(f"Added {len(additional_channels)} additional channels - final shape: {result.shape}")
        else:
            # print(f"✅ Already generated all {target_channels} channels at once - shape: {result.shape}")
            pass # Removed print
        
        # Scale final result to [-1, 1] range for the model - ONLY IF NO COLOR SCHEME APPLIED
        result = result * 2.0 - 1.0
        
        # Final check to ensure we have the right shape
        if result.shape[1] != target_channels:
            # print(f"⚠️ Channel mismatch: Generated {result.shape[1]} but need {target_channels}")
            
            # Log the issue
            # debugger.add_warning(f"Channel count mismatch: got {result.shape[1]} but expected {target_channels}", 
            #                   category="channel_count")
            
            # Create correctly shaped tensor
            corrected = torch.zeros((batch_size, target_channels, height, width), device=device)
            
            # Copy values from existing channels
            min_channels = min(result.shape[1], target_channels)
            corrected[:, :min_channels] = result[:, :min_channels]
            
            # Generate random values for missing channels
            if result.shape[1] < target_channels:
                std = result.std()
                mean = result.mean()
                for c in range(min_channels, target_channels):
                    # Add variation to each extra channel
                    phase = (c / target_channels) * 2 * 3.14159
                    corrected[:, c] = torch.sin(coords[:, 0] * 5 + phase) * std + mean

            result = corrected
            
            # Track final corrected tensor
            # debugger.track_tensor_shape_history(result, "result_final_corrected", "channel_correction")
            
            # print(f"✅ Adjusted curl noise to {target_channels} channels for compatibility")
        
        # Generate a debug report for this tensor generation process
        # debugger.visualize_tensor_shapes_report()
        
        return result
    
    @staticmethod
    def fbm_curl_noise(p, scale, warp_strength, phase_shift, octaves, time, device, seed, use_temporal_coherence=False):
        """
        Curl noise fbm implementation matching the WebGL version
        
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            scale: Scale factor for the noise 
            warp_strength: Amount of coordinate warping
            phase_shift: Controls the advection time step
            octaves: Number of detail levels and also determines visualization pattern
            time: Animation time
            device: Device to use for computation
            seed: Random seed value
            use_temporal_coherence: Whether to use 3D noise with time as third dimension
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        # Scale coordinates
        p_scaled = p * scale
        
        # Compute velocity field with temporal coherence if enabled
        velocity = CurlNoiseGenerator.get_velocity_field(
            p_scaled, time, octaves, device, seed, use_temporal_coherence
        )
        
        # Apply warp control to intensify curl
        velocity = CurlNoiseGenerator.apply_warp_intensity(velocity, warp_strength)
        
        # Vary the advection time step based on phase shift (between 0.2 and 2.0)
        dt = 0.2 + phase_shift * 1.8
        
        # Advect along the curl field - with temporal coherence if enabled
        result = CurlNoiseGenerator.advect(
            p_scaled, velocity, time, dt, octaves, scale, device, seed, use_temporal_coherence
        )
        
        # Ensure result is in [-1, 1] range
        result = torch.clamp(result, -1.0, 1.0)
        
        return result
    
    @staticmethod
    def compute_gradient(p, epsilon, seed):
        """
        Compute gradient of scalar field
        
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            epsilon: Small value for finite difference
            seed: Random seed for noise generation
            
        Returns:
            Gradient tensor [batch, height, width, 2]
        """
        batch, height, width, _ = p.shape
        device = p.device
        
        # Sample the potential field at nearby points
        p_plus_x = torch.clone(p)
        p_plus_x[:, :, :, 0] += epsilon
        
        p_minus_x = torch.clone(p)
        p_minus_x[:, :, :, 0] -= epsilon
        
        p_plus_y = torch.clone(p)
        p_plus_y[:, :, :, 1] += epsilon
        
        p_minus_y = torch.clone(p)
        p_minus_y[:, :, :, 1] -= epsilon
        
        # Get simplex noise at each point
        n_plus_x = CurlNoiseGenerator.simplex_noise(p_plus_x, seed)
        n_minus_x = CurlNoiseGenerator.simplex_noise(p_minus_x, seed)
        n_plus_y = CurlNoiseGenerator.simplex_noise(p_plus_y, seed)
        n_minus_y = CurlNoiseGenerator.simplex_noise(p_minus_y, seed)
        
        # Compute the gradient using finite difference
        dx = (n_plus_x - n_minus_x) / (2.0 * epsilon)
        dy = (n_plus_y - n_minus_y) / (2.0 * epsilon)
        
        # Combine into a gradient vector
        gradient = torch.cat([dx, dy], dim=-1)
        
        return gradient
    
    @staticmethod
    def compute_curl(p, epsilon, seed):
        """
        Compute curl of vector field (z component in 2D)
        
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            epsilon: Small value for finite difference
            seed: Random seed
            
        Returns:
            Curl tensor [batch, height, width, 1]
        """
        batch, height, width, _ = p.shape
        device = p.device
        
        # We'll use two offset perlin noise functions as our potential fields
        # For 2D curl, we need partial derivatives of two potential fields
        
        # Sample potential field 1
        p_plus_x_1 = torch.clone(p)
        p_plus_x_1[:, :, :, 0] += epsilon
        p_minus_x_1 = torch.clone(p)
        p_minus_x_1[:, :, :, 0] -= epsilon
        p_plus_y_1 = torch.clone(p)
        p_plus_y_1[:, :, :, 1] += epsilon
        p_minus_y_1 = torch.clone(p)
        p_minus_y_1[:, :, :, 1] -= epsilon
        
        # Sample potential field 2 (with offset for independence)
        offset = torch.tensor([0.0, 100.0], device=device).reshape(1, 1, 1, 2)
        p_plus_x_2 = torch.clone(p) + offset
        p_plus_x_2[:, :, :, 0] += epsilon
        p_minus_x_2 = torch.clone(p) + offset
        p_minus_x_2[:, :, :, 0] -= epsilon
        p_plus_y_2 = torch.clone(p) + offset
        p_plus_y_2[:, :, :, 1] += epsilon
        p_minus_y_2 = torch.clone(p) + offset
        p_minus_y_2[:, :, :, 1] -= epsilon
        
        # Get simplex noise at each point for both potential fields
        n_plus_x_1 = CurlNoiseGenerator.simplex_noise(p_plus_x_1, seed)
        n_minus_x_1 = CurlNoiseGenerator.simplex_noise(p_minus_x_1, seed)
        n_plus_y_1 = CurlNoiseGenerator.simplex_noise(p_plus_y_1, seed)
        n_minus_y_1 = CurlNoiseGenerator.simplex_noise(p_minus_y_1, seed)
        
        n_plus_x_2 = CurlNoiseGenerator.simplex_noise(p_plus_x_2, seed + 1)  # Different seed for independence
        n_minus_x_2 = CurlNoiseGenerator.simplex_noise(p_minus_x_2, seed + 1)
        n_plus_y_2 = CurlNoiseGenerator.simplex_noise(p_plus_y_2, seed + 1)
        n_minus_y_2 = CurlNoiseGenerator.simplex_noise(p_minus_y_2, seed + 1)
        
        # Normalize gradients
        pot1_dx = (n_plus_x_1 - n_minus_x_1) / (2.0 * epsilon)
        pot1_dy = (n_plus_y_1 - n_minus_y_1) / (2.0 * epsilon)
        pot2_dx = (n_plus_x_2 - n_minus_x_2) / (2.0 * epsilon)
        pot2_dy = (n_plus_y_2 - n_minus_y_2) / (2.0 * epsilon)
        
        # Compute curl (cross product in 2D: ∂pot2/∂x - ∂pot1/∂y)
        curl = pot2_dx - pot1_dy
        
        return curl
    
    @staticmethod
    def get_velocity_field(p, time, octaves, device, seed, use_temporal_coherence=False):
        """
        Get a fluid velocity field based on curl
        
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            time: Animation time
            octaves: Number of octaves to use
            device: Device to use
            seed: Random seed
            use_temporal_coherence: Whether to use 3D noise with time as third dimension
            
        Returns:
            Velocity field [batch, height, width, 2]
        """
        batch, height, width, _ = p.shape
        
        # Initialize velocity field
        velocity = torch.zeros_like(p)
        epsilon = 0.01
        
        # Base frequency and amplitude
        frequency = 1.0
        amplitude = 1.0
        
        # Loop through octaves
        for i in range(min(int(octaves), 3)):
            # Time-varied position
            time_offset = torch.tensor([time * 0.1 * frequency, 0.0], device=device).reshape(1, 1, 1, 2)
            pos = p * frequency + time_offset
            
            # Compute curl using appropriate method based on temporal coherence setting
            if use_temporal_coherence:
                # Create 3D position with time as third dimension
                # Use a different time offset per octave for varied animation
                time_scale = 0.3 + i * 0.2  # Different time scale per octave
                pos_3d = torch.cat([
                    pos,
                    torch.ones_like(pos[:, :, :, :1]) * time * time_scale
                ], dim=-1)
                
                # Compute curl using 3D sampling for temporal coherence
                curl = CurlNoiseGenerator.compute_curl_temporal(pos_3d, epsilon, seed + i)
                
                # Sample curl at offset positions to compute gradient of curl
                pos_plus_y = torch.clone(pos_3d)
                pos_plus_y[:, :, :, 1] += epsilon
                
                pos_plus_x = torch.clone(pos_3d)
                pos_plus_x[:, :, :, 0] += epsilon
                
                curl_plus_y = CurlNoiseGenerator.compute_curl_temporal(pos_plus_y, epsilon, seed + i)
                curl_plus_x = CurlNoiseGenerator.compute_curl_temporal(pos_plus_x, epsilon, seed + i)
            else:
                # Original approach
                curl = CurlNoiseGenerator.compute_curl(pos, epsilon, seed + i)
                
                # Sample curl at offset positions to compute gradient of curl
                pos_plus_y = torch.clone(pos)
                pos_plus_y[:, :, :, 1] += epsilon
                
                pos_plus_x = torch.clone(pos)
                pos_plus_x[:, :, :, 0] += epsilon
                
                curl_plus_y = CurlNoiseGenerator.compute_curl(pos_plus_y, epsilon, seed + i)
                curl_plus_x = CurlNoiseGenerator.compute_curl(pos_plus_x, epsilon, seed + i)
            
            # The gradient of the curl gives us a divergence-free field
            vel_x = (curl_plus_y - curl) / epsilon
            vel_y = (curl - curl_plus_x) / epsilon
            
            # Combine into velocity vector
            vel = torch.cat([vel_x, vel_y], dim=-1)
            
            # Add to total velocity with current amplitude
            velocity = velocity + vel * amplitude
            
            # Prepare for next octave
            frequency *= 2.0
            amplitude *= 0.5
            epsilon *= 0.5  # Adjust epsilon for higher frequencies
        
        return velocity
    
    @staticmethod
    def advect(p, velocity, time, dt, octaves, scale, device, seed, use_temporal_coherence=False):
        """
        Advect a property along the velocity field
        
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            velocity: Velocity field [batch, height, width, 2]
            time: Animation time
            dt: Advection time step
            octaves: Determines pattern type
            scale: Scale factor
            device: Device to use
            seed: Random seed
            use_temporal_coherence: Whether to use 3D noise with time as third dimension
            
        Returns:
            Advected property [batch, height, width, 1]
        """
        batch, height, width, _ = p.shape
        
        # Trace particle backward in time - apply dt more strongly
        particle_pos = p - velocity * dt
        
        # Sample different noise patterns based on octave setting
        pattern_type = int(octaves) % 4
        
        if pattern_type == 0:
            # Classic advected noise
            # Apply phase shift (via dt) to time offset for more visible effect
            time_offset = torch.tensor([time * 0.2 * dt, dt * 0.5], device=device).reshape(1, 1, 1, 2)
            
            if use_temporal_coherence:
                # For temporal coherence, add time as explicit third dimension
                particle_pos_3d = torch.cat([
                    particle_pos * scale + time_offset,
                    torch.ones_like(particle_pos[:, :, :, :1]) * time
                ], dim=-1)
                
                # Use 3D simplex noise for smooth transitions
                result = CurlNoiseGenerator.simplex_noise_3d(particle_pos_3d, seed)
            else:
                # Original implementation
                result = CurlNoiseGenerator.simplex_noise(particle_pos * scale + time_offset, seed)
            
        elif pattern_type == 1:
            # Dye injection visualization
            # Calculate distance from center of cell
            fract_pos = particle_pos - torch.floor(particle_pos)
            center_offset = fract_pos - 0.5
            
            # For temporal coherence, smoothly vary the distance calculation
            if use_temporal_coherence:
                # Use time to smoothly vary the visualization
                # Convert time to tensor for torch.sin
                time_tensor = torch.tensor(time, device=device)
                time_factor = torch.sin(time_tensor * 2.0 * math.pi) * 0.1 + 1.0
                dist = torch.norm(center_offset, dim=-1, keepdim=True) * (2.0 - dt * 0.5) * time_factor
            else:
                # Original implementation
                dist = torch.norm(center_offset, dim=-1, keepdim=True) * (2.0 - dt * 0.5)
            
            # Create spots using smoothstep
            t = torch.clamp((0.4 - dist) / 0.4, 0.0, 1.0)
            spots = t * t * (3.0 - 2.0 * t)  # Smoothstep implementation
            
            # Remap to [-1, 1]
            result = spots * 2.0 - 1.0
            
        elif pattern_type == 2:
            # Flow lines visualization
            # Normalize velocity
            vel_norm = torch.norm(velocity, dim=-1, keepdim=True)
            vel_norm = torch.where(vel_norm < 1e-6, torch.ones_like(vel_norm), vel_norm)
            normalized_vel = velocity / vel_norm
            
            # Calculate dot product for streak lines
            dot_product = torch.sum(particle_pos * normalized_vel, dim=-1, keepdim=True)
            
            if use_temporal_coherence:
                # Generate flow lines with temporal phase shift
                # This creates a smooth flowing pattern over time
                phase = time * dt * 2.0  # Temporal phase shift
                # Convert phase to tensor for torch.sin
                phase_tensor = torch.tensor(phase, device=device)
                result = torch.sin(dot_product * (10.0 + dt * 5.0) + phase_tensor)
            else:
                # Original implementation
                # Convert time to tensor for torch.sin
                time_dt_tensor = torch.tensor(time * dt, device=device)
                result = torch.sin(dot_product * (10.0 + dt * 5.0) + time_dt_tensor)
            
        else:  # pattern_type == 3
            # Vorticity visualization (shows rotations in the flow)
            # Use dt to affect the scale at which vorticity is computed
            if use_temporal_coherence:
                # Create 3D coordinates with time
                particle_pos_3d = torch.cat([
                    particle_pos * scale * (1.0 + dt * 0.5), 
                    torch.ones_like(particle_pos[:, :, :, :1]) * time
                ], dim=-1)
                
                # Use temporal coherent curl computation
                vorticity = CurlNoiseGenerator.compute_curl_temporal(particle_pos_3d, 0.01 / dt, seed)
            else:
                # Original implementation
                vorticity = CurlNoiseGenerator.compute_curl(particle_pos * scale * (1.0 + dt * 0.5), 0.01 / dt, seed)
            
            # Amplify for better visibility - scale by dt for phase shift effect
            result = vorticity * (2.0 + dt)
        
        return result
    
    @staticmethod
    def apply_warp_intensity(velocity, warp):
        """
        Apply the warp control to intensify curl
        
        Args:
            velocity: Velocity field [batch, height, width, 2]
            warp: Warp intensity
            
        Returns:
            Warped velocity field [batch, height, width, 2]
        """
        # Calculate vector lengths
        length = torch.norm(velocity, dim=-1, keepdim=True)
        
        # Avoid division by zero
        length = torch.maximum(length, torch.ones_like(length) * 0.001)
        
        # Normalize velocity
        normalized_vel = velocity / length
        
        # Apply non-linear scaling based on warp parameter
        log_scale = torch.log(length * 9.0 + 1.0) * warp
        
        # Apply scaling to normalized vectors
        return normalized_vel * log_scale
    
    @staticmethod
    def simplex_noise(coords, seed=0):
        """
        Simplex noise implementation in PyTorch
        Provides noise values similar to snoise in GLSL
        
        Args:
            coords: Coordinate tensor [batch, height, width, 2]
            seed: Random seed value
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        batch, height, width, dim = coords.shape
        device = coords.device
        
        # Set deterministic seed
        torch.manual_seed(seed)
        
        if dim == 2:
            # Constants
            F2 = 0.366025404  # (sqrt(3) - 1) / 2
            G2 = 0.211324865  # (3 - sqrt(3)) / 6
            
            # Skew the input space to determine which simplex cell we're in
            s = (coords[:, :, :, 0] + coords[:, :, :, 1]) * F2
            s = s.unsqueeze(-1)
            
            i = torch.floor(coords[:, :, :, 0] + s[:, :, :, 0])
            j = torch.floor(coords[:, :, :, 1] + s[:, :, :, 0])
            
            i = i.unsqueeze(-1)
            j = j.unsqueeze(-1)
            
            # Unskew the cell origin back to (x,y) space
            t = (i + j) * G2
            X0 = i - t
            Y0 = j - t
            
            # The x,y distances from the cell origin
            x0 = coords[:, :, :, 0:1] - X0
            y0 = coords[:, :, :, 1:2] - Y0
            
            # Determine which simplex we are in
            i1 = torch.zeros_like(i)
            j1 = torch.zeros_like(j)
            
            # Offsets for the second corner of simplex in (i,j) coords
            i1 = torch.where(x0 > y0, torch.ones_like(i), i1)
            j1 = torch.where(x0 <= y0, torch.ones_like(j), j1)
            
            # Offsets for the middle corner in (x,y) unskewed coords
            x1 = x0 - i1 + G2
            y1 = y0 - j1 + G2
            x2 = x0 - 1.0 + 2.0 * G2
            y2 = y0 - 1.0 + 2.0 * G2
            
            # Work out the hashed gradient indices of the three simplex corners
            # Using a simplified hash function that's deterministic yet varied
            def hash_func(i, j, seed_val=seed):
                return ((i * 1664525 + j * 1013904223) % 16384) / 16384.0 + seed_val/10.0
            
            # Get random gradients at corners
            h00 = hash_func(i, j)
            h10 = hash_func(i + 1, j)
            h01 = hash_func(i, j + 1)
            h11 = hash_func(i + 1, j + 1)
            
            # Compute gradient directions
            def get_grad(h):
                # Convert hash to angle
                angle = h * math.pi * 2.0
                # Return unit vector
                return torch.cat([
                    torch.cos(angle),
                    torch.sin(angle)
                ], dim=-1)
            
            # Get gradients at corners
            g00 = get_grad(h00)
            g10 = get_grad(h10)
            g01 = get_grad(h01)
            g11 = get_grad(h11)
            
            # Calculate noise contributions from each corner
            def calculate_corner(x, y, gx, gy):
                # Calculate t = 0.5 - x^2 - y^2 which is > 0 in the simplex
                t = 0.5 - x*x - y*y
                t = torch.maximum(torch.zeros_like(t), t)
                
                # Calculate the noise value: (t^4) * (gx*x + gy*y)
                n = t*t*t*t * (gx*x + gy*y)
                return n
            
            # Compute noise from all corners
            n0 = calculate_corner(x0, y0, g00[:, :, :, 0:1], g00[:, :, :, 1:2])
            n1 = calculate_corner(x1, y1, g10[:, :, :, 0:1] if torch.all(i1 > 0) else g01[:, :, :, 0:1],
                                  g10[:, :, :, 1:2] if torch.all(i1 > 0) else g01[:, :, :, 1:2])
            n2 = calculate_corner(x2, y2, g11[:, :, :, 0:1], g11[:, :, :, 1:2])
            
            # Sum them up and scale
            noise = 70.0 * (n0 + n1 + n2)
            
            # Ensure output is in valid [-1, 1] range
            noise = torch.clamp(noise, -1.0, 1.0)
            
            return noise
        else:
            # Fallback for non-2D coords
            # print(f"Warning: simplex_noise implementation only supports 2D coordinates")
            return torch.zeros(batch, height, width, 1, device=device)
    
    @staticmethod
    def compute_curl_temporal(p, epsilon, seed):
        """
        Compute curl of vector field (z component in 2D) using 3D coordinates for temporal coherence
        
        Args:
            p: Coordinate tensor [batch, height, width, 3] with time as third dimension
            epsilon: Small value for finite difference
            seed: Random seed
            
        Returns:
            Curl tensor [batch, height, width, 1]
        """
        batch, height, width, _ = p.shape
        device = p.device
        
        # We'll use two offset simplex noise functions as our potential fields
        # For 2D curl from 3D coordinates, we extract spatial components
        
        # Sample potential field 1
        p_plus_x_1 = torch.clone(p)
        p_plus_x_1[:, :, :, 0] += epsilon
        p_minus_x_1 = torch.clone(p)
        p_minus_x_1[:, :, :, 0] -= epsilon
        p_plus_y_1 = torch.clone(p)
        p_plus_y_1[:, :, :, 1] += epsilon
        p_minus_y_1 = torch.clone(p)
        p_minus_y_1[:, :, :, 1] -= epsilon
        
        # Sample potential field 2 (with offset for independence)
        offset = torch.tensor([0.0, 100.0, 50.0], device=device).reshape(1, 1, 1, 3)
        p_plus_x_2 = torch.clone(p) + offset
        p_plus_x_2[:, :, :, 0] += epsilon
        p_minus_x_2 = torch.clone(p) + offset
        p_minus_x_2[:, :, :, 0] -= epsilon
        p_plus_y_2 = torch.clone(p) + offset
        p_plus_y_2[:, :, :, 1] += epsilon
        p_minus_y_2 = torch.clone(p) + offset
        p_minus_y_2[:, :, :, 1] -= epsilon
        
        # Get simplex noise at each point for both potential fields using 3D noise
        n_plus_x_1 = CurlNoiseGenerator.simplex_noise_3d(p_plus_x_1, seed)
        n_minus_x_1 = CurlNoiseGenerator.simplex_noise_3d(p_minus_x_1, seed)
        n_plus_y_1 = CurlNoiseGenerator.simplex_noise_3d(p_plus_y_1, seed)
        n_minus_y_1 = CurlNoiseGenerator.simplex_noise_3d(p_minus_y_1, seed)
        
        n_plus_x_2 = CurlNoiseGenerator.simplex_noise_3d(p_plus_x_2, seed + 1)  # Different seed for independence
        n_minus_x_2 = CurlNoiseGenerator.simplex_noise_3d(p_minus_x_2, seed + 1)
        n_plus_y_2 = CurlNoiseGenerator.simplex_noise_3d(p_plus_y_2, seed + 1)
        n_minus_y_2 = CurlNoiseGenerator.simplex_noise_3d(p_minus_y_2, seed + 1)
        
        # Normalize gradients
        pot1_dx = (n_plus_x_1 - n_minus_x_1) / (2.0 * epsilon)
        pot1_dy = (n_plus_y_1 - n_minus_y_1) / (2.0 * epsilon)
        pot2_dx = (n_plus_x_2 - n_minus_x_2) / (2.0 * epsilon)
        pot2_dy = (n_plus_y_2 - n_minus_y_2) / (2.0 * epsilon)
        
        # Compute curl (cross product in 2D: ∂pot2/∂x - ∂pot1/∂y)
        curl = pot2_dx - pot1_dy
        
        return curl
    
    @staticmethod
    def simplex_noise_3d(coords, seed=0):
        """
        3D simplex noise implementation in PyTorch for temporal coherence
        This provides coherent noise values when the third dimension is time
        
        Args:
            coords: Coordinate tensor [batch, height, width, 3] with time as third dimension
            seed: Random seed value
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        batch, height, width, _ = coords.shape
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
        
        return noise.unsqueeze(-1)

    @staticmethod
    def add_temporal_coherent_curl_to_shader_params(animation_frame, animation_length, phase_shift, base_seed, use_temporal_coherence, color_scheme="none", color_intensity=0.8):
        """
        Helper function to prepare temporal coherence parameters for curl noise.
        
        Args:
            animation_frame: Current frame number in animation
            animation_length: Total number of frames in animation
            phase_shift: Phase shift amount
            base_seed: Base seed for consistent noise across frames
            use_temporal_coherence: Whether to use temporal coherence
            color_scheme: Color scheme to use for animation
            color_intensity: Intensity of color effect
            
        Returns:
            Dictionary with calculated time, seed and phase values
        """
        if animation_length <= 1:
            animation_length = 100  # Default for non-animations
        
        if use_temporal_coherence:
            # Calculate normalized time (0 to 1 over animation)
            time = animation_frame / animation_length
            # Scale to appropriate range for noise
            time = time * 10.0  # Scale time to make movement more apparent
            seed = base_seed
        else:
            # Without temporal coherence, use frame-dependent seed
            time = 0.0
            seed = base_seed + int(animation_frame * 1000)
        
        # Apply phase shift
        phase = (animation_frame / animation_length) * phase_shift
        
        # print(f"Temporal params - time: {time}, seed: {seed}, frame: {animation_frame}, phase: {phase}")
        
        return {
            "time": time,
            "seed": seed,
            "phase": phase,
            "colorScheme": color_scheme,
            "shaderColorIntensity": color_intensity
        }

def generate_curl_noise_tensor(shader_params, height, width, batch_size=1, device="cuda", seed=0, target_channels=4):
    """
    Generate curl noise tensor directly using shader_params
    
    Args:
        shader_params: Dictionary containing parameters from shader_params.json
        height: Height of tensor
        width: Width of tensor
        batch_size: Number of images in batch
        device: Device to create tensor on
        seed: Random seed to ensure consistent noise with same seed
        target_channels: Number of channels to generate (default: 4)
        
    Returns:
        Noise tensor with shape [batch_size, target_channels, height, width]
    """
    # Extract temporal coherence parameters if present
    time = shader_params.get("time", 0.0)
    base_seed = shader_params.get("base_seed", seed)
    use_temporal_coherence = shader_params.get("useTemporalCoherence", shader_params.get("temporal_coherence", False))
    
    # Extract or use provided target_channels - handle mismatches between parameters and dictionary
    if "target_channels" in shader_params:
        target_channels = shader_params["target_channels"]
        # print(f"Curl noise using target_channels={target_channels} from shader_params") # Redundant print
    else:
        shader_params["target_channels"] = target_channels  # Add to shader_params
        # print(f"Setting target_channels={target_channels} in shader_params") # Redundant print
    
    # Get color parameters
    color_scheme = shader_params.get("colorScheme", "none")
    color_intensity = shader_params.get("shaderColorIntensity", shader_params.get("intensity", 0.8))
    
    # Print target channels for debugging
    # print(f"Target channels: {target_channels}") # Redundant print
    
    # Set deterministic seed for this operation
    torch.manual_seed(base_seed if use_temporal_coherence else seed)
    
    # --- Removed Redundant Parameter Extraction & Printing ---
    # The parameters are extracted and printed within get_curl_noise now.
    # scale = shader_params.get("scale", 1.0)
    # warp_strength = shader_params.get("warp_strength", 0.5) 
    # phase_shift = shader_params.get("phase_shift", 0.5)
    # octaves = shader_params.get("octaves", 3.0)
    # shape_type = shader_params.get("shape_type", "none")
    # shape_mask_strength = shader_params.get("shapemaskstrength", 1.0)
    
    # Debug output of important parameters - Removed as it's handled in get_curl_noise
    # print(f"Curl noise using: scale={scale}, warp={warp_strength}, phase={phase_shift}, octaves={octaves}")
    # print(f"Temporal settings: time={time}, base_seed={base_seed}, coherence={use_temporal_coherence}")
    # print(f"Shape mask params: type={shape_type}, strength={shape_mask_strength}")
    # print(f"Color settings: scheme={color_scheme}, intensity={color_intensity}")
    # print(f"Target channels: {target_channels}")
    
    # Generate curl noise using the CurlNoiseGenerator
    noise_tensor = CurlNoiseGenerator.get_curl_noise(
        batch_size=batch_size,
        height=height,
        width=width,
        shader_params=shader_params,
        device=device,
        seed=base_seed if use_temporal_coherence else seed,
        target_channels=target_channels
    )
    
    # Reset random seed state to not affect other operations
    torch.manual_seed(torch.seed())
    
    # Normalize the noise to match typical noise distribution
    if noise_tensor.numel() > 0:
        noise_tensor = (noise_tensor - noise_tensor.mean()) / (noise_tensor.std() + 1e-8)
    
    # Ensure we have the right number of channels for the model
    if noise_tensor.shape[1] != target_channels:
        print(f"⚠️ Channel count mismatch: got {noise_tensor.shape[1]}, need {target_channels}")
        # Create a tensor with the correct number of channels
        corrected_tensor = torch.zeros((batch_size, target_channels, height, width), 
                                      device=noise_tensor.device, 
                                      dtype=noise_tensor.dtype)
        
        # Copy available channels
        channels_to_copy = min(noise_tensor.shape[1], target_channels)
        corrected_tensor[:, :channels_to_copy] = noise_tensor[:, :channels_to_copy]
        
        # Fill any remaining channels with appropriate noise
        if noise_tensor.shape[1] < target_channels:
            std = noise_tensor.std()
            mean = noise_tensor.mean()
            torch.manual_seed(seed + 999)
            for c in range(noise_tensor.shape[1], target_channels):
                corrected_tensor[:, c] = torch.randn((batch_size, height, width), 
                                                   device=noise_tensor.device, 
                                                   dtype=noise_tensor.dtype) * std + mean
        
        noise_tensor = corrected_tensor
        print(f"✅ Adjusted to {target_channels} channels")
    
    return noise_tensor

def add_curl_noise_to_tensor(tensor_class):
    """
    Add curl noise implementation to ShaderToTensor class
    Call this function to monkey patch the ShaderToTensor class
    
    Args:
        tensor_class: The ShaderToTensor class to modify
    """
    def curl_noise(cls, p, time, scale, warp_strength, phase_shift, octaves=3.0, seed=0, shape_type="none", shape_mask_strength=1.0, color_scheme="none", color_intensity=0.8):
        """
        Curl noise implementation for ShaderToTensor
        
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            time: Animation time
            scale: Scale factor
            warp_strength: Warp intensity
            phase_shift: Controls advection time step
            octaves: Number of detail levels (1-5) and pattern type
            seed: Random seed
            shape_type: Type of shape mask to apply ("none" for no mask)
            shape_mask_strength: Strength of shape mask application
            color_scheme: Color scheme to apply (e.g., "rainbow", "plasma", etc.)
            color_intensity: Intensity of color effect (0.0 to 1.0)
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        # Debug print to confirm this method is being called with the right params
        print(f">>> curl_noise method called with scale={scale}, warp={warp_strength}, phase={phase_shift}, octaves={octaves}")
        print(f">>> Shape params: type={shape_type}, strength={shape_mask_strength}")
        print(f">>> Color params: scheme={color_scheme}, intensity={color_intensity}")
        
        # Create parameter dictionary from individual params
        shader_params = {
            "shaderScale": scale,
            "shaderWarpStrength": warp_strength,
            "shaderPhaseShift": phase_shift,
            "shaderOctaves": octaves,
            "time": time,
            "shaderShapeType": shape_type,
            "shaderShapeStrength": shape_mask_strength,
            "colorScheme": color_scheme,
            "shaderColorIntensity": color_intensity
        }
        
        # Get shape from p
        batch, height, width, _ = p.shape
        
        # Use the CurlNoiseGenerator directly
        curl_noise = CurlNoiseGenerator.get_curl_noise(
            batch_size=batch,
            height=height,
            width=width,
            shader_params=shader_params,
            device=p.device,
            seed=seed  # Pass the seed to the generator
        )
        
        # Convert from [B, 4, H, W] to [B, H, W, 1] for consistency with other functions
        return curl_noise.permute(0, 2, 3, 1)[:, :, :, 0:1]
    
    # Add the method to the class
    setattr(tensor_class, 'curl_noise', classmethod(curl_noise))
    
    def curl_noise_with_params(cls, batch_size, height, width, shader_params, time, device, seed):
        """
        Method that directly takes the full shader_params dictionary
        
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
        print(f">>> curl_noise_with_params called with full params dictionary")
        print(f">>> Raw params: {shader_params}")
        
        # Add time to params if not present
        if "time" not in shader_params:
            shader_params["time"] = time
            
        # Add base_seed for temporal coherence if not present
        if "base_seed" not in shader_params:
            shader_params["base_seed"] = seed
        
        # Extract target_channels if present
        target_channels = shader_params.get("target_channels", 4)
        
        # Create coordinate grid for curl noise
        y, x = torch.meshgrid(
            torch.linspace(0, 1, height, device=device),
            torch.linspace(0, 1, width, device=device),
            indexing='ij'
        )
        
        # Combine into coordinate tensor [batch, height, width, 2]
        coords = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Set deterministic seed to ensure consistent results
        torch.manual_seed(seed)
        
        # Generate noise
        noise = CurlNoiseGenerator.get_curl_noise(
            batch_size=batch_size,
            height=height,
            width=width,
            shader_params=shader_params,
            device=device,
            seed=seed,
            target_channels=target_channels
        )
        
        # Convert to BHWC format for compatibility with expected outputs
        # Just take the first channel for compatibility with existing methods
        result = noise.permute(0, 2, 3, 1)[:, :, :, 0:1]
        print(f">>> curl_noise_with_params produced tensor shape: {result.shape}")
        
        return result
    
    # Add the new method to the class
    setattr(tensor_class, 'curl_noise_with_params', classmethod(curl_noise_with_params))
    
    print("=== Added curl_noise functionality to ShaderToTensor ===") # Added print statement

    # Update shader_noise_to_tensor to handle curl_noise type
    original_shader_noise_to_tensor = tensor_class.shader_noise_to_tensor
    
    @classmethod
    def new_shader_noise_to_tensor(cls, batch_size=1, height=64, width=64, shader_type="tensor_field", 
                               visualization_type=3, scale=1.0, phase_shift=0.0, 
                               warp_strength=0.0, time=0.0, device="cuda", seed=0, octaves=3.0,
                               shader_params=None, shape_type="none", shape_mask_strength=1.0,
                               color_scheme="none", color_intensity=0.8):
        """Updated shader_noise_to_tensor with curl_noise support"""
        print(f"Generating {shader_type} noise with scale={scale}, octaves={octaves}, warp={warp_strength}, phase={phase_shift}")
        print(f"Shape params: type={shape_type}, strength={shape_mask_strength}")
        print(f"Color params: scheme={color_scheme}, intensity={color_intensity}")
        
        # Check if we have a shader_params dictionary passed
        has_shader_params = shader_params is not None
        if has_shader_params:
            print(f"Full shader_params provided: {shader_params}")
        
        # Handle curl_noise shader type explicitly
        if shader_type == "curl_noise" or shader_type == "curl":
            print("EXECUTING CURL_NOISE BRANCH")
            
            # If we have the full params dict and the new method, use it
            if has_shader_params and hasattr(cls, 'curl_noise_with_params'):
                # Ensure shape parameters are in the shader_params
                if "shaderShapeType" not in shader_params and "shape_type" not in shader_params:
                    shader_params["shaderShapeType"] = shape_type
                if "shaderShapeStrength" not in shader_params and "shapemaskstrength" not in shader_params:
                    shader_params["shaderShapeStrength"] = shape_mask_strength
                # Ensure color parameters are in the shader_params
                if "colorScheme" not in shader_params:
                    shader_params["colorScheme"] = color_scheme
                if "shaderColorIntensity" not in shader_params and "intensity" not in shader_params:
                    shader_params["shaderColorIntensity"] = color_intensity
                
                return cls.curl_noise_with_params(
                    batch_size, height, width, shader_params, time, device, seed
                )
            
            # Otherwise use the original implementation
            # Create coordinate grid
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, height, device=device),
                torch.linspace(-1, 1, width, device=device),
                indexing='ij'
            )
            
            # Combine into coordinate tensor [batch, height, width, 2]
            p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            
            # Set deterministic seed to ensure consistent results with same seed
            torch.manual_seed(seed)
            
            # Generate curl_noise
            result = cls.curl_noise(
                p, time, scale, warp_strength, phase_shift, octaves, seed, 
                shape_type, shape_mask_strength, color_scheme, color_intensity
            )
            
            print(f"Curl noise raw result shape: {result.shape}")
            
            # Ensure result has the right shape [batch, height, width, 1]
            if len(result.shape) == 3:  # Missing dimension
                result = result.unsqueeze(-1)
                
            # Result should now be in [B, H, W, C] format
            print(f"Final curl noise shape (BHWC format): {result.shape}")
            
            return result
        else:
            # Call original method for other shader types
            return original_shader_noise_to_tensor(
                batch_size, height, width, shader_type, visualization_type, 
                scale, phase_shift, warp_strength, time, device, seed, octaves,
                shader_params, shape_type, shape_mask_strength
            )
    
    # Replace the original method only if it hasn't been replaced by other noise integrations already
    if hasattr(tensor_class, '_original_shader_noise_to_tensor'):
        # If already replaced, we need to update the existing method to handle curl_noise
        existing_method = tensor_class.shader_noise_to_tensor
        
        @classmethod
        def updated_shader_noise_to_tensor(cls, batch_size=1, height=64, width=64, shader_type="tensor_field", 
                                   visualization_type=3, scale=1.0, phase_shift=0.0, 
                                   warp_strength=0.0, time=0.0, device="cuda", seed=0, octaves=3.0,
                                   shader_params=None, shape_type="none", shape_mask_strength=1.0,
                                   color_scheme="none", color_intensity=0.8):
            # Handle curl_noise shader type explicitly
            if shader_type == "curl_noise" or shader_type == "curl":
                print("EXECUTING CURL_NOISE BRANCH")
                
                # If we have the full params dict and the new method, use it
                has_shader_params = shader_params is not None
                if has_shader_params:
                    # Ensure shape parameters are in the shader_params
                    if "shaderShapeType" not in shader_params and "shape_type" not in shader_params:
                        shader_params["shaderShapeType"] = shape_type
                    if "shaderShapeStrength" not in shader_params and "shapemaskstrength" not in shader_params:
                        shader_params["shaderShapeStrength"] = shape_mask_strength
                    # Ensure color parameters are in the shader_params
                    if "colorScheme" not in shader_params:
                        shader_params["colorScheme"] = color_scheme
                    if "shaderColorIntensity" not in shader_params and "intensity" not in shader_params:
                        shader_params["shaderColorIntensity"] = color_intensity
                        
                    return cls.curl_noise_with_params(
                        batch_size, height, width, shader_params, time, device, seed
                    )
                
                # Otherwise use the original implementation
                # Create coordinate grid
                y, x = torch.meshgrid(
                    torch.linspace(-1, 1, height, device=device),
                    torch.linspace(-1, 1, width, device=device),
                    indexing='ij'
                )
                
                # Combine into coordinate tensor [batch, height, width, 2]
                p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
                
                # Set deterministic seed
                torch.manual_seed(seed)
                
                # Generate curl_noise
                result = cls.curl_noise(
                    p, time, scale, warp_strength, phase_shift, octaves, seed,
                    shape_type, shape_mask_strength, color_scheme, color_intensity
                )
                
                # Ensure result has the right shape
                if len(result.shape) == 3:
                    result = result.unsqueeze(-1)
                    
                print(f"Final curl noise shape: {result.shape}")
                
                return result
            else:
                # Call existing method for other shader types
                return existing_method(
                    batch_size, height, width, shader_type, visualization_type, 
                    scale, phase_shift, warp_strength, time, device, seed, octaves,
                    shader_params, shape_type, shape_mask_strength
                )
        
        setattr(tensor_class, 'shader_noise_to_tensor', updated_shader_noise_to_tensor)
    else:
        # Save original method for later use
        setattr(tensor_class, '_original_shader_noise_to_tensor', original_shader_noise_to_tensor)
        setattr(tensor_class, 'shader_noise_to_tensor', new_shader_noise_to_tensor)

def integrate_curl_noise():
    """
    Integrate curl noise implementation into existing code
    This should be called in __init__.py to ensure curl noise is available
    """
    # Import the necessary classes
    from ..shader_to_tensor import ShaderToTensor
    
    # Add the curl noise implementation to ShaderToTensor
    add_curl_noise_to_tensor(ShaderToTensor)
    
    print("Curl Noise implementation integrated successfully")

def add_temporal_coherent_curl_to_shader_params(shader_params, animation_frame, base_seed, fps=24, duration=1.0):
    """
    Adds temporal coherence parameters to the shader_params for use with video frames
    
    Args:
        shader_params: Original shader parameters dictionary
        animation_frame: Current frame number (0-indexed)
        base_seed: Base seed to use for the entire animation
        fps: Frames per second (for time calculation)
        duration: Animation duration in seconds
        
    Returns:
        Updated shader_params with frame-specific temporal values
    """
    # Get the phase shift parameter (either with shaderPhaseShift or phase_shift key)
    phase_shift = shader_params.get("shaderPhaseShift", shader_params.get("phase_shift", 0.5))
    
    # Calculate animation length from fps and duration
    animation_length = int(fps * duration)
    
    # Get useTemporalCoherence parameter (default to True for better results)
    use_temporal_coherence = shader_params.get("useTemporalCoherence", True)
    
    # Get color scheme parameters and preserve them for video
    color_scheme = shader_params.get("colorScheme", "none")
    color_intensity = shader_params.get("shaderColorIntensity", shader_params.get("intensity", 0.8))
    
    # Get temporal parameters from helper function
    temporal_params = CurlNoiseGenerator.add_temporal_coherent_curl_to_shader_params(
        animation_frame, animation_length, phase_shift, base_seed, use_temporal_coherence,
        color_scheme, color_intensity
    )
    
    # Create a copy of shader_params to avoid modifying the original
    new_params = shader_params.copy()
    
    # Update with temporal parameters
    new_params["time"] = temporal_params["time"]
    new_params["base_seed"] = base_seed
    new_params["temporal_coherence"] = use_temporal_coherence
    new_params["seed"] = temporal_params["seed"]
    
    # Update the phase shift parameter for current frame
    new_params["shaderPhaseShift"] = temporal_params["phase"]
    
    # Ensure color parameters are preserved
    new_params["colorScheme"] = color_scheme
    new_params["shaderColorIntensity"] = color_intensity
    
    # Set these explicitly for cleaner integration
    print(f"Video frame {animation_frame}/{animation_length}: time={temporal_params['time']:.3f}, seed={temporal_params['seed']}, phase={temporal_params['phase']:.3f}")
    print(f"Color scheme: {color_scheme}, intensity: {color_intensity}")
    
    return new_params 