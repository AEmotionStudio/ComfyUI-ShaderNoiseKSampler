import torch
import math

class ShaderToTensor:
    """
    Class for converting shader operations to PyTorch tensors
    This class serves as the bridge between WebGL shaders and PyTorch tensor operations
    
    Methods are designed to closely match the shader implementations in the web UI
    """
    
    @classmethod
    def shader_noise_to_tensor(cls, batch_size=1, height=64, width=64, shader_type="tensor_field", 
                           visualization_type=3, scale=1.0, phase_shift=0.0, 
                           warp_strength=0.0, time=0.0, device="cuda", seed=0, octaves=3.0,
                           shape_type="none", shape_mask_strength=1.0, shader_params=None):
        """
        Generate noise tensors from shader parameters
        
        Args:
            batch_size: Number of images in batch
            height: Height of tensor
            width: Width of tensor
            shader_type: Type of shader ("tensor_field", "heterogeneous_fbm", "projection_3d")
            visualization_type: Type of visualization for tensor field (0-3)
            scale: Scale factor
            phase_shift: Contrast adjustment
            warp_strength: Warp amount
            time: Animation time
            device: Device to create tensor on
            seed: Random seed
            octaves: Number of octaves for fractal noise
            shape_type: Type of shape mask to apply
            shape_mask_strength: Strength of shape mask application
            shader_params: Full shader parameters dictionary (if provided)
            
        Returns:
            Tensor with shape [batch_size, 4, height, width]
        """
        # If shader_params is provided, use the dedicated method instead
        if shader_params is not None:
            return cls.shader_noise_to_tensor_with_params(
                batch_size, height, width, shader_type, 
                shader_params, time, device, seed
            )
            
        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing='ij'
        )
        
        # Combine into coordinate tensor [batch, height, width, 2]
        p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Generate different shader patterns
        if shader_type == "tensor_field":
            # Simple tensor field pattern
            shader_noise = cls.tensor_field(
                p, visualization_type, scale, warp_strength, phase_shift, time, device
            )
        elif shader_type == "heterogeneous_fbm":
            # Heterogeneous FBM noise
            shader_noise = cls.fbm_noise(
                p, scale, warp_strength, phase_shift, octaves, time, device
            )
        elif shader_type == "projection_3d":
            # 3D projection noise
            shader_noise = cls.projection3d(
                p, scale, warp_strength, phase_shift, octaves, time, device
            )
        else:
            # Raise error instead of falling back to random noise
            raise ValueError(f"Shader type '{shader_type}' is not directly supported in shader_noise_to_tensor. Please use shader_noise_to_tensor_with_params or ensure the shader type is registered.")
        
        # Convert from [B, H, W, 1] to [B, 1, H, W]
        shader_noise = shader_noise.permute(0, 3, 1, 2)
        
        # Normalize to have mean 0 and std 1 (like typical noise)
        shader_noise = (shader_noise - shader_noise.mean()) / (shader_noise.std() + 1e-8)
        
        # Expand channels if needed
        if shader_noise.shape[1] == 1:
            # Expand to 9 channels for latent space operations instead of just 4
            shader_noise = shader_noise.expand(-1, 9, -1, -1) # [B, 1, H, W] -> [B, 9, H, W]
        # If we have 4 channels but need 9 for latent operations
        elif shader_noise.shape[1] == 4:
            # Get existing channels
            r, g, b, a = shader_noise.chunk(4, dim=1)
            
            # Create 5 more channels as variations of the existing ones
            c5 = (r + g) / 2.0  # Average of red and green
            c6 = (g + b) / 2.0  # Average of green and blue
            c7 = (b + a) / 2.0  # Average of blue and alpha
            c8 = (r + b) / 2.0  # Average of red and blue
            c9 = (r + g + b + a) / 4.0  # Average of all channels
            
            # Combine all 9 channels
            shader_noise = torch.cat([r, g, b, a, c5, c6, c7, c8, c9], dim=1)
        
        return shader_noise
    
    @classmethod
    def tensor_field(cls, p, viz_type, scale, warp_strength, phase_shift, time, device):
        """
        Simple implementation of tensor field visualization
        
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            viz_type: Visualization type (0-3)
            scale: Scale factor
            warp_strength: Warp amount
            phase_shift: Contrast adjustment
            time: Animation parameter
            device: Torch device
            
        Returns:
            Field tensor [batch, height, width, 1]
        """
        batch, height, width, _ = p.shape
        
        # Scale coordinates by user parameter
        p = p * scale
        
        # Apply domain warping if warp strength > 0
        if warp_strength > 0.0:
            # Create a smooth flow field for warping
            warp_coords = p * 0.3 + torch.tensor([time * 0.05, 0.0], device=device).reshape(1, 1, 1, 2)
            warp_noise = cls.simplex_noise(warp_coords)
            
            # Apply warp to coordinates
            p = p + warp_noise * warp_strength
        
        # Calculate angle field (similar to atan2 in shaders)
        angle = torch.atan2(p[:, :, :, 1], p[:, :, :, 0])
        
        # Calculate sines and cosines with 2 offset frequencies
        sin1 = torch.sin(angle + time * 0.1)
        cos1 = torch.cos(angle + time * 0.1)
        sin2 = torch.sin(angle * 2.0 + time * 0.2)
        cos2 = torch.cos(angle * 2.0 + time * 0.2)
        
        # Different visualization types
        if viz_type == 0:
            # Simple sine wave
            result = sin1
        elif viz_type == 1:
            # Combined sine waves
            result = sin1 * cos2
        elif viz_type == 2:
            # Squared combination
            result = sin1 * sin1 + cos2 * cos2
        else:  # viz_type == 3 (default - ellipses)
            # Create elliptical patterns
            radius = torch.sqrt(torch.sum(p * p, dim=-1, keepdim=True))
            radius = radius + 0.01  # Avoid division by zero
            result = (sin1 + cos2) / radius
        
        # Apply phase shift (contrast adjustment)
        contrast = 1.0 + phase_shift
        result = result * contrast
        
        # Ensure output is in valid [-1, 1] range
        result = torch.clamp(result, -1.0, 1.0)
        
        return result
    
    @classmethod
    def fbm_noise(cls, p, scale, warp_strength, phase_shift, octaves, time, device):
        """
        Heterogeneous FBM noise implementation
        
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            scale: Scale factor
            warp_strength: Warp amount
            phase_shift: Contrast adjustment
            octaves: Number of detail levels
            time: Animation time
            device: Torch device
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        batch, height, width, _ = p.shape
        
        # Scale coordinates by user parameter
        p = p * scale
        
        # Apply domain warping if warp strength > 0
        if warp_strength > 0.0:
            # Create a smooth flow field for warping
            warp_coords = p * 0.3 + torch.tensor([time * 0.05, 0.0], device=device).reshape(1, 1, 1, 2)
            warp_noise = cls.simplex_noise(warp_coords)
            
            # Apply warp to coordinates
            p = p + warp_noise * warp_strength
        
        # Generate multi-octave FBM noise
        result = torch.zeros(batch, height, width, 1, device=device)
        amplitude = 1.0
        frequency = 1.0
        
        # Loop through octaves
        for i in range(min(int(octaves), 8)):
            current_p = p * frequency + torch.tensor([i * 2.0, time * 0.1], device=device).reshape(1, 1, 1, 2)
            value = cls.simplex_noise(current_p)
            
            # Calculate heterogeneous weight
            if i > 0:
                # This creates areas with more or less detail
                weight = torch.clamp(cls.simplex_noise(p * 0.1 + i * 0.5), 0.0, 1.0)
                value = value * weight
            
            # Add weighted noise at current frequency
            result += amplitude * value
            
            # Prepare for next octave
            frequency *= 2.0
            amplitude *= 0.5
        
        # Apply phase shift (contrast adjustment)
        contrast = 1.0 + phase_shift
        result = result * contrast
        
        # Ensure output is in valid [-1, 1] range
        result = torch.clamp(result, -1.0, 1.0)
        
        return result
    
    @classmethod
    def projection3d(cls, p, scale, warp_strength, phase_shift, octaves, time, device):
        """
        3D projection noise implementation
        
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            scale: Scale factor
            warp_strength: Warp amount
            phase_shift: Contrast adjustment
            octaves: Number of detail levels
            time: Animation time
            device: Torch device
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        batch, height, width, _ = p.shape
        
        # Scale coordinates by user parameter
        p = p * scale
        
        # Apply domain warping if warp strength > 0
        if warp_strength > 0.0:
            # Create a smooth flow field for warping
            warp_coords1 = p * 0.3 + torch.tensor([0.0, time * 0.05], device=device).reshape(1, 1, 1, 2)
            warp_coords2 = p * 0.3 + torch.tensor([time * 0.05, 0.0], device=device).reshape(1, 1, 1, 2)
            
            # Generate noise for warping
            warp_noise1 = cls.simplex_noise(warp_coords1)
            warp_noise2 = cls.simplex_noise(warp_coords2, seed=13.5)
            
            # Apply warp to coordinates
            p = p + torch.cat([warp_noise1, warp_noise2], dim=-1) * warp_strength
        
        # Create 3D coordinates for projection
        z = torch.ones(batch, height, width, 1, device=device) * (time * 0.1)
        coords_3d = torch.cat([p, z], dim=-1)
        
        # Generate multi-octave 3D noise
        result = torch.zeros(batch, height, width, 1, device=device)
        amplitude = 1.0
        frequency = 1.0
        
        # Loop through octaves
        for i in range(min(int(octaves), 8)):
            # Scale and add phase offset for each octave
            current_coords = coords_3d * frequency
            value = cls.simplex_noise_3d(current_coords, seed=i * 13.5)
            
            # Add to result with decreasing amplitude
            result += amplitude * value
            
            # Prepare for next octave
            frequency *= 2.0
            amplitude *= 0.5
        
        # Apply phase shift (contrast adjustment)
        contrast = 1.0 + phase_shift
        result = result * contrast
        
        # Ensure output is in valid [-1, 1] range
        result = torch.clamp(result, -1.0, 1.0)
        
        return result
    
    @classmethod
    def simplex_noise(cls, coords, seed=0):
        """
        Simplex-like noise implementation in PyTorch
        
        Args:
            coords: Coordinate tensor [batch, height, width, 2]
            seed: Random seed value
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        batch, height, width, _ = coords.shape
        device = coords.device
        
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
        return 2.0 * noise
    
    @classmethod
    def simplex_noise_3d(cls, coords, seed=0):
        """
        Simplified 3D noise implementation
        This is a basic approximation for the 3D projection
        
        Args:
            coords: 3D coordinate tensor [batch, height, width, 3]
            seed: Random seed value
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        batch, height, width, _ = coords.shape
        device = coords.device
        
        # Slice the 3D space in multiple 2D planes
        xy_coords = coords[:, :, :, 0:2]
        yz_coords = coords[:, :, :, 1:3]
        xz_coords = torch.cat([coords[:, :, :, 0:1], coords[:, :, :, 2:3]], dim=-1)
        
        # Get 2D noise for each plane
        xy_noise = cls.simplex_noise(xy_coords, seed=seed)
        yz_noise = cls.simplex_noise(yz_coords, seed=seed+1)
        xz_noise = cls.simplex_noise(xz_coords, seed=seed+2)
        
        # Combine the noises
        noise = (xy_noise + yz_noise + xz_noise) / 3.0
        
        return noise
    
    @classmethod
    def shader_noise_to_tensor_with_params(cls, batch_size=1, height=64, width=64, shader_type="tensor_field", 
                                      shader_params=None, time=0.0, device="cuda", seed=0):
        """
        Generate noise tensors using the full shader_params dictionary
        
        Args:
            batch_size: Number of images in batch
            height: Height of tensor
            width: Width of tensor
            shader_type: Type of shader ("tensor_field", "heterogeneous_fbm", "projection_3d", "cellular", "fractal", "perlin", "waves", "gaussian")
            shader_params: Full dictionary of shader parameters from JSON
            time: Animation time
            device: Device to create tensor on
            seed: Random seed
            
        Returns:
            Tensor with shape [batch_size, height, width, 1] in BHWC format
        """
        # Fallback to empty dict if None
        if shader_params is None:
            shader_params = {}
            
        # Extract common parameters with fallbacks
        scale = shader_params.get("shaderScale", shader_params.get("scale", 1.0))
        phase_shift = shader_params.get("shaderPhaseShift", shader_params.get("phase_shift", 0.0))
        warp_strength = shader_params.get("shaderWarpStrength", shader_params.get("warp_strength", 0.0))
        octaves = shader_params.get("shaderOctaves", shader_params.get("octaves", 3.0))
        visualization_type = shader_params.get("visualization_type", 3)
        
        # Extract shape parameters for all noise types
        shape_type = shader_params.get("shaderShapeType", shader_params.get("shape_type", "none"))
        shape_strength = shader_params.get("shaderShapeStrength", shader_params.get("shapemaskstrength", shader_params.get("shape_strength", 1.0)))
        
        # Ensure consistent parameter naming in shader_params
        shader_params["shaderShapeType"] = shape_type
        shader_params["shaderShapeStrength"] = shape_strength
        
        # Extract temporal coherence parameters
        use_temporal_coherence = shader_params.get("useTemporalCoherence", shader_params.get("temporal_coherence", False))
        base_seed = shader_params.get("base_seed", seed)
        
        # Make sure we have a proper time parameter for animation/temporal coherence
        if "time" not in shader_params or shader_params["time"] is None:
            shader_params["time"] = time
        else:
            # If time is already in shader_params, ensure it's used
            time = shader_params["time"]
        
        # Extract color parameters for debugging
        color_scheme = shader_params.get("colorScheme", "none")
        color_intensity = shader_params.get("shaderColorIntensity", 0.8)
        
        # Set deterministic seed for this operation
        # Use base_seed if temporal coherence is enabled
        effective_seed = base_seed if use_temporal_coherence else seed
        torch.manual_seed(effective_seed)
        
        # Use the specialized methods for specific shader types
        if shader_type == "cellular" and hasattr(cls, 'cellular_noise_with_params'):
            noise_bhwc = cls.cellular_noise_with_params(batch_size, height, width, shader_params, time, device, effective_seed)
        elif shader_type == "domain_warp" and hasattr(cls, 'domain_warp_with_params'):
            noise_bhwc = cls.domain_warp_with_params(batch_size, height, width, shader_params, time, device, effective_seed)
        elif shader_type == "fractal" and hasattr(cls, 'fractal_noise_with_params'):
            noise_bhwc = cls.fractal_noise_with_params(batch_size, height, width, shader_params, time, device, effective_seed)
        elif shader_type == "perlin" and hasattr(cls, 'perlin_noise_with_params'):
            noise_bhwc = cls.perlin_noise_with_params(batch_size, height, width, shader_params, time, device, effective_seed)
        elif shader_type == "waves" and hasattr(cls, 'waves_noise_with_params'):
            noise_bhwc = cls.waves_noise_with_params(batch_size, height, width, shader_params, time, device, effective_seed)
        elif shader_type == "gaussian" and hasattr(cls, 'gaussian_noise_with_params'):
            noise_bhwc = cls.gaussian_noise_with_params(batch_size, height, width, shader_params, time, device, effective_seed)
        elif shader_type == "tensor_field" and hasattr(cls, 'tensor_field_with_params'):
            noise_bhwc = cls.tensor_field_with_params(batch_size, height, width, shader_params, time, device, effective_seed)
        elif shader_type == "heterogeneous_fbm" and hasattr(cls, 'heterogeneous_fbm_with_params'):
            noise_bhwc = cls.heterogeneous_fbm_with_params(batch_size, height, width, shader_params, time, device, effective_seed)
        elif shader_type == "interference_patterns" and hasattr(cls, 'interference_patterns_with_params'):
            noise_bhwc = cls.interference_patterns_with_params(batch_size, height, width, shader_params, time, device, effective_seed)
        elif (shader_type == "spectral" or shader_type == "spectral_noise") and hasattr(cls, 'spectral_noise_with_params'):
            noise_bhwc = cls.spectral_noise_with_params(batch_size, height, width, shader_params, time, device, effective_seed)
        elif (shader_type == "projection_3d" or shader_type == "3d_projection") and hasattr(cls, 'projection_3d_with_params'):
            noise_bhwc = cls.projection_3d_with_params(batch_size, height, width, shader_params, time, device, effective_seed)
        elif (shader_type == "curl" or shader_type == "curl_noise") and hasattr(cls, 'curl_noise_with_params'):
            noise_bhwc = cls.curl_noise_with_params(batch_size, height, width, shader_params, time, device, effective_seed)
        else:
            # Use the standard method for other shader types
            noise_bhwc = cls.shader_noise_to_tensor(
                batch_size=batch_size,
                height=height,
                width=width,
                shader_type=shader_type,
                visualization_type=visualization_type,
                scale=scale,
                phase_shift=phase_shift,
                warp_strength=warp_strength,
                time=time,
                device=device,
                seed=effective_seed,
                octaves=octaves
            )
            
        # Convert from BHWC to BCHW for color processing
        noise_bchw = noise_bhwc.permute(0, 3, 1, 2)
        
        # Expand channels if needed for latent space operations
        if noise_bchw.shape[1] == 1:
            # Expand to 9 channels for latent space operations instead of just 4
            noise_bchw = noise_bchw.expand(-1, 9, -1, -1)
        # If we have 4 channels but need 9 for latent operations
        elif noise_bchw.shape[1] == 4:
            # Get existing channels
            r, g, b, a = noise_bchw.chunk(4, dim=1)
            
            # Create 5 more channels as variations of the existing ones
            c5 = (r + g) / 2.0  # Average of red and green
            c6 = (g + b) / 2.0  # Average of green and blue
            c7 = (b + a) / 2.0  # Average of blue and alpha
            c8 = (r + b) / 2.0  # Average of red and blue
            c9 = (r + g + b + a) / 4.0  # Average of all channels
            
            # Combine all 9 channels
            noise_bchw = torch.cat([r, g, b, a, c5, c6, c7, c8, c9], dim=1)
            
        # Apply color scheme if we have one
        if color_scheme != "none" and color_intensity > 0:
            # Import here to avoid circular import
            from .shader_params_reader import ShaderParamsReader
            
            # Get stats before color application
            before_min = noise_bchw.min().item()
            before_max = noise_bchw.max().item()
            before_mean = noise_bchw.mean().item()
            before_std = noise_bchw.std().item()
            
            # Apply color transformation
            noise_bchw = ShaderParamsReader.apply_color_scheme(noise_bchw, shader_params)
            
            # Get stats after color application
            after_min = noise_bchw.min().item()
            after_max = noise_bchw.max().item()
            after_mean = noise_bchw.mean().item()
            after_std = noise_bchw.std().item()
            
            # Check if the stats changed significantly (which would indicate the color was applied)
            if abs(before_std - after_std) < 0.001 and abs(before_mean - after_mean) < 0.001:
                pass
        else:
            pass
            
        # Convert back to BHWC format for return
        noise_bhwc = noise_bchw.permute(0, 2, 3, 1)
        
        return noise_bhwc 

    @classmethod
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
        # Add time to shader params if not already present
        if "time" not in shader_params:
            shader_params["time"] = time
            
        # Ensure shape parameters are properly extracted and included
        shape_type = shader_params.get("shaderShapeType", shader_params.get("shape_type", "none"))
        shape_mask_strength = shader_params.get("shaderShapeStrength", shader_params.get("shapemaskstrength", 1.0))
        
        # Ensure phase shift is properly extracted and used with the correct parameter name
        phase_shift = shader_params.get("shaderPhaseShift", shader_params.get("phase_shift", 0.5))
        
        # Extract temporal coherence parameters
        base_seed = shader_params.get("base_seed", seed)  # Use provided seed as default base_seed
        use_temporal_coherence = shader_params.get("temporal_coherence", shader_params.get("useTemporalCoherence", False))
        
        # Set deterministic seed for this operation
        effective_seed = base_seed if use_temporal_coherence else seed
        
        # Make sure the correct parameter names are set for consistency
        shader_params["shaderShapeType"] = shape_type
        shader_params["shaderShapeStrength"] = shape_mask_strength
        shader_params["shaderPhaseShift"] = phase_shift  # Ensure it's consistently named
        shader_params["base_seed"] = base_seed  # Ensure base_seed is set
        shader_params["temporal_coherence"] = use_temporal_coherence  # Ensure temporal_coherence is set
        
        # Determine target channels for the generator call
        # Prioritize from shader_params, then ACEStep/ACE logic, then default.
        gen_target_channels = 4 # Default
        if "target_channels" in shader_params:
            gen_target_channels = int(shader_params["target_channels"])
        elif shader_params.get("inner_model_class") in ["ACEStep", "ACE"]:
            gen_target_channels = 8

        # Use the CurlNoiseGenerator directly with the full params
        from .shaders.curl_noise import CurlNoiseGenerator
        curl_noise = CurlNoiseGenerator.get_curl_noise(
            batch_size=batch_size,
            height=height,
            width=width,
            shader_params=shader_params,  # Pass the full parameter dictionary
            device=device,
            seed=effective_seed,  # Use base_seed if temporal coherence is enabled
            target_channels=gen_target_channels # Explicitly pass determined target channels
        )
        
        # Convert to BHWC format to match expected output
        result = curl_noise.permute(0, 2, 3, 1)[:, :, :, 0:1]
        
        return result 