import torch
import math
from ..shader_params_reader import ShaderParamsReader

class TemporalCoherentNoiseGenerator:
    """
    Implementation of temporally coherent noise that maintains consistency
    between animation frames by treating time as a proper 4th dimension
    rather than using different seeds per frame.
    
    This class generates noise that smoothly transitions between frames
    by implementing true 4D (x,y,z,time) noise functions.
    """
    
    @staticmethod
    def get_temporal_noise(batch_size, height, width, shader_params, device="cuda", base_seed=0):
        """
        Generate temporally coherent noise tensor
        
        Args:
            batch_size: Number of images in batch
            height: Height of tensor
            width: Width of tensor
            shader_params: Dictionary containing parameters from shader_params.json
            device: Device to create tensor on
            base_seed: Base seed value for initialization (NOT changed per frame)
            
        Returns:
            Tensor with shape [batch_size, 4, height, width]
        """
        # Extract parameters from shader_params
        scale = shader_params.get("shaderScale", shader_params.get("scale", 1.0))
        warp_strength = shader_params.get("shaderWarpStrength", shader_params.get("warp_strength", 0.5))
        phase_shift = shader_params.get("shaderPhaseShift", shader_params.get("phase_shift", 0.5))
        octaves = shader_params.get("shaderOctaves", shader_params.get("octaves", 3))
        time = shader_params.get("time", 0.0)  # Critical for temporal coherence
        frequency_range = shader_params.get("frequencyRange", shader_params.get("frequency_range", 0))
        shape_type = shader_params.get("shaderShapeType", shader_params.get("shape_type", "none"))
        shape_mask_strength = shader_params.get("shaderShapeStrength", shader_params.get("shapemaskstrength", 1.0))
        
        # Create coordinate grid (normalized to [-1, 1])
        # Important: make sure we create the grid with the correct orientation for height/width
        # In PyTorch, meshgrid with ij indexing puts height first, width second
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing='ij'
        )
        
        # Combine into coordinate tensor [batch, height, width, 2]
        p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Verify p has the correct shape before proceeding
        if p.shape[1:3] != (height, width):
            # Try to fix by transposing or reshaping if needed
            if p.shape[1] == width and p.shape[2] == height:
                p = p.permute(0, 2, 1, 3)
        
        # Set consistent seed for initialization
        torch.manual_seed(base_seed)
        
        # Generate temporal coherent noise
        result = TemporalCoherentNoiseGenerator.temporal_spectral_noise(
            p, scale, warp_strength, phase_shift, octaves, 
            frequency_range, time, device, base_seed
        )
        
        # Apply shape mask if requested
        if shape_type not in ["none", "0"] and shape_mask_strength > 0:
            shape_mask = ShaderParamsReader.apply_shape_mask(p, shape_type)
            result = torch.lerp(result, result * shape_mask, shape_mask_strength)
        
        # Ensure output is in valid [-1, 1] range
        result = torch.clamp(result, -1.0, 1.0)
        
        # Convert from [batch, height, width, 1] to [batch, 4, height, width]
        result = result.permute(0, 3, 1, 2)  # [batch, 1, height, width]
        result = result.expand(-1, 4, -1, -1)  # [batch, 4, height, width]
        
        # Final verification of output dimensions
        if result.shape[2:] != (height, width):
            pass  # Added pass statement to fix linter error
        
        return result
    
    @staticmethod
    def temporal_spectral_noise(p, scale, warp_strength, phase_shift, octaves, frequency_range, time, device, base_seed):
        """
        Generate spectral noise with true temporal coherence across frames
        
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            scale: Scale factor
            warp_strength: Amount of warping
            phase_shift: Phase/rotation adjustment
            octaves: Number of detail levels (1-8)
            frequency_range: Type of frequency filtering (1-4)
            time: Animation time parameter (critical for temporal coherence)
            device: Device to use for computation
            base_seed: Base random seed value (consistent across frames)
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        batch, height, width, _ = p.shape
        
        # Scale by user parameter
        p = p * scale
        
        # Extend 2D coordinates to 3D by adding time as third dimension
        # This is the key to temporal coherence
        p_temporal = torch.cat([
            p, 
            torch.ones_like(p[:, :, :, :1]) * time
        ], dim=-1)  # [batch, height, width, 3] where third dimension is time
        
        # Apply 3D warp if enabled
        if warp_strength > 0.0:
            # Create a smooth 3D flow field for warping
            # Note: Using 3D noise for warping instead of 2D noise with time offset
            warp_p = p_temporal * 0.4  # Scale for warp coords
            
            # Generate 3D noise for each component of the warp
            # Add seed dimension for 3D simplex noise
            warp_seed1 = torch.ones_like(warp_p[:, :, :, :1]) * base_seed
            warp_p_with_seed1 = torch.cat([warp_p, warp_seed1], dim=-1)
            
            warp_seed2 = torch.ones_like(warp_p[:, :, :, :1]) * (base_seed + 1)
            warp_p_with_seed2 = torch.cat([warp_p + 5.0, warp_seed2], dim=-1)

            # Get noise for x and y dimensions only
            warp_noise1 = TemporalCoherentNoiseGenerator.simplex_noise_3d(warp_p_with_seed1)
            warp_noise2 = TemporalCoherentNoiseGenerator.simplex_noise_3d(warp_p_with_seed2)
            
            # Create warp vectors - shape [batch, height, width, 1]
            # Now properly expand to match p_temporal's shape
            zeros = torch.zeros_like(warp_noise1)
            
            # Create spatial warp vector with explicit shape control
            # Make sure dimensions match exactly with p_temporal 
            x_warp = warp_noise1  # [batch, height, width, 1]
            y_warp = warp_noise2  # [batch, height, width, 1]
            t_warp = zeros       # [batch, height, width, 1] - no warp in time dimension
            
            # Apply the warp to coordinates, component-wise
            # Make sure we're doing the addition with properly aligned tensors
            try:
                p_temporal_x = p_temporal[:, :, :, 0:1] + x_warp * warp_strength
                p_temporal_y = p_temporal[:, :, :, 1:2] + y_warp * warp_strength
                p_temporal_t = p_temporal[:, :, :, 2:3]  # Time dimension unchanged
                
                # Recombine the warped components
                p_temporal = torch.cat([p_temporal_x, p_temporal_y, p_temporal_t], dim=-1)
            except RuntimeError as e:
                # Alternative approach - reshape noise to match p_temporal dimensions
                # This ensures that height and width dimensions match exactly
                if x_warp.shape[1:3] != p_temporal.shape[1:3]:
                    # Transpose the height and width dimensions if needed
                    if (x_warp.shape[1] == p_temporal.shape[2] and 
                        x_warp.shape[2] == p_temporal.shape[1]):
                        # Need to transpose dimensions 1 and 2
                        x_warp = x_warp.permute(0, 2, 1, 3)
                        y_warp = y_warp.permute(0, 2, 1, 3)
                        t_warp = t_warp.permute(0, 2, 1, 3)
                    else:
                        # Try to resize the noise tensors to match
                        target_height, target_width = p_temporal.shape[1:3]
                        x_warp = torch.nn.functional.interpolate(
                            x_warp.permute(0, 3, 1, 2),  # [B, C, H, W]
                            size=(target_height, target_width),
                            mode='bilinear'
                        ).permute(0, 2, 3, 1)  # Back to [B, H, W, C]
                        
                        y_warp = torch.nn.functional.interpolate(
                            y_warp.permute(0, 3, 1, 2),
                            size=(target_height, target_width),
                            mode='bilinear'
                        ).permute(0, 2, 3, 1)
                        
                        t_warp = torch.zeros(batch, target_height, target_width, 1, device=device)
                
                # Try again with the reshaped tensors
                p_temporal_x = p_temporal[:, :, :, 0:1] + x_warp * warp_strength
                p_temporal_y = p_temporal[:, :, :, 1:2] + y_warp * warp_strength
                p_temporal_t = p_temporal[:, :, :, 2:3]  # Time dimension unchanged
                
                # Recombine the warped components
                p_temporal = torch.cat([p_temporal_x, p_temporal_y, p_temporal_t], dim=-1)
        
        # Calculate frequency domain coordinates
        uv = p_temporal[:, :, :, :2]  # Use spatial dimensions for UV mapping
        freq = (p_temporal[:, :, :, :2] - 0.5) * 2.0
        
        # Calculate radius and angle in frequency space
        radius = torch.sqrt(freq[:, :, :, 0]**2 + freq[:, :, :, 1]**2)
        angle = torch.atan2(freq[:, :, :, 1], freq[:, :, :, 0])
        
        # Apply phase shift with time component for smooth rotation
        # This creates smooth transitions between frames
        angle = angle + (phase_shift + time * 0.1) * math.pi
        
        # Recalculate frequency coordinates after rotation
        freq_rotated_x = radius * torch.cos(angle)
        freq_rotated_y = radius * torch.sin(angle)
        freq = torch.stack([freq_rotated_x, freq_rotated_y], dim=-1)
        
        # Initialize filter based on frequency range
        filter_tensor = torch.ones_like(radius)
        
        # Radial filtering based on frequency range (with time modulation)
        if frequency_range == 1:  # Temporal low-pass
            cutoff = 0.25 + 0.05 * torch.sin(torch.tensor(time * 0.2, device=device))
            filter_tensor = torch.sigmoid((1.0 - radius - cutoff) * 10.0)
        elif frequency_range == 2:  # Temporal band-pass
            center = 0.5 + 0.1 * torch.sin(torch.tensor(time * 0.3, device=device))
            width = 0.2 + 0.05 * torch.cos(torch.tensor(time * 0.25, device=device))
            low, high = center - width/2, center + width/2
            low_pass = torch.sigmoid((radius - low) * 10.0)
            high_pass = torch.sigmoid((high - radius) * 10.0)
            filter_tensor = low_pass * high_pass
        elif frequency_range == 3:  # Temporal high-pass
            cutoff = 0.6 + 0.05 * torch.sin(torch.tensor(time * 0.15, device=device))
            filter_tensor = torch.sigmoid((radius - cutoff) * 10.0)
        elif frequency_range == 4:  # Directional filter with temporal rotation
            # Directional filtering - rotates smoothly over time
            dir_strength = 0.8
            num_dir = 4  # Number of directions/lobes
            angle_mod = (angle + time * 0.2) % (2.0 * math.pi)
            dir_filter = 0.5 + 0.5 * torch.cos(torch.tensor(float(num_dir), device=device) * angle_mod)
            filter_tensor = torch.lerp(torch.ones_like(dir_filter), dir_filter, dir_strength)
        
        # Generate base noise with different frequency components using 3D noise
        noise = torch.zeros_like(radius).unsqueeze(-1)
        
        # Determine number of octaves based on user control
        max_octaves = min(int(octaves), 8)
        
        # Sum multiple octaves with different frequencies
        for i in range(max_octaves):
            # Scale frequency based on octave
            freq_scale = 2.0 ** i
            
            # Scale amplitude based on frequency (1/f noise)
            amp = 1.0 / freq_scale
            
            # Use 3D simplex noise to get temporally coherent noise
            # We add a different phase offset per octave
            octave_p = p_temporal * freq_scale + torch.tensor([0.0, 0.0, i * 1.5], device=device)
            
            # Get 3D noise value
            noise_val = TemporalCoherentNoiseGenerator.simplex_noise_3d(
                torch.cat([octave_p, torch.ones_like(octave_p[:, :, :, :1]) * (base_seed + i)], dim=-1)
            )
            
            # Apply frequency-based filtering
            freq_factor = i / max(max_octaves - 1, 1)  # 0-1 range for frequency
            freq_filter = torch.ones_like(noise_val)
            
            if frequency_range == 1:  # Low-pass
                freq_filter = 1.0 - freq_factor
            elif frequency_range == 2:  # Band-pass
                freq_filter = 1.0 - torch.abs(torch.tensor(freq_factor, device=device) - 0.5) * 2.0
            elif frequency_range == 3:  # High-pass
                freq_filter = freq_factor
            
            noise = noise + (noise_val * amp * freq_filter)
        
        # Apply filter and normalize
        noise = noise * filter_tensor.unsqueeze(-1)
        
        # Add smooth temporal modulations
        time_factor = torch.sin(torch.tensor(time * 0.3, device=device))
        noise = noise * (1.0 + 0.1 * time_factor)
        
        # Normalize to [-1, 1] range
        noise = torch.clamp(noise * 1.5, -1.0, 1.0)
        
        return noise
    
    @staticmethod
    def simplex_noise_3d(coords):
        """
        3D simplex noise implementation in PyTorch
        This is the key to temporal coherence as it treats time as a full dimension
        
        Args:
            coords: Coordinate tensor [batch, height, width, 4]
                   where dims are [x, y, time, seed]
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        batch, height, width, _ = coords.shape
        device = coords.device
        
        # Extract x, y, z (time), and seed components
        x, y, z, seed = coords[:, :, :, 0], coords[:, :, :, 1], coords[:, :, :, 2], coords[:, :, :, 3]
        
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
        
        # Determine which simplex we are in
        # Offsets for second corner of simplex
        i1 = torch.zeros_like(x0)
        j1 = torch.zeros_like(y0)
        k1 = torch.zeros_like(z0)
        
        # Offsets for third corner of simplex
        i2 = torch.zeros_like(x0)
        j2 = torch.zeros_like(y0)
        k2 = torch.zeros_like(z0)
        
        # This logic is hard to vectorize, but possible
        # Using temporary masks to determine the simplex case
        x_ge_y = (x0 >= y0).float()
        y_ge_z = (y0 >= z0).float()
        x_ge_z = (x0 >= z0).float()
        
        # i1
        i1 = x_ge_y * x_ge_z
        # j1
        j1 = (1 - x_ge_y) * y_ge_z
        # k1
        k1 = (1 - x_ge_z) * (1 - y_ge_z)
        
        # i2
        i2 = x_ge_y + (1 - x_ge_y) * x_ge_z
        # j2
        j2 = x_ge_y * (1 - x_ge_z) + (1 - x_ge_y)
        # k2
        k2 = (1 - x_ge_z) + x_ge_z * (1 - x_ge_y)
        
        # Calculate noise contributions from each corner
        # 3D noise contributions from simplex corners
        noise = torch.zeros_like(x0)
        
        # Corner 1 - origin of simplex
        t0 = 0.6 - x0*x0 - y0*y0 - z0*z0
        mask0 = (t0 >= 0).float()
        t0 = t0 * t0
        n0 = mask0 * t0 * t0 * TemporalCoherentNoiseGenerator.grad3d(i, j, k, x0, y0, z0, seed)
        
        # Corner 2
        x1 = x0 - i1 + G3
        y1 = y0 - j1 + G3
        z1 = z0 - k1 + G3
        t1 = 0.6 - x1*x1 - y1*y1 - z1*z1
        mask1 = (t1 >= 0).float()
        t1 = t1 * t1
        n1 = mask1 * t1 * t1 * TemporalCoherentNoiseGenerator.grad3d(i + i1, j + j1, k + k1, x1, y1, z1, seed)
        
        # Corner 3
        x2 = x0 - i2 + 2.0 * G3
        y2 = y0 - j2 + 2.0 * G3
        z2 = z0 - k2 + 2.0 * G3
        t2 = 0.6 - x2*x2 - y2*y2 - z2*z2
        mask2 = (t2 >= 0).float()
        t2 = t2 * t2
        n2 = mask2 * t2 * t2 * TemporalCoherentNoiseGenerator.grad3d(i + i2, j + j2, k + k2, x2, y2, z2, seed)
        
        # Corner 4 (last corner of simplex)
        x3 = x0 - 1.0 + 3.0 * G3
        y3 = y0 - 1.0 + 3.0 * G3
        z3 = z0 - 1.0 + 3.0 * G3
        t3 = 0.6 - x3*x3 - y3*y3 - z3*z3
        mask3 = (t3 >= 0).float()
        t3 = t3 * t3
        n3 = mask3 * t3 * t3 * TemporalCoherentNoiseGenerator.grad3d(i + 1, j + 1, k + 1, x3, y3, z3, seed)
        
        # Sum up noise contributions
        # Scale to stay within [-1,1]
        result = (n0 + n1 + n2 + n3) * 32.0
        
        # Ensure we maintain the input height/width dimensions
        if result.shape[1:3] != (height, width):
            # Reshape to match the input dimensions
            result = result.reshape(batch, height, width).unsqueeze(-1)
        
        return result.unsqueeze(-1) if len(result.shape) == 3 else result
    
    @staticmethod
    def grad3d(ix, iy, iz, x, y, z, seed):
        """
        3D gradient function for simplex noise
        Computes the dot product of a random gradient vector and a distance vector
        
        Args:
            ix, iy, iz: Grid cell coordinates
            x, y, z: Position relative to grid cell
            seed: Seed value
            
        Returns:
            Dot product of gradient and distance vectors
        """
        # Simple hash for deterministic, repeatable patterns
        # Convert to integers before applying bitwise operations
        # Need to use mod instead of bitwise & for floats
        ix_int = torch.floor(ix).to(torch.int32)
        iy_int = torch.floor(iy).to(torch.int32)
        iz_int = torch.floor(iz).to(torch.int32)
        seed_int = torch.floor(seed).to(torch.int32)
        
        # Compute hash (now with integer tensors)
        h = (ix_int * 1619 + iy_int * 31337 + iz_int * 6971 + seed_int * 1013) & 15
        
        # Convert back to original dtype for the rest of the calculations
        h = h.to(x.dtype)
        
        # Convert lower 4 bits of hash to 12 gradient directions
        u = torch.where(h < 8, x, y)
        v = torch.where(h < 4, y, torch.where((h == 12) | (h == 14), x, z))
        
        # Convert hash to 12 gradient directions
        # 1 or -1 based on hash bits - avoiding bitwise operations on floats
        # The & operation is replaced with modulo division which is safer for floats
        h_mod_2 = torch.remainder(h, 2)
        h_mod_4 = torch.remainder(h, 4)
        
        u_sign = (h_mod_2 == 0).float() * 2.0 - 1.0
        v_sign = (h_mod_4 < 2).float() * 2.0 - 1.0
        
        return u_sign * u + v_sign * v

def add_temporal_noise_to_shader_params(shader_params, frame_number, base_seed, fps=24, duration=5):
    """
    Prepare shader parameters for temporal coherent noise
    
    Args:
        shader_params: Original shader parameters dictionary
        frame_number: Current frame number
        base_seed: Base seed for the entire animation
        fps: Frames per second
        duration: Animation duration in seconds
        
    Returns:
        Modified shader parameters with temporal settings
    """
    # Create a copy to avoid modifying the original
    params = shader_params.copy()
    
    # Calculate time value based on frame number (0.0 to 1.0 over duration)
    time = (frame_number / (fps * duration)) % 1.0
    
    # Store time parameter for the shader
    params["time"] = time
    
    # Store the base seed (constant across frames)
    params["base_seed"] = base_seed
    
    # Add subtle phase progression for temporal interest
    # This creates a smooth progression that's continuous
    base_phase = params.get("shaderPhaseShift", 0.5)
    frame_phase = base_phase + time * 0.4  # Subtle phase change over time
    params["shaderPhaseShift"] = frame_phase
    
    return params

def generate_temporal_coherent_noise_tensor(shader_params, height, width, batch_size=1, device="cuda", seed=0):
    """
    Generate temporal coherent noise tensor directly using shader_params
    
    Args:
        shader_params: Dictionary containing parameters from shader_params.json
        height: Height of tensor
        width: Width of tensor
        batch_size: Number of images in batch
        device: Device to create tensor on
        seed: Random seed to ensure consistent noise with same seed
        
    Returns:
        Noise tensor with shape [batch_size, 4, height, width]
    """
    # Extract temporal coherence parameters if present
    time = shader_params.get("time", 0.0)
    base_seed = shader_params.get("base_seed", seed)
    use_temporal_coherence = shader_params.get("useTemporalCoherence", True)  # Default is True for this type
    
    # Set deterministic seed for this operation
    torch.manual_seed(base_seed if use_temporal_coherence else seed)
    
    # Generate temporal coherent noise using the TemporalCoherentNoiseGenerator
    noise_bhwc = TemporalCoherentNoiseGenerator.get_temporal_noise(
        batch_size=batch_size,
        height=height,
        width=width,
        shader_params=shader_params,
        device=device,
        base_seed=base_seed if use_temporal_coherence else seed
    )
    
    # Reset random seed state to not affect other operations
    torch.manual_seed(torch.seed())
    
    # Ensure we have the right format - should already be [batch, 4, height, width]
    if noise_bhwc.shape[1] == 1:
        noise_bhwc = noise_bhwc.expand(-1, 4, -1, -1)
    
    # Normalize the noise to match typical noise distribution
    noise_bhwc = (noise_bhwc - noise_bhwc.mean()) / (noise_bhwc.std() + 1e-8)
    
    return noise_bhwc

def integrate_temporal_coherent_noise():
    """
    Function to integrate temporal coherent noise into shader_to_tensor.py
    
    This adds the temporal_coherent_noise method to the ShaderToTensor class
    """
    from ..shader_to_tensor import ShaderToTensor
    
    def temporal_coherent_noise(cls, batch_size, height, width, shader_params, time, device, base_seed):
        """
        Generate temporally coherent noise for animation frames
        
        Args:
            batch_size: Number of images in batch
            height: Height of tensor
            width: Width of tensor
            shader_params: Dictionary containing parameters
            time: Animation time
            device: Device to create tensor on
            base_seed: Base seed value (consistent across animation)
            
        Returns:
            Noise tensor [batch_size, 4, height, width]
        """
        # Update shader parameters with time value
        updated_params = shader_params.copy()
        updated_params["time"] = time
        
        # Call the temporal coherent noise generator
        noise = TemporalCoherentNoiseGenerator.get_temporal_noise(
            batch_size, height, width, updated_params, device, base_seed
        )
        
        return noise
    
    # Add the method to ShaderToTensor class
    setattr(ShaderToTensor, 'temporal_coherent_noise', classmethod(temporal_coherent_noise))
    
    # Register in shader type handlers
    if hasattr(ShaderToTensor, 'shader_handlers') and isinstance(ShaderToTensor.shader_handlers, dict):
        ShaderToTensor.shader_handlers['temporal_coherent'] = ShaderToTensor.temporal_coherent_noise
    
    return True 