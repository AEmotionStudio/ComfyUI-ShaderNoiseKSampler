// Auto-extracted from v260 backup - DO NOT EDIT MANUALLY
// These are the complete GLSL shader sources restored from
// the pre-TypeScript-refactor version
export const FRAGMENT_SHADER_HEADER = `
                    precision mediump float;
                    uniform float u_time;
                    uniform float u_intensity;
                    uniform float u_scale;
                    uniform float u_octaves;
                    uniform float u_persistence;
                    uniform float u_lacunarity;
                    uniform int u_shapeType;
                    uniform float u_shapeStrength;
                    uniform float u_warpStrength;
                    uniform float u_phaseShift;
                    uniform int u_frequencyRange;
                    uniform int u_distribution;
                    uniform float u_adaptationStrength;
                    uniform float u_resolutionScale;
                    uniform int u_colorScheme;
                    varying vec2 v_texCoord;
                    
                    // Note: We no longer use resolution scaling in the shader
                    // The pixel density is now directly controlled by the canvas resolution
                    
                    // Shared utility functions
                    vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
                    vec4 permute(vec4 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
                    vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
                    vec2 fade(vec2 t) { return t*t*t*(t*(t*6.0-15.0)+10.0); }

                    float snoise(vec2 v){
                        const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                               -0.577350269189626, 0.024390243902439);
                        vec2 i  = floor(v + dot(v, C.yy));
                        vec2 x0 = v -   i + dot(i, C.xx);
                        vec2 i1;
                        i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
                        vec4 x12 = x0.xyxy + C.xxzz;
                        x12.xy -= i1;
                        i = mod(i, 289.0);
                        vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
                               + i.x + vec3(0.0, i1.x, 1.0 ));
                        vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
                                                dot(x12.zw,x12.zw)), 0.0);
                        m = m*m;
                        m = m*m;
                        vec3 x = 2.0 * fract(p * C.www) - 1.0;
                        vec3 h = abs(x) - 0.5;
                        vec3 ox = floor(x + 0.5);
                        vec3 a0 = x - ox;
                        m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
                        vec3 g;
                        g.x  = a0.x  * x0.x  + h.x  * x0.y;
                        g.yz = a0.yz * x12.xz + h.yz * x12.yw;
                        return 130.0 * dot(m, g);
                    }
                    
                    // Random function
                    float random(vec2 st) {
                        return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453);
                    }
                    
                    // Shape mask function that varies based on type
                    float applyShapeMask(vec2 st, int type) {
                        if (type == 0) { // none
                            return 1.0;
                        } else if (type == 1) { // radial
                            // Create a radial gradient from center with animation
                            vec2 center = vec2(0.5, 0.5);
                            
                            // Animate center position
                            center += 0.2 * vec2(cos(u_time), sin(u_time));
                            
                            float dist = distance(st, center) * 2.0;
                            return clamp(1.0 - dist, 0.0, 1.0);
                        } else if (type == 2) { // linear
                            // Create a linear gradient with animation
                            float x_offset = fract(u_time * 0.2) * 2.0;
                            float shifted_x = fract(st.x + x_offset);
                            return shifted_x;
                        } else if (type == 3) { // spiral
                            // Create a spiral pattern with animation
                            vec2 centered = st - vec2(0.5, 0.5);
                            float theta = atan(centered.y, centered.x);
                            float r = length(centered) * 2.0;
                            theta += u_time;
                            return fract((theta / (2.0 * 3.14159265) + r));
                        } else if (type == 4) { // checkerboard
                            // Create animated checkerboard
                            float grid_size = 8.0;
                            float x_offset = u_time * grid_size * 0.2;
                            float y_offset = u_time * grid_size * 0.1;
                            float x_grid = floor((st.x + x_offset / grid_size) * grid_size) * 0.5;
                            float y_grid = floor((st.y + y_offset / grid_size) * grid_size) * 0.5;
                            return mod(x_grid + y_grid, 1.0);
                        } else if (type == 5) { // spots
                            // Create animated spots
                            float mask = 0.0;
                            int num_spots = 10;
                            
                            // Use deterministic pseudo-random positions based on index
                            for (int i = 0; i < 10; i++) {
                                if (i >= num_spots) break;
                                // Better randomization
                                float rand_x = fract(sin(float(i) * 78.233) * 43758.5453);
                                float rand_y = fract(sin(float(i) * 12.9898) * 43758.5453);
                                float size = fract(sin(float(i) * 93.719) * 43758.5453) * 0.3 + 0.1;
                                
                                // Animate spots
                                float angle = u_time + float(i);
                                vec2 spot_pos = vec2(
                                    0.5 + cos(angle) * 0.4 * rand_x,
                                    0.5 + sin(angle) * 0.4 * rand_y
                                );
                                
                                // Pulse size
                                size *= 1.0 + 0.2 * sin(u_time * 2.0 + float(i));
                                
                                // Calculate spot mask
                                float dist = distance(st, spot_pos);
                                float spot_mask = clamp(1.0 - dist / size, 0.0, 1.0);
                                mask = max(mask, spot_mask);
                            }
                            
                            return mask;
                        } else if (type == 6) { // hexgrid
                            // Create animated hexagonal grid
                            vec2 hex_uv = st * 6.0; // Scale for hex grid
                            
                            // Apply animation
                            hex_uv.x += sin(u_time * 0.5) * 0.5;
                            hex_uv.y += cos(u_time * 0.3) * 0.5;
                            
                            // Hexagon grid math
                            vec2 r = vec2(1.0, 1.73); // Hexagon ratio
                            vec2 h = r * 0.5;
                            vec2 a = mod(hex_uv, r) - h;
                            vec2 b = mod(hex_uv + h, r) - h;
                            
                            // Determine distance to hexagon centers
                            float dist = min(length(a), length(b));
                            
                            // Create cells with smooth borders 
                            float cell_size = 0.3 + 0.1 * sin(u_time);
                            return smoothstep(cell_size + 0.05, cell_size - 0.05, dist);
                        } else if (type == 7) { // stripes
                            // Animated stripes pattern
                            float freq = 10.0;
                            float angle = 0.5 * sin(u_time * 0.2);
                            
                            // Compute rotated coordinates
                            vec2 rotated = vec2(
                                st.x * cos(angle) - st.y * sin(angle),
                                st.x * sin(angle) + st.y * cos(angle)
                            );
                            
                            // Animated stripe pattern
                            float stripes = sin(rotated.x * freq + u_time);
                            
                            // Create binary stripes with smoothed edges
                            return smoothstep(0.0, 0.1, stripes) * smoothstep(0.0, -0.1, -stripes);
                        } else if (type == 8) { // gradient
                            // Animated moving gradient
                            float angle = u_time * 0.2;
                            vec2 dir = vec2(cos(angle), sin(angle));
                            
                            // Project position onto direction vector
                            float proj = dot(st - 0.5, dir) + 0.5;
                            
                            // Smooth gradient
                            return proj;
                        } else if (type == 9) { // vignette
                            // Animated vignette effect
                            vec2 center = vec2(0.5) + vec2(
                                0.2 * sin(u_time * 0.3),
                                0.2 * cos(u_time * 0.4)
                            );
                            
                            float dist = distance(st, center);
                            
                            // Animated vignette radius
                            float radius = 0.6 + 0.2 * sin(u_time * 0.5);
                            float smoothness = 0.3;
                            
                            return 1.0 - smoothstep(radius - smoothness, radius, dist);
                        } else if (type == 10) { // cross
                            // Animated cross pattern
                            float thickness = 0.1 + 0.05 * sin(u_time);
                            float rotation = u_time * 0.2;
                            
                            // Rotate the coordinates
                            vec2 centered = st - 0.5;
                            vec2 rotated = vec2(
                                centered.x * cos(rotation) - centered.y * sin(rotation),
                                centered.x * sin(rotation) + centered.y * cos(rotation)
                            );
                            rotated += 0.5;
                            
                            // Create horizontal and vertical bars
                            float h_bar = smoothstep(0.5 - thickness, 0.5 - thickness + 0.02, rotated.y) * 
                                         smoothstep(0.5 + thickness, 0.5 + thickness - 0.02, rotated.y);
                            float v_bar = smoothstep(0.5 - thickness, 0.5 - thickness + 0.02, rotated.x) * 
                                         smoothstep(0.5 + thickness, 0.5 + thickness - 0.02, rotated.x);
                            
                            return max(h_bar, v_bar);
                        } else if (type == 11) { // stars
                            // Animated star field
                            float mask = 0.0;
                            int num_stars = 20;
                            
                            // Generate star field
                            for (int i = 0; i < 20; i++) {
                                if (i >= num_stars) break;
                                
                                // Deterministic star positions
                                float rand_x = fract(sin(float(i) * 78.233) * 43758.5453);
                                float rand_y = fract(sin(float(i) * 12.9898) * 43758.5453);
                                
                                // Star position with slow drift
                                vec2 star_pos = vec2(
                                    fract(rand_x + 0.05 * sin(u_time * 0.1 + float(i))),
                                    fract(rand_y + 0.05 * cos(u_time * 0.15 + float(i) * 1.5))
                                );
                                
                                // Star size and brightness (twinkling)
                                float brightness = 0.5 + 0.5 * sin(u_time * (0.5 + rand_x * 0.5) + float(i));
                                float size = 0.01 + 0.015 * rand_y * brightness;
                                
                                // Calculate star mask with softer edge
                                float dist = distance(st, star_pos);
                                float star_mask = smoothstep(size, size * 0.5, dist) * brightness;
                                
                                // Accumulate stars
                                mask = max(mask, star_mask);
                            }
                            
                            return mask;
                        } else if (type == 12) { // triangles
                            // Animated triangle pattern
                            float time = u_time * 0.2;
                            float scale = 5.0;
                            
                            // Apply animation to coordinates
                            vec2 uv = st * scale;
                            uv.x += sin(time) * 0.5;
                            uv.y += cos(time * 0.7) * 0.5;
                            
                            // Triangle grid
                            vec2 grid = floor(uv);
                            vec2 gv = fract(uv) - 0.5;
                            
                            // Determine which half of the square we're in
                            float t = step(gv.x, gv.y);
                            
                            // Calculate distance to triangle edge
                            vec2 ab = vec2(t, t);
                            vec2 bc = vec2(0.5 - gv.y, 0.5 - gv.x) * (1.0 - t) + vec2(-0.5 - gv.y, 0.5 - gv.x) * t;
                            vec2 ca = vec2(-0.5 - gv.x, -0.5 - gv.y) * (1.0 - t) + vec2(0.5 - gv.x, -0.5 - gv.y) * t;
                            
                            // Minimum distance to the triangle edges
                            float d_ab = dot(gv - ab * 0.5, normalize(vec2(-ab.y, ab.x)));
                            float d_bc = dot(gv - ab - bc * 0.5, normalize(vec2(-bc.y, bc.x)));
                            float d_ca = dot(gv - ab - bc - ca * 0.5, normalize(vec2(-ca.y, ca.x)));
                            
                            float d = min(min(d_ab, d_bc), d_ca);
                            
                            // Create triangle pattern with pulsing border width
                            float border_width = 0.05 + 0.03 * sin(time * 1.5);
                            return smoothstep(border_width, border_width - 0.02, abs(d));
                        } else if (type == 13) { // concentric
                            // Animated concentric circles
                            vec2 center = vec2(0.5) + vec2(
                                0.2 * sin(u_time * 0.3),
                                0.2 * cos(u_time * 0.4)
                            );
                            
                            float dist = distance(st, center);
                            
                            // Animated frequency and phase
                            float freq = 10.0 + 5.0 * sin(u_time * 0.1);
                            float phase = u_time * 0.5;
                            
                            // Create concentric rings
                            float rings = sin(dist * freq + phase);
                            
                            // Create binary rings with smoothed edges
                            return smoothstep(0.0, 0.1, rings) * smoothstep(0.0, -0.1, -rings);
                        } else if (type == 14) { // rays
                            // Animated rays from center
                            vec2 center = vec2(0.5) + vec2(
                                0.1 * sin(u_time * 0.3),
                                0.1 * cos(u_time * 0.4)
                            );
                            
                            vec2 toCenter = st - center;
                            float angle = atan(toCenter.y, toCenter.x);
                            
                            // Animated frequency and phase for rays
                            float freq = 8.0;
                            float phase = u_time * 0.5;
                            
                            // Create rays with smooth transitions
                            float rays = sin(angle * freq + phase);
                            
                            // Create binary rays with smoothed edges and distance falloff
                            float dist = length(toCenter);
                            float falloff = 1.0 - smoothstep(0.0, 0.8, dist);
                            
                            return smoothstep(0.0, 0.3, rays) * falloff;
                        } else if (type == 15) { // zigzag
                            // Animated zigzag pattern
                            float freq = 10.0;
                            float angle = 0.5 * sin(u_time * 0.2);
                            
                            // Compute rotated coordinates
                            vec2 rotated = vec2(
                                st.x * cos(angle) - st.y * sin(angle),
                                st.x * sin(angle) + st.y * cos(angle)
                            );
                            
                            // Create two perpendicular triangle waves
                            float zigzag1 = abs(2.0 * fract(rotated.x * freq - u_time * 0.5) - 1.0);
                            float zigzag2 = abs(2.0 * fract(rotated.y * freq + u_time * 0.3) - 1.0);
                            
                            // Combine zigzag patterns
                            float zigzag = min(zigzag1, zigzag2);
                            
                            // Create crisp zigzag lines with varying thickness
                            float thickness = 0.3 + 0.1 * sin(u_time);
                            return step(thickness, zigzag);
                        }
                        
                        return 1.0; // Fallback
                    }
                    
                    // Color mapping function based on the selected color scheme
                    vec3 getColor(float t, int colorScheme) {
                        // Map t from [-1, 1] to [0, 1] for color mapping
                        float normalized = (t + 1.0) * 0.5;
                        
                        vec3 color;
                        
                        // Switch based on color scheme
                        if (colorScheme == 0) { // none (Black & White)
                            return vec3(normalized);
                        }
                        else if (colorScheme == 1) { // blue_red
                            return mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), normalized);
                        }
                        else if (colorScheme == 2) { // viridis
                            vec3 c0 = vec3(0.267, 0.005, 0.329); // #440154
                            vec3 c1 = vec3(0.188, 0.407, 0.553); // #30678D
                            vec3 c2 = vec3(0.208, 0.718, 0.471); // #35B778
                            vec3 c3 = vec3(0.992, 0.906, 0.143); // #FDE724
                            
                            if (normalized < 0.33) {
                                return mix(c0, c1, normalized * 3.0);
                            } else if (normalized < 0.66) {
                                return mix(c1, c2, (normalized - 0.33) * 3.0);
                            } else {
                                return mix(c2, c3, (normalized - 0.66) * 3.0);
                            }
                        }
                        else if (colorScheme == 3) { // plasma
                            vec3 c0 = vec3(0.050, 0.031, 0.529); // #0D0887
                            vec3 c1 = vec3(0.494, 0.012, 0.659); // #7E03A8
                            vec3 c2 = vec3(0.800, 0.275, 0.471); // #CC4678
                            vec3 c3 = vec3(0.973, 0.584, 0.255); // #F89441
                            vec3 c4 = vec3(0.941, 0.973, 0.129); // #F0F921
                            
                            if (normalized < 0.25) {
                                return mix(c0, c1, normalized * 4.0);
                            } else if (normalized < 0.5) {
                                return mix(c1, c2, (normalized - 0.25) * 4.0);
                            } else if (normalized < 0.75) {
                                return mix(c2, c3, (normalized - 0.5) * 4.0);
                            } else {
                                return mix(c3, c4, (normalized - 0.75) * 4.0);
                            }
                        }
                        else if (colorScheme == 4) { // inferno
                            vec3 c0 = vec3(0.001, 0.001, 0.016); // #000004
                            vec3 c1 = vec3(0.259, 0.039, 0.408); // #420A68
                            vec3 c2 = vec3(0.576, 0.149, 0.404); // #932667
                            vec3 c3 = vec3(0.867, 0.318, 0.227); // #DD513A
                            vec3 c4 = vec3(0.988, 0.647, 0.039); // #FCA50A
                            vec3 c5 = vec3(0.988, 1.000, 0.643); // #FCFFA4
                            
                            if (normalized < 0.2) {
                                return mix(c0, c1, normalized * 5.0);
                            } else if (normalized < 0.4) {
                                return mix(c1, c2, (normalized - 0.2) * 5.0);
                            } else if (normalized < 0.6) {
                                return mix(c2, c3, (normalized - 0.4) * 5.0);
                            } else if (normalized < 0.8) {
                                return mix(c3, c4, (normalized - 0.6) * 5.0);
                            } else {
                                return mix(c4, c5, (normalized - 0.8) * 5.0);
                            }
                        }
                        else if (colorScheme == 5) { // magma
                            vec3 c0 = vec3(0.001, 0.001, 0.016); // #000004
                            vec3 c1 = vec3(0.231, 0.059, 0.439); // #3B0F70
                            vec3 c2 = vec3(0.549, 0.161, 0.506); // #8C2981
                            vec3 c3 = vec3(0.871, 0.288, 0.408); // #DE4968
                            vec3 c4 = vec3(0.996, 0.624, 0.427); // #FE9F6D
                            vec3 c5 = vec3(0.988, 0.992, 0.749); // #FCFDBF
                            
                            if (normalized < 0.2) {
                                return mix(c0, c1, normalized * 5.0);
                            } else if (normalized < 0.4) {
                                return mix(c1, c2, (normalized - 0.2) * 5.0);
                            } else if (normalized < 0.6) {
                                return mix(c2, c3, (normalized - 0.4) * 5.0);
                            } else if (normalized < 0.8) {
                                return mix(c3, c4, (normalized - 0.6) * 5.0);
                            } else {
                                return mix(c4, c5, (normalized - 0.8) * 5.0);
                            }
                        } 
                        else if (colorScheme == 6) { // turbo
                            vec3 c0 = vec3(0.188, 0.071, 0.235); // #30123b
                            vec3 c1 = vec3(0.275, 0.408, 0.859); // #4669db
                            vec3 c2 = vec3(0.149, 0.749, 0.549); // #26bf8c
                            vec3 c3 = vec3(0.831, 1.000, 0.314); // #d4ff50
                            vec3 c4 = vec3(0.980, 0.718, 0.298); // #fab74c
                            vec3 c5 = vec3(0.729, 0.004, 0.000); // #ba0100
                            
                            if (normalized < 0.2) {
                                return mix(c0, c1, normalized * 5.0);
                            } else if (normalized < 0.4) {
                                return mix(c1, c2, (normalized - 0.2) * 5.0);
                            } else if (normalized < 0.6) {
                                return mix(c2, c3, (normalized - 0.4) * 5.0);
                            } else if (normalized < 0.8) {
                                return mix(c3, c4, (normalized - 0.6) * 5.0);
                            } else {
                                return mix(c4, c5, (normalized - 0.8) * 5.0);
                            }
                        }
                        else if (colorScheme == 7) { // jet
                            vec3 c0 = vec3(0.000, 0.000, 0.498); // #00007f
                            vec3 c1 = vec3(0.000, 0.000, 1.000); // #0000ff
                            vec3 c2 = vec3(0.000, 1.000, 1.000); // #00ffff
                            vec3 c3 = vec3(1.000, 1.000, 0.000); // #ffff00
                            vec3 c4 = vec3(1.000, 0.000, 0.000); // #ff0000
                            vec3 c5 = vec3(0.498, 0.000, 0.000); // #7f0000
                            
                            if (normalized < 0.2) {
                                return mix(c0, c1, normalized * 5.0);
                            } else if (normalized < 0.4) {
                                return mix(c1, c2, (normalized - 0.2) * 5.0);
                            } else if (normalized < 0.6) {
                                return mix(c2, c3, (normalized - 0.4) * 5.0);
                            } else if (normalized < 0.8) {
                                return mix(c3, c4, (normalized - 0.6) * 5.0);
                            } else {
                                return mix(c4, c5, (normalized - 0.8) * 5.0);
                            }
                        }
                        else if (colorScheme == 8) { // rainbow
                            vec3 c0 = vec3(0.431, 0.251, 0.667); // #6e40aa
                            vec3 c1 = vec3(0.075, 0.600, 0.851); // #1399d9
                            vec3 c2 = vec3(0.122, 0.745, 0.243); // #1fbe3e
                            vec3 c3 = vec3(0.816, 0.757, 0.004); // #d0c101
                            vec3 c4 = vec3(0.694, 0.027, 0.478); // #b1077a
                            
                            if (normalized < 0.25) {
                                return mix(c0, c1, normalized * 4.0);
                            } else if (normalized < 0.5) {
                                return mix(c1, c2, (normalized - 0.25) * 4.0);
                            } else if (normalized < 0.75) {
                                return mix(c2, c3, (normalized - 0.5) * 4.0);
                            } else {
                                return mix(c3, c4, (normalized - 0.75) * 4.0);
                            }
                        }
                        else if (colorScheme == 9) { // cool
                            vec3 c0 = vec3(0.000, 1.000, 1.000); // #00ffff
                            vec3 c1 = vec3(1.000, 0.000, 1.000); // #ff00ff
                            
                            return mix(c0, c1, normalized);
                        }
                        else if (colorScheme == 10) { // hot
                            vec3 c0 = vec3(0.000, 0.000, 0.000); // #000000
                            vec3 c1 = vec3(1.000, 0.000, 0.000); // #ff0000
                            vec3 c2 = vec3(1.000, 1.000, 0.000); // #ffff00
                            vec3 c3 = vec3(1.000, 1.000, 1.000); // #ffffff
                            
                            if (normalized < 0.33) {
                                return mix(c0, c1, normalized * 3.0);
                            } else if (normalized < 0.66) {
                                return mix(c1, c2, (normalized - 0.33) * 3.0);
                            } else {
                                return mix(c2, c3, (normalized - 0.66) * 3.0);
                            }
                        }
                        else if (colorScheme == 11) { // parula
                            vec3 c0 = vec3(0.208, 0.165, 0.529); // #352a87
                            vec3 c1 = vec3(0.059, 0.361, 0.867); // #0f5cdd
                            vec3 c2 = vec3(0.000, 0.710, 0.651); // #00b5a6
                            vec3 c3 = vec3(1.000, 0.765, 0.216); // #ffc337
                            vec3 c4 = vec3(0.988, 0.996, 0.643); // #fcfea4
                            
                            if (normalized < 0.25) {
                                return mix(c0, c1, normalized * 4.0);
                            } else if (normalized < 0.5) {
                                return mix(c1, c2, (normalized - 0.25) * 4.0);
                            } else if (normalized < 0.75) {
                                return mix(c2, c3, (normalized - 0.5) * 4.0);
                            } else {
                                return mix(c3, c4, (normalized - 0.75) * 4.0);
                            }
                        }
                        else if (colorScheme == 12) { // hsv
                            // HSV color wheel implemented directly
                            float h = normalized * 6.0;
                            int i = int(floor(h));
                            float f = h - float(i);
                            
                            float v = 1.0;
                            float s = 1.0;
                            float p = v * (1.0 - s);
                            float q = v * (1.0 - s * f);
                            float t = v * (1.0 - s * (1.0 - f));
                            
                            if (i == 0) return vec3(v, t, p);
                            else if (i == 1) return vec3(q, v, p);
                            else if (i == 2) return vec3(p, v, t);
                            else if (i == 3) return vec3(p, q, v);
                            else if (i == 4) return vec3(t, p, v);
                            else return vec3(v, p, q);
                        }
                        else if (colorScheme == 13) { // autumn
                            vec3 c0 = vec3(1.000, 0.000, 0.000); // #ff0000
                            vec3 c1 = vec3(1.000, 1.000, 0.000); // #ffff00
                            
                            return mix(c0, c1, normalized);
                        }
                        else if (colorScheme == 14) { // winter
                            vec3 c0 = vec3(0.000, 0.000, 1.000); // #0000ff
                            vec3 c1 = vec3(0.000, 1.000, 1.000); // #00ffff
                            
                            return mix(c0, c1, normalized);
                        }
                        else if (colorScheme == 15) { // spring
                            vec3 c0 = vec3(1.000, 0.000, 1.000); // #ff00ff
                            vec3 c1 = vec3(1.000, 1.000, 0.000); // #ffff00
                            
                            return mix(c0, c1, normalized);
                        }
                        else if (colorScheme == 16) { // summer
                            vec3 c0 = vec3(0.000, 0.502, 0.400); // #008066
                            vec3 c1 = vec3(1.000, 1.000, 0.400); // #ffff66
                            
                            return mix(c0, c1, normalized);
                        }
                        else if (colorScheme == 17) { // copper
                            vec3 c0 = vec3(0.000, 0.000, 0.000); // #000000
                            vec3 c1 = vec3(1.000, 0.600, 0.400); // #ff9966
                            
                            return mix(c0, c1, normalized);
                        }
                        else if (colorScheme == 18) { // pink
                            vec3 c0 = vec3(0.051, 0.051, 0.051); // #0d0d0d
                            vec3 c1 = vec3(1.000, 0.000, 1.000); // #ff00ff
                            vec3 c2 = vec3(1.000, 1.000, 1.000); // #ffffff
                            
                            if (normalized < 0.5) {
                                return mix(c0, c1, normalized * 2.0);
                            } else {
                                return mix(c1, c2, (normalized - 0.5) * 2.0);
                            }
                        }
                        else if (colorScheme == 19) { // bone
                            vec3 c0 = vec3(0.000, 0.000, 0.000); // #000000
                            vec3 c1 = vec3(0.329, 0.329, 0.455); // #545474
                            vec3 c2 = vec3(0.627, 0.757, 0.757); // #a0c1c1
                            vec3 c3 = vec3(1.000, 1.000, 1.000); // #ffffff
                            
                            if (normalized < 0.33) {
                                return mix(c0, c1, normalized * 3.0);
                            } else if (normalized < 0.66) {
                                return mix(c1, c2, (normalized - 0.33) * 3.0);
                            } else {
                                return mix(c2, c3, (normalized - 0.66) * 3.0);
                            }
                        }
                        else if (colorScheme == 20) { // ocean
                            vec3 c0 = vec3(0.000, 0.000, 0.000); // #000000
                            vec3 c1 = vec3(0.000, 0.000, 0.600); // #000099
                            vec3 c2 = vec3(0.000, 0.600, 1.000); // #0099ff
                            vec3 c3 = vec3(0.600, 1.000, 1.000); // #99ffff
                            
                            if (normalized < 0.33) {
                                return mix(c0, c1, normalized * 3.0);
                            } else if (normalized < 0.66) {
                                return mix(c1, c2, (normalized - 0.33) * 3.0);
                            } else {
                                return mix(c2, c3, (normalized - 0.66) * 3.0);
                            }
                        }
                        else if (colorScheme == 21) { // terrain
                            vec3 c0 = vec3(0.200, 0.200, 0.600); // #333399
                            vec3 c1 = vec3(0.000, 0.800, 0.400); // #00cc66
                            vec3 c2 = vec3(1.000, 0.800, 0.000); // #ffcc00
                            vec3 c3 = vec3(1.000, 1.000, 1.000); // #ffffff
                            
                            if (normalized < 0.33) {
                                return mix(c0, c1, normalized * 3.0);
                            } else if (normalized < 0.66) {
                                return mix(c1, c2, (normalized - 0.33) * 3.0);
                            } else {
                                return mix(c2, c3, (normalized - 0.66) * 3.0);
                            }
                        }
                        else if (colorScheme == 22) { // neon
                            vec3 c0 = vec3(1.000, 0.000, 1.000); // #ff00ff
                            vec3 c1 = vec3(0.000, 1.000, 1.000); // #00ffff
                            vec3 c2 = vec3(1.000, 1.000, 0.000); // #ffff00
                            
                            if (normalized < 0.5) {
                                return mix(c0, c1, normalized * 2.0);
                            } else {
                                return mix(c1, c2, (normalized - 0.5) * 2.0);
                            }
                        }
                        else if (colorScheme == 23) { // fire
                            vec3 c0 = vec3(0.000, 0.000, 0.000); // #000000
                            vec3 c1 = vec3(1.000, 0.000, 0.000); // #ff0000
                            vec3 c2 = vec3(1.000, 1.000, 0.000); // #ffff00
                            vec3 c3 = vec3(1.000, 1.000, 1.000); // #ffffff
                            
                            if (normalized < 0.33) {
                                return mix(c0, c1, normalized * 3.0);
                            } else if (normalized < 0.66) {
                                return mix(c1, c2, (normalized - 0.33) * 3.0);
                            } else {
                                return mix(c2, c3, (normalized - 0.66) * 3.0);
                            }
                        }
                        else {
                            // Default to blue-red gradient
                            return mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), normalized);
                        }
                    }
                `;
export const SHADER_SOURCES = {
    "domain_warp": `
                    // More precise domain warping implementation
                    float fbm_base(vec2 p) {
                        // Standard FBM using user-controlled octaves
                        float sum = 0.0;
                        float amp = 1.0;
                        float freq = 1.0;
                        
                        for (int i = 0; i < 8; i++) {
                            if (float(i) >= u_octaves) break;
                            sum += amp * snoise(p * freq);
                            freq *= 2.0;
                            amp *= 0.5;
                        }
                        
                        return sum;
                    }
                    
                    // Improved domain warping with different warp types
                    float domainWarp(vec2 p, int warpType) {
                        // Note: Resolution is now controlled by canvas size
                        
                        // Base unwarped coordinates scaled by user scale parameter
                        vec2 p0 = p * u_scale;
                        
                        // Basic domain warp using noise as a displacement vector field
                        if (warpType == 0) {
                            // Create distortion vector
                            float angle = u_time * 0.1;
                            vec2 d = vec2(cos(angle), sin(angle));
                            
                            // Generate the warp field
                            float warpNoise1 = snoise(p0 * 0.5); 
                            float warpNoise2 = snoise(p0 * 0.5 + vec2(5.2, 1.3));
                            
                            // Create the warp displacement
                            vec2 warpVec = vec2(warpNoise1, warpNoise2) * u_warpStrength;
                            
                            // Apply the warp
                            vec2 warped = p0 + warpVec;
                            
                            // Sample the base pattern with warped coordinates
                            return snoise(warped);
                        }
                        // Fractal domain warp (warp the warp)
                        else if (warpType == 1) {
                            // Apply progressive warping with multiple layers
                            vec2 warped = p0;
                            
                            // Progressive warp layers
                            for (int i = 0; i < 3; i++) {
                                // Scale decreases for each iteration
                                float warpScale = 1.0 / pow(2.0, float(i));
                                
                                // Generate warp vectors
                                float warpNoise1 = snoise(warped * warpScale + vec2(u_time * 0.05, 0.0));
                                float warpNoise2 = snoise(warped * warpScale + vec2(0.0, u_time * 0.05) + vec2(43.13, 17.21));
                                
                                // Apply warp with decreasing strength for each iteration
                                float warpFactor = u_warpStrength * warpScale;
                                warped += vec2(warpNoise1, warpNoise2) * warpFactor;
                            }
                            
                            // Sample final fbm with fully warped coordinates
                            return fbm_base(warped);
                        }
                        // Advanced vector field warping
                        else if (warpType == 2) {
                            // Use a separate noise function to determine flow direction
                            float flowNoise = snoise(p0 * 0.2 + vec2(u_time * 0.1, 0.0));
                            float flowAngle = flowNoise * 6.28318530718; // Map to full rotation
                            
                            // Create flow direction vector
                            vec2 flowDir = vec2(cos(flowAngle), sin(flowAngle));
                            
                            // Apply directional warp
                            vec2 warped = p0 + flowDir * u_warpStrength * snoise(p0 * 0.4);
                            
                            // Add secondary orthogonal flow
                            vec2 perpDir = vec2(-flowDir.y, flowDir.x); // Perpendicular vector
                            warped += perpDir * u_warpStrength * 0.5 * snoise(p0 * 0.3 + vec2(10.0, 20.0));
                            
                            return fbm_base(warped);
                        }
                        // Swirl warp
                        else {
                            // Calculate distance from center
                            vec2 centered = p0 - 0.5;
                            float dist = length(centered);
                            
                            // Calculate angle based on distance
                            float angle = u_time + dist * u_warpStrength * 10.0;
                            
                            // Create rotation matrix
                            float s = sin(angle);
                            float c = cos(angle);
                            mat2 rot = mat2(c, -s, s, c);
                            
                            // Apply rotational warping
                            vec2 warped = rot * centered + 0.5;
                            
                            return fbm_base(warped);
                        }
                    }
                    
                    float fbm(vec2 p) {
                        int warpType = int(mod(u_octaves, 4.0));
                        float result = domainWarp(p, warpType);
                        
                        // Apply user's phase shift to control contrast and distribution
                        float contrast = 1.0 + u_phaseShift;
                        result *= contrast;
                        
                        // Ensure output is in valid [-1,1] range
                        return clamp(result, -1.0, 1.0);
                    }
                `,
    "tensor_field": `
                    // Improved tensor field implementation with better mathematical representation
                    
                    // Helper function for computing tensor eigenvectors and eigenvalues
                    void computeTensorProperties(vec2 p, out float magnitude1, out float magnitude2, 
                                               out vec2 direction1, out vec2 direction2) {
                        // Note: Resolution is now controlled by canvas size
                        
                        // Generate a tensor field using noise gradients
                        vec2 offset = vec2(u_time * 0.05);
                        vec2 p1 = p * u_scale + offset;
                        
                        // Apply warp to coordinates if warp strength is non-zero
                        if (u_warpStrength > 0.0) {
                            // Generate warp field based on noise
                            float warpNoise1 = snoise(p1 * 0.3 + vec2(0.0, 1.0));
                            float warpNoise2 = snoise(p1 * 0.3 + vec2(1.0, 0.0));
                            
                            // Apply warp to coordinates
                            p1 += vec2(warpNoise1, warpNoise2) * u_warpStrength;
                        }
                        
                        // Use noise derivatives to generate tensor field
                        float eps = 0.01;
                        
                        // Compute approximate derivatives of noise field
                        float n00 = snoise(p1);
                        float n10 = snoise(p1 + vec2(eps, 0.0));
                        float n01 = snoise(p1 + vec2(0.0, eps));
                        float n11 = snoise(p1 + vec2(eps, eps));
                        
                        // Calculate derivatives (gradient components)
                        float dx = (n10 - n00) / eps;
                        float dy = (n01 - n00) / eps;
                        
                        // Second order derivatives for tensor components
                        float dxx = (n10 - 2.0 * n00 + snoise(p1 - vec2(eps, 0.0))) / (eps * eps);
                        float dyy = (n01 - 2.0 * n00 + snoise(p1 - vec2(0.0, eps))) / (eps * eps);
                        float dxy = (n11 - n10 - n01 + n00) / (eps * eps);
                        
                        // Construct tensor matrix components
                        float T00 = dxx;
                        float T01 = dxy;
                        float T10 = dxy;
                        float T11 = dyy;
                        
                        // Calculate eigenvalues
                        float trace = T00 + T11;
                        float det = T00 * T11 - T01 * T10;
                        float discriminant = sqrt(trace * trace - 4.0 * det);
                        
                        // Two eigenvalues
                        magnitude1 = (trace + discriminant) * 0.5;
                        magnitude2 = (trace - discriminant) * 0.5;
                        
                        // Calculate first eigenvector
                        if (abs(T01) > 0.0001) {
                            direction1 = normalize(vec2(T01, magnitude1 - T00));
                        } else if (abs(T10) > 0.0001) {
                            direction1 = normalize(vec2(magnitude1 - T11, T10));
                        } else {
                            // Diagonal tensor
                            direction1 = vec2(1.0, 0.0);
                        }
                        
                        // Second eigenvector is perpendicular to first
                        direction2 = vec2(-direction1.y, direction1.x);
                    }
                    
                    // Improved tensor field visualization
                    float tensorField(vec2 p) {
                        // Calculate tensor field components
                        float lambda1, lambda2;
                        vec2 v1, v2;
                        computeTensorProperties(p, lambda1, lambda2, v1, v2);
                        
                        // Choose visualization based on octaves
                        int visualizationType = int(mod(u_octaves, 4.0));
                        
                        // Different visualization modes
                        if (visualizationType == 0) {
                            // Eigenvalue visualization - shows magnitude of deformation
                            float maxEig = max(abs(lambda1), abs(lambda2));
                            return clamp(maxEig, -1.0, 1.0);
                        }
                        else if (visualizationType == 1) {
                            // Eigenvector streamlines - shows direction of principal stress
                            
                            // Direction-based visualization with animating flow
                            float lineWidth = 0.08;
                            vec2 st = p;
                            
                            // Calculate distance to streamline along first eigenvector
                            float flowPhase = u_time * 0.2;
                            float t = st.x * v1.x + st.y * v1.y;
                            float streamline1 = abs(fract(t * 5.0 + flowPhase) - 0.5) * 2.0;
                            
                            // Calculate distance to streamline along second eigenvector
                            t = st.x * v2.x + st.y * v2.y;
                            float streamline2 = abs(fract(t * 5.0 - flowPhase) - 0.5) * 2.0;
                            
                            // Combine streamlines
                            float pattern = min(streamline1, streamline2);
                            return 1.0 - smoothstep(0.0, lineWidth, pattern) * 2.0;
                        }
                        else if (visualizationType == 2) {
                            // Hyperstreamlines - thickness varies with eigenvalue magnitude
                            
                            vec2 dir1 = v1 * sign(lambda1);
                            vec2 dir2 = v2 * sign(lambda2);
                            
                            float weight1 = abs(lambda1) / (abs(lambda1) + abs(lambda2) + 0.001);
                            float weight2 = 1.0 - weight1;
                            
                            float angle1 = atan(dir1.y, dir1.x);
                            float angle2 = atan(dir2.y, dir2.x);
                            
                            float t1 = cos(5.0 * (p.x * dir1.x + p.y * dir1.y) + u_time);
                            float t2 = cos(5.0 * (p.x * dir2.x + p.y * dir2.y) - u_time);
                            
                            return (t1 * weight1 + t2 * weight2) * 0.5;
                        }
                        else {
                            // Tensor ellipses
                            
                            // Create an elliptical pattern aligned with eigenvectors and scaled by eigenvalues
                            vec2 centered = p - floor(p * 4.0 + 0.5) / 4.0; // Create grid
                            
                            // Transform point to eigenvector basis
                            float x = dot(centered, v1);
                            float y = dot(centered, v2);
                            
                            // Scale by eigenvalues (normalized to prevent distortion)
                            float maxEig = max(abs(lambda1), abs(lambda2)) + 0.1;
                            float scaledX = x * abs(lambda1) / maxEig;
                            float scaledY = y * abs(lambda2) / maxEig;
                            
                            // Create ellipse
                            float ellipse = length(vec2(scaledX, scaledY));
                            float radius = 0.05;
                            
                            // Animate pulsing ellipses
                            radius *= 1.0 + 0.3 * sin(u_time * 2.0);
                            
                            // Return elliptical pattern
                            return 1.0 - smoothstep(0.0, 0.01, ellipse - radius) * 2.0;
                        }
                    }
                    
                    float fbm(vec2 p) {
                        float result = tensorField(p);
                        
                        // Apply user's phase shift to control contrast and distribution
                        float contrast = 1.0 + u_phaseShift;
                        result *= contrast;
                        
                        // Ensure output is in valid [-1,1] range
                        return clamp(result, -1.0, 1.0);
                    }
                `,
    "curl_noise": `
                    // Compute gradient of scalar field
                    vec2 computeGradient(vec2 p, float epsilon) {
                        // Sample the potential field at nearby points
                        float dx = snoise(vec2(p.x + epsilon, p.y)) - snoise(vec2(p.x - epsilon, p.y));
                        float dy = snoise(vec2(p.x, p.y + epsilon)) - snoise(vec2(p.x, p.y - epsilon));
                        
                        // Normalize by epsilon and return
                        return vec2(dx, dy) / (2.0 * epsilon);
                    }
                    
                    // Compute curl of vector field (z component in 2D)
                    float computeCurl(vec2 p, float epsilon) {
                        // For 2D curl, we need partial derivatives of two potential fields
                        // We'll use two offset perlin noise functions as our potential fields
                        
                        // Sample two potential fields (offset for independence)
                        float pot1_dx = snoise(vec2(p.x + epsilon, p.y)) - snoise(vec2(p.x - epsilon, p.y));
                        float pot1_dy = snoise(vec2(p.x, p.y + epsilon)) - snoise(vec2(p.x, p.y - epsilon));
                        
                        float pot2_dx = snoise(vec2(p.x + epsilon, p.y + 100.0)) - snoise(vec2(p.x - epsilon, p.y + 100.0));
                        float pot2_dy = snoise(vec2(p.x, p.y + epsilon + 100.0)) - snoise(vec2(p.x, p.y - epsilon + 100.0));
                        
                        // Normalize gradients
                        pot1_dx /= (2.0 * epsilon);
                        pot1_dy /= (2.0 * epsilon);
                        pot2_dx /= (2.0 * epsilon);
                        pot2_dy /= (2.0 * epsilon);
                        
                        // Compute curl (cross product in 2D: ∂pot2/∂x - ∂pot1/∂y)
                        return pot2_dx - pot1_dy;
                    }
                    
                    // Get a fluid velocity field based on curl
                    vec2 getVelocityField(vec2 p, float time) {
                        // Use multiple frequencies for more detailed flow
                        vec2 velocity = vec2(0.0);
                        float epsilon = 0.01;
                        
                        // Base frequency
                        float frequency = 1.0;
                        float amplitude = 1.0;
                        
                        for (int i = 0; i < 3; i++) {
                            if (float(i) >= u_octaves) break;
                            
                            // Time-varied position
                            vec2 pos = p * frequency + vec2(time * 0.1 * frequency);
                            
                            // Compute curl at this frequency
                            float curl = computeCurl(pos, epsilon);
                            
                            // Use curl to derive a velocity field
                            // The gradient of the curl gives us a divergence-free field
                            vec2 vel = vec2(
                                computeCurl(pos + vec2(0.0, epsilon), epsilon) - curl,
                                curl - computeCurl(pos + vec2(epsilon, 0.0), epsilon)
                            ) / epsilon;
                            
                            // Add to total velocity
                            velocity += vel * amplitude;
                            
                            // Prepare for next octave
                            frequency *= 2.0;
                            amplitude *= 0.5;
                            epsilon *= 0.5; // Adjust epsilon for higher frequencies
                        }
                        
                        return velocity;
                    }
                    
                    // Advect a property along the velocity field
                    float advect(vec2 p, vec2 velocity, float time, float dt) {
                        // Trace particle backward in time
                        vec2 particlePos = p - velocity * dt;
                        
                        // Sample different noise patterns based on octave setting
                        int patternType = int(mod(u_octaves, 4.0));
                        float result;
                        
                        if (patternType == 0) {
                            // Classic advected noise
                            result = snoise(particlePos * u_scale + vec2(time * 0.2));
                        } 
                        else if (patternType == 1) {
                            // Dye injection visualization
                            float dist = length(fract(particlePos) - 0.5) * 2.0;
                            float spots = smoothstep(0.4, 0.0, dist);
                            result = spots * 2.0 - 1.0; // Remap to [-1, 1]
                        }
                        else if (patternType == 2) {
                            // Flow lines visualization
                            float stream = sin(dot(particlePos, normalize(velocity)) * 10.0 + time);
                            result = stream;
                        }
                        else {
                            // Vorticity visualization (shows rotations in the flow)
                            float vorticity = computeCurl(particlePos * u_scale, 0.01);
                            result = vorticity * 2.0; // Amplify for better visibility
                        }
                        
                        return result;
                    }
                    
                    // Apply the warp control to intensify curl
                    vec2 applyWarpIntensity(vec2 velocity, float warp) {
                        float length = max(length(velocity), 0.001);
                        float logScale = log(length * 9.0 + 1.0) * warp;
                        return normalize(velocity) * logScale;
                    }
                    
                    float fbm(vec2 p) {
                        // Note: Resolution is now controlled by canvas size
                        
                        // Scale coordinates
                        p *= u_scale;
                        
                        // Compute velocity field
                        vec2 velocity = getVelocityField(p, u_time);
                        
                        // Apply warp control to intensify curl
                        velocity = applyWarpIntensity(velocity, u_warpStrength);
                        
                        // Vary the advection time step based on phase shift
                        float dt = mix(0.2, 2.0, u_phaseShift);
                        
                        // Advect along the curl field
                        float result = advect(p, velocity, u_time, dt);
                        
                        // Return noise within proper range
                        return clamp(result, -1.0, 1.0);
                    }
                `
};
export const FRAGMENT_SHADER_FOOTER = `
                    void main() {
                        // Generate noise
                        float noise = fbm(v_texCoord);
                        
                        // Apply shape mask if enabled
                        float shape_mask = applyShapeMask(v_texCoord, u_shapeType);
                        noise = noise * (shape_mask * u_shapeStrength + (1.0 - u_shapeStrength));
                        
                        // Map to [0,1] range for color
                        vec3 color = getColor(noise, u_colorScheme);
                        
                        gl_FragColor = vec4(color * u_intensity, 1.0);
                    }
                `;
//# sourceMappingURL=glsl_shaders.js.map