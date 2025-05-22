// web/noise_visualizer.js
if (!window.NoiseVisualizer) {
    window.NoiseVisualizer = {
        kofiCupImageBitmap: null, // To store the preloaded ImageBitmap object
        kofiImageLoaded: false,
        kofiImageLoadAttempted: false,

        _preloadKofiImage: async function() { // Made async
            if (this.kofiImageLoadAttempted) return;
            this.kofiImageLoadAttempted = true;

            const absolutePath = '/extensions/ComfyUI-ShaderNoiseKsampler/images/kofi_symbol.svg';

            // Attempt 1: Direct Image load
            try {
                console.log("Attempting Ko-fi SVG load using new Image() with direct path (Primary Attempt)...");
                const img = new Image();
                await new Promise((resolve, reject) => {
                    img.onload = () => {
                        this.kofiCupImageBitmap = img; // Store the HTMLImageElement
                        this.kofiImageLoaded = true;
                        console.log("Ko-fi symbol SVG loaded via Image() successfully (Primary).");
                        resolve();
                    };
                    img.onerror = (e) => {
                        console.warn("Primary Ko-fi symbol SVG load via Image() failed. Proceeding to secondary attempt.", e);
                        reject(e); // Reject to proceed to the next try block
                    };
                    img.src = absolutePath;
                });
                return; // Successfully loaded, no need to proceed
            } catch (error) {
                // Error logged by the promise, clear flags before next attempt
                this.kofiImageLoaded = false;
                this.kofiCupImageBitmap = null;
            }

            // Attempt 2: Fetch -> Blob -> Intermediate Image -> ImageBitmap
            let objectURL = null;
            try {
                console.log("Attempting Ko-fi SVG load via Blob -> Image -> ImageBitmap (Secondary Attempt)...");
                const response = await fetch(absolutePath);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status} for ${absolutePath}`);
                }
                const svgText = await response.text();
                const blob = new Blob([svgText], { type: 'image/svg+xml' });
                objectURL = URL.createObjectURL(blob);

                const intermediateImg = new Image();
                await new Promise((resolve, reject) => {
                    intermediateImg.onload = resolve;
                    intermediateImg.onerror = (e) => reject(new Error("Intermediate Image() load failed for SVG blob (Secondary)."));
                    intermediateImg.src = objectURL;
                });
                
                this.kofiCupImageBitmap = await createImageBitmap(intermediateImg);
                this.kofiImageLoaded = true;
                console.log("Ko-fi symbol SVG processed to ImageBitmap successfully (Secondary).");

            } catch (error) {
                this.kofiImageLoaded = false;
                this.kofiCupImageBitmap = null;
                console.error("All Ko-fi SVG load attempts failed: ", error, ". Will use fallback drawing.");
            } finally {
                if (objectURL) {
                    URL.revokeObjectURL(objectURL); // Clean up the object URL
                }
            }
            // If all attempts fail, kofiImageLoaded remains false, and manual drawing will be used.
        },

        renderAllInModal: async function(modalContentElement) { // Made async
            if (!this.kofiImageLoadAttempted) {
                await this._preloadKofiImage(); // Ensure preloading is complete
            }

            const noiseCanvases = modalContentElement.querySelectorAll('.noise-canvas');
            noiseCanvases.forEach(canvasDiv => {
                const canvasId = canvasDiv.id;
                if (!canvasId || !canvasId.startsWith("noise-canvas-")) return;
                const noiseType = canvasId.substring("noise-canvas-".length);
                
                let canvas = canvasDiv.querySelector('canvas');
                if (!canvas) {
                    canvas = document.createElement('canvas');
                    canvas.width = 130;
                    canvas.height = 130;
                    canvasDiv.innerHTML = '';
                    canvasDiv.appendChild(canvas);
                }

                const functionNameSuffix = noiseType.split('_').map(word => {
                    if (word.toLowerCase() === 'fbm') return 'FBM';
                    return word.charAt(0).toUpperCase() + word.slice(1);
                }).join('');
                const renderFunctionName = `render${functionNameSuffix.replace("3d", "3D")}`;

                if (typeof this[renderFunctionName] === 'function') {
                    this[renderFunctionName](canvas);
                } else {
                    console.warn("No renderer function found:", renderFunctionName, "for noise type:", noiseType);
                    this.renderPlaceholder(canvas, noiseType.replace(/_/g, ' '));
                }
            });
            
            // Add rendering for mask canvases
            const maskCanvases = modalContentElement.querySelectorAll('.mask-canvas');
            maskCanvases.forEach(canvasDiv => {
                const canvasId = canvasDiv.id;
                if (!canvasId || !canvasId.startsWith("mask-canvas-")) return;
                const maskType = canvasId.substring("mask-canvas-".length);
                
                let canvas = canvasDiv.querySelector('canvas');
                if (!canvas) {
                    canvas = document.createElement('canvas');
                    canvas.width = 100;
                    canvas.height = 70;
                    canvasDiv.innerHTML = '';
                    canvasDiv.appendChild(canvas);
                }

                const functionNameSuffix = maskType.split('_').map(word => {
                    return word.charAt(0).toUpperCase() + word.slice(1);
                }).join('');
                const renderFunctionName = `renderMask${functionNameSuffix}`;

                if (typeof this[renderFunctionName] === 'function') {
                    this[renderFunctionName](canvas);
                } else {
                    console.warn("No renderer function found:", renderFunctionName, "for mask type:", maskType);
                    this.renderMaskPlaceholder(canvas, maskType.replace(/_/g, ' '));
                }
            });
            
            // Add rendering for color scheme swatches
            const colorSwatches = modalContentElement.querySelectorAll('.color-swatch');
            colorSwatches.forEach(swatchDiv => {
                // Skip if it already has a gradient background
                if (swatchDiv.style.background) return;
                
                // Extract color scheme name from the div's text content
                const schemeName = swatchDiv.textContent.trim().toLowerCase();
                const renderFunctionName = `renderColor${schemeName.charAt(0).toUpperCase() + schemeName.slice(1)}`;
                
                if (typeof this[renderFunctionName] === 'function') {
                    this[renderFunctionName](swatchDiv);
                } else {
                    // For schemes without specific renderers, use the style already in the HTML
                    console.log("Using existing style for color scheme:", schemeName);
                }
            });

            // Add rendering for the temporal animation demo
            const animationDemoContainer = modalContentElement.querySelector('#animation-demo-placeholder');
            if (animationDemoContainer) {
                let canvas = animationDemoContainer.querySelector('canvas');
                if (!canvas) {
                    canvas = document.createElement('canvas');
                    // Set canvas dimensions based on its container, respecting aspect ratio if possible
                    // For simplicity, let's use fixed dimensions or dimensions derived from container style
                    const containerStyle = getComputedStyle(animationDemoContainer);
                    canvas.width = parseInt(containerStyle.width) || 300; // Default if not specified
                    canvas.height = parseInt(containerStyle.height) || 200; // Default if not specified
                    animationDemoContainer.innerHTML = ''; // Clear placeholder text/content
                    animationDemoContainer.appendChild(canvas);
                    animationDemoContainer.style.display = 'block'; // Ensure container is block for canvas
                }
                if (typeof this.renderTemporalAnimation === 'function') {
                    this.renderTemporalAnimation(canvas);
                } else {
                    console.warn("renderTemporalAnimation function not found in NoiseVisualizer.");
                    this.renderMaskPlaceholder(canvas, "Animation Demo"); // Fallback placeholder
                }
            }

            // Add rendering for the intro noise demo
            const introNoiseDemoContainer = modalContentElement.querySelector('#intro-noise-demo');
            if (introNoiseDemoContainer) {
                let canvas = introNoiseDemoContainer.querySelector('canvas');
                if (!canvas) {
                    canvas = document.createElement('canvas');
                    const containerStyle = getComputedStyle(introNoiseDemoContainer);
                    canvas.width = parseInt(containerStyle.width) || 300;
                    canvas.height = parseInt(containerStyle.height) || 350; // Match defined height in CSS
                    introNoiseDemoContainer.innerHTML = ''; // Clear placeholder
                    introNoiseDemoContainer.appendChild(canvas);
                    introNoiseDemoContainer.style.display = 'block';
                }
                if (typeof this.renderIntroNoiseDemo === 'function') {
                    this.renderIntroNoiseDemo(canvas);
                } else {
                    console.warn("renderIntroNoiseDemo function not found in NoiseVisualizer.");
                    this.renderMaskPlaceholder(canvas, "Intro Noise Demo");
                }
            }
        },

        _clearCanvas: function(canvas, backgroundColor = '#1a1a2e') {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = backgroundColor;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            return ctx;
        },

        _drawKofiIcon: function(ctx) {
            const iconSize = 18; 
            const padding = 3;
            const x = ctx.canvas.width - iconSize - padding;
            const y = ctx.canvas.height - iconSize - padding;

            // Now checks for kofiCupImageBitmap
            if (this.kofiImageLoaded && this.kofiCupImageBitmap) { 
                try {
                    // drawImage can take an ImageBitmap, HTMLImageElement, HTMLVideoElement, HTMLCanvasElement, etc.
                    ctx.drawImage(this.kofiCupImageBitmap, x, y, iconSize, iconSize);
                } catch (e) {
                    console.error("Error drawing local Ko-fi SVG ImageBitmap, falling back to manual draw:", e);
                    this._drawManualKofiCup(ctx, x, y, iconSize); 
                }
            } else {
                this._drawManualKofiCup(ctx, x, y, iconSize);
            }
        },

        _drawManualKofiCup: function(ctx, x, y, iconSize) {
            ctx.save();
            ctx.fillStyle = '#FFDD99'; 
            ctx.strokeStyle = '#D2B48C'; 
            ctx.lineWidth = 1;

            ctx.beginPath();
            ctx.moveTo(x, y + iconSize * 0.2);
            ctx.lineTo(x, y + iconSize * 0.9);
            ctx.quadraticCurveTo(x + iconSize * 0.5, y + iconSize * 1.1, x + iconSize, y + iconSize * 0.9);
            ctx.lineTo(x + iconSize, y + iconSize * 0.2);
            ctx.quadraticCurveTo(x + iconSize * 0.5, y, x, y + iconSize * 0.2); 
            ctx.fill();
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(x + iconSize * 0.9, y + iconSize * 0.5, iconSize * 0.25, -Math.PI/2, Math.PI/2);
            ctx.stroke();
            
            ctx.strokeStyle = '#FFFFFF'; 
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(x + iconSize * 0.3, y + iconSize * 0.1);
            ctx.quadraticCurveTo(x + iconSize * 0.2, y - iconSize * 0.2, x + iconSize * 0.4, y - iconSize * 0.3);
            ctx.moveTo(x + iconSize * 0.6, y + iconSize * 0.05);
            ctx.quadraticCurveTo(x + iconSize * 0.5, y - iconSize * 0.3, x + iconSize * 0.7, y - iconSize * 0.4);
            ctx.stroke();

            ctx.restore();
        },

        renderPlaceholder: function(canvas, noiseName) {
            const ctx = this._clearCanvas(canvas, '#2c2c34');
            const nameToShow = noiseName.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
            
            ctx.fillStyle = '#e0e0e8';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.font = 'bold 12px "Segoe UI", Tahoma, Geneva, Verdana, sans-serif';
            
            const words = nameToShow.split(' ');
            if (words.length > 2) {
                 ctx.fillText(words.slice(0,2).join(' '), canvas.width / 2, canvas.height / 2 - 7);
                 ctx.fillText(words.slice(2).join(' '), canvas.width / 2, canvas.height / 2 + 7);
            } else {
                ctx.fillText(nameToShow, canvas.width / 2, canvas.height / 2);
            }
            ctx.strokeStyle = '#555';
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },

        // --- Visualization Functions ---
        renderTensorField: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            ctx.strokeStyle = 'rgba(138, 43, 226, 0.7)';
            ctx.lineWidth = 1;
            for (let i = 0; i < 10; i++) {
                ctx.beginPath();
                const y = (i + 0.5) * (canvas.height / 10);
                ctx.moveTo(0, y);
                for (let x = 0; x <= canvas.width; x += 5) {
                    const angle = Math.sin(x * 0.1 + y * 0.05) * Math.PI * 0.25;
                    ctx.lineTo(x + Math.cos(angle) * 5, y + Math.sin(angle) * 5);
                    ctx.moveTo(x,y);
                }
                ctx.stroke();
            }
        },

        renderCellular: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            const numCells = 8;
            const points = [];
            for (let i = 0; i < numCells; i++) {
                points.push({x: Math.random() * canvas.width, y: Math.random() * canvas.height});
            }
            for (let x = 0; x < canvas.width; x += 4) {
                for (let y = 0; y < canvas.height; y += 4) {
                    let minDist = Infinity;
                    points.forEach(p => { minDist = Math.min(minDist, Math.hypot(p.x - x, p.y - y)); });
                    const intensity = Math.min(255, minDist * 2);
                    ctx.fillStyle = `rgb(${intensity},${intensity},${intensity + 50})`;
                    ctx.fillRect(x, y, 4, 4);
                }
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d'));
        },

        renderDomainWarp: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            ctx.lineWidth = 1.5;
            for (let i = 0; i < 20; i++) {
                ctx.beginPath();
                ctx.moveTo(Math.random() * canvas.width, Math.random() * canvas.height);
                ctx.strokeStyle = `rgba(${100 + Math.random()*155}, ${100 + Math.random()*155}, ${200 + Math.random()*55}, 0.6)`;
                for (let j = 0; j < 5; j++) {
                    const x = Math.random() * canvas.width;
                    const y = Math.random() * canvas.height;
                    const cp1x = Math.random() * canvas.width;
                    const cp1y = Math.random() * canvas.height;
                    const cp2x = Math.random() * canvas.width;
                    const cp2y = Math.random() * canvas.height;
                    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x, y);
                }
                ctx.stroke();
            }
        },

        renderFractal: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            const scale = 0.05;
            const octaves = 4;
            const persistence = 0.5;
            const lacunarity = 2.0;

            for (let x = 0; x < canvas.width; x += 2) {
                for (let y = 0; y < canvas.height; y += 2) {
                    let total = 0;
                    let frequency = 1;
                    let amplitude = 1;
                    let maxValue = 0;

                    for (let i = 0; i < octaves; i++) {
                        // Using a simple pseudo-random noise based on sin/cos
                        // A more robust noise function (like Simplex or Perlin) would be better for true FBM.
                        // This is a visual approximation for the small canvas.
                        const noiseVal = (Math.sin(x * scale * frequency + y * scale * frequency * 0.7) +
                                          Math.cos(y * scale * frequency - x * scale * frequency * 0.3)) / 2;
                        total += noiseVal * amplitude;
                        
                        maxValue += amplitude;
                        amplitude *= persistence;
                        frequency *= lacunarity;
                    }
                    
                    const normalizedTotal = (total / maxValue + 1) / 2; // Normalize to 0-1 range
                    const colorVal = Math.floor(normalizedTotal * 200) + 55; // Map to 55-255 range
                    
                    // A more neutral, less blue color scheme
                    ctx.fillStyle = `rgb(${colorVal}, ${colorVal * 0.95}, ${colorVal * 0.9})`; 
                    ctx.fillRect(x, y, 2, 2);
                }
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d'));
        },

        renderPerlin: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            for (let x = 0; x < canvas.width; x+=3) {
                for (let y = 0; y < canvas.height; y+=3) {
                    const noiseVal = (Math.sin(x * 0.05 + Math.cos(y*0.08)) + Math.cos(y * 0.06)) / 2;
                    const intensity = (noiseVal + 1) / 2 * 200 + 55;
                    ctx.fillStyle = `rgba(${intensity * 0.8}, ${intensity * 0.9}, ${intensity}, 1)`;
                    ctx.fillRect(x, y, 3, 3);
                }
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d'));
        },

        renderWaves: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            ctx.strokeStyle = 'rgba(52, 152, 219, 0.7)';
            ctx.lineWidth = 1.5;
            for (let i = 0; i < 15; i++) {
                ctx.beginPath();
                const startY = i * (canvas.height / 15);
                ctx.moveTo(0, startY);
                for (let x = 0; x <= canvas.width; x += 5) {
                    const yOffset = Math.sin(x * 0.1 + i * 0.5) * 10;
                    ctx.lineTo(x, startY + yOffset);
                }
                ctx.stroke();
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d'));
        },

        renderGaussian: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            for (let i = 0; i < 10000; i++) {
                const x = Math.random() * canvas.width;
                const y = Math.random() * canvas.height;
                const intensity = Math.floor(Math.random() * 100) + 100;
                ctx.fillStyle = `rgba(${intensity * 0.8}, ${intensity * 0.9}, ${intensity}, ${Math.random() * 0.5 + 0.1})`;
                ctx.fillRect(x - 1, y - 1, 2, 2);
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d'));
        },

        renderHeterogeneousFBM: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            for (let region = 0; region < 3; region++) {
                const regionX = (canvas.width / 3) * region;
                const regionW = canvas.width / 3;
                const step = 2 + region * 2;
                for (let x = 0; x < regionW; x += step) {
                    for (let y = 0; y < canvas.height; y += step) {
                        const noiseVal = Math.random(); 
                        const intensity = noiseVal * 255;
                        ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
                        ctx.fillRect(regionX + x, y, step, step);
                    }
                }
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d'));
        },

        renderInterference: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            ctx.lineWidth = 0.5;
            const sources = [
                {x: canvas.width * 0.2, y: canvas.height * 0.3, phase: 0},
                {x: canvas.width * 0.8, y: canvas.height * 0.7, phase: Math.PI/2}
            ];
            for (let x = 0; x < canvas.width; x += 3) {
                for (let y = 0; y < canvas.height; y += 3) {
                    let sum = 0;
                    sources.forEach(s => {
                        const dist = Math.hypot(s.x - x, s.y - y);
                        sum += Math.sin(dist * 0.1 + s.phase);
                    });
                    const intensity = (sum / sources.length + 1) / 2 * 255;
                    ctx.fillStyle = `rgb(${intensity * 0.7}, ${intensity}, ${intensity * 0.8})`;
                    ctx.fillRect(x, y, 3, 3);
                }
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d'));
        },

        renderSpectral: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            for (let i = 0; i < canvas.height; i+=4) {
                const freqComponent = Math.sin(i * 0.1) * 0.3 + Math.cos(i*0.05) * 0.3 + Math.random()*0.4;
                const intensity = (freqComponent + 1)/2 * 255;
                ctx.fillStyle = `rgb(${intensity}, ${intensity*0.8}, ${intensity*0.6})`;
                ctx.fillRect(0, i, canvas.width, 4);
                ctx.strokeStyle = 'rgba(0,0,0,0.2)';
                ctx.strokeRect(0, i, canvas.width, 4);
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d'));
        },

        renderProjection3D: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            for (let i = 0; i < 50; i++) {
                const z = Math.random();
                const x = (Math.random() - 0.5) * canvas.width * (1 + z) + canvas.width / 2;
                const y = (Math.random() - 0.5) * canvas.height * (1 + z) + canvas.height / 2;
                const size = (1 - z) * 10 + 2;
                const opacity = (1 - z) * 0.7 + 0.1;
                ctx.fillStyle = `rgba(${150 + z*105}, ${150 + z*105}, ${200 + z*55}, ${opacity})`;
                ctx.beginPath();
                ctx.arc(x, y, size, 0, Math.PI * 2);
                ctx.fill();
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d'));
        },

        renderCurlNoise: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            ctx.lineWidth = 1;
            for (let i = 0; i < 50; i++) {
                let x = Math.random() * canvas.width;
                let y = Math.random() * canvas.height;
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.strokeStyle = `rgba(${100 + Math.random()*100}, ${150 + Math.random()*105}, ${200 + Math.random()*55}, 0.5)`;
                for (let step = 0; step < 20; step++) {
                    const angle = Math.sin(x * 0.02 + y * 0.03) * Math.PI + Math.cos(y * 0.02 - x*0.01) * Math.PI;
                    x += Math.cos(angle) * 5;
                    y += Math.sin(angle) * 5;
                    if (x < 0 || x > canvas.width || y < 0 || y > canvas.height) break;
                    ctx.lineTo(x, y);
                }
                ctx.stroke();
            }
        },

        // Mask Placeholder
        renderMaskPlaceholder: function(canvas, maskName) {
            const ctx = this._clearCanvas(canvas, '#2c2c34');
            const nameToShow = maskName.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
            
            ctx.fillStyle = '#e0e0e8';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.font = 'bold 10px "Segoe UI", Tahoma, Geneva, Verdana, sans-serif';
            ctx.fillText(nameToShow, canvas.width / 2, canvas.height / 2);
            ctx.strokeStyle = '#555';
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },
        
        // Mask Visualization Functions
        renderMaskRadial: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const maxRadius = Math.min(canvas.width, canvas.height) / 2;
            
            // Create gradient for radial mask
            const gradient = ctx.createRadialGradient(
                centerX, centerY, 0,
                centerX, centerY, maxRadius
            );
            gradient.addColorStop(0, 'rgba(255, 255, 255, 1.0)');
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0.0)');
            
            // Draw the gradient as the mask
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Add a subtle border
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },
        
        renderMaskLinear: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            
            // Create gradient for linear mask
            const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
            gradient.addColorStop(0, 'rgba(255, 255, 255, 1.0)');
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0.0)');
            
            // Draw the gradient as the mask
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Add a subtle border
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },
        
        renderMaskGrid: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            const cellSize = 20;
            
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            
            // Draw a grid pattern
            for (let x = 0; x < canvas.width; x += cellSize) {
                for (let y = 0; y < canvas.height; y += cellSize) {
                    if ((Math.floor(x / cellSize) + Math.floor(y / cellSize)) % 2 === 0) {
                        ctx.fillRect(x, y, cellSize, cellSize);
                    }
                }
            }
            
            // Add a subtle border
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },
        
        renderMaskVignette: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const maxRadius = Math.sqrt(centerX * centerX + centerY * centerY);
            
            // Create gradient for vignette mask (inverse of radial)
            const gradient = ctx.createRadialGradient(
                centerX, centerY, 0,
                centerX, centerY, maxRadius
            );
            gradient.addColorStop(0, 'rgba(255, 255, 255, 0.0)');
            gradient.addColorStop(0.6, 'rgba(255, 255, 255, 0.3)');
            gradient.addColorStop(1, 'rgba(255, 255, 255, 1.0)');
            
            // Draw the gradient as the mask
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Add a subtle border
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },
        
        renderMaskSpiral: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const maxRadius = Math.min(canvas.width, canvas.height) / 2;
            
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.strokeStyle = 'rgba(255, 255, 255, 1.0)';
            ctx.lineWidth = 5;
            
            // Draw a spiral
            ctx.beginPath();
            for (let theta = 0; theta < 8 * Math.PI; theta += 0.1) {
                const radius = (maxRadius / (8 * Math.PI)) * theta;
                const x = centerX + radius * Math.cos(theta);
                const y = centerY + radius * Math.sin(theta);
                
                if (theta === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
            
            // Add a subtle border
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1;
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },
        
        renderMaskHexgrid: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            const hexSize = 15;
            const hexHeight = hexSize * Math.sqrt(3);
            
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1;
            
            // Draw a hexagonal grid
            for (let row = 0; row < canvas.height / hexHeight + 1; row++) {
                for (let col = 0; col < canvas.width / (hexSize * 3) + 1; col++) {
                    const offsetX = (row % 2) * hexSize * 1.5;
                    const x = col * hexSize * 3 + offsetX;
                    const y = row * hexHeight;
                    
                    ctx.beginPath();
                    for (let i = 0; i < 6; i++) {
                        const angle = (i * Math.PI / 3);
                        const pX = x + hexSize * Math.cos(angle);
                        const pY = y + hexSize * Math.sin(angle);
                        
                        if (i === 0) {
                            ctx.moveTo(pX, pY);
                        } else {
                            ctx.lineTo(pX, pY);
                        }
                    }
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                }
            }
            
            // Add a subtle border
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },
        
        renderMaskWavy: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.strokeStyle = 'rgba(255, 255, 255, 1.0)';
            ctx.lineWidth = 3;
            
            const amplitude = 10;
            const frequency = 0.05;
            
            // Draw wavy lines
            for (let y = 20; y < canvas.height; y += 20) {
                ctx.beginPath();
                for (let x = 0; x <= canvas.width; x += 2) {
                    const yOffset = Math.sin(x * frequency) * amplitude;
                    if (x === 0) {
                        ctx.moveTo(x, y + yOffset);
                    } else {
                        ctx.lineTo(x, y + yOffset);
                    }
                }
                ctx.stroke();
            }
            
            // Add a subtle border
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1;
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },
        
        renderMaskConcentricRings: function(canvas) {
            const ctx = this._clearCanvas(canvas);
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            
            ctx.strokeStyle = 'rgba(255, 255, 255, 1.0)';
            ctx.lineWidth = 2;
            
            // Draw concentric rings
            const ringCount = 6;
            const maxRadius = Math.min(canvas.width, canvas.height) / 2;
            const step = maxRadius / ringCount;
            
            for (let i = 1; i <= ringCount; i++) {
                const radius = i * step;
                ctx.beginPath();
                ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
                ctx.stroke();
                
                // Add some fill for alternating rings
                if (i % 2 === 0) {
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
                    ctx.beginPath();
                    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
            
            // Add a subtle border around the canvas
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1;
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },

        // Color scheme rendering functions
        renderColorInferno: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #000004, #3b0f70, #8c2981, #de4968, #fe9f6d, #fcfdbf)';
        },
        
        renderColorMagma: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #000004, #3b0f70, #8c2981, #de4968, #fe9f6d, #fcfdbf)';
        },
        
        renderColorPlasma: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #0d0887, #7e03a8, #cc4678, #f89441, #f0f921)';
        },
        
        renderColorViridis: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #440154, #30678d, #35b778, #fde724)';
        },
        
        renderColorTurbo: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #30123b, #4669db, #26bf8c, #d4ff50, #fab74c, #ba0100)';
        },
        
        renderColorJet: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #00007f, #0000ff, #00ffff, #ffff00, #ff0000, #7f0000)';
        },
        
        renderColorParula: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #352a87, #0f5cdd, #00b5a6, #ffc337, #fcfea4)';
        },
        
        renderColorRainbow: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000)';
        },
        
        renderColorHot: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #000000, #ff0000, #ffff00, #ffffff)';
        },
        
        renderColorBlueRed: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #0000ff, #7777ff, #ffffff, #ff7777, #ff0000)';
        },
        
        renderColorCool: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #00ffff, #77aaff, #aa77ff, #ff00ff)';
        },
        
        renderColorHsv: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff, #ff00ff, #ff0000)';
        },
        
        renderColorAutumn: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #ff0000, #ff7700, #ffaa00, #ffdd00, #ffff00)';
        },
        
        renderColorWinter: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #0000ff, #0077cc, #00aabb, #00ddaa)';
        },
        
        renderColorSpring: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #ff00ff, #ff33cc, #ff77aa, #ffaa77, #ffdd44, #ffff00)';
        },
        
        renderColorSummer: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #004433, #008855, #44aa66, #88cc77, #ccee88, #ffff66)';
        },
        
        renderColorCopper: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #000000, #331100, #662200, #993300, #cc6644, #ff9966)';
        },
        
        renderColorPink: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #0a0a0a, #550055, #aa0066, #ff44aa, #ffaadd, #ffffff)';
        },
        
        renderColorBone: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #000000, #2a2a3a, #5a748a, #9ebacb, #dfdfef, #ffffff)';
        },
        
        renderColorOcean: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #000000, #000066, #0000bb, #0066cc, #00ccff, #99ffff)';
        },
        
        renderColorTerrain: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #333399, #009966, #66cc33, #cccc33, #ff9933, #ffffff)';
        },
        
        renderColorNeon: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #ff00ff, #aa00ff, #5500ff, #0000ff, #00aaff, #00ffff, #00ff00, #aaff00, #ffff00)';
        },
        
        renderColorFire: function(swatchDiv) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #000000, #330000, #660000, #bb0000, #ff0000, #ff7700, #ffdd00, #ffffff)';
        },
        
        // New function for temporal animation
        renderTemporalAnimation: function(canvas) {
            const ctx = this._clearCanvas(canvas, '#0a0a0a'); // Dark background for animation
            let time = 0;

            function drawPattern(currentTime) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#0a0a0a';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                const numLines = 20;
                const maxOffset = 20;
                
                ctx.strokeStyle = 'rgba(52, 152, 219, 0.6)';
                ctx.lineWidth = 1.5;

                for (let i = 0; i < numLines; i++) {
                    ctx.beginPath();
                    const startY = (i / numLines) * canvas.height;
                    ctx.moveTo(0, startY);
                    for (let x = 0; x <= canvas.width; x += 5) {
                        // Create a wave that evolves with time and differs per line
                        const yOffset = Math.sin(x * 0.02 + i * 0.3 + currentTime * 0.05) * maxOffset * Math.sin(currentTime * 0.02 + i * 0.1);
                        const actualY = startY + yOffset;
                        ctx.lineTo(x, actualY);
                    }
                    ctx.stroke();
                }
            }

            function animate() {
                drawPattern(time);
                time += 0.1; // Increment time for animation
                // Only request next frame if the canvas is still in the document (modal is open)
                if (canvas.isConnected) {
                    requestAnimationFrame(animate);
                }
            }
            
            // Check if an animation is already running on this canvas
            if (!canvas.dataset.animationRunning) {
                canvas.dataset.animationRunning = 'true';
                animate();
            }
        },

        // New function for the intro noise demo
        renderIntroNoiseDemo: function(canvas) {
            const ctx = this._clearCanvas(canvas, '#0f0f12'); // Darker background for intro
            let time = 0;
            const particles = [];
            const numParticles = 120; // Slightly reduced for clarity with trails
            const particleColorBase = [138, 43, 226]; 
            const particleColorHighlight = [224, 224, 232];
            const loopCycleDuration = 420; // Extended to 7 seconds
            let lastLoopInstance = -1;
            let currentBackgroundPattern = { type: null, variantSeed: 0, alpha: 0, startTime: 0 };

            // Helper function to get spawn points based on pattern
            function getSpawnPointForPattern(patternType, variantSeed, canvas, patternTimeVal) {
                const centerX = canvas.width / 2;
                const centerY = canvas.height / 2;
                const maxPatternSize = Math.min(centerX, centerY) * 0.95;
                let spawnX = Math.random() * canvas.width;
                let spawnY = Math.random() * canvas.height;

                // timeVal for spawning should reflect the pattern at its initial state (or very early evolution)
                const timeVal = patternTimeVal * 0.00015; // Match background pattern evolution rate

                if (patternType === 'tensor_field') {
                    const gridSize = 30;
                    const gx = Math.floor(Math.random() * (canvas.width / gridSize)) * gridSize + gridSize/2;
                    const gy = Math.floor(Math.random() * (canvas.height / gridSize)) * gridSize + gridSize/2;
                    const tfScale = 0.025 + (variantSeed % 5) * 0.003;
                    const angleOffset = variantSeed * 0.3;
                    const fieldAngle = Math.sin(gx * tfScale + timeVal * 5 + angleOffset) * Math.PI + 
                                     Math.cos(gy * tfScale - timeVal * 3) * Math.PI;
                    const fieldMagnitude = (Math.sin(gx * tfScale * 0.7 + gy * tfScale * 0.5 - timeVal * 2) + 1) / 2;
                    const lineLength = 10 * (0.5 + fieldMagnitude * 0.7);
                    // Spawn along the vector line
                    const r = Math.random() * lineLength - lineLength/2;
                    spawnX = gx + Math.cos(fieldAngle) * r;
                    spawnY = gy + Math.sin(fieldAngle) * r;
                } else if (patternType === 'cellular') {
                    const numPoints = 3 + Math.floor(variantSeed % 4);
                    const pointIndex = Math.floor(Math.random() * numPoints);
                    const angle = (pointIndex / numPoints) * Math.PI * 2 + timeVal * (0.1 + (variantSeed % 3)*0.03) + variantSeed * 0.7;
                    const radius = maxPatternSize * (0.4 + ((pointIndex + variantSeed) % 4) * 0.12) * (0.85 + Math.sin(timeVal*1.5 + pointIndex) * 0.15);
                    const pX = centerX + Math.cos(angle) * radius;
                    const pY = centerY + Math.sin(angle) * radius;
                    // Spawn near a cell center point
                    spawnX = pX + (Math.random() - 0.5) * 30;
                    spawnY = pY + (Math.random() - 0.5) * 30;
                } else if (patternType === 'domain_warp') {
                    const gridSize = 25;
                    const c = Math.floor(Math.random() * (canvas.width / gridSize));
                    const r = Math.floor(Math.random() * (canvas.height / gridSize));
                    const origX = c * gridSize;
                    const origY = r * gridSize;
                    const warpScale = 0.015 + (variantSeed % 5) * 0.002;
                    const warpIntensity = canvas.width * (0.05 + (variantSeed % 3) * 0.02);
                    const offsetX = Math.sin(origY * warpScale * 1.2 + timeVal * 8 + variantSeed) * warpIntensity;
                    const offsetY = Math.cos(origX * warpScale * 0.8 - timeVal * 6 + variantSeed) * warpIntensity;
                    spawnX = origX + offsetX;
                    spawnY = origY + offsetY;
                } else if (patternType === 'perlin') {
                    // Try a few random spots and pick one with mid-range noise intensity
                    let bestX = spawnX, bestY = spawnY, minDiff = Infinity;
                    for(let i=0; i<10; i++) {
                        let sx = Math.random() * canvas.width;
                        let sy = Math.random() * canvas.height;
                        let noiseVal = 0;
                        let freq = 0.015 + (variantSeed % 7) * 0.003; 
                        let amp = 0.6;
                        for (let oct = 0; oct < 3; oct++) {
                            noiseVal += Math.sin(sx * freq + timeVal*10 + variantSeed*0.2) * Math.cos(sy * freq - timeVal*12 - variantSeed*0.1) * amp;
                            freq *= 1.8 + (variantSeed % 3) * 0.1; 
                            amp *= 0.45 + (variantSeed % 4) * 0.02; 
                        }
                        const currentDiff = Math.abs(noiseVal); // Target noiseVal around 0 (mid-range for bipolar noise)
                        if (currentDiff < minDiff) {
                            minDiff = currentDiff;
                            bestX = sx; bestY = sy;
                        }
                    }
                    spawnX = bestX; spawnY = bestY;
                } else if (patternType === 'curl_noise') {
                    // Spawn along a few conceptual streamlines
                    const numStreamlines = 5;
                    const streamlineIndex = Math.floor(Math.random() * numStreamlines);
                    let pX = canvas.width * (streamlineIndex / numStreamlines + Math.random() * 0.1 - 0.05);
                    let pY = canvas.height * Math.random();
                    const curlScale = 0.018 + (variantSeed % 5) * 0.002;
                    const Gx_val = Math.sin(pY*curlScale + timeVal*7 + variantSeed + streamlineIndex*0.1);
                    const Gy_val = Math.cos(pX*curlScale - timeVal*9 + variantSeed + streamlineIndex*0.1);
                    const flowX = Gy_val; 
                    const flowY = -Gx_val;
                    const norm = Math.hypot(flowX, flowY);
                    if (norm > 0.01) {
                        const step = (Math.random() - 0.5) * 50; // Spawn somewhere along the local flow vector
                        spawnX = pX + (flowX / norm) * step;
                        spawnY = pY + (flowY / norm) * step;
                    } else {
                        spawnX = pX; spawnY = pY;
                    }
                } else if (patternType === 'domain_warp') {
                    const gridSize = 25;
                    const c = Math.floor(Math.random() * (canvas.width / gridSize));
                    const r = Math.floor(Math.random() * (canvas.height / gridSize));
                    const origX = c * gridSize;
                    const origY = r * gridSize;
                    const warpScale = 0.015 + (variantSeed % 5) * 0.002;
                    const warpIntensity = canvas.width * (0.05 + (variantSeed % 3) * 0.02);
                    const offsetX = Math.sin(origY * warpScale * 1.2 + timeVal * 8 + variantSeed) * warpIntensity;
                    const offsetY = Math.cos(origX * warpScale * 0.8 - timeVal * 6 + variantSeed) * warpIntensity;
                    spawnX = origX + offsetX;
                    spawnY = origY + offsetY;
                } else if (patternType === 'perlin') {
                    // Try a few random spots and pick one with mid-range noise intensity
                    let bestX = spawnX, bestY = spawnY, minDiff = Infinity;
                    for(let i=0; i<10; i++) {
                        let sx = Math.random() * canvas.width;
                        let sy = Math.random() * canvas.height;
                        let noiseVal = 0;
                        let freq = 0.015 + (variantSeed % 7) * 0.003; 
                        let amp = 0.6;
                        for (let oct = 0; oct < 3; oct++) {
                            noiseVal += Math.sin(sx * freq + timeVal*10 + variantSeed*0.2) * Math.cos(sy * freq - timeVal*12 - variantSeed*0.1) * amp;
                            freq *= 1.8 + (variantSeed % 3) * 0.1; 
                            amp *= 0.45 + (variantSeed % 4) * 0.02; 
                        }
                        const currentDiff = Math.abs(noiseVal); // Target noiseVal around 0 (mid-range for bipolar noise)
                        if (currentDiff < minDiff) {
                            minDiff = currentDiff;
                            bestX = sx; bestY = sy;
                        }
                    }
                    spawnX = bestX; spawnY = bestY;
                } else if (patternType === 'curl_noise') {
                    // Spawn along a few conceptual streamlines
                    const numStreamlines = 5;
                    const streamlineIndex = Math.floor(Math.random() * numStreamlines);
                    let pX = canvas.width * (streamlineIndex / numStreamlines + Math.random() * 0.1 - 0.05);
                    let pY = canvas.height * Math.random();
                    const curlScale = 0.018 + (variantSeed % 5) * 0.002;
                    const Gx_val = Math.sin(pY*curlScale + timeVal*7 + variantSeed + streamlineIndex*0.1);
                    const Gy_val = Math.cos(pX*curlScale - timeVal*9 + variantSeed + streamlineIndex*0.1);
                    const flowX = Gy_val; 
                    const flowY = -Gx_val;
                    const norm = Math.hypot(flowX, flowY);
                    if (norm > 0.01) {
                        const step = (Math.random() - 0.5) * 50; // Spawn somewhere along the local flow vector
                        spawnX = pX + (flowX / norm) * step;
                        spawnY = pY + (flowY / norm) * step;
                    } else {
                        spawnX = pX; spawnY = pY;
                    }
                } else if (patternType === 'waves_interference') {
                    const numWaveSets = 2 + Math.floor(variantSeed % 2);
                    const ws = Math.floor(Math.random()*numWaveSets); // Pick a wave set
                    const lineIdx = Math.floor(Math.random()*15); // Pick a line within the set
                    
                    const amplitude = maxPatternSize * (0.15 + (variantSeed % 3) * 0.05);
                    const frequency = 0.025 + ((variantSeed+ws*2) % 7) * 0.004;
                    const angle = (variantSeed * 0.2 + ws * Math.PI / (numWaveSets * 1.5)) + timeVal * (3 + ws*0.5);
                    const cosA = Math.cos(angle);
                    const sinA = Math.sin(angle);

                    const lineOffset = (lineIdx - 15/2) * maxPatternSize * 0.1;
                    const j = (Math.random() - 0.5) * maxPatternSize * 2.0; // Random point along the line
                    const waveDisplacement = Math.sin(j * frequency + timeVal * (10 + ws*3) + ws*Math.PI) * amplitude;
                    const originalX = j;
                    const originalY = lineOffset + waveDisplacement;
                    
                    spawnX = centerX + (originalX * cosA - originalY * sinA);
                    spawnY = centerY + (originalX * sinA + originalY * cosA);
                }
                
                return { 
                    x: Math.max(0, Math.min(canvas.width, spawnX)), 
                    y: Math.max(0, Math.min(canvas.height, spawnY))
                };
            }

            // Noise function modified to include influence from background patterns
            function noiseField(x, y, t, patternType, patternVariantSeed, patternTime) {
                const baseScale = 0.009;
                const timeScale = 0.0023;
                const loopTime = t % loopCycleDuration;
                const currentLoopInstance = Math.floor(t / loopCycleDuration);

                const globalRotationAngle = currentLoopInstance * 0.15; 
                const cosA = Math.cos(globalRotationAngle);
                const sinA = Math.sin(globalRotationAngle);

                let rotatedX = x * cosA - y * sinA;
                let rotatedY = x * sinA + y * cosA;

                // --- Base Chaotic Noise Calculation (similar to before) ---
                let baseChaosVal = 0;
                const scale1 = baseScale * 1.8;
                baseChaosVal += Math.sin(rotatedX * scale1 + loopTime * timeScale * 1.1) * 
                               Math.cos(rotatedY * scale1 * 0.8 - loopTime * timeScale * 0.9);
                const scale2 = baseScale * 0.9;
                baseChaosVal += Math.sin(rotatedY * scale2 * 1.1 - loopTime * timeScale * 1.3) * 
                               Math.cos(rotatedX * scale2 * 0.9 + loopTime * timeScale * 1.0) * 0.8;
                const scale3 = baseScale * 1.2;
                const innerAngleOffset = Math.sin(loopTime * 0.0008 + currentLoopInstance * 0.05) * 0.5;
                const N_x_chaos = rotatedX * Math.cos(innerAngleOffset) - rotatedY * Math.sin(innerAngleOffset);
                const N_y_chaos = rotatedX * Math.sin(innerAngleOffset) + rotatedY * Math.cos(innerAngleOffset);
                baseChaosVal += Math.sin(N_x_chaos * scale3 * 0.6 + loopTime * timeScale * 0.7) * 
                               Math.cos(N_y_chaos * scale3 * 0.7 - loopTime * timeScale * 0.5) * 0.9;
                let baseAngle = (baseChaosVal / 2.7) * Math.PI * 4.0; // Base movement angle

                // --- Background Pattern Influence Calculation ---
                let influenceAngle = baseAngle;
                let influenceStrength = 0.0; // 0.0 to 1.0 (max influence)
                const patternEvoTime = patternTime * 0.00015; 
                const centerX = canvas.width / 2;
                const centerY = canvas.height / 2;

                if (patternType === 'tensor_field') { 
                    const tfScale = 0.025 + (patternVariantSeed % 5) * 0.003;
                    const angleOffset = patternVariantSeed * 0.3;
                    // Match the angle calculation from drawBackgroundShaderPattern
                    influenceAngle = Math.sin(x * tfScale + patternEvoTime * 5 + angleOffset) * Math.PI + 
                                   Math.cos(y * tfScale - patternEvoTime * 3) * Math.PI;
                    influenceStrength = 0.75; 
                } else if (patternType === 'cellular') {
                    const numPoints = 3 + Math.floor(patternVariantSeed % 4);
                    const points = [];
                    for (let i = 0; i < numPoints; i++) {
                        const angle = (i / numPoints) * Math.PI * 2 + patternEvoTime * (0.1 + (patternVariantSeed % 3)*0.03) + patternVariantSeed * 0.7;
                        const radiusFactor = (0.4 + ((i + patternVariantSeed) % 4) * 0.12) * (0.85 + Math.sin(patternEvoTime*1.5 + i) * 0.15);
                        points.push({
                            x: centerX + Math.cos(angle) * (Math.min(centerX,centerY)*0.95) * radiusFactor,
                            y: centerY + Math.sin(angle) * (Math.min(centerX,centerY)*0.95) * radiusFactor
                        });
                    }

                    let p1_idx = -1, p2_idx = -1;
                    let minDist1 = Infinity, minDist2 = Infinity;

                    for(let k=0; k<points.length; ++k) {
                        const dSq = (points[k].x - x)*(points[k].x - x) + (points[k].y - y)*(points[k].y - y);
                        if (dSq < minDist1) {
                            minDist2 = minDist1; p2_idx = p1_idx;
                            minDist1 = dSq; p1_idx = k;
                        } else if (dSq < minDist2) {
                            minDist2 = dSq; p2_idx = k;
                        }
                    }

                    if (p1_idx !== -1 && p2_idx !== -1) {
                        const d1 = Math.sqrt(minDist1);
                        const d2 = Math.sqrt(minDist2);
                        // If particle is roughly equidistant from two closest points (i.e., near an edge)
                        if (Math.abs(d1 - d2) < canvas.width * 0.1) { // Threshold for being "near an edge"
                            const pA = points[p1_idx];
                            const pB = points[p2_idx];
                            // Angle parallel to the edge P_A P_B
                            influenceAngle = Math.atan2(pB.y - pA.y, pB.x - pA.x);
                            // Could be either direction, let's check dot product with current velocity or baseAngle
                            // For simplicity, just pick one. Or, add Math.PI based on which side of the line particle is.
                            // To make it flow along, maybe alternate direction based on something?
                            // Let's try making it consistently one way along the edge.
                            // To decide which of the two parallel directions, project particle's relative position to pA onto the perpendicular of (pB-pA)
                            const edgeDx = pB.x - pA.x;
                            const edgeDy = pB.y - pA.y;
                            const particleRelX = x - pA.x;
                            const particleRelY = y - pA.y;
                            // Perpendicular vector to edge: (-edgeDy, edgeDx)
                            const crossProdSign = particleRelX * edgeDx + particleRelY * edgeDy; // Simplified from cross for side check
                            if (crossProdSign < 0) { // If on one side, add PI
                                // This depends on how pA and pB were ordered.
                                // A robust way for flow: If the particle is to the "left" of vector pA->pB, flow one way, else other.
                                // Vector from pA to particle (particleRelX, particleRelY)
                                // Vector pA to pB (edgeDx, edgeDy)
                                // 2D cross product: particleRelX * edgeDy - particleRelY * edgeDx
                                if (particleRelX * edgeDy - particleRelY * edgeDx > 0) {
                                     influenceAngle += Math.PI; // Reverse direction
                                }
                            }
                             influenceStrength = 0.7;
                        } else {
                            // If not near an edge, maybe weakly attract to the closest point or a mix?
                            // For now, let's stick to edge following for clear "lines"
                            influenceStrength = 0.1; // Weak influence if not near edge
                        }
                    }
                } else if (patternType === 'domain_warp') {
                    const warpScale = 0.015 + (patternVariantSeed % 5) * 0.002;
                    const warpIntensity = 25 + (patternVariantSeed % 3) * 5;
                    const displacementX = Math.sin(y * warpScale * 1.2 + patternEvoTime * 8 + patternVariantSeed) * warpIntensity;
                    const displacementY = Math.cos(x * warpScale * 0.8 - patternEvoTime * 6 + patternVariantSeed) * warpIntensity;
                    if (Math.hypot(displacementX, displacementY) > 0.1) {
                         influenceAngle = Math.atan2(displacementY, displacementX);
                         influenceStrength = 0.7; // Increased strength
                    }
                } else if (patternType === 'perlin') {
                    let noiseValCenter = 0, noiseValX = 0, noiseValY = 0;
                    let freq = 0.015 + (patternVariantSeed % 7) * 0.003; 
                    let amp = 0.6;
                    for (let oct = 0; oct < 3; oct++) { 
                        noiseValCenter += Math.sin(x * freq + patternEvoTime*10 + patternVariantSeed*0.2) * Math.cos(y * freq - patternEvoTime*12 - patternVariantSeed*0.1) * amp;
                        noiseValX += Math.sin((x+1) * freq + patternEvoTime*10+ patternVariantSeed*0.2) * Math.cos(y * freq - patternEvoTime*12 - patternVariantSeed*0.1) * amp;
                        noiseValY += Math.sin(x * freq + patternEvoTime*10+ patternVariantSeed*0.2) * Math.cos((y+1) * freq - patternEvoTime*12 - patternVariantSeed*0.1) * amp;
                        freq *= 1.8 + (patternVariantSeed % 3) * 0.1; 
                        amp *= 0.45 + (patternVariantSeed % 4) * 0.02; 
                    }
                    const gradX = noiseValX - noiseValCenter;
                    const gradY = noiseValY - noiseValCenter;
                    if (Math.hypot(gradX, gradY) > 0.005) {
                        influenceAngle = Math.atan2(gradY, gradX) + (Math.PI / 2); // Perpendicular to gradient
                        influenceStrength = 0.7; // Increased strength
                    }
                } else if (patternType === 'waves_interference') {
                    let waveSum = 0;
                    const numWaveSources = 2 + (patternVariantSeed % 2); 
                    for(let i=0; i<numWaveSources; i++) {
                        const sourceAngle = (i / numWaveSources) * Math.PI*2 + patternVariantSeed * 0.5 + patternEvoTime*0.5;
                        const sourceRadius = canvas.width * (0.1 + (i%2)*0.3);
                        const sourceX = centerX + Math.cos(sourceAngle) * sourceRadius;
                        const sourceY = centerY + Math.sin(sourceAngle) * sourceRadius;
                        const distToSource = Math.hypot(x - sourceX, y - sourceY);
                        const waveFreq = 0.05 + (i + patternVariantSeed % 3) * 0.01;
                        waveSum += Math.sin(distToSource * waveFreq - patternEvoTime * 15);
                    }
                    let waveSumX = 0, waveSumY = 0;
                     for(let i=0; i<numWaveSources; i++) { 
                        const sourceAngle = (i / numWaveSources) * Math.PI*2 + patternVariantSeed * 0.5 + patternEvoTime*0.5;
                        const sourceRadius = canvas.width * (0.1 + (i%2)*0.3);
                        const sourceX = centerX + Math.cos(sourceAngle) * sourceRadius;
                        const sourceY = centerY + Math.sin(sourceAngle) * sourceRadius;
                        const waveFreq = 0.05 + (i + patternVariantSeed % 3) * 0.01;
                        
                        waveSumX += Math.sin(Math.hypot(x+1 - sourceX, y - sourceY) * waveFreq - patternEvoTime * 15);
                        waveSumY += Math.sin(Math.hypot(x - sourceX, y+1 - sourceY) * waveFreq - patternEvoTime * 15);
                    }
                    const gradX = waveSumX - waveSum;
                    const gradY = waveSumY - waveSum;
                    
                    if (Math.hypot(gradX, gradY) > 0.01) {
                         influenceAngle = Math.atan2(gradY, gradX) + (Math.PI / 2); // Perpendicular to gradient
                         influenceStrength = 0.65; 
                    }
                } else if (patternType === 'curl_noise') { 
                    const curlScale = 0.018 + (patternVariantSeed % 5) * 0.002;
                    const Gx_val = Math.sin(y*curlScale + patternEvoTime*7 + patternVariantSeed);
                    const Gy_val = Math.cos(x*curlScale - patternEvoTime*9 + patternVariantSeed);
                    const flowX = Gy_val; 
                    const flowY = -Gx_val;
                    if (Math.hypot(flowX, flowY) > 0.01) {
                        influenceAngle = Math.atan2(flowY, flowX);
                        influenceStrength = 0.85; // Increased strength
                    }
                }

                // Blend base chaos with pattern influence
                // Normalize angles for smoother blending if they wrap around PI
                let finalAngle;
                let diff = influenceAngle - baseAngle;
                while (diff > Math.PI) diff -= 2 * Math.PI;
                while (diff < -Math.PI) diff += 2 * Math.PI;
                finalAngle = baseAngle + diff * influenceStrength;

                return finalAngle;
            }

            for (let i = 0; i < numParticles; i++) {
                particles.push({
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    vx: 0,
                    vy: 0,
                    life: 0, // Start dead, will be revived in primordial state
                    initialLife: Math.random() * 100 + 90,
                    radius: Math.random() * 1.7 + 0.6, 
                    trail: [],
                    isPrimordial: true, // New flag
                    primordialTime: 0 // Counter for primordial phase
                });
            }
            const primordialDuration = 60; // Approx 1 second for primordial phase

            function drawBackgroundShaderPattern(ctx, type, variantSeed, targetAlpha, patternTime) {
                ctx.save();
                ctx.globalAlpha = targetAlpha * 0.07; // Slightly increased base visibility for new patterns
                ctx.strokeStyle = 'rgba(180, 190, 200, 0.75)'; 
                ctx.lineWidth = 0.7;
                
                const centerX = canvas.width / 2;
                const centerY = canvas.height / 2;
                const maxPatternSize = Math.min(centerX, centerY) * 0.95;
                const timeVal = patternTime * 0.00015; // Consistent slow evolution

                if (type === 'cellular') {
                    const numPoints = 3 + Math.floor(variantSeed % 4); // 3 to 6 points
                    const points = [];
                    for (let i = 0; i < numPoints; i++) {
                        const angle = (i / numPoints) * Math.PI * 2 + timeVal * (0.1 + (variantSeed % 3)*0.03) + variantSeed * 0.7;
                        const radius = maxPatternSize * (0.4 + ((i + variantSeed) % 4) * 0.12) * (0.85 + Math.sin(timeVal*1.5 + i) * 0.15);
                        points.push({
                            x: centerX + Math.cos(angle) * radius,
                            y: centerY + Math.sin(angle) * radius
                        });
                    }
                    // Draw Voronoi edges (simplified)
                    ctx.beginPath();
                    const step = 15;
                    for (let cy = 0; cy < canvas.height; cy += step) {
                        for (let cx = 0; cx < canvas.width; cx += step) {
                            let minDist1 = Infinity, minDist2 = Infinity;
                            let p1_idx = -1;
                            for(let k=0; k<points.length; ++k) {
                                const d = Math.hypot(points[k].x - cx, points[k].y - cy);
                                if (d < minDist1) { minDist2 = minDist1; minDist1 = d; p1_idx = k; }
                                else if (d < minDist2) { minDist2 = d; }
                            }
                            if (Math.abs(minDist1 - minDist2) < step * 0.7) { // If close to an edge
                                ctx.rect(cx -1, cy -1, 2, 2);
                            }
                        }
                    }
                    ctx.fillStyle = ctx.strokeStyle; // Fill tiny rects with stroke color
                    ctx.fill();

                } else if (type === 'perlin') { // Was perlin_fractal
                    const patchSize = 10; // Smaller patches for smoother look
                    ctx.globalAlpha = targetAlpha * 0.04; // Perlin is more subtle
                    for (let py = 0; py < canvas.height; py += patchSize) {
                        for (let px = 0; px < canvas.width; px += patchSize) {
                            let noiseVal = 0;
                            let freq = 0.015 + (variantSeed % 7) * 0.003; 
                            let amp = 0.6;
                            for (let oct = 0; oct < 3; oct++) {
                                noiseVal += Math.sin((px + variantSeed*10) * freq + timeVal*10) * Math.cos((py - variantSeed*5) * freq - timeVal*12) * amp;
                                freq *= 1.8 + (variantSeed % 3) * 0.1; 
                                amp *= 0.45 + (variantSeed % 4) * 0.02; 
                            }
                            const alpha = (noiseVal / 1.5 + 0.5) * 0.3; // Normalize and map to low alpha
                            ctx.fillStyle = `rgba(160, 170, 180, ${Math.max(0, alpha)})`;
                            ctx.fillRect(px, py, patchSize, patchSize);
                        }
                    }
                } else if (type === 'waves_interference') { // Was waves
                    const numWaveSets = 2 + Math.floor(variantSeed % 2);
                    ctx.lineWidth = 0.5;
                     ctx.globalAlpha = targetAlpha * 0.06;
                    for (let ws = 0; ws < numWaveSets; ws++) {
                        const amplitude = maxPatternSize * (0.15 + (variantSeed % 3) * 0.05);
                        const frequency = 0.025 + ((variantSeed+ws*2) % 7) * 0.004;
                        const angle = (variantSeed * 0.2 + ws * Math.PI / (numWaveSets * 1.5)) + timeVal * (3 + ws*0.5);
                        const cosA = Math.cos(angle);
                        const sinA = Math.sin(angle);
                        
                        const numLines = 15;
                        for (let i = 0; i < numLines; i++) {
                             ctx.beginPath();
                            let first = true;
                            const lineOffset = (i - numLines/2) * maxPatternSize * 0.1;
                            for (let j = -maxPatternSize * 1.2; j < maxPatternSize * 1.2; j += 8) {
                                const waveDisplacement = Math.sin(j * frequency + timeVal * (10 + ws*3) + ws*Math.PI) * amplitude;
                                const originalX = j;
                                const originalY = lineOffset + waveDisplacement;
                                
                                const xPos = centerX + (originalX * cosA - originalY * sinA);
                                const yPos = centerY + (originalX * sinA + originalY * cosA);
                                if (first) { ctx.moveTo(xPos, yPos); first = false; } else { ctx.lineTo(xPos, yPos); }
                            }
                            ctx.stroke();
                        }
                    }
                } else if (type === 'tensor_field') { // Was flow_field
                    const gridSize = 30;
                    const lineLengthBase = 10;
                    ctx.lineWidth = 0.8;
                    for (let gy = 0; gy < canvas.height; gy += gridSize) {
                        for (let gx = 0; gx < canvas.width; gx += gridSize) {
                            const tfScale = 0.025 + (variantSeed % 5) * 0.003;
                            const angleOffset = variantSeed * 0.3;
                            const fieldAngle = Math.sin(gx * tfScale + timeVal * 5 + angleOffset) * Math.PI + 
                                             Math.cos(gy * tfScale - timeVal * 3) * Math.PI;
                            const fieldMagnitude = (Math.sin(gx * tfScale * 0.7 + gy * tfScale * 0.5 - timeVal * 2) + 1) / 2; // 0 to 1
                            const lineLength = lineLengthBase * (0.5 + fieldMagnitude * 0.7);
                            ctx.save();
                            ctx.translate(gx, gy);
                            ctx.rotate(fieldAngle);
                            ctx.beginPath();
                            ctx.moveTo(-lineLength / 2, 0);
                            ctx.lineTo(lineLength / 2, 0);
                            // Arrowhead
                            ctx.lineTo(lineLength/2 - 3, -2);
                            ctx.moveTo(lineLength/2,0);
                            ctx.lineTo(lineLength/2 - 3, 2);
                            ctx.globalAlpha = targetAlpha * 0.07 * (0.6 + fieldMagnitude*0.4);
                            ctx.stroke();
                            ctx.restore();
                        }
                    }
                } else if (type === 'domain_warp') {
                    const gridSize = 25;
                     ctx.globalAlpha = targetAlpha * 0.055;
                    for (let r = 0; r < Math.floor(canvas.height / gridSize) + 1; r++) {
                        ctx.beginPath();
                        for (let c = 0; c < Math.floor(canvas.width / gridSize) + 1; c++) {
                            const origX = c * gridSize;
                            const origY = r * gridSize;
                            const warpScale = 0.015 + (variantSeed % 5) * 0.002;
                            const warpIntensity = canvas.width * (0.05 + (variantSeed % 3) * 0.02);
                            
                            const offsetX = Math.sin(origY * warpScale * 1.2 + timeVal * 8 + variantSeed) * warpIntensity;
                            const offsetY = Math.cos(origX * warpScale * 0.8 - timeVal * 6 + variantSeed) * warpIntensity;
                            
                            if (c === 0) ctx.moveTo(origX + offsetX, origY + offsetY);
                            else ctx.lineTo(origX + offsetX, origY + offsetY);
                        }
                        ctx.stroke();
                    }
                     for (let c = 0; c < Math.floor(canvas.width / gridSize) + 1; c++) {
                        ctx.beginPath();
                        for (let r = 0; r < Math.floor(canvas.height / gridSize) + 1; r++) {
                            const origX = c * gridSize;
                            const origY = r * gridSize;
                             const warpScale = 0.015 + (variantSeed % 5) * 0.002;
                            const warpIntensity = canvas.width * (0.05 + (variantSeed % 3) * 0.02);
                            const offsetX = Math.sin(origY * warpScale * 1.2 + timeVal * 8 + variantSeed) * warpIntensity;
                            const offsetY = Math.cos(origX * warpScale * 0.8 - timeVal * 6 + variantSeed) * warpIntensity;
                            if (r === 0) ctx.moveTo(origX + offsetX, origY + offsetY);
                            else ctx.lineTo(origX + offsetX, origY + offsetY);
                        }
                        ctx.stroke();
                    }
                } else if (type === 'curl_noise') {
                    const numStreaks = 30;
                    ctx.lineWidth = 0.6;
                     ctx.globalAlpha = targetAlpha * 0.065;
                    for(let i=0; i<numStreaks; i++) {
                        let pX = centerX + (Math.random()-0.5) * canvas.width * 0.8 + Math.sin(variantSeed*0.1*i + timeVal*2)*canvas.width*0.1;
                        let pY = centerY + (Math.random()-0.5) * canvas.height* 0.8 + Math.cos(variantSeed*0.1*i + timeVal*2)*canvas.height*0.1;
                        ctx.beginPath();
                        ctx.moveTo(pX, pY);
                        const streakLength = 20 + Math.random()*20;
                        for(let k=0; k<streakLength; k++) {
                            const curlScale = 0.018 + (variantSeed % 5) * 0.002;
                            const Gx_val = Math.sin(pY*curlScale + timeVal*7 + variantSeed + i*0.1);
                            const Gy_val = Math.cos(pX*curlScale - timeVal*9 + variantSeed + i*0.1);
                            // For drawing, use the vector (Gy, -Gx) to simulate particle path on paper
                            const flowX = Gy_val; 
                            const flowY = -Gx_val;
                            const norm = Math.hypot(flowX, flowY);
                            if (norm < 0.01) break;
                            pX += (flowX / norm) * 3; // Step size
                            pY += (flowY / norm) * 3;
                            if (pX < 0 || pX > canvas.width || pY < 0 || pY > canvas.height) break;
                            ctx.lineTo(pX,pY);
                        }
                        ctx.stroke();
                    }
                }

                ctx.restore();
            }

            function drawIntroPattern() {
                const currentLoopInstance = Math.floor(time / loopCycleDuration);
                const isNewLoopStart = (time % loopCycleDuration) < (60 / 2.5); 

                if (isNewLoopStart && currentLoopInstance !== lastLoopInstance) {
                    ctx.fillStyle = 'rgba(15, 15, 18, 0.45)'; 
                    lastLoopInstance = currentLoopInstance;
                    
                    const shaderPatternTypes = ['tensor_field', 'cellular', 'domain_warp', 'perlin', 'curl_noise', 'waves_interference'];
                    currentBackgroundPattern.type = shaderPatternTypes[currentLoopInstance % shaderPatternTypes.length];
                    currentBackgroundPattern.variantSeed = currentLoopInstance; // Use loop instance for variation
                    currentBackgroundPattern.alpha = 1.0; 
                    currentBackgroundPattern.startTime = time; 
                    
                    const patternSpawnTimeVal = 0; // Use 0 for the conceptual "start time" of the pattern for consistent spawning

                    particles.forEach(p => {
                        const spawnPoint = getSpawnPointForPattern(currentBackgroundPattern.type, currentBackgroundPattern.variantSeed, canvas, patternSpawnTimeVal);
                        p.x = spawnPoint.x;
                        p.y = spawnPoint.y;
                        // p.x = Math.random() * canvas.width; // Old random spawn
                        // p.y = Math.random() * canvas.height; // Old random spawn
                        p.vx = (Math.random() - 0.5) * 0.1;
                        p.vy = (Math.random() - 0.5) * 0.1;
                        p.life = p.initialLife;
                        p.trail = [];
                        p.isPrimordial = true;
                        p.primordialTime = 0;
                    });

                } else {
                    ctx.fillStyle = 'rgba(15, 15, 18, 0.015)'; 
                }
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                if (currentBackgroundPattern.type && currentBackgroundPattern.alpha > 0) {
                    drawBackgroundShaderPattern(ctx, currentBackgroundPattern.type, currentBackgroundPattern.variantSeed, currentBackgroundPattern.alpha, time - currentBackgroundPattern.startTime);
                    currentBackgroundPattern.alpha = Math.max(0, 1.0 - ((time - currentBackgroundPattern.startTime) / loopCycleDuration) * 0.95 ); // Adjusted fade for longer loop
                }

                particles.forEach(p => {
                    if (p.isPrimordial) {
                        p.primordialTime++;
                        if (p.primordialTime >= primordialDuration) {
                            p.isPrimordial = false;
                        }
                    }

                    const finalAngle = noiseField(p.x, p.y, time, currentBackgroundPattern.type, currentBackgroundPattern.variantSeed, time - currentBackgroundPattern.startTime);
                    // const angle = noiseVal * Math.PI * 4.0; // Old way

                    p.vx += Math.cos(finalAngle) * 0.10; 
                    p.vy += Math.sin(finalAngle) * 0.10;

                    p.vx *= 0.94; 
                    p.vy *= 0.94;

                    p.x += p.vx;
                    p.y += p.vy;

                    p.trail.push({x: p.x, y: p.y});
                    if (p.trail.length > 25) { 
                        p.trail.shift();
                    }

                    let trailBaseAlpha = 0.8;
                    let particleHeadBaseAlpha = 0.65;
                    let currentParticleRadius = p.radius;
                    let r = particleColorBase[0], g = particleColorBase[1], b = particleColorBase[2];

                    if (p.isPrimordial) {
                        const primordialRatio = p.primordialTime / primordialDuration;
                        trailBaseAlpha *= primordialRatio * 0.5; // Trails fainter during primordial
                        particleHeadBaseAlpha = primordialRatio * 0.5; 
                        currentParticleRadius = p.radius * (0.3 + primordialRatio * 0.7); // Grow from small
                        // Desaturated primordial color (e.g., grayish-purple)
                        const primordialColorLerp = primordialRatio;
                        r = (particleColorBase[0] * primordialColorLerp) + (70 * (1 - primordialColorLerp));
                        g = (particleColorBase[1] * primordialColorLerp) + (70 * (1 - primordialColorLerp));
                        b = (particleColorBase[2] * primordialColorLerp) + (90 * (1 - primordialColorLerp));
                    }

                    if (p.trail.length > 1) {
                        ctx.beginPath();
                        ctx.moveTo(p.trail[0].x, p.trail[0].y);
                        for (let i = 1; i < p.trail.length; i++) {
                            const trailSegmentAlpha = (i / p.trail.length) * trailBaseAlpha; 
                            const trailColor = `rgba(200, 200, 200, ${trailSegmentAlpha})`; 
                            ctx.strokeStyle = trailColor;
                            ctx.lineWidth = currentParticleRadius * (i / p.trail.length) * 1.1; 
                            ctx.lineTo(p.trail[i].x, p.trail[i].y);
                        }
                        ctx.stroke();
                    }

                    const lifeRatio = p.life / p.initialLife;
                    const emergenceFactor = p.isPrimordial ? (p.primordialTime / primordialDuration) : (1.0 - Math.pow(1.0 - lifeRatio, 2));
                    const finalRadius = p.isPrimordial ? currentParticleRadius : p.radius * emergenceFactor;
                    const mainParticleAlpha = emergenceFactor * particleHeadBaseAlpha; 
                    const highlightAlpha = emergenceFactor * (particleHeadBaseAlpha + 0.1);

                    ctx.beginPath();
                    ctx.arc(p.x, p.y, Math.max(0.1, finalRadius), 0, Math.PI * 2);
                    if (!p.isPrimordial) { // Use speed-based color only after primordial phase
                        const speed = Math.hypot(p.vx, p.vy);
                        r = Math.min(255, particleColorBase[0] + speed * 15);
                        g = Math.min(255, particleColorBase[1] - speed * 5); 
                        b = Math.min(255, particleColorBase[2] + speed * 5);
                    }
                    ctx.fillStyle = `rgba(${Math.floor(r)}, ${Math.floor(g)}, ${Math.floor(b)}, ${mainParticleAlpha})`;
                    ctx.fill();
                    
                    if (finalRadius > 0.5) { 
                        ctx.beginPath();
                        ctx.arc(p.x, p.y, Math.max(0.1, finalRadius * 0.5), 0, Math.PI * 2);
                        ctx.fillStyle = `rgba(${particleColorHighlight[0]}, ${particleColorHighlight[1]}, ${particleColorHighlight[2]}, ${highlightAlpha})`;
                        ctx.fill();
                    }

                    p.life--;

                    if (p.life <= 0 || p.x < -finalRadius * 2 || p.x > canvas.width + finalRadius*2 || p.y < -finalRadius*2 || p.y > canvas.height + finalRadius*2) {
                        // Reset to primordial state when respawning
                        p.x = Math.random() * canvas.width; 
                        p.y = Math.random() * canvas.height;  
                        p.vx = (Math.random() - 0.5) * 0.1;
                        p.vy = (Math.random() - 0.5) * 0.1;
                        p.life = p.initialLife;
                        p.trail = [];
                        p.isPrimordial = true;
                        p.primordialTime = 0;
                    }
                });
            }

            function animateIntro() {
                drawIntroPattern();
                time++;
                if (canvas.isConnected) {
                    requestAnimationFrame(animateIntro);
                }
            }

            if (!canvas.dataset.animationRunningIntro) {
                canvas.dataset.animationRunningIntro = 'true';
                animateIntro();
            }
        },

        // Color scheme placeholder (fallback)
        renderColorPlaceholder: function(swatchDiv, schemeName) {
            swatchDiv.style.background = 'linear-gradient(to bottom, #333333, #666666, #999999, #cccccc)';
            
            // Add text
            const nameSpan = document.createElement('span');
            nameSpan.textContent = schemeName;
            nameSpan.style.position = 'absolute';
            nameSpan.style.top = '50%';
            nameSpan.style.left = '50%';
            nameSpan.style.transform = 'translate(-50%, -50%)';
            nameSpan.style.color = 'white';
            nameSpan.style.textShadow = '1px 1px 1px black';
            nameSpan.style.fontSize = '10px';
            
            swatchDiv.appendChild(nameSpan);
        }
    };
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        // Make it async call immediately
        (async () => { await window.NoiseVisualizer._preloadKofiImage(); })();
    } else {
        document.addEventListener('DOMContentLoaded', async () => { 
            await window.NoiseVisualizer._preloadKofiImage(); 
        });
    }
} 