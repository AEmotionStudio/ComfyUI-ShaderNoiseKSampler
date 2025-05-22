// shader_renderer.js - Adds shader visualization to ShaderNoiseKSampler node
// This file has a single responsibility: rendering shaders using a canvas approach

import { app } from "../../scripts/app.js";

//-------------------------------------------------------------------------
// HELPER FUNCTIONS
//-------------------------------------------------------------------------
// Helper function to get color scheme name from string representation
function getColorSchemeName(scheme) {
    const schemeNames = {
        "none": "Black & White",
        "blue_red": "Blue to Red",
        "viridis": "Viridis",
        "plasma": "Plasma",
        "inferno": "Inferno",
        "magma": "Magma",
        "turbo": "Turbo",
        "jet": "Jet",
        "rainbow": "Rainbow",
        "cool": "Cool",
        "hot": "Hot",
        "parula": "Parula",
        "hsv": "HSV",
        "autumn": "Autumn",
        "winter": "Winter",
        "spring": "Spring",
        "summer": "Summer",
        "copper": "Copper",
        "pink": "Pink",
        "bone": "Bone",
        "ocean": "Ocean",
        "terrain": "Terrain",
        "neon": "Neon",
        "fire": "Fire"
    };
    return schemeNames[scheme] || scheme;
}

// Register a callback to run when ComfyUI is fully loaded
app.registerExtension({
    name: "ShaderNoiseKSampler.ShaderRenderer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Apply only to the shader noise sampler node
        if (nodeData.name === "ShaderNoiseKSampler") {
            // Store the original onNodeCreated function if it exists
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            
            // Add our own onNodeCreated function
            nodeType.prototype.onNodeCreated = function() {
                // Call the original onNodeCreated if it exists
                if (origOnNodeCreated) {
                    origOnNodeCreated.apply(this, arguments);
                }
                
                // Add shader properties
                this.properties = this.properties || {};
                this.properties.shaderVisible = false;
                this.properties.tooltipsVisible = true; // Add new property for tooltips visibility
                this.properties.shaderType = "domain_warp";
                this.properties.shaderSpeed = 0.2;
                this.properties.shaderColorIntensity = 0.8;
                this.properties.shaderTime = 0;
                this.properties.lastRenderTime = 0;
                this.properties.shaderScale = 1.0;
                this.properties.shaderOctaves = 1;
                this.properties.shaderShapeType = "none";
                this.properties.shaderShapeStrength = 1.0;
                this.properties.shaderWarpStrength = 0.5;
                this.properties.shaderPhaseShift = 0.5;
                this.properties.shaderFrequencyRange = 0; // 0=all, 1=low, 2=mid, 3=high, 4=custom
                this.properties.shaderDistribution = 0; // 0=cauchy, 1=laplace
                this.properties.shaderAdaptationStrength = 0.5;
                this.properties.shaderResolutionScale = 512; // Medium resolution
                
                // Store the shader height
                this.shaderHeight = 200;
                
                // Track animation state
                this.animationFrameId = null;
                this.isShaderActive = false;
                
                // Initialize shader canvas once
                this.initShaderCanvas();
                
                // Custom widget for shader rendering
                this.addWidget("toggle", "Show Shader", this.properties.shaderVisible, (v) => {
                    this.properties.shaderVisible = v;
                    
                    // Update node size
                    const baseSize = origComputeSize ? origComputeSize.call(this, [this.size[0], 0]) : [this.size[0], 0];
                    const baseHeight = baseSize[1];
                    
                    if (v && this.gl) {
                        // If turning on, calculate shader height for current node size
                        if (!this.shaderHeight || this.shaderHeight < 50) {
                            this.shaderHeight = 200; // Default height
                        }
                        
                        // Update node size to include shader
                        this.size[1] = baseHeight + this.shaderHeight;
                        this.resizeShaderCanvas(this.size[0], this.shaderHeight);
                        
                        // Activate shader
                        this.isShaderActive = true;
                        
                        // Reset render time to avoid jumps in animation
                        this.properties.lastRenderTime = 0;
                    } else {
                        // If turning off, remove shader height
                        this.size[1] = baseHeight;
                        
                        // Deactivate shader
                        this.isShaderActive = false;
                        
                        // Cancel any pending animation frame
                        if (this.animationFrameId) {
                            cancelAnimationFrame(this.animationFrameId);
                            this.animationFrameId = null;
                        }
                    }
                    
                    this.setDirtyCanvas(true, true);
                });
                // Set tooltip directly on the widget
                this.widgets[this.widgets.length-1].tooltip = "Toggle the visibility of the shader preview visualization [visual aid to help you see the shader noise pattern before it is applied for sampling]";
                
                // Add toggle for tooltips visibility
                this.addWidget("toggle", "Show Tooltips", this.properties.tooltipsVisible, (v) => {
                    this.properties.tooltipsVisible = v;
                    
                    // Apply tooltip visibility to all widgets
                    for (let i = 0; i < this.widgets.length; i++) {
                        // Skip the Show Tooltips widget itself to always show its tooltip
                        if (this.widgets[i].name === "Show Tooltips") continue;
                        
                        // Store original tooltip if not already stored
                        if (!this.widgets[i]._originalTooltip && this.widgets[i].tooltip) {
                            this.widgets[i]._originalTooltip = this.widgets[i].tooltip;
                        }
                        
                        // Set or clear tooltip based on visibility setting
                        if (v && this.widgets[i]._originalTooltip) {
                            this.widgets[i].tooltip = this.widgets[i]._originalTooltip;
                        } else {
                            this.widgets[i].tooltip = "";
                        }
                    }
                    
                    this.setDirtyCanvas(true, true);
                });
                // Set tooltip directly on the widget
                this.widgets[this.widgets.length-1].tooltip = "Toggle the visibility of tooltips for all shader parameters [disable if tooltips get in the way of controls]";
                
                // Updated shader noise type combo with all the new types
                this.addWidget("combo", "Shader Noise Type 🔄", this.properties.shaderType, (v) => {
                    this.properties.shaderType = v;
                    
                    // Load the shader if it's not already loaded and shader is active
                    if (this.loadShader && (this.isShaderActive || this.properties.shaderVisible)) {
                        this.loadShader(v);
                    }
                    
                    this.setDirtyCanvas(true, true);
                }, { 
                    values: [
                        "domain_warp", "tensor_field", "curl_noise"
                    ]
                });
                // Set tooltip directly on the widget
                this.widgets[this.widgets.length-1].tooltip = "Select the type of shader noise pattern to use in the visualization [different types have different characteristic outputs]";

                // Add shape mask type widget
                this.addWidget("combo", "Shape Mask Type 🔄", this.properties.shaderShapeType, (v) => {
                    this.properties.shaderShapeType = v;
                    this.setDirtyCanvas(true, true);
                }, { 
                    values: ["none", "radial", "linear", "spiral", "checkerboard", "spots", "hexgrid", "stripes", "gradient", "vignette", "cross", "stars", "triangles", "concentric", "rays", "zigzag"]
                });
                // Set tooltip directly on the widget
                this.widgets[this.widgets.length-1].tooltip = "Apply a shape mask to the shader noise pattern to create more complex structures [not post processing - is applied to shader noise pattern before rendering]";

                // Add color scheme widget
                this.properties.colorScheme = this.properties.colorScheme || "none";
                this.addWidget("combo", "Color Scheme 🔄", this.properties.colorScheme, (v) => {
                    this.properties.colorScheme = v;
                    this.setDirtyCanvas(true, true);
                }, { 
                    values: [
                        "none", "blue_red", "viridis", "plasma", "inferno", 
                        "magma", "turbo", "jet", "rainbow", "cool", 
                        "hot", "parula", "hsv", "autumn", "winter", 
                        "spring", "summer", "copper", "pink", "bone", 
                        "ocean", "terrain", "neon", "fire"
                    ]
                });
                // Set tooltip directly on the widget
                this.widgets[this.widgets.length-1].tooltip = "Choose a color palette to apply to the shader noise visualization [not post processing - is applied to the shader noise pattern before rendering]";

                this.addWidget("slider", "Noise Scale 🔄", this.properties.shaderScale, (v) => {
                    this.properties.shaderScale = v;
                    this.setDirtyCanvas(true, true);
                }, { min: 0.1, max: 10.0, step: 0.001, precision: 3 });
                // Set tooltip directly on the widget
                this.widgets[this.widgets.length-1].tooltip = "Adjust the scale of the shader noise pattern - lower values create larger, zoomed-in features; higher values create smaller, zoomed-out features [small value shifts can lead to larger variations]";
                
                this.addWidget("slider", "Octaves 🔄", this.properties.shaderOctaves, (v) => {
                    this.properties.shaderOctaves = v;
                    this.setDirtyCanvas(true, true);
                }, { min: 1, max: 8, step: 0.1, precision: 1 });
                // Set tooltip directly on the widget
                this.widgets[this.widgets.length-1].tooltip = "Number of shader noise layers to combine - higher values add more detail and complexity [small value shifts can lead to larger variations]";
                
                this.addWidget("slider", "Warp Strength 🔄", this.properties.shaderWarpStrength, (v) => {
                    this.properties.shaderWarpStrength = v;
                    this.setDirtyCanvas(true, true);
                }, { min: 0.0, max: 5.0, step: 0.001, precision: 3 });
                // Set tooltip directly on the widget
                this.widgets[this.widgets.length-1].tooltip = "Control how much the shader noise pattern warps and distorts - higher values create more swirling or complex transformations [small adjustments are good for subtle variations]";

                this.addWidget("slider", "Shape Mask Strength 🔄", this.properties.shaderShapeStrength, (v) => {
                    this.properties.shaderShapeStrength = v;
                    this.setDirtyCanvas(true, true);
                }, { min: 0.0, max: 2.0, step: 0.0005, precision: 4 });
                // Set tooltip directly on the widget
                this.widgets[this.widgets.length-1].tooltip = "Adjust the intensity of the shape mask's effect on the shader noise pattern - higher values make the shape more prominent [small adjustments are good for subtle variations - not effective without shape mask]";
                
                this.addWidget("slider", "Phase Shift 🔄", this.properties.shaderPhaseShift, (v) => {
                    this.properties.shaderPhaseShift = v;
                    this.setDirtyCanvas(true, true);
                }, { min: 0.0, max: 2.0, step: 0.0005, precision: 4 });
                // Set tooltip directly on the widget
                this.widgets[this.widgets.length-1].tooltip = "Shift the phase of the shader noise pattern to create different variations or animate patterns over time [small adjustments are good for subtle variations]";
                
                this.addWidget("slider", "Color Intensity 🔄", this.properties.shaderColorIntensity, (v) => {
                    this.properties.shaderColorIntensity = v;
                    this.setDirtyCanvas(true, true);
                }, { min: 0.0, max: 1.0, step: 0.0005, precision: 4 });
                // Set tooltip directly on the widget
                this.widgets[this.widgets.length-1].tooltip = "Adjust the intensity of the color scheme application - lower values are more desaturated, higher values are more vibrant [small adjustments are good for subtle variations - not effective without color scheme]";

                this.addWidget("slider", "Animation Speed 🖥️", this.properties.shaderSpeed, (v) => {
                    this.properties.shaderSpeed = v;
                    this.setDirtyCanvas(true, true);
                }, { min: 0.1, max: 3.0, step: 0.001, precision: 3 });
                // Set tooltip directly on the widget
                this.widgets[this.widgets.length-1].tooltip = "Control how quickly the shader noise pattern animates in the preview [UI effect only - not used in final image generation]";
            
                // Add resolution quality slider
                this.addWidget("slider", "Pixel Resolution 🖥️", this.properties.shaderResolutionScale, (v) => {
                    this.properties.shaderResolutionScale = v;
                    // Trigger canvas resize to apply the new resolution
                    this.resizeShaderCanvas(this.size[0], this.shaderHeight);
                    this.setDirtyCanvas(true, true);
                }, { min: 128, max: 1024, step: 1, precision: 0 });
                // Set tooltip directly on the widget
                    this.widgets[this.widgets.length-1].tooltip = "Set the resolution of the shader preview - higher values increase quality but require more processing power [UI effect only - not used in final image generation]";
                
                // Ensure initial tooltip visibility is correctly applied and _originalTooltip is captured
                const initialTooltipsStateIsOn = this.properties.tooltipsVisible;
                for (let i = 0; i < this.widgets.length; i++) {
                    const widget = this.widgets[i];

                    if (widget.name === "Show Tooltips") {
                        // The "Show Tooltips" widget's tooltip is managed by its own definition
                        // and should always be visible.
                        continue;
                    }

                    // Capture the hardcoded tooltip text as _originalTooltip.
                    // This ensures _originalTooltip always reflects the intended full tooltip text.
                    if (widget.tooltip && typeof widget.tooltip === 'string' && widget.tooltip.trim() !== "") {
                        widget._originalTooltip = widget.tooltip;
                    } else {
                        widget._originalTooltip = ""; // Mark as having no original tooltip if none was defined.
                    }

                    // Set the currently visible tooltip based on the initial state.
                    if (initialTooltipsStateIsOn && widget._originalTooltip) {
                        widget.tooltip = widget._originalTooltip;
                    } else {
                        widget.tooltip = ""; // Hide tooltip if initially off or no original text.
                    }
                }
                
                // Make the node resizable
                this.resizable = true;
                
                // Set minimum size for the node to prevent it from becoming too small
                this.min_height = 100; // Minimum height including the shader area
                this.min_width = 300;  // Minimum width
                
                // Add a method to handle node resize
                nodeType.prototype.onResize = function(size) {
                    // Set flag to indicate we're in a resize operation
                    this._isResizing = true;
                    
                    // Enforce minimum size constraints
                    size[0] = Math.max(this.min_width || 300, size[0]);
                    
                    // Get base height without shader area
                    const baseSize = origComputeSize ? origComputeSize.call(this, [size[0], 0]) : [size[0], 0];
                    const baseHeight = baseSize[1];
                    
                    // Calculate minimum required height
                    const minTotalHeight = baseHeight + 50; // base + minimum shader height
                    size[1] = Math.max(minTotalHeight, size[1]);
                    
                    // Calculate new shader height based on requested size
                    if (this.properties.shaderVisible) {
                        // Update shader height based on the requested node size
                        this.shaderHeight = Math.max(50, size[1] - baseHeight);
                        
                        // Resize the shader canvas to match new dimensions
                        this.resizeShaderCanvas(size[0], this.shaderHeight);
                    }
                    
                    // Update node size
                    this.size = size;
                    
                    // Clear the resize flag
                    this._isResizing = false;
                    
                    // Tell the system to recalculate positions
                    this.setDirtyCanvas(true, true);
                };
                
                // Method to resize the shader canvas
                nodeType.prototype.resizeShaderCanvas = function(width, height) {
                    if (!this.gl || !this.shaderCanvas) return;
                    
                    // Ensure minimum canvas height
                    height = Math.max(50, height);
                    
                    // Calculate aspect ratio
                    const aspectRatio = width / height;
                    
                    // Get target resolution (square resolution that maintains aspect ratio)
                    const targetRes = Math.round(this.properties.shaderResolutionScale);
                    
                    // Calculate width and height at target resolution, maintaining aspect ratio
                    let canvasWidth, canvasHeight;
                    if (aspectRatio >= 1) {
                        // Wider than tall
                        canvasWidth = targetRes;
                        canvasHeight = Math.round(targetRes / aspectRatio);
                    } else {
                        // Taller than wide
                        canvasHeight = targetRes;
                        canvasWidth = Math.round(targetRes * aspectRatio);
                    }
                    
                    // Set internal canvas resolution (actual pixels)
                    this.shaderCanvas.width = canvasWidth;
                    this.shaderCanvas.height = canvasHeight;
                    
                    // Set displayed size via CSS (this is what the user sees)
                    this.shaderCanvas.style.width = width + "px";
                    this.shaderCanvas.style.height = height + "px";
                    
                    // Update WebGL viewport to match internal resolution
                    this.gl.viewport(0, 0, canvasWidth, canvasHeight);
                    
                    // Store the display dimensions
                    this.displayWidth = width;
                    this.displayHeight = height;
                    
                    // Update shader height property to match the passed height
                    // This ensures all parts of the code are using the same value
                    this.shaderHeight = height;
                };
            };
            
            // Override getExtraMenuOptions to add shader visibility toggle
            const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(canvas, options) {
                // Call original if it exists
                if (origGetExtraMenuOptions) {
                    origGetExtraMenuOptions.call(this, canvas, options);
                }

                // Add toggle for shader visibility
                const toggleShaderPreview = {
                    content: this.properties.shaderVisible ? "Hide Shader Preview" : "Show Shader Preview",
                    callback: () => {
                        // Toggle shader visibility
                        this.properties.shaderVisible = !this.properties.shaderVisible;
                        
                        // Update the shader toggle widget to match
                        const shaderWidget = this.widgets?.find(w => w.name === "Show Shader");
                        if (shaderWidget) {
                            shaderWidget.value = this.properties.shaderVisible;
                        }
                        
                        // Update node size
                        const baseSize = origComputeSize ? origComputeSize.call(this, [this.size[0], 0]) : [this.size[0], 0];
                        const baseHeight = baseSize[1];
                        
                        if (this.properties.shaderVisible && this.gl) {
                            // If turning on, calculate shader height for current node size
                            if (!this.shaderHeight || this.shaderHeight < 50) {
                                this.shaderHeight = 200; // Default height
                            }
                            
                            // Update node size to include shader
                            this.size[1] = baseHeight + this.shaderHeight;
                            this.resizeShaderCanvas(this.size[0], this.shaderHeight);
                            
                            // Activate shader
                            this.isShaderActive = true;
                            
                            // Reset render time to avoid jumps in animation
                            this.properties.lastRenderTime = 0;
                        } else {
                            // If turning off, remove shader height
                            this.size[1] = baseHeight;
                            
                            // Deactivate shader
                            this.isShaderActive = false;
                            
                            // Cancel any pending animation frame
                            if (this.animationFrameId) {
                                cancelAnimationFrame(this.animationFrameId);
                                this.animationFrameId = null;
                            }
                        }
                        
                        this.setDirtyCanvas(true, true);
                    }
                };
                
                // Add toggle for tooltips visibility
                const toggleTooltips = {
                    content: this.properties.tooltipsVisible ? "Hide Tooltips" : "Show Tooltips",
                    callback: () => {
                        // Toggle tooltips visibility
                        this.properties.tooltipsVisible = !this.properties.tooltipsVisible;
                        
                        // Update the tooltips toggle widget to match
                        const tooltipsWidget = this.widgets?.find(w => w.name === "Show Tooltips");
                        if (tooltipsWidget) {
                            tooltipsWidget.value = this.properties.tooltipsVisible;
                            
                            // Trigger the widget's callback to update all tooltips
                            if (tooltipsWidget.callback) {
                                tooltipsWidget.callback(this.properties.tooltipsVisible);
                            }
                        }
                        
                        this.setDirtyCanvas(true, true);
                    }
                };
                
                // Add resolution quality submenu
                const resolutionLevels = [
                    ["Very Low (128px)", 128],
                    ["Low (256px)", 256],
                    ["Medium (512px)", 512],
                    ["High (768px)", 768],
                    ["Very High (1024px)", 1024]
                ];
                
                const resolutionSubmenu = {
                    content: "Pixel Resolution",
                    submenu: {
                        options: resolutionLevels.map(([label, value]) => ({
                            content: label,
                            callback: () => {
                                // Update resolution quality property
                                this.properties.shaderResolutionScale = value;
                                
                                // Update the resolution quality widget to match
                                const resolutionWidget = this.widgets?.find(w => w.name === "Pixel Resolution");
                                if (resolutionWidget) {
                                    resolutionWidget.value = value;
                                }
                                
                                // Trigger canvas resize to apply the new resolution
                                this.resizeShaderCanvas(this.size[0], this.shaderHeight);
                                
                                // Redraw canvas
                                this.setDirtyCanvas(true, true);
                            },
                            checked: Math.abs(this.properties.shaderResolutionScale - value) < 1.0
                        }))
                    }
                };
                
                // Add shader noise type selection submenu
                const shaderTypes = [
                    ["Domain Warping", "domain_warp"],
                    ["Tensor Field", "tensor_field"],
                    ["Curl Noise", "curl_noise"],
                ];
                
                const shaderTypeSubmenu = {
                    content: "Shader Noise Type",
                    submenu: {
                        options: shaderTypes.map(([label, type]) => ({
                            content: label,
                            callback: () => {
                                // Update shader noise type property
                                this.properties.shaderType = type;
                                
                                // Update the shader noise type widget to match
                                const typeWidget = this.widgets?.find(w => w.name === "Shader Noise Type");
                                if (typeWidget) {
                                    typeWidget.value = type;
                                }
                                
                                // Load the shader if it's not already loaded and shader is active
                                if (this.loadShader && (this.isShaderActive || this.properties.shaderVisible)) {
                                    this.loadShader(type);
                                }
                                
                                // Redraw canvas
                                this.setDirtyCanvas(true, true);
                            },
                            checked: this.properties.shaderType === type
                        }))
                    }
                };
                
                // Add shape mask type selection submenu
                const shapeTypes = [
                    ["None", "none", 0],
                    ["Radial", "radial", 1],
                    ["Linear", "linear", 2],
                    ["Spiral", "spiral", 3],
                    ["Checkerboard", "checkerboard", 4],
                    ["Spots", "spots", 5],
                    ["Hex Grid", "hexgrid", 6],
                    ["Stripes", "stripes", 7],
                    ["Gradient", "gradient", 8],
                    ["Vignette", "vignette", 9],
                    ["Cross", "cross", 10],
                    ["Stars", "stars", 11],
                    ["Triangles", "triangles", 12],
                    ["Concentric", "concentric", 13],
                    ["Rays", "rays", 14],
                    ["ZigZag", "zigzag", 15]
                ];
                
                const shapeTypeSubmenu = {
                    content: "Shape Mask Type",
                    submenu: {
                        options: shapeTypes.map(([label, type, index]) => ({
                            content: label,
                            callback: () => {
                                // Update shape mask type property
                                this.properties.shaderShapeType = type;
                                
                                // Update the shape mask type widget to match
                                const shapeWidget = this.widgets?.find(w => w.name === "Shape Mask Type");
                                if (shapeWidget) {
                                    shapeWidget.value = type;
                                }
                                
                                // Redraw canvas
                                this.setDirtyCanvas(true, true);
                            },
                            checked: this.properties.shaderShapeType === type
                        }))
                    }
                };
                
                // Add color scheme selection submenu
                const colorSchemes = [
                    ["Black & White", "none"],
                    ["Blue to Red", "blue_red"],
                    ["Viridis", "viridis"],
                    ["Plasma", "plasma"],
                    ["Inferno", "inferno"],
                    ["Magma", "magma"],
                    ["Turbo", "turbo"],
                    ["Jet", "jet"],
                    ["Rainbow", "rainbow"],
                    ["Cool", "cool"],
                    ["Hot", "hot"],
                    ["Parula", "parula"],
                    ["HSV", "hsv"],
                    ["Autumn", "autumn"],
                    ["Winter", "winter"],
                    ["Spring", "spring"],
                    ["Summer", "summer"],
                    ["Copper", "copper"],
                    ["Pink", "pink"],
                    ["Bone", "bone"],
                    ["Ocean", "ocean"],
                    ["Terrain", "terrain"],
                    ["Neon", "neon"],
                    ["Fire", "fire"]
                ];
                
                const colorSchemeSubmenu = {
                    content: "Color Scheme",
                    submenu: {
                        options: colorSchemes.map(([label, scheme]) => ({
                            content: label,
                            callback: () => {
                                // Update color scheme property
                                this.properties.colorScheme = scheme;
                                
                                // Update the color scheme widget to match
                                const colorSchemeWidget = this.widgets?.find(w => w.name === "Color Scheme");
                                if (colorSchemeWidget) {
                                    colorSchemeWidget.value = scheme;
                                }
                                
                                // Redraw canvas
                                this.setDirtyCanvas(true, true);
                            },
                            checked: this.properties.colorScheme === scheme
                        }))
                    }
                };
                
                options.push(null, toggleShaderPreview, toggleTooltips, shaderTypeSubmenu, shapeTypeSubmenu, colorSchemeSubmenu, resolutionSubmenu);
                
                return options;
            };
            
            // Override onConfigure to ensure widgets reflect loaded properties
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                if (origOnConfigure) {
                    origOnConfigure.apply(this, arguments);
                }
            
                // Ensure widgets display the correct values after loading from JSON
                if (this.widgets && this.properties) {
                    const updateWidgetValue = (widgetName, propertyName) => {
                        const widget = this.widgets.find(w => w.name === widgetName);
                        if (widget && this.properties[propertyName] !== undefined) {
                            // Check if the value exists in the widget's options (for combos)
                            if (widget.options?.values && !widget.options.values.includes(this.properties[propertyName])) {
                                console.warn(`Value "${this.properties[propertyName]}" for property "${propertyName}" not found in options for widget "${widgetName}". Using default or first option.`);
                                // Optionally set to default or first value if invalid
                                // widget.value = widget.options.values[0]; 
                            } else {
                                widget.value = this.properties[propertyName];
                            }
                        }
                    };
            
                    updateWidgetValue("Show Shader", "shaderVisible");
                    updateWidgetValue("Show Tooltips", "tooltipsVisible");
                    updateWidgetValue("Shader Noise Type 🔄", "shaderType");
                    updateWidgetValue("Shape Mask Type 🔄", "shaderShapeType");
                    updateWidgetValue("Color Scheme 🔄", "colorScheme");
                    updateWidgetValue("Noise Scale 🔄", "shaderScale");
                    updateWidgetValue("Octaves 🔄", "shaderOctaves");
                    updateWidgetValue("Warp Strength 🔄", "shaderWarpStrength");
                    updateWidgetValue("Shape Mask Strength 🔄", "shaderShapeStrength");
                    updateWidgetValue("Phase Shift 🔄", "shaderPhaseShift");
                    updateWidgetValue("Animation Speed 🖥️", "shaderSpeed");
                    updateWidgetValue("Color Intensity 🔄", "shaderColorIntensity");
                    updateWidgetValue("Pixel Resolution 🖥️", "shaderResolutionScale");
                    // Add any other widgets that need synchronization here
                    
                    // After updating widget values, if the tooltips widget exists and has a callback,
                    // call it to ensure tooltip visibility is correctly applied based on the loaded state.
                    const tooltipsWidget = this.widgets.find(w => w.name === "Show Tooltips");
                    if (tooltipsWidget && tooltipsWidget.callback) {
                        // Ensure the widget.value (this.properties.tooltipsVisible) is up-to-date before calling back
                        // The updateWidgetValue call above should have handled this.
                        tooltipsWidget.callback(tooltipsWidget.value);
                    }
                }
            
                // If the shader should be visible based on loaded properties, ensure size is correct
                if (this.properties?.shaderVisible) {
                    // Calculate base height
                    const baseSize = origComputeSize ? origComputeSize.call(this, [this.size[0], 0]) : [this.size[0], 0];
                    const baseHeight = baseSize[1];
            
                    // Ensure shader height is valid
                    if (!this.shaderHeight || this.shaderHeight < 50) {
                        this.shaderHeight = 200; // Default height if needed
                    }
            
                    // Update node size if necessary
                    const expectedHeight = baseHeight + this.shaderHeight;
                    if (Math.abs(this.size[1] - expectedHeight) > 1) { // Use tolerance for float comparison
                       this.size[1] = expectedHeight;
                       // Also resize the canvas to match the potentially updated shaderHeight
                       this.resizeShaderCanvas(this.size[0], this.shaderHeight); 
                       this.setDirtyCanvas(true, true);
                    }
                    
                    // Ensure shader is active if it's supposed to be visible
                    this.isShaderActive = true;
                } else {
                    // Ensure shader is inactive if not visible
                    this.isShaderActive = false;
                }
            };
            
            // Initialize WebGL for shader rendering
            nodeType.prototype.initShaderCanvas = function() {
                // Create a canvas for shader rendering
                this.shaderCanvas = document.createElement('canvas');
                this.shaderCanvas.width = this.size ? this.size[0] : 250;
                this.shaderCanvas.height = this.shaderHeight;
                
                // Get WebGL context
                this.gl = this.shaderCanvas.getContext('webgl');
                if (!this.gl) {
                    console.error('WebGL not supported');
                    return;
                }
                
                // Set the viewport to match the canvas size
                this.gl.viewport(0, 0, this.shaderCanvas.width, this.shaderCanvas.height);
                
                // Setup shader programs for each shader type
                this.shaderPrograms = {};
                
                // Create shader programs
                const vertexShaderSource = `
                    attribute vec2 a_position;
                    varying vec2 v_texCoord;
                    
                    void main() {
                        v_texCoord = a_position * 0.5 + 0.5;
                        gl_Position = vec4(a_position, 0.0, 1.0);
                    }
                `;
                
                // Common fragment shader header
                const fragmentShaderHeader = `
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
                
                // Common fragment shader footer
                const fragmentShaderFooter = `
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
                
                // Domain Warp shader
                const domainWarpShader = `
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
                `;
                
                // Tensor Field shader
                const tensorFieldShader = `
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
                `;
                
                // Curl Noise shader
                const curlNoiseShader = `
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
                `;
                
                // Store shader sources for lazy loading
                this.shaderSources = {
                    // gaussian: gaussianShader,
                    // perlin: perlinShader,
                    // cellular: cellularShader,
                    // fractal: fractalShader,
                    // waves: wavesShader,
                    domain_warp: domainWarpShader,
                    tensor_field: tensorFieldShader,
                    // heterogeneous_fbm: heterogeneousFbmShader,
                    // interference: interferenceShader,
                    // spectral: spectralShader,
                    // "3d_projection": projectionShader,
                    curl_noise: curlNoiseShader
                };
                
                // Initialize shader programs object
                this.shaderPrograms = {};
                
                // Store common shader parts
                this.vertexShaderSource = vertexShaderSource;
                this.fragmentShaderHeader = fragmentShaderHeader;
                this.fragmentShaderFooter = fragmentShaderFooter;
                
                // Track shader loading
                this.pendingShaders = [];
                this.loadingShader = false;
                
                // Load only the default shader initially
                const defaultShader = this.properties.shaderType || "domain_warp";
                this.loadShader(defaultShader);
                
                // Remove the background loading timer completely
                // this.backgroundLoadTimer = setTimeout(() => {
                //     this.startBackgroundLoading();
                // }, 2000); // Wait 2 seconds before starting background loads
                
                // Create position buffer
                const positionBuffer = this.gl.createBuffer();
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
                this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([
                    -1.0, -1.0,
                     1.0, -1.0,
                    -1.0,  1.0,
                     1.0,  1.0
                ]), this.gl.STATIC_DRAW);
                
                this.positionBuffer = positionBuffer;
            };
            
            // Helper function to create shader program
            nodeType.prototype.createShaderProgram = function(vsSource, fsSource) {
                const gl = this.gl;
                
                // Create shaders
                const vertexShader = gl.createShader(gl.VERTEX_SHADER);
                gl.shaderSource(vertexShader, vsSource);
                gl.compileShader(vertexShader);
                
                const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
                gl.shaderSource(fragmentShader, fsSource);
                gl.compileShader(fragmentShader);
                
                // Create program
                const program = gl.createProgram();
                gl.attachShader(program, vertexShader);
                gl.attachShader(program, fragmentShader);
                gl.linkProgram(program);
                
                if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
                    console.error('Shader program error:', gl.getProgramInfoLog(program));
                    return null;
                }
                
                return program;
            };
            
            // Load shader on demand
            nodeType.prototype.loadShader = function(shaderType) {
                // Skip shader loading if inactive
                if (!this.isShaderActive && !this.properties.shaderVisible) {
                    return;
                }
                
                // Check if shader is already loaded
                if (this.shaderPrograms[shaderType]) {
                    return;
                }
                
                // Check if shader source exists
                if (!this.shaderSources[shaderType]) {
                    console.error('Shader source not found:', shaderType);
                    return;
                }
                
                // Add to pending shaders queue if not already there
                if (!this.pendingShaders.includes(shaderType)) {
                    this.pendingShaders.push(shaderType);
                }
                
                // Start loading process if not already in progress
                if (!this.loadingShader) {
                    this.processNextShader();
                }
            };
            
            // Process the next shader in the queue
            nodeType.prototype.processNextShader = function() {
                // Skip processing if shader is now inactive
                if (!this.isShaderActive && !this.properties.shaderVisible) {
                    this.loadingShader = false;
                    this.pendingShaders = []; // Clear pending shaders
                    return;
                }
                
                if (this.pendingShaders.length === 0) {
                    this.loadingShader = false;
                    return;
                }
                
                this.loadingShader = true;
                const shaderType = this.pendingShaders.shift();
                
                // Compile the shader program
                console.log('Loading shader:', shaderType);
                this.shaderPrograms[shaderType] = this.createShaderProgram(
                    this.vertexShaderSource, 
                    this.fragmentShaderHeader + this.shaderSources[shaderType] + this.fragmentShaderFooter
                );
                
                // Process next shader on next frame to avoid blocking UI
                setTimeout(() => {
                    this.processNextShader();
                }, 10); // Short delay to let UI breathe
            };
            
            // Render the current shader
            nodeType.prototype.renderShader = function() {
                // Skip rendering if shader is inactive or WebGL context is not available
                if (!this.isShaderActive || !this.gl || !this.shaderPrograms) return;
                
                const gl = this.gl;
                const currentShaderType = this.properties.shaderType;
                
                // Check if the current shader program is loaded
                if (!this.shaderPrograms[currentShaderType]) {
                    // Try to use a fallback shader while loading
                    const fallbackShader = Object.keys(this.shaderPrograms)[0];
                    
                    // Request loading of the current shader
                    this.loadShader(currentShaderType);
                    
                    // If no shaders are loaded at all, we can't render
                    if (!fallbackShader) {
                        // Draw loading indicator (using WebGL since 2D context may not be available)
                        gl.clearColor(0.1, 0.1, 0.1, 1.0); // Dark gray background
                        gl.clear(gl.COLOR_BUFFER_BIT);
                        
                        // We can't use text in WebGL easily, so we'll draw a simple loading indicator
                        // using a pulsing rectangle in the center
                        const now = performance.now() / 1000;
                        const pulseSize = 0.2 + 0.1 * Math.sin(now * 3); // Pulsing size
                        
                        // Create a simple program for drawing a colored rectangle
                        if (!this.loadingIndicatorProgram) {
                            const vsSource = `
                                attribute vec2 a_position;
                                void main() {
                                    gl_Position = vec4(a_position, 0.0, 1.0);
                                }
                            `;
                            const fsSource = `
                                precision mediump float;
                                uniform vec4 u_color;
                                void main() {
                                    gl_FragColor = u_color;
                                }
                            `;
                            this.loadingIndicatorProgram = this.createShaderProgram(vsSource, fsSource);
                        }
                        
                        if (this.loadingIndicatorProgram) {
                            gl.useProgram(this.loadingIndicatorProgram);
                            
                            // Set color (light blue)
                            const colorLocation = gl.getUniformLocation(this.loadingIndicatorProgram, "u_color");
                            gl.uniform4f(colorLocation, 0.2, 0.6, 1.0, 1.0);
                            
                            // Draw a rectangle in the center
                            const positionLocation = gl.getAttribLocation(this.loadingIndicatorProgram, "a_position");
                            gl.enableVertexAttribArray(positionLocation);
                            gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
                            gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
                            
                            // Set viewport to maintain aspect ratio
                            gl.viewport(0, 0, this.shaderCanvas.width, this.shaderCanvas.height);
                            
                            // Draw the rectangle
                            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
                        }
                        
                        return;
                    }
                    
                    // Use fallback shader
                    const program = this.shaderPrograms[fallbackShader];
                    if (!program) return;
                    
                    // Clear canvas
                    gl.clearColor(0, 0, 0, 1);
                    gl.clear(gl.COLOR_BUFFER_BIT);
                    
                    // Update animation time
                    this.updateAnimationTime();
                    
                    // Draw with fallback shader
                    this.drawShader(program);
                    return;
                }
                
                const program = this.shaderPrograms[currentShaderType];
                
                // Clear canvas
                gl.clearColor(0, 0, 0, 1);
                gl.clear(gl.COLOR_BUFFER_BIT);
                
                // Update animation time
                this.updateAnimationTime();
                
                // Draw with current shader
                this.drawShader(program);
            };
            
            // Update animation time
            nodeType.prototype.updateAnimationTime = function() {
                // Only update if shader is active
                if (!this.isShaderActive) {
                    return;
                }
                
                const now = performance.now();
                if (this.properties.lastRenderTime > 0) {
                    const delta = (now - this.properties.lastRenderTime) / 1000;
                    this.properties.shaderTime += delta * this.properties.shaderSpeed;
                }
                this.properties.lastRenderTime = now;
            };
            
            // Draw using the provided shader program
            nodeType.prototype.drawShader = function(program) {
                const gl = this.gl;
                
                // Use the provided shader program
                gl.useProgram(program);
                
                // Set common uniform values
                const uniformLocations = {
                    time: gl.getUniformLocation(program, 'u_time'),
                    intensity: gl.getUniformLocation(program, 'u_intensity'),
                    scale: gl.getUniformLocation(program, 'u_scale'),
                    octaves: gl.getUniformLocation(program, 'u_octaves'),
                    persistence: gl.getUniformLocation(program, 'u_persistence'),
                    lacunarity: gl.getUniformLocation(program, 'u_lacunarity'),
                    shapeType: gl.getUniformLocation(program, 'u_shapeType'),
                    shapeStrength: gl.getUniformLocation(program, 'u_shapeStrength'),
                    warpStrength: gl.getUniformLocation(program, 'u_warpStrength'),
                    phaseShift: gl.getUniformLocation(program, 'u_phaseShift'),
                    frequencyRange: gl.getUniformLocation(program, 'u_frequencyRange'),
                    distribution: gl.getUniformLocation(program, 'u_distribution'),
                    adaptationStrength: gl.getUniformLocation(program, 'u_adaptationStrength'),
                    resolutionScale: gl.getUniformLocation(program, 'u_resolutionScale'),
                    colorScheme: gl.getUniformLocation(program, 'u_colorScheme')
                };
                
                // Set all uniforms
                gl.uniform1f(uniformLocations.time, this.properties.shaderTime);
                gl.uniform1f(uniformLocations.intensity, this.properties.shaderColorIntensity);
                gl.uniform1f(uniformLocations.scale, this.properties.shaderScale);
                gl.uniform1f(uniformLocations.octaves, this.properties.shaderOctaves);
                gl.uniform1f(uniformLocations.persistence, 0.5); // Default value
                gl.uniform1f(uniformLocations.lacunarity, 2.0);  // Default value
                
                // Map shapeType string to integer value
                const shapeTypeMap = {
                    'none': 0,
                    'radial': 1,
                    'linear': 2,
                    'spiral': 3,
                    'checkerboard': 4,
                    'spots': 5,
                    'hexgrid': 6,
                    'stripes': 7,
                    'gradient': 8,
                    'vignette': 9,
                    'cross': 10,
                    'stars': 11,
                    'triangles': 12,
                    'concentric': 13,
                    'rays': 14,
                    'zigzag': 15
                };
                gl.uniform1i(uniformLocations.shapeType, shapeTypeMap[this.properties.shaderShapeType] || 0);
                
                // Map colorScheme string to integer value
                const colorSchemeMap = {
                    'none': 0,
                    'blue_red': 1,
                    'viridis': 2,
                    'plasma': 3,
                    'inferno': 4,
                    'magma': 5,
                    'turbo': 6,
                    'jet': 7,
                    'rainbow': 8,
                    'cool': 9,
                    'hot': 10,
                    'parula': 11,
                    'hsv': 12,
                    'autumn': 13,
                    'winter': 14,
                    'spring': 15,
                    'summer': 16,
                    'copper': 17,
                    'pink': 18,
                    'bone': 19,
                    'ocean': 20,
                    'terrain': 21,
                    'neon': 22,
                    'fire': 23
                };
                // Original line had an issue with " none\ (0) being treated as falsy: // gl.uniform1i(uniformLocations.colorScheme, colorSchemeMap[this.properties.colorScheme] || 1); // Default to blue_red
                
                // Set colorScheme - properly handle "none" (value 0) without defaulting to blue_red
                const colorSchemeValue = colorSchemeMap[this.properties.colorScheme];
                gl.uniform1i(uniformLocations.colorScheme, colorSchemeValue !== undefined ? colorSchemeValue : 1); // Default to blue_red only if not found
                
                gl.uniform1f(uniformLocations.shapeStrength, this.properties.shaderShapeStrength);
                gl.uniform1f(uniformLocations.warpStrength, this.properties.shaderWarpStrength);
                gl.uniform1f(uniformLocations.phaseShift, this.properties.shaderPhaseShift);
                gl.uniform1i(uniformLocations.frequencyRange, this.properties.shaderFrequencyRange);
                gl.uniform1i(uniformLocations.distribution, this.properties.shaderDistribution);
                gl.uniform1f(uniformLocations.adaptationStrength, this.properties.shaderAdaptationStrength);
                
                // We no longer need to send resolution to shader since we directly control pixel density via canvas size
                // gl.uniform1f(uniformLocations.resolutionScale, this.properties.shaderResolutionScale);
                
                // Set up position attribute
                const positionAttributeLocation = gl.getAttribLocation(program, 'a_position');
                gl.enableVertexAttribArray(positionAttributeLocation);
                gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
                gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
                
                // Draw
                gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            };
            
            // Override onDrawForeground to render the shader
            const origOnDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                // Render shader if visible
                if (this.properties.shaderVisible && this.gl) {
                    // Calculate where to position the shader
                    let shaderWidth, shaderHeight, shaderY;
                    
                    if (this.flags.collapsed) {
                        shaderWidth = 190;
                        shaderHeight = 20;
                        shaderY = 20; // Position below the title text
                    } else {
                        // Get the height of the node without shader area
                        const baseSize = origComputeSize ? origComputeSize.call(this, [this.size[0], 0]) : [this.size[0], 0];
                        const baseHeight = baseSize[1];
                        
                        // Set shader dimensions based on available space
                        shaderWidth = this.size[0];
                        
                        // Use the current shaderHeight that's been set during resize
                        shaderHeight = this.shaderHeight;
                        
                        // Ensure minimum height
                        if (!shaderHeight || shaderHeight < 50) {
                            shaderHeight = 50;
                            this.shaderHeight = 50;
                        }
                        
                        shaderY = baseHeight;
                        
                        // Resize the canvas if needed to match current dimensions
                        this.resizeShaderCanvas(shaderWidth, shaderHeight);
                    }
                    
                    // Render shader to its canvas
                    this.renderShader();
                    
                    // Call original onDrawForeground to draw the title
                    if (origOnDrawForeground) {
                        origOnDrawForeground.call(this, ctx);
                    }
                    
                    // Save context for drawing with rounded corners
                    ctx.save();
                    
                    // Define corner radius - not too large to maintain clean look
                    const cornerRadius = 8;
                    
                    // Create clipping path with rounded corners for the shader display
                    if (!this.flags.collapsed) {
                        ctx.beginPath();
                        ctx.moveTo(0, shaderY + cornerRadius);
                        ctx.lineTo(0, shaderY + shaderHeight - cornerRadius);
                        ctx.arcTo(0, shaderY + shaderHeight, cornerRadius, shaderY + shaderHeight, cornerRadius);
                        ctx.lineTo(shaderWidth - cornerRadius, shaderY + shaderHeight);
                        ctx.arcTo(shaderWidth, shaderY + shaderHeight, shaderWidth, shaderY + shaderHeight - cornerRadius, cornerRadius);
                        ctx.lineTo(shaderWidth, shaderY + cornerRadius);
                        ctx.arcTo(shaderWidth, shaderY, shaderWidth - cornerRadius, shaderY, cornerRadius);
                        ctx.lineTo(cornerRadius, shaderY);
                        ctx.arcTo(0, shaderY, 0, shaderY + cornerRadius, cornerRadius);
                        ctx.closePath();
                        ctx.clip();
                    }
                    
                    // Draw shader canvas
                    ctx.drawImage(
                        this.shaderCanvas, 
                        0, shaderY, 
                        shaderWidth, shaderHeight
                    );
                    
                    // Restore context after clipped drawing
                    ctx.restore();
                    
                    // Draw a styled border with rounded corners around the shader
                    if (!this.flags.collapsed) {
                        // Save context for border drawing
                        ctx.save();
                        
                        // Create path for border with rounded corners
                        ctx.beginPath();
                        ctx.moveTo(0, shaderY + cornerRadius);
                        ctx.lineTo(0, shaderY + shaderHeight - cornerRadius);
                        ctx.arcTo(0, shaderY + shaderHeight, cornerRadius, shaderY + shaderHeight, cornerRadius);
                        ctx.lineTo(shaderWidth - cornerRadius, shaderY + shaderHeight);
                        ctx.arcTo(shaderWidth, shaderY + shaderHeight, shaderWidth, shaderY + shaderHeight - cornerRadius, cornerRadius);
                        ctx.lineTo(shaderWidth, shaderY + cornerRadius);
                        ctx.arcTo(shaderWidth, shaderY, shaderWidth - cornerRadius, shaderY, cornerRadius);
                        ctx.lineTo(cornerRadius, shaderY);
                        ctx.arcTo(0, shaderY, 0, shaderY + cornerRadius, cornerRadius);
                        ctx.closePath();
                        
                        // Create a subtle glow effect
                        ctx.shadowColor = "rgba(255, 255, 255, 0.2)";
                        ctx.shadowBlur = 2;
                        ctx.strokeStyle = "rgba(200, 200, 200, 0.6)";
                        ctx.lineWidth = 1;
                        ctx.stroke();
                        
                        // Restore context
                        ctx.restore();
                    }
                    
                    // Mark shader as active
                    this.isShaderActive = true;
                    
                    // Request animation frame to keep updating - only when shader is active
                    // Store the requestAnimationFrame ID for potential cancellation
                    this.animationFrameId = requestAnimationFrame(() => {
                        // Only continue rendering if the shader is still active
                        if (this.isShaderActive && this.properties.shaderVisible) {
                            this.setDirtyCanvas(true, true);
                        }
                    });
                } else {
                    // If shader not visible, just call original onDrawForeground
                    if (origOnDrawForeground) {
                        origOnDrawForeground.call(this, ctx);
                    }
                    
                    // Ensure shader is marked as inactive
                    this.isShaderActive = false;
                    
                    // Make sure no animation frame is pending
                    if (this.animationFrameId) {
                        cancelAnimationFrame(this.animationFrameId);
                        this.animationFrameId = null;
                    }
                }
            };
            
            // Override computeSize to adjust node height
            const origComputeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function(size) {
                size = size || [this.size[0] || 300, 0];
                
                // Handle collapsed state
                if (this.flags && this.flags.collapsed) {
                    size[0] = 190; // Collapsed width
                    size[1] = 40; // Collapsed height (title + shader)
                    return size;
                }
                
                // Get base size without shader
                if (origComputeSize) {
                    origComputeSize.call(this, size);
                }
                
                // Store the base height before adding shader height
                const baseHeight = size[1];
                
                // Add height for shader area if visible
                if (this.properties && this.properties.shaderVisible) {
                    // If this is called during user resize, respect the provided size
                    // This is crucial for allowing Y-axis shrinking
                    if (this._isResizing) {
                        // The actual shader height will be calculated in onResize
                        return size;
                    }
                    
                    // Otherwise use the stored shader height for normal rendering
                    const minShaderHeight = 50;
                    const currentShaderHeight = this.shaderHeight || 200;
                    
                    // Set the size to be exactly baseHeight + shaderHeight
                    size[1] = baseHeight + Math.max(minShaderHeight, currentShaderHeight);
                }
                
                return size;
            };
            
            // Add clean-up on node removal
            const origOnRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                if (origOnRemoved) {
                    origOnRemoved.call(this);
                }
                
                // Cancel any pending animation frame
                if (this.animationFrameId) {
                    cancelAnimationFrame(this.animationFrameId);
                    this.animationFrameId = null;
                }
                
                // Clean up WebGL resources
                if (this.gl) {
                    // Delete shader programs
                    for (const programType in this.shaderPrograms) {
                        if (this.shaderPrograms[programType]) {
                            this.gl.deleteProgram(this.shaderPrograms[programType]);
                        }
                    }
                    
                    // Delete loading indicator program if it exists
                    if (this.loadingIndicatorProgram) {
                        this.gl.deleteProgram(this.loadingIndicatorProgram);
                        this.loadingIndicatorProgram = null;
                    }
                    
                    // Delete buffers
                    if (this.positionBuffer) {
                        this.gl.deleteBuffer(this.positionBuffer);
                    }
                    
                    // Clear references
                    this.shaderPrograms = null;
                    this.positionBuffer = null;
                    this.gl = null;
                }
                
                // Clear shader sources to save memory
                this.shaderSources = null;
                
                // Remove canvas reference
                this.shaderCanvas = null;
                
                // Clear any pending shaders
                this.pendingShaders = [];
                this.loadingShader = false;
            };
            
            // Start loading remaining shaders in the background with longer delays between each
            nodeType.prototype.startBackgroundLoading = function() {
                // This function is now a no-op - we only want to load shaders on demand
                return;
            };
        }
    }
});
