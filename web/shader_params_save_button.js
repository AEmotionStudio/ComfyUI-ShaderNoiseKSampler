// Add a button to manually save shader parameters to file
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ShaderParamsSaveButton",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ShaderNoiseKSampler") {
            // Store original onNodeCreated to maintain the node's behavior
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            
            // Add our button to the node
            nodeType.prototype.onNodeCreated = function() {
                // Call original function to preserve existing behavior
                if (origOnNodeCreated) {
                    origOnNodeCreated.apply(this, arguments);
                }
                
                // Keep track of the node instance
                const node = this;
                
                // Track if we're currently saving (to prevent double-clicks)
                let isSaving = false;
                
                // Add indicator widget that will show when parameters need saving
                // Create a custom widget object instead of using the built-in text widget
                const indicatorWidget = {
                    name: "⚠️ Parameters with 🔄 must be saved ⚠️",
                    type: "custom_indicator",
                    value: "",
                    options: { className: "shader-params-indicator" },
                    tooltip: "parameters marked with 🔄 require saving to take effect in the generation process",
                    disabled: true,
                    // Add computeSize method to properly handle resizing
                    computeSize: function() {
                        // Return fixed height but variable width based on parent node width
                        if (this.parent && this.parent.size) {
                            // Adjust width to match node width with some padding
                            return [this.parent.size[0] - 30, 20]; 
                        }
                        return [220, 28]; // Default size if parent not available
                    },
                    // Custom draw method for the widget
                    draw: function(ctx, node, widget_width, y, widget_height) {
                        if (!ctx) return;
                        
                        // Draw background
                        ctx.fillStyle = "rgba(255, 119, 0, 0.1)";
                        ctx.strokeStyle = "#ff7700";
                        ctx.lineWidth = 1;
                        
                        // Draw rounded rectangle for the widget background
                        const radius = 4;
                        const x = 15; // Padding from left edge
                        const width = widget_width - 30; // Subtract padding from both sides
                        
                        ctx.beginPath();
                        ctx.moveTo(x + radius, y);
                        ctx.lineTo(x + width - radius, y);
                        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
                        ctx.lineTo(x + width, y + widget_height - radius);
                        ctx.quadraticCurveTo(x + width, y + widget_height, x + width - radius, y + widget_height);
                        ctx.lineTo(x + radius, y + widget_height);
                        ctx.quadraticCurveTo(x, y + widget_height, x, y + widget_height - radius);
                        ctx.lineTo(x, y + radius);
                        ctx.quadraticCurveTo(x, y, x + radius, y);
                        ctx.closePath();
                        
                        ctx.fill();
                        ctx.stroke();
                        
                        // Draw text
                        ctx.fillStyle = "#ff7700";
                        ctx.font = "bold 12px Arial";
                        ctx.textAlign = "center";
                        ctx.fillText(this.name, x + width / 2, y + widget_height / 2 + 4);
                        
                        return widget_height;
                    }
                };
                
                // Add widget reference to node
                indicatorWidget.parent = this;
                this.widgets.push(indicatorWidget);
                
                // --- Refactored Save Function ---
                const saveParameters = () => {
                    if (isSaving) return;
                    isSaving = true;
                    
                    // Change button text to indicate saving
                    if (saveButtonWidget) saveButtonWidget.name = "Saving...";
                    console.log("Starting save process (triggered)...");
                    
                    // Print each widget and its properties for deep debugging
                    if (node.widgets) {
                        console.log("Widget details:");
                        node.widgets.forEach((widget, index) => {
                            console.log(`Widget ${index}: name="${widget.name}", type=${widget.type}, value=${widget.value}`);
                        });
                    }
                    
                    // Find the widgets that contain our values
                    // Default values
                    let shaderScale = 1.0;
                    let shaderOctaves = 1;
                    let shaderWarpStrength = 0.5;
                    let shaderShapeStrength = 1.0;
                    let shaderPhaseShift = 0.5;
                    let shaderColorIntensity = 0.8;
                    let shaderType = "tensor_field";
                    let shaderShapeType = "none";
                    let colorScheme = "none";
                    
                    // Try to find widgets by partial name match
                    if (node.widgets) {
                        for (const widget of node.widgets) {
                            // Ensure widget and widget.name exist before accessing
                             if (!widget || typeof widget.name !== 'string') continue;
                             
                            const name = widget.name.toLowerCase();
                            
                            // Use partial name matching for more flexibility
                            if (name.includes("scale") && !name.includes("color")) {
                                shaderScale = widget.value;
                                console.log(`Found shader scale: ${widget.value}`);
                            } 
                            else if (name.includes("octaves")) {
                                shaderOctaves = widget.value;
                                console.log(`Found octaves: ${widget.value}`);
                            }
                            else if (name.includes("warp")) {
                                shaderWarpStrength = widget.value;
                                console.log(`Found warp strength: ${widget.value}`);
                            }
                            else if (name.includes("shape") && name.includes("strength")) {
                                shaderShapeStrength = widget.value;
                                console.log(`Found shape strength: ${widget.value}`);
                            }
                            else if (name.includes("phase")) {
                                shaderPhaseShift = widget.value;
                                console.log(`Found phase: ${widget.value}`);
                            }
                            else if (name.includes("color") && name.includes("intensity")) {
                                shaderColorIntensity = widget.value;
                                console.log(`Found color intensity: ${widget.value}`);
                            }
                            else if (name.includes("shader") && name.includes("type")) {
                                shaderType = widget.value;
                                console.log(`Found shader type: ${widget.value}`);
                            }
                            else if (name.includes("shape") && name.includes("type")) {
                                shaderShapeType = widget.value;
                                console.log(`Found shape type: ${widget.value}`);
                            }
                            else if (name.includes("color") && name.includes("scheme")) {
                                colorScheme = widget.value;
                                console.log(`Found color scheme: ${widget.value}`);
                            }
                        }
                    }
                    
                    // Create the properties object with the gathered values
                    const currentProps = {
                        shaderType: shaderType,
                        shaderScale: shaderScale,
                        shaderOctaves: shaderOctaves,
                        shaderWarpStrength: shaderWarpStrength,
                        shaderShapeType: shaderShapeType,
                        shaderShapeStrength: shaderShapeStrength,
                        shaderPhaseShift: shaderPhaseShift,
                        colorScheme: colorScheme,
                        shaderColorIntensity: shaderColorIntensity
                    };
                    
                    console.log("Saving shader properties:", currentProps);
                    
                    try {
                        // Serialize with pretty printing
                        const jsonData = JSON.stringify(currentProps, null, 2);
                        
                        // Save to localStorage with size limit and error handling
                        try {
                            // Check if data is too large for localStorage
                            const dataSize = new Blob([jsonData]).size;
                            if (dataSize > 1024 * 1024) { // 1MB limit
                                console.warn("Shader params data too large for localStorage, skipping localStorage save");
                            } else {
                                // Remove old shader params first to free space
                                const oldKeys = Object.keys(localStorage).filter(key => 
                                    key.startsWith('shader_params') || key.includes('shader')
                                );
                                oldKeys.forEach(key => {
                                    if (key !== 'shader_params') { // Keep only the main one
                                        try {
                                            localStorage.removeItem(key);
                                        } catch (e) { /* ignore */ }
                                    }
                                });
                                
                                localStorage.setItem('shader_params', jsonData);
                                console.log("Saved to localStorage successfully");
                            }
                        } catch (localErr) {
                            if (localErr.name === 'QuotaExceededError') {
                                console.warn("localStorage quota exceeded, skipping localStorage save:", localErr.message);
                                // Try to free up space by removing old workflow data
                                if (window.storageOptimizer) {
                                    window.storageOptimizer.forceCleanup();
                                }
                            } else {
                                console.error("Failed to save to localStorage:", localErr);
                            }
                        }
                        
                        // Send the file directly to the data directory - most reliable method
                        // Create a simple download link to trigger the file save
                        const blob = new Blob([jsonData], {type: 'application/json'});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'shader_params.json';
                        document.body.appendChild(a);
                        a.click();
                        
                        // Show instructions in console AFTER download is triggered
                        console.log("%cIMPORTANT: Please save the downloaded file to the following location (overwrite if exists):", "color: red; font-weight: bold");
                        console.log("%ccustom_nodes/ComfyUI-ShaderNoiseKsampler/data/shader_params.json", "color: blue; font-weight: bold");
                        
                        // Clean up and update button state on success
                        setTimeout(() => {
                            try {
                                document.body.removeChild(a);
                                URL.revokeObjectURL(url);
                                console.log("Shader params JSON download link removed.");
                            } catch (cleanupError) {
                                console.warn("Could not clean up download link:", cleanupError);
                            }
                            
                            // if (saveButtonWidget) saveButtonWidget.name = "✗ Parameters Not Saved";
                            setTimeout(() => {
                                if (saveButtonWidget) saveButtonWidget.name = "💾 Save Shader Parameters";
                                isSaving = false;
                            }, 2000);
                        }, 500);
                        
                    } catch (error) {
                        console.error("Error saving shader parameters:", error);
                        if (saveButtonWidget) saveButtonWidget.name = "Error Saving!";
                         setTimeout(() => {
                            if (saveButtonWidget) saveButtonWidget.name = "💾 Save Shader Parameters";
                            isSaving = false;
                        }, 3000);
                    }
                }; // --- End of Refactored Save Function ---
                
                // Add save button using the refactored function
                const saveButtonWidget = this.addWidget("button", " 💾 Save Shader Parameters", null, saveParameters);
                
                // Extend the widget with a tooltip property that ComfyUI's system recognizes
                saveButtonWidget.options = saveButtonWidget.options || {};
                saveButtonWidget.options.className = "save-shader-params";
                
                // Add tooltip following ComfyUI's pattern
                saveButtonWidget.tooltip = "Save parameters (Alt+S) -- file must be named shader_params.json -- always overwrite old file -- [Save location: `custom_nodes/ComfyUI-ShaderNoiseKsampler/data/shader_params.json`][WIP]";
                
                // Move both the indicator and save button widgets to the end of all widgets
                // This ensures they appear at the bottom of all settings
                setTimeout(() => {
                    if (this.widgets && this.widgets.length > 0) {
                        // Get the indicators's current index
                        const indicatorIndex = this.widgets.indexOf(indicatorWidget);
                        if (indicatorIndex !== -1) {
                            // Remove it from its current position
                            this.widgets.splice(indicatorIndex, 1);
                        }
                        
                        // Get the save button's current index
                        const buttonIndex = this.widgets.indexOf(saveButtonWidget);
                        if (buttonIndex !== -1) {
                            // Remove it from its current position
                            this.widgets.splice(buttonIndex, 1);
                        }
                        
                        // Add them back at the end, indicator first then save button
                        this.widgets.push(indicatorWidget);
                        this.widgets.push(saveButtonWidget);
                        
                        // Ensure the node is redrawn to show the updated widget positions
                        this.setDirtyCanvas(true, true);
                    }
                }, 100);

                // --- Keybinding Logic ---
                const handleKeyDown = (event) => {
                    // Check for Alt+S and if this node is currently selected
                    if (event.altKey && event.key === 's') {
                         // Check if the graph canvas and selected nodes exist
                        if (app.canvas && app.canvas.current_node) {
                            // Check if the currently selected node is this node
                            if (app.canvas.current_node === node) {
                                console.log("Alt+S detected for selected ShaderNoiseKSampler node.");
                                event.preventDefault(); // Prevent browser's default Alt+S action
                                event.stopPropagation(); // Stop event from bubbling up
                                saveParameters(); // Trigger the save function
                            }
                        } else if (app.canvas && app.canvas.selected_nodes && Object.keys(app.canvas.selected_nodes).length === 1 && app.canvas.selected_nodes[node.id]) {
                           // Fallback check for selected_nodes if current_node isn't reliable
                            console.log("Alt+S detected for selected ShaderNoiseKSampler node (using selected_nodes).");
                            event.preventDefault();
                            event.stopPropagation();
                            saveParameters();
                        }
                    }
                };

                // Attach the event listener to the document
                document.addEventListener('keydown', handleKeyDown);
                
                // Store handler reference for removal
                this.handleKeyDown = handleKeyDown; 

                // Original onRemoved method if it exists
                const origOnRemoved = this.onRemoved;
                
                // Add logic to remove the event listener when the node is removed
                this.onRemoved = function() {
                    console.log("Removing keydown listener for node:", this.id);
                    document.removeEventListener('keydown', this.handleKeyDown);
                    
                    // Call original onRemoved if it existed
                    if (origOnRemoved) {
                        origOnRemoved.apply(this, arguments);
                    }
                };
                 // --- End Keybinding Logic ---
            };
        }
    },
    // Add CSS styling for the button
    async setup(app) {
        // Add a small CSS rule for spacing and tooltip styling
        const style = document.createElement("style");
        style.textContent = `
            .save-shader-params {
                padding: 6px;
                background-color: #5c5c5c;
                color: white;
                border-radius: 4px;
                cursor: pointer;
            }
            
            .shader-params-indicator {
                padding: 4px;
                font-weight: bold;
                color: #ff7700;
                background-color: rgba(255, 119, 0, 0.1);
                border-left: 3px solid #ff7700;
                border-radius: 2px;
                text-align: center;
                pointer-events: none;
                user-select: none;
            }
            
            /* The tooltip container needs positioning */
            .tooltip-container {
                position: relative;
                display: inline-block;
            }
            
            /* Style for ComfyUI-compatible tooltips */
            .comfy-tooltip {
                visibility: hidden;
                background-color: rgba(40, 40, 40, 0.95);
                color: #fff;
                text-align: center;
                padding: 8px;
                border-radius: 6px;
                position: absolute;
                z-index: 1000;
                bottom: 125%;
                left: 50%;
                transform: translateX(-50%);
                white-space: nowrap;
                font-size: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            
            /* Show tooltip on hover */
            .tooltip-container:hover .comfy-tooltip {
                visibility: visible;
            }
        `;
        document.head.appendChild(style);
        
        // Hook into the app's widget drawing system to add tooltip support
        // This is done after initial setup to ensure proper integration
        const originalDrawNodeWidgets = LGraphCanvas.prototype.drawNodeWidgets;
        if (originalDrawNodeWidgets) {
            LGraphCanvas.prototype.drawNodeWidgets = function(node, pos, ctx, active_widget) {
                // Call the original method first
                const result = originalDrawNodeWidgets.call(this, node, pos, ctx, active_widget);
                
                // After rendering widgets, check for our tooltip property
                if (node && node.widgets) {
                    for (const widget of node.widgets) {
                        // If widget has our tooltip property and mouse is over it
                        if (widget === active_widget && widget.tooltip) {
                            // Get canvas position
                            const rect = this.canvas.getBoundingClientRect();
                            
                            // Create tooltip if it doesn't exist yet
                            if (!widget._tooltip_elem) {
                                const tooltipContainer = document.createElement('div');
                                tooltipContainer.className = 'tooltip-container';
                                
                                const tooltip = document.createElement('span');
                                tooltip.className = 'comfy-tooltip';
                                tooltip.textContent = widget.tooltip;
                                
                                tooltipContainer.appendChild(tooltip);
                                document.body.appendChild(tooltipContainer);
                                
                                widget._tooltip_elem = tooltipContainer;
                            }
                            
                            // Position tooltip
                            if (widget._tooltip_elem) {
                                const x = pos[0] + rect.left; 
                                const y = pos[1] + rect.top;
                                
                                widget._tooltip_elem.style.left = x + 'px';
                                widget._tooltip_elem.style.top = y - 20 + 'px';
                                widget._tooltip_elem.style.display = 'block';
                            }
                        } else if (widget._tooltip_elem) {
                            // Hide tooltip when not hovering
                            widget._tooltip_elem.style.display = 'none';
                        }
                    }
                }
                
                return result;
            };
        }
    }
}); 