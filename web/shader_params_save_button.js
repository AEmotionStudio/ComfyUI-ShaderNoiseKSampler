/**
 * shader_params_save_button.ts - Adds a button to manually save shader parameters
 */
// Import app from ComfyUI at runtime
// @ts-ignore - ComfyUI provides this at runtime
import { app as comfyApp } from '../../scripts/app.js';
// Cast the runtime import to our typed interface
const appInstance = comfyApp;
/**
 * Helper function to show toast notifications
 */
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('comfy-toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'comfy-toast-container';
        document.body.appendChild(toastContainer);
    }
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `comfy-toast comfy-toast-${type}`;
    toast.setAttribute('role', 'alert');
    toast.textContent = message;
    // Add to container
    toastContainer.appendChild(toast);
    // Trigger animation
    requestAnimationFrame(() => {
        toast.classList.add('show');
    });
    // Remove after delay
    setTimeout(() => {
        toast.classList.remove('show');
        // Wait for transition to finish, with fallback
        const removeToast = () => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
                // Ensure listener is removed if called by fallback
                toast.removeEventListener('transitionend', removeToast);
            }
        };
        toast.addEventListener('transitionend', removeToast, { once: true });
        // Fallback cleanup if transitions are disabled
        setTimeout(removeToast, 350);
    }, 3000);
}
// Expose toast function globally
window.showComfyToast = showToast;
// Define the extension
const extension = {
    name: 'ShaderParamsSaveButton',
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name === 'ShaderNoiseKSampler') {
            // Store original onNodeCreated to maintain the node's behavior
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            // Add our button to the node
            nodeType.prototype.onNodeCreated = function () {
                // Call original function to preserve existing behavior
                if (origOnNodeCreated) {
                    origOnNodeCreated.call(this);
                }
                // Keep track of the node instance
                const node = this;
                // Track if we're currently saving (to prevent double-clicks)
                let isSaving = false;
                // Add indicator widget that will show when parameters need saving
                const indicatorWidget = {
                    name: '⚠️ Parameters with 🔄 must be saved ⚠️',
                    type: 'custom_indicator',
                    value: '',
                    options: { className: 'shader-params-indicator' },
                    tooltip: 'parameters marked with 🔄 require saving to take effect in the generation process',
                    disabled: true,
                    // Add computeSize method to properly handle resizing
                    computeSize() {
                        // Return fixed height but variable width based on parent node width
                        if (this.parent && this.parent.size) {
                            // Adjust width to match node width with some padding
                            return [this.parent.size[0] - 30, 20];
                        }
                        return [220, 28]; // Default size if parent not available
                    },
                    // Custom draw method for the widget
                    draw(ctx, _node, widget_width, y, widget_height) {
                        if (!ctx)
                            return;
                        // Draw background
                        ctx.fillStyle = 'rgba(255, 119, 0, 0.1)';
                        ctx.strokeStyle = '#ff7700';
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
                        ctx.fillStyle = '#ff7700';
                        ctx.font = 'bold 12px Arial';
                        ctx.textAlign = 'center';
                        ctx.fillText(this.name, x + width / 2, y + widget_height / 2 + 4);
                    },
                };
                // Add widget reference to node
                indicatorWidget.parent = this;
                this.widgets.push(indicatorWidget);
                // --- Refactored Save Function ---
                const saveParameters = () => {
                    if (isSaving)
                        return;
                    isSaving = true;
                    // Change button text to indicate saving
                    if (saveButtonWidget)
                        saveButtonWidget.name = 'Saving...';
                    console.log('Starting save process (triggered)...');
                    // Print each widget and its properties for deep debugging
                    if (node.widgets) {
                        console.log('Widget details:');
                        node.widgets.forEach((widget, index) => {
                            console.log(`Widget ${index}: name="${widget.name}", type=${widget.type}, value=${widget.value}`);
                        });
                    }
                    // Find the widgets that contain our values
                    // Default values
                    const currentProps = {
                        shaderType: 'tensor_field',
                        shaderScale: 1.0,
                        shaderOctaves: 1,
                        shaderWarpStrength: 0.5,
                        shaderShapeType: 'none',
                        shaderShapeStrength: 1.0,
                        shaderPhaseShift: 0.5,
                        colorScheme: 'none',
                        shaderColorIntensity: 0.8,
                    };
                    // Try to find widgets by partial name match
                    if (node.widgets) {
                        for (const widget of node.widgets) {
                            // Ensure widget and widget.name exist before accessing
                            if (!widget || typeof widget.name !== 'string')
                                continue;
                            const name = widget.name.toLowerCase();
                            // Use partial name matching for more flexibility
                            if (name.includes('scale') && !name.includes('color')) {
                                currentProps.shaderScale = widget.value;
                                console.log(`Found shader scale: ${widget.value}`);
                            }
                            else if (name.includes('octaves')) {
                                currentProps.shaderOctaves = widget.value;
                                console.log(`Found octaves: ${widget.value}`);
                            }
                            else if (name.includes('warp')) {
                                currentProps.shaderWarpStrength = widget.value;
                                console.log(`Found warp strength: ${widget.value}`);
                            }
                            else if (name.includes('shape') && name.includes('strength')) {
                                currentProps.shaderShapeStrength = widget.value;
                                console.log(`Found shape strength: ${widget.value}`);
                            }
                            else if (name.includes('phase')) {
                                currentProps.shaderPhaseShift = widget.value;
                                console.log(`Found phase: ${widget.value}`);
                            }
                            else if (name.includes('color') && name.includes('intensity')) {
                                currentProps.shaderColorIntensity = widget.value;
                                console.log(`Found color intensity: ${widget.value}`);
                            }
                            else if (name.includes('shader') && name.includes('type')) {
                                currentProps.shaderType = widget.value;
                                console.log(`Found shader type: ${widget.value}`);
                            }
                            else if (name.includes('shape') && name.includes('type')) {
                                currentProps.shaderShapeType = widget.value;
                                console.log(`Found shape type: ${widget.value}`);
                            }
                            else if (name.includes('color') && name.includes('scheme')) {
                                currentProps.colorScheme = widget.value;
                                console.log(`Found color scheme: ${widget.value}`);
                            }
                        }
                    }
                    console.log('Saving shader properties:', currentProps);
                    try {
                        // Serialize with pretty printing
                        const jsonData = JSON.stringify(currentProps, null, 2);
                        // Save to localStorage with size limit and error handling
                        try {
                            // Check if data is too large for localStorage
                            const dataSize = new Blob([jsonData]).size;
                            if (dataSize > 1024 * 1024) {
                                // 1MB limit
                                console.warn('Shader params data too large for localStorage, skipping localStorage save');
                            }
                            else {
                                // Remove old shader params first to free space
                                const oldKeys = Object.keys(localStorage).filter((key) => key.startsWith('shader_params') || key.includes('shader'));
                                oldKeys.forEach((key) => {
                                    if (key !== 'shader_params') {
                                        // Keep only the main one
                                        try {
                                            localStorage.removeItem(key);
                                        }
                                        catch {
                                            /* ignore */
                                        }
                                    }
                                });
                                localStorage.setItem('shader_params', jsonData);
                                console.log('Saved to localStorage successfully');
                            }
                        }
                        catch (localErr) {
                            const error = localErr;
                            if (error.name === 'QuotaExceededError') {
                                console.warn('localStorage quota exceeded, skipping localStorage save:', error.message);
                                // Try to free up space by removing old workflow data
                                if (window.storageOptimizer) {
                                    window.storageOptimizer.forceCleanup();
                                }
                            }
                            else {
                                console.error('Failed to save to localStorage:', localErr);
                            }
                        }
                        // Send parameters to server API for automatic save
                        fetch('/shader_noise_ksampler/save_params', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: jsonData
                        })
                            .then(response => response.json())
                            .then(result => {
                            if (result.status === 'success') {
                                console.log('Shader params saved successfully to server');
                                showToast('Parameters saved successfully!', 'success');
                            }
                            else {
                                throw new Error(result.message || 'Unknown error');
                            }
                        })
                            .catch(error => {
                            console.error('Error saving shader parameters to server:', error);
                            showToast('Error saving parameters!', 'error');
                            if (saveButtonWidget)
                                saveButtonWidget.name = 'Error Saving!';
                        })
                            .finally(() => {
                            setTimeout(() => {
                                if (saveButtonWidget)
                                    saveButtonWidget.name = '💾 Save Shader Parameters';
                                isSaving = false;
                            }, 500);
                        });
                    }
                    catch (error) {
                        console.error('Error saving shader parameters:', error);
                        showToast('Error saving parameters!', 'error');
                        if (saveButtonWidget)
                            saveButtonWidget.name = 'Error Saving!';
                        setTimeout(() => {
                            if (saveButtonWidget)
                                saveButtonWidget.name = '💾 Save Shader Parameters';
                            isSaving = false;
                        }, 3000);
                    }
                };
                // --- End of Refactored Save Function ---
                // Add save button using the refactored function
                const saveButtonWidget = this.addWidget('button', ' 💾 Save Shader Parameters', null, saveParameters);
                // Extend the widget with a tooltip property that ComfyUI's system recognizes
                saveButtonWidget.options = saveButtonWidget.options || {};
                saveButtonWidget.options.className = 'save-shader-params';
                // Add tooltip following ComfyUI's pattern
                saveButtonWidget.tooltip =
                    "Save parameters (Alt+S) -- file must be named shader_params.json -- always overwrite old file -- [Save location: `custom_nodes/ComfyUI-ShaderNoiseKsampler/data/shader_params.json`][WIP]";
                // Move both the indicator and save button widgets to the end of all widgets
                // This ensures they appear at the bottom of all settings
                setTimeout(() => {
                    if (this.widgets && this.widgets.length > 0) {
                        // Get the indicator's current index
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
                        const canvas = appInstance.canvas;
                        // Check if the graph canvas and selected nodes exist
                        if (canvas && canvas.current_node) {
                            // Check if the currently selected node is this node
                            if (canvas.current_node === node) {
                                console.log('Alt+S detected for selected ShaderNoiseKSampler node.');
                                event.preventDefault(); // Prevent browser's default Alt+S action
                                event.stopPropagation(); // Stop event from bubbling up
                                saveParameters(); // Trigger the save function
                            }
                        }
                        else if (canvas &&
                            canvas.selected_nodes &&
                            Object.keys(canvas.selected_nodes).length === 1 &&
                            canvas.selected_nodes[node.id]) {
                            // Fallback check for selected_nodes if current_node isn't reliable
                            console.log('Alt+S detected for selected ShaderNoiseKSampler node (using selected_nodes).');
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
                this.onRemoved = function () {
                    console.log('Removing keydown listener for node:', this.id);
                    if (this.handleKeyDown) {
                        document.removeEventListener('keydown', this.handleKeyDown);
                    }
                    // Call original onRemoved if it existed
                    if (origOnRemoved) {
                        origOnRemoved.call(this);
                    }
                };
                // --- End Keybinding Logic ---
            };
        }
    },
    // Add CSS styling for the button
    async setup(_app) {
        // Add a small CSS rule for spacing and tooltip styling
        const style = document.createElement('style');
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

      /* Toast Notification */
      #comfy-toast-container {
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        display: flex;
        flex-direction: column;
        gap: 10px;
        pointer-events: none;
      }

      .comfy-toast {
        background-color: rgba(40, 40, 40, 0.95);
        color: #fff;
        padding: 12px 24px;
        border-radius: 6px;
        font-size: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        opacity: 0;
        transform: translateY(-20px);
        transition: opacity 0.3s ease, transform 0.3s ease;
        pointer-events: auto;
        text-align: center;
        min-width: 250px;
        border-left: 4px solid #4a9eff;
      }

      .comfy-toast.show {
        opacity: 1;
        transform: translateY(0);
      }

      .comfy-toast-success {
        border-left-color: #2ecc71;
      }

      .comfy-toast-error {
        border-left-color: #e74c3c;
      }

      .comfy-toast-warning {
        border-left-color: #f39c12;
      }
    `;
        document.head.appendChild(style);
        // Hook into the app's widget drawing system to add tooltip support
        // This is done after initial setup to ensure proper integration
        const originalDrawNodeWidgets = LGraphCanvas.prototype.drawNodeWidgets;
        if (originalDrawNodeWidgets) {
            LGraphCanvas.prototype.drawNodeWidgets = function (node, pos, ctx, active_widget) {
                // Call the original method first
                const result = originalDrawNodeWidgets.call(this, node, pos, ctx, active_widget);
                // After rendering widgets, check for our tooltip property
                if (node && node.widgets) {
                    for (const widget of node.widgets) {
                        const buttonWidget = widget;
                        // If widget has our tooltip property and mouse is over it
                        if (widget === active_widget && buttonWidget.tooltip) {
                            // Get canvas position
                            const rect = this.canvas.getBoundingClientRect();
                            // Create tooltip if it doesn't exist yet
                            if (!buttonWidget._tooltip_elem) {
                                const tooltipContainer = document.createElement('div');
                                tooltipContainer.className = 'tooltip-container';
                                const tooltip = document.createElement('span');
                                tooltip.className = 'comfy-tooltip';
                                tooltip.textContent = buttonWidget.tooltip;
                                tooltipContainer.appendChild(tooltip);
                                document.body.appendChild(tooltipContainer);
                                buttonWidget._tooltip_elem = tooltipContainer;
                            }
                            // Position tooltip
                            if (buttonWidget._tooltip_elem) {
                                const x = pos[0] + rect.left;
                                const y = pos[1] + rect.top;
                                buttonWidget._tooltip_elem.style.left = x + 'px';
                                buttonWidget._tooltip_elem.style.top = y - 20 + 'px';
                                buttonWidget._tooltip_elem.style.display = 'block';
                            }
                        }
                        else if (buttonWidget._tooltip_elem) {
                            // Hide tooltip when not hovering
                            buttonWidget._tooltip_elem.style.display = 'none';
                        }
                    }
                }
                return result;
            };
        }
    },
};
// Register the extension
appInstance.registerExtension(extension);
//# sourceMappingURL=shader_params_save_button.js.map