// gradient_title.js - Adds a custom gradient title to ShaderNoiseKSampler node

import { app } from "../../scripts/app.js";

// Cache for rendering optimization
const CACHE = {
    titleCanvas: null,
    titleCtx: null,
    lastWidth: 0,
    lastHeight: 0,
    lastTime: 0,
    frameCount: 0,
    frameSkip: 2, // Only update animation every X frames
    collapsed: {
        canvas: null,
        ctx: null,
        lastWidth: 0
    }
};

// Register a callback to run when ComfyUI is fully loaded
app.registerExtension({
    name: "ShaderNoiseKSampler.GradientTitle",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Apply to both shader noise ksampler nodes
        if (nodeData.name === "ShaderNoiseKSampler" || nodeData.name === "ShaderNoiseKSamplerDirect") {
            // Store the original onDrawForeground function if it exists
            const origOnDrawForeground = nodeType.prototype.onDrawForeground;
            
            // Add our own onDrawForeground function
            nodeType.prototype.onDrawForeground = function(ctx) {
                // Call the original onDrawForeground if it exists
                if (origOnDrawForeground) {
                    origOnDrawForeground.apply(this, arguments);
                }
                
                // Draw a custom gradient title
                drawGradientTitle(this, ctx);
            };

            // Clean up resources when node is removed
            const origOnRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                if (origOnRemoved) {
                    origOnRemoved.apply(this, arguments);
                }
                
                // Clear cached canvases to prevent memory leaks
                CACHE.titleCanvas = null;
                CACHE.titleCtx = null;
                CACHE.collapsed.canvas = null;
                CACHE.collapsed.ctx = null;
            };
        }
    }
});

/**
 * Draws a gradient title directly on the canvas
 * @param {LGraphNode} node - The node to apply the gradient to
 * @param {CanvasRenderingContext2D} ctx - The canvas context
 */
function drawGradientTitle(node, ctx) {
    // Get title area dimensions
    const titleHeight = node.flags.collapsed ? 20 : 30; // Smaller height when collapsed
    const width = node.flags.collapsed ? 190 : node.size[0]; // Smaller width when collapsed
    const fullHeight = node.size[1]; // Get actual node height
    const equationY = 45; // Y position for the equation, moved lower
    
    // Choose appropriate equation based on node type
    let equation, collapsedEquation;
    
    if (node.type === "ShaderNoiseKSamplerDirect") {
        equation = "Lt = Sα(N) ∘ Kβ(t) ⟿";
        collapsedEquation = "Lt = Sα(N) ∘ Kβ(t) ⟿";
    } else {
        equation = "Lt = Sα(N) ∘ Kβ(t)";
        collapsedEquation = "Lt = Sα(N) ∘ Kβ(t)";
    }
    
    // Update animation frame counter (performance optimization)
    CACHE.frameCount = (CACHE.frameCount + 1) % (CACHE.frameSkip + 1);
    const shouldUpdateAnimation = CACHE.frameCount === 0;
    
    // Save current state
    ctx.save();
    
    // Reset shadow properties for gradient drawing
    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    
    // Create vertical black gradient for entire background that fills node height
    const gradient = ctx.createLinearGradient(0, 0, 0, fullHeight);
    gradient.addColorStop(0, "#000000");     // Pure black at top
    gradient.addColorStop(0.2, "#101010");   // Transition to very dark gray
    gradient.addColorStop(1, "#101010");     // Very dark gray at bottom
    
    // Create smooth shimmer effect for mathematical formula - only calculate if animation should update
    let shimmerPosition = 0.5; // Default middle position
    if (shouldUpdateAnimation) {
        const time = Date.now() / 3000; // Faster time factor (changed from 4000)
        shimmerPosition = (Math.sin(time) + 1) / 2; // Changed from cos to sin for left-to-right only
        // Store for later use if needed
        CACHE.lastTime = time;
    } else {
        // Reuse last calculation for animation frames we're skipping
        const time = CACHE.lastTime || Date.now() / 3000;
        shimmerPosition = (Math.sin(time) + 1) / 2;
    }
    
    // Add collapse button handler
    if (node.flags.collapsed) {
        // If node is collapsed, adjust the title rendering
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, titleHeight);
        
        // Draw etched shadow for collapsed version
        ctx.fillStyle = "rgba(0,0,0,0.3)";
        ctx.font = "italic 11px Arial"; // Smaller font for collapsed state
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(collapsedEquation, width / 2 + 1, titleHeight / 2 + 1);
        
        // Create base golden gradient for collapsed version
        const baseGradient = ctx.createLinearGradient(0, titleHeight/2 - 5, 0, titleHeight/2 + 5);
        baseGradient.addColorStop(0, "#B8860B");    // Darker gold
        baseGradient.addColorStop(0.5, "#FFD700");  // Bright gold
        baseGradient.addColorStop(1, "#B8860B");    // Darker gold
        
        // Draw base golden text
        ctx.fillStyle = baseGradient;
        ctx.font = "italic 11px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(collapsedEquation, width / 2, titleHeight / 2);
        
        // Create moving highlight effect for collapsed version
        const highlightWidth = width * 0.4; // Width of the highlight
        const highlightX = -highlightWidth + (width + highlightWidth) * shimmerPosition; // Adjusted range
        
        const shimmerGradient = ctx.createLinearGradient(
            highlightX - highlightWidth/2, 0,
            highlightX + highlightWidth/2, 0
        );
        
        // Create smooth highlight transition
        shimmerGradient.addColorStop(0, "rgba(255, 255, 200, 0)");
        shimmerGradient.addColorStop(0.1, "rgba(255, 255, 200, 0)");
        shimmerGradient.addColorStop(0.5, "rgba(255, 255, 200, 0.3)");
        shimmerGradient.addColorStop(0.9, "rgba(255, 255, 200, 0)");
        shimmerGradient.addColorStop(1, "rgba(255, 255, 200, 0)");
        
        // Apply highlight
        ctx.fillStyle = shimmerGradient;
        ctx.fillText(collapsedEquation, width / 2, titleHeight / 2);
        
        // Add outline glow that follows the highlight
        const glowIntensity = Math.max(0, 1 - Math.abs(width/2 - highlightX)/(width/4));
        ctx.shadowColor = `rgba(255, 255, 200, ${glowIntensity * 0.3})`;
        ctx.shadowBlur = 4; // Less blur for collapsed version
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
        ctx.fillText(collapsedEquation, width / 2, titleHeight / 2);
        
        // Skip the rest of the rendering when collapsed
        ctx.restore();
        return;
    }
    
    // Draw background that fills the entire node
    ctx.fillStyle = gradient;
    
    // Use rounded rectangle for the background with rounded corners at the bottom
    if (!node.flags.collapsed) {
        const cornerRadius = 8; // Adjust radius as needed
        
        // Create path for rounded rectangle
        ctx.beginPath();
        ctx.moveTo(0, 0); // Top-left corner (no rounding)
        ctx.lineTo(width, 0); // Top-right corner (no rounding)
        ctx.lineTo(width, fullHeight - cornerRadius); // Right edge before bottom-right corner
        ctx.arcTo(width, fullHeight, width - cornerRadius, fullHeight, cornerRadius); // Bottom-right rounded corner
        ctx.lineTo(cornerRadius, fullHeight); // Bottom edge before bottom-left corner
        ctx.arcTo(0, fullHeight, 0, fullHeight - cornerRadius, cornerRadius); // Bottom-left rounded corner
        ctx.lineTo(0, 0); // Left edge back to top
        ctx.closePath();
        ctx.fill();
    } else {
        // Keep regular rectangle for collapsed state
        ctx.fillRect(0, 0, width, fullHeight); // No extra padding
    }
    
    // Draw etched shadow
    ctx.fillStyle = "rgba(0,0,0,0.3)";
    ctx.font = "italic 14px Arial"; // Smaller font for equation
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(equation, width / 2 + 2, equationY + 2);
    
    // Create base golden gradient
    const baseGradient = ctx.createLinearGradient(0, equationY - 7, 0, equationY + 7);
    baseGradient.addColorStop(0, "#B8860B");    // Darker gold
    baseGradient.addColorStop(0.5, "#FFD700");  // Bright gold
    baseGradient.addColorStop(1, "#B8860B");    // Darker gold
    
    // Draw base golden text
    ctx.fillStyle = baseGradient;
    ctx.font = "italic 14px Arial"; // Smaller font for equation
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(equation, width / 2, equationY);
    
    // Create moving highlight effect
    const highlightWidth = width * 0.4; // Width of the highlight
    const highlightX = -highlightWidth + (width + highlightWidth) * shimmerPosition; // Adjusted range
    
    const shimmerGradient = ctx.createLinearGradient(
        highlightX - highlightWidth/2, 0,
        highlightX + highlightWidth/2, 0
    );
    
    // Create smooth highlight transition
    shimmerGradient.addColorStop(0, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(0.1, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(0.5, "rgba(255, 255, 200, 0.3)");
    shimmerGradient.addColorStop(0.9, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(1, "rgba(255, 255, 200, 0)");
    
    // Apply highlight
    ctx.fillStyle = shimmerGradient;
    ctx.fillText(equation, width / 2, equationY);
    
    // Add outline glow that follows the highlight
    const glowIntensity = Math.max(0, 1 - Math.abs(width/2 - highlightX)/(width/4));
    ctx.shadowColor = `rgba(255, 255, 200, ${glowIntensity * 0.3})`;
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.fillText(equation, width / 2, equationY);
    
    // Restore context state
    ctx.restore();
}