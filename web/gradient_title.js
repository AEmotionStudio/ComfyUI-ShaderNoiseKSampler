import { shouldUpdateFrame, resetShadowContext, createTitleGradient, calculateShimmerPosition, TITLE_CORNER_RADIUS, } from './golden_eyeball.js';
// Cache for rendering optimization (uses shared AnimationCache interface)
const CACHE = {
    lastTime: 0,
    frameCount: 0,
    frameSkip: 2, // Only update animation every X frames
};
// Import app from ComfyUI at runtime (this import is resolved by the browser)
// @ts-ignore - ComfyUI provides this at runtime
import { app } from '../../scripts/app.js';
// Define the extension
const extension = {
    name: 'ShaderNoiseKSampler.GradientTitle',
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        // Apply to both shader noise ksampler nodes
        if (nodeData.name === 'ShaderNoiseKSampler' ||
            nodeData.name === 'ShaderNoiseKSamplerDirect') {
            // Store the original onDrawForeground function if it exists
            const origOnDrawForeground = nodeType.prototype.onDrawForeground;
            // Add our own onDrawForeground function
            nodeType.prototype.onDrawForeground = function (ctx) {
                // Draw gradient title FIRST so it acts as background layer
                drawGradientTitle(this, ctx);
                // Call the original onDrawForeground after, so shader renders on top
                if (origOnDrawForeground) {
                    origOnDrawForeground.call(this, ctx);
                }
            };
            // Clean up resources when node is removed
            const origOnRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                if (origOnRemoved) {
                    origOnRemoved.call(this);
                }
                // Note: CACHE only holds animation timing state, no cleanup needed
            };
        }
    },
};
// Register the extension
app.registerExtension(extension);
/**
 * Draws a gradient title directly on the canvas
 * @param node - The node to apply the gradient to
 * @param ctx - The canvas context
 */
function drawGradientTitle(node, ctx) {
    // Get title area dimensions
    const titleHeight = node.flags.collapsed ? 20 : 30; // Smaller height when collapsed
    const width = node.flags.collapsed ? 190 : node.size[0]; // Smaller width when collapsed
    const fullHeight = node.size[1]; // Get actual node height
    const equationY = 45; // Y position for the equation, moved lower
    // Choose appropriate equation based on node type
    let equation;
    let collapsedEquation;
    if (node.type === 'ShaderNoiseKSamplerDirect') {
        equation = 'Lt = Sα(N) ∘ Kβ(t) ⟿';
        collapsedEquation = 'Lt = Sα(N) ∘ Kβ(t) ⟿';
    }
    else {
        equation = 'Lt = Sα(N) ∘ Kβ(t)';
        collapsedEquation = 'Lt = Sα(N) ∘ Kβ(t)';
    }
    // Update animation frame counter using shared utility
    const shouldUpdateAnimation = shouldUpdateFrame(CACHE);
    // Save current state
    ctx.save();
    // Reset shadow properties using shared utility
    resetShadowContext(ctx);
    // Create vertical background gradient using shared utility
    const gradient = createTitleGradient(ctx, fullHeight);
    // Calculate shimmer position (always compute for smooth animation)
    const shimmerPosition = calculateShimmerPosition(1.0);
    if (shouldUpdateAnimation) {
        CACHE.lastTime = Date.now() / 3000;
    }
    // Add collapse button handler
    if (node.flags.collapsed) {
        // If node is collapsed, adjust the title rendering
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, titleHeight);
        // Draw etched shadow for collapsed version
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.font = 'italic 11px Arial'; // Smaller font for collapsed state
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(collapsedEquation, width / 2 + 1, titleHeight / 2 + 1);
        // Create base golden gradient for collapsed version
        const baseGradient = ctx.createLinearGradient(0, titleHeight / 2 - 5, 0, titleHeight / 2 + 5);
        baseGradient.addColorStop(0, '#B8860B'); // Darker gold
        baseGradient.addColorStop(0.5, '#FFD700'); // Bright gold
        baseGradient.addColorStop(1, '#B8860B'); // Darker gold
        // Draw base golden text
        ctx.fillStyle = baseGradient;
        ctx.font = 'italic 11px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(collapsedEquation, width / 2, titleHeight / 2);
        // Create moving highlight effect for collapsed version
        const highlightWidth = width * 0.4; // Width of the highlight
        const highlightX = -highlightWidth + (width + highlightWidth) * shimmerPosition; // Adjusted range
        const shimmerGradient = ctx.createLinearGradient(highlightX - highlightWidth / 2, 0, highlightX + highlightWidth / 2, 0);
        // Create smooth highlight transition
        shimmerGradient.addColorStop(0, 'rgba(255, 255, 200, 0)');
        shimmerGradient.addColorStop(0.1, 'rgba(255, 255, 200, 0)');
        shimmerGradient.addColorStop(0.5, 'rgba(255, 255, 200, 0.3)');
        shimmerGradient.addColorStop(0.9, 'rgba(255, 255, 200, 0)');
        shimmerGradient.addColorStop(1, 'rgba(255, 255, 200, 0)');
        // Apply highlight
        ctx.fillStyle = shimmerGradient;
        ctx.fillText(collapsedEquation, width / 2, titleHeight / 2);
        // Add outline glow that follows the highlight
        const glowIntensity = Math.max(0, 1 - Math.abs(width / 2 - highlightX) / (width / 4));
        ctx.shadowColor = `rgba(255, 255, 200, ${glowIntensity * 0.3})`;
        ctx.shadowBlur = 4; // Less blur for collapsed version
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
        ctx.fillText(collapsedEquation, width / 2, titleHeight / 2);
        // Skip the rest of the rendering when collapsed
        ctx.restore();
        return;
    }
    // Draw background that fills the entire node (non-collapsed state, reached after early return above)
    ctx.fillStyle = gradient;
    // Use rounded rectangle for the background with rounded corners at the bottom
    ctx.beginPath();
    ctx.moveTo(0, 0); // Start at top-left
    ctx.lineTo(width, 0); // Top edge
    ctx.lineTo(width, fullHeight - TITLE_CORNER_RADIUS); // Right edge before bottom-right corner
    ctx.arcTo(width, fullHeight, width - TITLE_CORNER_RADIUS, fullHeight, TITLE_CORNER_RADIUS); // Bottom-right rounded corner
    ctx.lineTo(TITLE_CORNER_RADIUS, fullHeight); // Bottom edge before bottom-left corner
    ctx.arcTo(0, fullHeight, 0, fullHeight - TITLE_CORNER_RADIUS, TITLE_CORNER_RADIUS); // Bottom-left rounded corner
    ctx.lineTo(0, 0); // Left edge back to top
    ctx.closePath();
    ctx.fill();
    // Draw etched shadow
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.font = 'italic 14px Arial'; // Smaller font for equation
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(equation, width / 2 + 2, equationY + 2);
    // Create base golden gradient
    const baseGradient = ctx.createLinearGradient(0, equationY - 7, 0, equationY + 7);
    baseGradient.addColorStop(0, '#B8860B'); // Darker gold
    baseGradient.addColorStop(0.5, '#FFD700'); // Bright gold
    baseGradient.addColorStop(1, '#B8860B'); // Darker gold
    // Draw base golden text
    ctx.fillStyle = baseGradient;
    ctx.font = 'italic 14px Arial'; // Smaller font for equation
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(equation, width / 2, equationY);
    // Create moving highlight effect
    const highlightWidth = width * 0.4; // Width of the highlight
    const highlightX = -highlightWidth + (width + highlightWidth) * shimmerPosition; // Adjusted range
    const shimmerGradient = ctx.createLinearGradient(highlightX - highlightWidth / 2, 0, highlightX + highlightWidth / 2, 0);
    // Create smooth highlight transition
    shimmerGradient.addColorStop(0, 'rgba(255, 255, 200, 0)');
    shimmerGradient.addColorStop(0.1, 'rgba(255, 255, 200, 0)');
    shimmerGradient.addColorStop(0.5, 'rgba(255, 255, 200, 0.3)');
    shimmerGradient.addColorStop(0.9, 'rgba(255, 255, 200, 0)');
    shimmerGradient.addColorStop(1, 'rgba(255, 255, 200, 0)');
    // Apply highlight
    ctx.fillStyle = shimmerGradient;
    ctx.fillText(equation, width / 2, equationY);
    // Add outline glow that follows the highlight
    const glowIntensity = Math.max(0, 1 - Math.abs(width / 2 - highlightX) / (width / 4));
    ctx.shadowColor = `rgba(255, 255, 200, ${glowIntensity * 0.3})`;
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.fillText(equation, width / 2, equationY);
    // Restore context state
    ctx.restore();
}
//# sourceMappingURL=gradient_title.js.map