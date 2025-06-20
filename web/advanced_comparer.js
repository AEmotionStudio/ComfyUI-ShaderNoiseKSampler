// AdvancedImageComparer.js
// Implementation based on rgthree's image_comparer.js but simplified and enhanced for batch comparison
// Now includes auto-fill slot functionality

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("AdvancedImageComparer module loaded");

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

function imageDataToUrl(data) {
    return api.apiURL(`/view?filename=${encodeURIComponent(data.filename)}&type=${data.type || ""}&subfolder=${data.subfolder || ""}${app.getPreviewFormatParam()}${app.getRandParam()}`);
}

/**
 * Draws a custom golden eyeball using canvas drawing commands
 * @param {CanvasRenderingContext2D} ctx - The canvas context
 * @param {number} centerX - X center position
 * @param {number} centerY - Y center position
 * @param {number} size - Size of the eyeball
 * @param {number} shimmerPosition - Position of the shimmer effect (0-1)
 */
function drawGoldenEyeball(ctx, centerX, centerY, size, shimmerPosition) {
    // Refined proportions for smaller, cleaner look
    const eyeWidth = size * 1.6;
    const eyeHeight = size * 1.0;
    const irisRadius = size * 0.35;
    const pupilRadius = size * 0.15;
    
    ctx.save();
    
    // Create base golden gradient exactly like the mathematical formula
    const baseGradient = ctx.createLinearGradient(0, centerY - size*0.7, 0, centerY + size*0.7);
    baseGradient.addColorStop(0, "#B8860B");    // Darker gold
    baseGradient.addColorStop(0.5, "#FFD700");  // Bright gold
    baseGradient.addColorStop(1, "#B8860B");    // Darker gold
    
    // Create moving highlight effect (matching gradient_title.js exactly)
    const highlightWidth = eyeWidth * 0.4; // Width of the highlight (same as formula)
    const highlightX = -highlightWidth + (eyeWidth + highlightWidth) * shimmerPosition; // Adjusted range
    
    const shimmerGradient = ctx.createLinearGradient(
        centerX + highlightX - highlightWidth/2, 0,
        centerX + highlightX + highlightWidth/2, 0
    );
    
    // Create smooth highlight transition (exact same as formula)
    shimmerGradient.addColorStop(0, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(0.1, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(0.5, "rgba(255, 255, 200, 0.3)");
    shimmerGradient.addColorStop(0.9, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(1, "rgba(255, 255, 200, 0)");
    
    // Draw etched shadow for all outlines (matching formula style)
    ctx.strokeStyle = "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1.5;
    ctx.lineCap = "round";
    
    // Shadow for main eye outline
    ctx.beginPath();
    ctx.ellipse(centerX + 2, centerY + 2, eyeWidth/2, eyeHeight/2, 0, 0, Math.PI * 2);
    ctx.stroke();
    
    // Shadow for iris
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX + 2, centerY + 2, irisRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Shadow for pupil
    ctx.beginPath();
    ctx.arc(centerX + 2, centerY + 2, pupilRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Draw 8 eyelashes/rays around the eye - shadows first
    const rayCount = 8;
    const rayLength = size * 0.7;
    ctx.lineWidth = 1;
    
    for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        const startX = centerX + Math.cos(angle) * (eyeWidth/2 + 1);
        const startY = centerY + Math.sin(angle) * (eyeHeight/2 + 1);
        const endX = centerX + Math.cos(angle) * (eyeWidth/2 + rayLength);
        const endY = centerY + Math.sin(angle) * (eyeHeight/2 + rayLength);
        
        ctx.beginPath();
        ctx.moveTo(startX + 2, startY + 2);
        ctx.lineTo(endX + 2, endY + 2);
        ctx.stroke();
    }
    
    // Now draw the golden base outlines
    ctx.strokeStyle = baseGradient;
    ctx.lineWidth = 1.5;
    
    // Main eye outline
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, eyeWidth/2, eyeHeight/2, 0, 0, Math.PI * 2);
    ctx.stroke();
    
    // Iris outline
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX, centerY, irisRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Pupil outline
    ctx.beginPath();
    ctx.arc(centerX, centerY, pupilRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Draw 8 eyelashes/rays with base golden color
    for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        const startX = centerX + Math.cos(angle) * (eyeWidth/2 + 1);
        const startY = centerY + Math.sin(angle) * (eyeHeight/2 + 1);
        const endX = centerX + Math.cos(angle) * (eyeWidth/2 + rayLength);
        const endY = centerY + Math.sin(angle) * (eyeHeight/2 + rayLength);
        
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
    }
    
    // Add subtle iris texture lines (fewer for cleaner look)
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 8; i++) {
        const angle = (i / 8) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(centerX + Math.cos(angle) * pupilRadius * 1.1, centerY + Math.sin(angle) * pupilRadius * 1.1);
        ctx.lineTo(centerX + Math.cos(angle) * irisRadius * 0.9, centerY + Math.sin(angle) * irisRadius * 0.9);
        ctx.stroke();
    }
    
    // Apply shimmer highlight effect to all outlines
    ctx.strokeStyle = shimmerGradient;
    ctx.lineWidth = 1.5;
    
    // Shimmer on main eye outline
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, eyeWidth/2, eyeHeight/2, 0, 0, Math.PI * 2);
    ctx.stroke();
    
    // Shimmer on iris
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX, centerY, irisRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Shimmer on pupil
    ctx.beginPath();
    ctx.arc(centerX, centerY, pupilRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Shimmer on 8 eyelashes/rays
    for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        const startX = centerX + Math.cos(angle) * (eyeWidth/2 + 1);
        const startY = centerY + Math.sin(angle) * (eyeHeight/2 + 1);
        const endX = centerX + Math.cos(angle) * (eyeWidth/2 + rayLength);
        const endY = centerY + Math.sin(angle) * (eyeHeight/2 + rayLength);
        
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
    }
    
    // Add outline glow that follows the highlight (matching formula exactly)
    const glowIntensity = Math.max(0, 1 - Math.abs(centerX - (centerX + highlightX))/(eyeWidth/4));
    ctx.shadowColor = `rgba(255, 255, 200, ${glowIntensity * 0.3})`;
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    
    // Final glow pass on main outline
    ctx.strokeStyle = baseGradient;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, eyeWidth/2, eyeHeight/2, 0, 0, Math.PI * 2);
    ctx.stroke();
    
    ctx.restore();
}

/**
 * Draws a gradient background with golden eyeball title
 * @param {LGraphNode} node - The node to apply the gradient to
 * @param {CanvasRenderingContext2D} ctx - The canvas context
 */
function drawGradientTitle(node, ctx) {
    // Get title area dimensions
    const titleHeight = node.flags.collapsed ? 20 : 30; // Smaller height when collapsed
    const width = node.flags.collapsed ? 190 : node.size[0]; // Smaller width when collapsed
    const fullHeight = node.size[1]; // Get actual node height
    const eyeballY = node.flags.collapsed ? titleHeight / 2 : 25; // Much closer to top
    
    // Eyeball size based on collapsed state (made smaller)
    const eyeballSize = node.flags.collapsed ? 6 : 10;
    
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
    
    // Create smooth shimmer effect for eyeball - only calculate if animation should update
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
        
        // Draw custom golden eyeball for collapsed version
        drawGoldenEyeball(ctx, width / 2, titleHeight / 2, eyeballSize, shimmerPosition);
        
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
    
    // Draw custom golden eyeball for expanded version
    drawGoldenEyeball(ctx, width / 2, eyeballY, eyeballSize, shimmerPosition);
    
    // Restore context state
    ctx.restore();
}

class AdvancedImageComparerWidget {
    constructor(name, node) {
        this.name = name;
        this.type = "custom";
        this.node = node;
        this._value = { images: [] };
        this.selected = [];
        this.imgs = [];
        this.options = { serialize: false };
        this.y = 0;
        this.last_y = 0;
        
        // Enhanced batch handling properties
        this.imagesA = [];
        this.imagesB = [];
        this.currentPairIndex = 0;
        this.maxPairs = 0;
        this.animationFrame = null;
        this.autoPlayEnabled = false;
        this.autoPlaySpeed = 2000; // 2 seconds per pair
        
        // Batch pagination properties
        this.currentBatchPage = 0;
        this.pairsPerPage = 3; // Show 3 pairs per page in batch mode
        this.maxBatchPages = 0;
    }

    set value(v) {
        // Process the images from the execution result
        const images = v.images || [];
        
        const imagesA = images.filter(img => img.is_image_a);
        const imagesB = images.filter(img => img.is_image_b);
        
        // Store all images for batch processing
        this.imagesA = imagesA.map((img, index) => ({
            name: `A${index + 1}`,
                selected: true,
            url: imageDataToUrl(img),
            img: null,
            index: index
        }));
        
        this.imagesB = imagesB.map((img, index) => ({
            name: `B${index + 1}`,
                selected: true,
            url: imageDataToUrl(img),
            img: null,
            index: index
        }));
        
        // Calculate max pairs for comparison
        this.maxPairs = Math.max(this.imagesA.length, this.imagesB.length);
        this.currentPairIndex = 0;
        
        // Calculate batch pagination
        this.maxBatchPages = Math.ceil(this.maxPairs / this.pairsPerPage);
        this.currentBatchPage = 0;
        
        // Set the value and update selected pair
        this._value = { images: [...this.imagesA, ...this.imagesB] };
        this.updateSelectedPair();
        
        // Load all images for batch modes
        this.loadAllImages();
        
        // Update controls when images change
        if (this.node && this.node.updateControlsVisibility) {
            this.node.updateControlsVisibility();
        }
    }

    get value() {
        return this._value || { images: [] };
    }

    loadAllImages() {
        const allImages = [...this.imagesA, ...this.imagesB];
        
        allImages.forEach(imageData => {
            if (!imageData.img && imageData.url) {
                imageData.img = new Image();
                imageData.img.onload = () => {
                    this.node.setDirtyCanvas(true, false);
                };
                imageData.img.onerror = (error) => {
                    console.error("[AdvancedImageComparer] Image failed to load:", imageData.name, "URL:", imageData.url, "Error:", error);
                };
                imageData.img.src = imageData.url;
            }
        });
    }

    setSelected(selected) {
        this.selected = selected;
        this.imgs = [];
        
        for (const sel of selected) {
            if (!sel.img && sel.url) {
                sel.img = new Image();
                sel.img.onload = () => {
                    this.node.setDirtyCanvas(true, false);
                };
                sel.img.onerror = (error) => {
                    console.error("[AdvancedImageComparer] Image failed to load:", sel.name, "URL:", sel.url, "Error:", error);
                };
                sel.img.src = sel.url;
            }
            if (sel.img) {
                this.imgs.push(sel.img);
            }
        }
    }

    draw(ctx, node, width, y, height) {
        this.y = y;
        this.last_y = y;
        
        // Calculate the actual available height for images
        const [nodeWidth, nodeHeight] = node.size;
        const availableHeight = nodeHeight - y - 10;
        
        const mode = node.properties?.comparer_mode || "Slider";
        
        switch (mode) {
            case "Click":
                this.drawClickMode(ctx, y, width, availableHeight);
                break;
                
            case "Side-by-Side":
                this.drawSideBySideMode(ctx, y, width, availableHeight);
                break;
                
            case "Stacked":
                this.drawStackedMode(ctx, y, width, availableHeight);
                break;
                
            case "Grid":
                this.drawGridMode(ctx, y, width, availableHeight);
                break;
                
            case "Carousel":
                this.drawCarouselMode(ctx, y, width, availableHeight);
                break;
                
            case "Batch":
                this.drawBatchMode(ctx, y, width, availableHeight);
                break;
                
            case "Onion Skin":
                this.drawOnionSkinMode(ctx, y, width, availableHeight);
                break;
                
            default: // "Slider"
                this.drawSliderMode(ctx, y, width, availableHeight);
                break;
        }
        
        // Draw controls for batch modes (excluding Carousel which uses widgets)
        if (["Grid", "Batch"].includes(mode) && this.maxPairs > 1) {
            this.drawBatchControls(ctx, y, width, availableHeight);
        }
        
        // Draw pair indicator for Carousel mode
        if (mode === "Carousel" && this.maxPairs > 1) {
            this.drawPairIndicator(ctx, y + availableHeight - 25, width);
        }
    }

    drawClickMode(ctx, y, width, availableHeight) {
        const imageIndex = this.node.isPointerDown ? 1 : 0;
        this.drawImage(ctx, this.selected[imageIndex], y, width, availableHeight);
    }

    drawSideBySideMode(ctx, y, width, availableHeight) {
        if (this.selected[0]) {
            this.drawImageSideBySide(ctx, this.selected[0], y, width, availableHeight, 0);
        }
        if (this.selected[1]) {
            this.drawImageSideBySide(ctx, this.selected[1], y, width, availableHeight, 1);
        }
    }
                
    drawStackedMode(ctx, y, width, availableHeight) {
        if (this.selected[0]) {
            this.drawImageStacked(ctx, this.selected[0], y, width, availableHeight, 0);
        }
        if (this.selected[1]) {
            this.drawImageStacked(ctx, this.selected[1], y, width, availableHeight, 1);
        }
    }
                
    drawSliderMode(ctx, y, width, availableHeight) {
        if (this.selected[0]) {
            this.drawImage(ctx, this.selected[0], y, width, availableHeight);
        }
        
        if (this.selected[1] && this.node.isPointerOver) {
            const cropX = this.node.pointerOverPos[0];
            this.drawImage(ctx, this.selected[1], y, width, availableHeight, cropX);
        }
    }

    drawGridMode(ctx, y, width, availableHeight) {
        // Calculate grid layout
        const pairs = Math.min(this.maxPairs, 64); // Show max 64 pairs in grid
        const cols = Math.ceil(Math.sqrt(pairs * 2)); // 2 images per pair
        const rows = Math.ceil((pairs * 2) / cols);
        
        const cellWidth = width / cols;
        const cellHeight = (availableHeight - 40) / rows; // Reserve space for controls
        
        let cellIndex = 0;
        for (let i = 0; i < pairs; i++) {
            const imageA = this.imagesA[i];
            const imageB = this.imagesB[i];
            
            if (imageA && imageA.img) {
                const col = cellIndex % cols;
                const row = Math.floor(cellIndex / cols);
                this.drawImageInCell(ctx, imageA, y + row * cellHeight, col * cellWidth, cellWidth, cellHeight, `A${i + 1}`);
                cellIndex++;
            }
            
            if (imageB && imageB.img) {
                const col = cellIndex % cols;
                const row = Math.floor(cellIndex / cols);
                this.drawImageInCell(ctx, imageB, y + row * cellHeight, col * cellWidth, cellWidth, cellHeight, `B${i + 1}`);
                cellIndex++;
            }
        }
    }

    drawCarouselMode(ctx, y, width, availableHeight) {
        const imageA = this.imagesA[this.currentPairIndex];
        const imageB = this.imagesB[this.currentPairIndex];
        
        // Draw current pair side by side (use full available height since controls are now widgets)
        if (imageA && imageA.img) {
            this.drawImageSideBySide(ctx, imageA, y, width, availableHeight, 0);
        }
        if (imageB && imageB.img) {
            this.drawImageSideBySide(ctx, imageB, y, width, availableHeight, 1);
        }
    }

    drawBatchMode(ctx, y, width, availableHeight) {
        const pairHeight = (availableHeight - 40) / this.pairsPerPage; // Reserve space for controls
        const startPairIndex = this.currentBatchPage * this.pairsPerPage;
        const endPairIndex = Math.min(startPairIndex + this.pairsPerPage, this.maxPairs);
        
        for (let i = 0; i < this.pairsPerPage; i++) {
            const pairIndex = startPairIndex + i;
            if (pairIndex >= this.maxPairs) break;
            
            const imageA = this.imagesA[pairIndex];
            const imageB = this.imagesB[pairIndex];
            const pairY = y + i * pairHeight;
            
            if (imageA && imageA.img) {
                this.drawImageInPair(ctx, imageA, pairY, 0, width / 2, pairHeight, 0);
            }
            if (imageB && imageB.img) {
                this.drawImageInPair(ctx, imageB, pairY, width / 2, width / 2, pairHeight, 1);
            }
            
            // Draw separator between pairs
            if (i < this.pairsPerPage - 1 && pairIndex < this.maxPairs - 1) {
                ctx.beginPath();
                ctx.moveTo(0, pairY + pairHeight);
                ctx.lineTo(width, pairY + pairHeight);
                ctx.strokeStyle = "rgba(255,255,255,0.3)";
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }
    }

    drawImageInCell(ctx, imageData, y, x, cellWidth, cellHeight, label) {
        if (!imageData || !imageData.img || !imageData.img.naturalWidth || !imageData.img.naturalHeight) {
            return;
        }

        const image = imageData.img;
        const padding = 4;
        const usableWidth = cellWidth - padding * 2;
        const usableHeight = cellHeight - padding * 2;
        
        const imageAspect = image.naturalWidth / image.naturalHeight;
        const cellAspect = usableWidth / usableHeight;
        
        let targetWidth, targetHeight;
        
        if (imageAspect > cellAspect) {
            targetWidth = usableWidth;
            targetHeight = usableWidth / imageAspect;
        } else {
            targetHeight = usableHeight;
            targetWidth = usableHeight * imageAspect;
        }

        const destX = x + padding + (usableWidth - targetWidth) / 2;
        const destY = y + padding + (usableHeight - targetHeight) / 2;

        ctx.save();
        
        // Draw border
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x + padding, y + padding, usableWidth, usableHeight);
        
        // Draw image
        ctx.drawImage(
            image,
            0, 0, image.naturalWidth, image.naturalHeight,
            destX, destY, targetWidth, targetHeight
        );
        
        // Draw label
        ctx.fillStyle = "rgba(0,0,0,0.8)";
        ctx.fillRect(destX, destY, 25, 18);
        ctx.fillStyle = "white";
        ctx.font = "11px Arial";
        ctx.textAlign = "center";
        ctx.fillText(label, destX + 12, destY + 13);

        ctx.restore();
    }

    drawImageInPair(ctx, imageData, y, x, pairWidth, pairHeight, imageIndex) {
        if (!imageData || !imageData.img || !imageData.img.naturalWidth || !imageData.img.naturalHeight) {
            return;
        }

        const image = imageData.img;
        const padding = 2;
        const usableWidth = pairWidth - padding * 2;
        const usableHeight = pairHeight - padding * 2;
        
        const imageAspect = image.naturalWidth / image.naturalHeight;
        const pairAspect = usableWidth / usableHeight;
        
        let targetWidth, targetHeight;
        
        if (imageAspect > pairAspect) {
            targetWidth = usableWidth;
            targetHeight = usableWidth / imageAspect;
        } else {
            targetHeight = usableHeight;
            targetWidth = usableHeight * imageAspect;
        }

        const destX = x + padding + (usableWidth - targetWidth) / 2;
        const destY = y + padding + (usableHeight - targetHeight) / 2;

        ctx.save();
        
        // Draw image
        ctx.drawImage(
            image,
            0, 0, image.naturalWidth, image.naturalHeight,
            destX, destY, targetWidth, targetHeight
        );
        
        // Draw label
        ctx.fillStyle = "rgba(0,0,0,0.8)";
        ctx.fillRect(destX, destY, 25, 18);
        ctx.fillStyle = "white";
        ctx.font = "12px Arial";
        ctx.textAlign = "center";
        ctx.fillText(imageData.name, destX + 12, destY + 13);
        
        // Draw separator line for side-by-side in batch mode
        if (imageIndex === 0 && pairWidth < this.node.size[0]) {
            ctx.beginPath();
            ctx.moveTo(x + pairWidth, y);
            ctx.lineTo(x + pairWidth, y + pairHeight);
            ctx.strokeStyle = "rgba(255,255,255,0.5)";
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        ctx.restore();
    }

    drawBatchControls(ctx, y, width, availableHeight) {
        const controlY = y + availableHeight - 30;
        const mode = this.node.properties?.comparer_mode || "Slider";
        
        ctx.save();
        
        // Draw control background
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(0, controlY, width, 30);
        
        ctx.fillStyle = "white";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        
        if (mode === "Carousel") {
            // Carousel controls
            const buttonWidth = 60;
            const buttonHeight = 20;
            const buttonY = controlY + 5;
            
            // Previous button
            ctx.fillStyle = "rgba(100,100,100,0.8)";
            ctx.fillRect(10, buttonY, buttonWidth, buttonHeight);
            ctx.fillStyle = "white";
            ctx.textAlign = "center";
            ctx.fillText("◀ Prev", 10 + buttonWidth/2, buttonY + 14);
            
            // Next button
            ctx.fillStyle = "rgba(100,100,100,0.8)";
            ctx.fillRect(80, buttonY, buttonWidth, buttonHeight);
            ctx.fillStyle = "white";
            ctx.fillText("Next ▶", 80 + buttonWidth/2, buttonY + 14);
            
            // Auto-play button
            ctx.fillStyle = this.autoPlayEnabled ? "rgba(0,150,0,0.8)" : "rgba(100,100,100,0.8)";
            ctx.fillRect(150, buttonY, buttonWidth, buttonHeight);
            ctx.fillStyle = "white";
            ctx.fillText(this.autoPlayEnabled ? "⏸ Pause" : "▶ Play", 150 + buttonWidth/2, buttonY + 14);
            
            // Pair indicator
            ctx.textAlign = "right";
            ctx.fillStyle = "white";
            ctx.fillText(`${this.currentPairIndex + 1} / ${this.maxPairs}`, width - 10, buttonY + 14);
        } else {
            // General batch info
            ctx.fillText(`Images: A(${this.imagesA.length}) B(${this.imagesB.length})`, 10, controlY + 18);
            
            if (mode === "Grid") {
                ctx.textAlign = "right";
                ctx.fillText(`Showing ${Math.min(this.maxPairs, 64)} pairs`, width - 10, controlY + 18);
            } else if (mode === "Batch") {
                const startPair = this.currentBatchPage * this.pairsPerPage + 1;
                const endPair = Math.min((this.currentBatchPage + 1) * this.pairsPerPage, this.maxPairs);
                ctx.textAlign = "right";
                ctx.fillText(`Showing pairs ${startPair}-${endPair} of ${this.maxPairs}`, width - 10, controlY + 18);
            }
        }
        
        ctx.restore();
    }

    drawPairIndicator(ctx, y, width) {
        ctx.save();
        
        const dotSize = 8;
        const dotSpacing = 12;
        const totalWidth = this.maxPairs * dotSpacing - (dotSpacing - dotSize);
        const startX = (width - totalWidth) / 2;
        
        for (let i = 0; i < this.maxPairs; i++) {
            const x = startX + i * dotSpacing;
            
            ctx.beginPath();
            ctx.arc(x + dotSize/2, y + dotSize/2, dotSize/2, 0, Math.PI * 2);
            
            if (i === this.currentPairIndex) {
                ctx.fillStyle = "rgba(255,255,255,1)";
            } else {
                ctx.fillStyle = "rgba(255,255,255,0.4)";
            }
            ctx.fill();
        }
        
        ctx.restore();
    }

    drawImage(ctx, imageData, y, nodeWidth, availableHeight, cropX) {
        if (!imageData) {
            return;
        }
        
        if (!imageData.img) {
            return;
        }
        
        if (!imageData.img.naturalWidth || !imageData.img.naturalHeight) {
            return;
        }

        const image = imageData.img;
        
        const imageAspect = image.naturalWidth / image.naturalHeight;
        const widgetAspect = nodeWidth / availableHeight;
        
        let targetWidth, targetHeight, offsetX = 0;
        
        // Ensure minimum size for visibility
        const minSize = 100;
        const effectiveHeight = Math.max(availableHeight, minSize);
        const effectiveWidth = Math.max(nodeWidth, minSize);
        
        if (imageAspect > widgetAspect) {
            targetWidth = effectiveWidth;
            targetHeight = effectiveWidth / imageAspect;
        } else {
            targetHeight = effectiveHeight;
            targetWidth = effectiveHeight * imageAspect;
            offsetX = (nodeWidth - targetWidth) / 2;
        }

        const widthMultiplier = image.naturalWidth / targetWidth;
        const sourceX = 0;
        const sourceY = 0;
        const sourceWidth = cropX != null ? (cropX - offsetX) * widthMultiplier : image.naturalWidth;
        const sourceHeight = image.naturalHeight;
        
        const destX = (nodeWidth - targetWidth) / 2;
        const destY = y + (availableHeight - targetHeight) / 2;
        const destWidth = cropX != null ? cropX - offsetX : targetWidth;
        const destHeight = targetHeight;

        ctx.save();
        
        if (cropX) {
            ctx.beginPath();
            ctx.rect(destX, destY, destWidth, destHeight);
            ctx.clip();
        }
        
        ctx.drawImage(
            image,
            sourceX, sourceY, sourceWidth, sourceHeight,
            destX, destY, destWidth, destHeight
        );
        
        // Draw slider line
        if (cropX != null && cropX >= (nodeWidth - targetWidth) / 2 && cropX <= targetWidth + offsetX) {
            ctx.beginPath();
            ctx.moveTo(cropX, destY);
            ctx.lineTo(cropX, destY + destHeight);
            ctx.globalCompositeOperation = "difference";
            ctx.strokeStyle = "rgba(255,255,255,1)";
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        ctx.restore();
    }

    // Navigation methods for carousel mode
    nextPair() {
        if (this.currentPairIndex < this.maxPairs - 1) {
            this.currentPairIndex++;
            this.updateSelectedPair();
            this.node.setDirtyCanvas(true, false);
        }
    }

    previousPair() {
        if (this.currentPairIndex > 0) {
            this.currentPairIndex--;
            this.updateSelectedPair();
            this.node.setDirtyCanvas(true, false);
        }
    }

    toggleAutoPlay() {
        this.autoPlayEnabled = !this.autoPlayEnabled;
        
        if (this.autoPlayEnabled) {
            this.startAutoPlay();
        } else {
            this.stopAutoPlay();
        }
        
        this.updateNodeControls();
        this.node.setDirtyCanvas(true, false);
    }

    // Update the node's control widgets
    updateNodeControls() {
        if (this.node.pairInfoWidget) {
            this.node.pairInfoWidget.value = `${this.currentPairIndex + 1} / ${this.maxPairs}`;
        }
        if (this.node.autoPlayButton) {
            this.node.autoPlayButton.name = this.autoPlayEnabled ? "⏸ Pause" : "▶ Play";
        }
        if (this.node.batchSelectorWidget) {
            this.node.batchSelectorWidget.value = (this.currentPairIndex + 1).toString();
        }
        if (this.node.batchPageInfoWidget) {
            this.node.batchPageInfoWidget.value = `Page ${this.currentBatchPage + 1} / ${this.maxBatchPages}`;
        }
    }

    // Update the selected pair for modes that show individual pairs
    updateSelectedPair() {
        const mode = this.node.properties?.comparer_mode || "Slider";
        
        // For modes that show individual pairs, update the selected images
        if (["Slider", "Click", "Side-by-Side", "Stacked", "Onion Skin"].includes(mode)) {
            const processedImages = [];
            
            if (this.imagesA[this.currentPairIndex]) {
                processedImages.push(this.imagesA[this.currentPairIndex]);
            }
            if (this.imagesB[this.currentPairIndex]) {
                processedImages.push(this.imagesB[this.currentPairIndex]);
            }
            
            this.setSelected(processedImages);
        }
        
        this.updateNodeControls();
    }

    startAutoPlay() {
        if (this.animationFrame) {
            clearInterval(this.animationFrame);
        }
        
        this.animationFrame = setInterval(() => {
            if (this.currentPairIndex >= this.maxPairs - 1) {
                // Loop back to the beginning
                this.currentPairIndex = 0;
            } else {
                this.currentPairIndex++;
            }
            this.updateSelectedPair();
            this.node.setDirtyCanvas(true, false);
        }, this.autoPlaySpeed);
    }

    stopAutoPlay() {
        if (this.animationFrame) {
            clearInterval(this.animationFrame);
            this.animationFrame = null;
        }
    }

    // Batch pagination methods
    nextBatchPage() {
        if (this.currentBatchPage < this.maxBatchPages - 1) {
            this.currentBatchPage++;
            this.updateNodeControls();
            this.node.setDirtyCanvas(true, false);
        }
    }

    previousBatchPage() {
        if (this.currentBatchPage > 0) {
            this.currentBatchPage--;
            this.updateNodeControls();
            this.node.setDirtyCanvas(true, false);
        }
    }

    drawImageSideBySide(ctx, imageData, y, nodeWidth, availableHeight, imageIndex) {
        if (!imageData || !imageData.img || !imageData.img.naturalWidth || !imageData.img.naturalHeight) {
            return;
        }

        const image = imageData.img;
        const halfWidth = nodeWidth / 2;
        const padding = 2; // Small padding between images
        const usableWidth = halfWidth - padding;
        
        // Ensure minimum size for visibility
        const minSize = 100;
        const effectiveHeight = Math.max(availableHeight, minSize);
        const effectiveWidth = Math.max(usableWidth, minSize / 2);
        
        const imageAspect = image.naturalWidth / image.naturalHeight;
        const widgetAspect = effectiveWidth / effectiveHeight;
        
        let targetWidth, targetHeight;
        
        if (imageAspect > widgetAspect) {
            targetWidth = effectiveWidth;
            targetHeight = effectiveWidth / imageAspect;
        } else {
            targetHeight = effectiveHeight;
            targetWidth = effectiveHeight * imageAspect;
        }

        const destX = imageIndex === 0 ? 
            (halfWidth - targetWidth) / 2 : 
            halfWidth + padding + (usableWidth - targetWidth) / 2;
        const destY = y + (availableHeight - targetHeight) / 2;

        ctx.save();
        
        // Draw image
        ctx.drawImage(
            image,
            0, 0, image.naturalWidth, image.naturalHeight,
            destX, destY, targetWidth, targetHeight
        );
        
        // Draw label
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(destX, destY, 20, 20);
        ctx.fillStyle = "white";
        ctx.font = "12px Arial";
        ctx.textAlign = "center";
        ctx.fillText(imageData.name, destX + 10, destY + 14);
        
        // Draw separator line
        if (imageIndex === 0) {
            ctx.beginPath();
            ctx.moveTo(halfWidth, y);
            ctx.lineTo(halfWidth, y + availableHeight);
            ctx.strokeStyle = "rgba(255,255,255,0.5)";
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        ctx.restore();
    }

    drawImageStacked(ctx, imageData, y, nodeWidth, availableHeight, imageIndex) {
        if (!imageData || !imageData.img || !imageData.img.naturalWidth || !imageData.img.naturalHeight) {
            return;
        }

        const image = imageData.img;
        const halfHeight = availableHeight / 2;
        const padding = 2; // Small padding between images
        const usableHeight = halfHeight - padding;
        
        // Ensure minimum size for visibility
        const minSize = 100;
        const effectiveHeight = Math.max(usableHeight, minSize / 2);
        const effectiveWidth = Math.max(nodeWidth, minSize);
        
        const imageAspect = image.naturalWidth / image.naturalHeight;
        const widgetAspect = effectiveWidth / effectiveHeight;
        
        let targetWidth, targetHeight;
        
        if (imageAspect > widgetAspect) {
            targetWidth = effectiveWidth;
            targetHeight = effectiveWidth / imageAspect;
        } else {
            targetHeight = effectiveHeight;
            targetWidth = effectiveHeight * imageAspect;
        }

        const destX = (nodeWidth - targetWidth) / 2;
        const destY = imageIndex === 0 ? 
            y + (halfHeight - targetHeight) / 2 : 
            y + halfHeight + padding + (usableHeight - targetHeight) / 2;

        ctx.save();
        
        // Draw image
        ctx.drawImage(
            image,
            0, 0, image.naturalWidth, image.naturalHeight,
            destX, destY, targetWidth, targetHeight
        );
        
        // Draw label
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(destX, destY, 20, 20);
        ctx.fillStyle = "white";
        ctx.font = "12px Arial";
        ctx.textAlign = "center";
        ctx.fillText(imageData.name, destX + 10, destY + 14);
        
        // Draw separator line
        if (imageIndex === 0) {
            ctx.beginPath();
            ctx.moveTo(0, y + halfHeight);
            ctx.lineTo(nodeWidth, y + halfHeight);
            ctx.strokeStyle = "rgba(255,255,255,0.5)";
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        ctx.restore();
    }

    drawOnionSkinMode(ctx, y, width, availableHeight) {
        const opacity = this.node.properties?.onionSkinOpacity || 0.5; // Default to 50% opacity

        if (this.selected[0]) {
            this.drawImage(ctx, this.selected[0], y, width, availableHeight);
        }

        if (this.selected[1]) {
            ctx.save();
            ctx.globalAlpha = opacity;
            this.drawImage(ctx, this.selected[1], y, width, availableHeight);
            ctx.restore();
        }
    }

    computeSize(width) {
        const mode = this.node?.properties?.comparer_mode || "Slider";
        
        // Base height calculation - reduced minimum sizes for better sizing flexibility
        let height = Math.max(150, width * 0.5); // Reduced from 300 and 0.75
        
        // Adjust height based on layout mode
        switch (mode) {
            case "Stacked":
                // Stacked mode needs more height to show both images vertically
                height = Math.max(200, width * 0.8); // Reduced from 500 and 1.2
                break;
            case "Side-by-Side":
                // Side-by-side can use standard height but smaller than before
                height = Math.max(120, width * 0.4); // Reduced from 300 and 0.6
                break;
            case "Grid":
                // Grid mode needs more height to show multiple pairs
                const pairs = Math.min(this.maxPairs || 1, 64);
                const cols = Math.ceil(Math.sqrt(pairs * 2));
                const rows = Math.ceil((pairs * 2) / cols);
                height = Math.max(180, (width / cols) * rows + 60); // Reduced from 400 and 80
                break;
            case "Carousel":
                // Carousel mode uses standard height plus controls
                height = Math.max(150, width * 0.4 + 60); // Reduced from 350, 0.6, and 80
                break;
            case "Batch":
                // Batch mode shows multiple pairs vertically
                const visiblePairs = Math.min(this.maxPairs || 1, 3);
                height = Math.max(200, visiblePairs * (width * 0.3) + 60); // Reduced from 500, 0.4, and 80
                break;
            case "Onion Skin":
                height = Math.max(150, width * 0.5); // Reduced from 300 and 0.75
                break;
            default:
                // Slider and Click modes use standard height but smaller
                height = Math.max(150, width * 0.5); // Reduced from 300 and 0.75
                break;
        }
        
        return [width, height];
    }

    mouse(event, pos, node) {
        // Handle mouse events for the widget
        const mode = node.properties?.comparer_mode || "Slider";
        
        if (event.type === "pointermove") {
            node.pointerOverPos = [...pos];
            if (mode === "Slider") {
                node.setDirtyCanvas(true, false);
            }
            return true;
        }
        
        if (event.type === "pointerdown") {
            // Handle clicks on grid cells for selection
            if (mode === "Grid" && this.maxPairs > 1) {
                const pairs = Math.min(this.maxPairs, 64);
                const cols = Math.ceil(Math.sqrt(pairs * 2));
                const rows = Math.ceil((pairs * 2) / cols);
                const cellWidth = node.size[0] / cols;
                const widgetHeight = node.size[1] - this.y - 10;
                const cellHeight = (widgetHeight - 40) / rows; // Reserve space for controls
                
                const col = Math.floor(pos[0] / cellWidth);
                const row = Math.floor(pos[1] / cellHeight);
                const cellIndex = row * cols + col;
                
                // Determine which image was clicked
                let imageIndex = Math.floor(cellIndex / 2);
                let isImageB = cellIndex % 2 === 1;
                
                if (imageIndex < this.maxPairs) {
                    // Switch to carousel mode to focus on this pair
                    this.currentPairIndex = imageIndex;
                    node.properties.comparer_mode = "Carousel";
                    node.layoutWidget.value = "Carousel";
                    node.setSize(node.computeSize());
                    node.setDirtyCanvas(true, false);
                    return true;
                }
            }
        }
        
        return false;
    }

    // Cleanup method
    onRemoved() {
        this.stopAutoPlay();
    }
}

app.registerExtension({
    name: "AdvancedImageComparer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only process if this is our target node
        if (nodeData.name !== "AdvancedImageComparer") {
            return;
        }
        
        // Add properties
        nodeType.prototype.properties = nodeType.prototype.properties || {};
        nodeType.prototype.properties.comparer_mode = "Slider";
        nodeType.prototype.properties.onionSkinOpacity = 0.5; // Default opacity for Onion Skin mode
        
        nodeType["@comparer_mode"] = {
            type: "combo",
            values: ["Slider", "Click", "Side-by-Side", "Stacked", "Grid", "Carousel", "Batch", "Onion Skin"],
        };

        // Store the original onDrawForeground function if it exists
        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        
        // Add our own onDrawForeground function for gradient background
        nodeType.prototype.onDrawForeground = function(ctx) {
            // Call the original onDrawForeground if it exists
            if (origOnDrawForeground) {
                origOnDrawForeground.apply(this, arguments);
            }
            
            // Draw our custom gradient title
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

        // Initialize state variables
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            
            // Ensure properties are properly initialized
            this.properties = this.properties || {};
            if (!this.properties.comparer_mode) {
                this.properties.comparer_mode = "Slider";
            }
            
            this.isPointerDown = false;
            this.isPointerOver = false;
            this.pointerOverPos = [0, 0];
            this.imageIndex = 0;
            
            // Add layout control widget
            this.layoutWidget = this.addWidget("combo", "Layout Mode", this.properties.comparer_mode, (value) => {
                this.properties.comparer_mode = value;
                
                // Show/hide controls based on mode
                this.updateControlsVisibility();
                
                this.setDirtyCanvas(true, false);
            }, {
                values: ["Slider", "Click", "Side-by-Side", "Stacked", "Grid", "Carousel", "Batch", "Onion Skin"]
            });

            // Add batch selector widget for modes that need it
            this.batchSelectorWidget = this.addWidget("combo", "View Pair", "1", (value) => {
                console.log("[AdvancedImageComparer] Batch selector changed to:", value);
                const pairIndex = parseInt(value) - 1;
                if (this.comparerWidget && pairIndex >= 0 && pairIndex < this.comparerWidget.maxPairs) {
                    this.comparerWidget.currentPairIndex = pairIndex;
                    this.comparerWidget.updateSelectedPair();
                    this.setDirtyCanvas(true, false);
                }
            }, {
                values: ["1"]
            });

            // Add carousel control widgets (initially hidden)
            this.prevButton = this.addWidget("button", "◀ Previous", null, () => {
                if (this.comparerWidget) {
                    this.comparerWidget.previousPair();
                }
            });
            
            this.nextButton = this.addWidget("button", "Next ▶", null, () => {
                if (this.comparerWidget) {
                    this.comparerWidget.nextPair();
                }
            });
            
            this.autoPlayButton = this.addWidget("button", "▶ Play", null, () => {
                if (this.comparerWidget) {
                    this.comparerWidget.toggleAutoPlay();
                    // Update button text
                    this.autoPlayButton.name = this.comparerWidget.autoPlayEnabled ? "⏸ Pause" : "▶ Play";
                }
            });
            
            this.pairInfoWidget = this.addWidget("text", "Pair Info", "1 / 1", () => {}, {});
            this.pairInfoWidget.disabled = true;

            // Add batch pagination control widgets (initially hidden)
            this.batchPrevButton = this.addWidget("button", "◀ Prev Page", null, () => {
                if (this.comparerWidget) {
                    this.comparerWidget.previousBatchPage();
                }
            });
            
            this.batchNextButton = this.addWidget("button", "Next Page ▶", null, () => {
                if (this.comparerWidget) {
                    this.comparerWidget.nextBatchPage();
                }
            });
            
            this.batchPageInfoWidget = this.addWidget("text", "Page Info", "Page 1 / 1", () => {}, {});
            this.batchPageInfoWidget.disabled = true;

            // Add Onion Skin opacity slider (initially hidden)
            this.onionSkinOpacitySlider = this.addWidget("slider", "Opacity B", this.properties.onionSkinOpacity, (value) => {
                this.properties.onionSkinOpacity = parseFloat(value);
                this.setDirtyCanvas(true, false);
            }, {
                min: 0.0,
                max: 1.0,
                step: 0.01
            });

            // Add cache management buttons for debugging
            this.clearCacheButton = this.addWidget("button", "Clear Cache", null, () => {
                // Call the Python class method to clear cache
                console.log("[AdvancedImageComparer] Clearing image cache");
                // We can't directly call Python methods from JS, but we can log this action
                // The cache clearing happens automatically in the Python code
            });
            this.clearCacheButton.hidden = true; // Hide by default, can be shown for debugging

            // Create the custom widget
            this.comparerWidget = this.addCustomWidget(new AdvancedImageComparerWidget("advanced_comparer", this));
            
            // Initialize controls visibility
            this.updateControlsVisibility();
            
            this.setSize(this.computeSize());
            this.setDirtyCanvas(true, true);
        };

        // Method to show/hide controls based on mode
        nodeType.prototype.updateControlsVisibility = function() {
            const mode = this.properties.comparer_mode;
            const hasMultiplePairs = this.comparerWidget && this.comparerWidget.maxPairs > 1;
            
            // Show batch selector for modes that need individual pair selection
            const showBatchSelector = hasMultiplePairs && ["Slider", "Click", "Side-by-Side", "Stacked", "Onion Skin"].includes(mode);
            
            // Show carousel controls for carousel mode
            const showCarouselControls = mode === "Carousel" && hasMultiplePairs;
            
            // Show batch pagination controls for batch mode with multiple pages
            const showBatchPagination = mode === "Batch" && this.comparerWidget && this.comparerWidget.maxBatchPages > 1;
            
            // Show Onion Skin opacity slider for Onion Skin mode
            const showOnionSkinSlider = mode === "Onion Skin";
            
            // Update batch selector
            if (this.batchSelectorWidget) {
                this.batchSelectorWidget.hidden = !showBatchSelector;
                if (showBatchSelector && this.comparerWidget) {
                    // Update the options for the batch selector
                    const options = [];
                    for (let i = 1; i <= this.comparerWidget.maxPairs; i++) {
                        options.push(i.toString());
                    }
                    this.batchSelectorWidget.options.values = options;
                    this.batchSelectorWidget.value = (this.comparerWidget.currentPairIndex + 1).toString();
                }
            }
            
            // Update carousel controls
            if (this.prevButton) {
                this.prevButton.hidden = !showCarouselControls;
            }
            if (this.nextButton) {
                this.nextButton.hidden = !showCarouselControls;
            }
            if (this.autoPlayButton) {
                this.autoPlayButton.hidden = !showCarouselControls;
            }
            if (this.pairInfoWidget) {
                this.pairInfoWidget.hidden = !showCarouselControls;
                if (showCarouselControls && this.comparerWidget) {
                    this.pairInfoWidget.value = `${this.comparerWidget.currentPairIndex + 1} / ${this.comparerWidget.maxPairs}`;
                }
            }
            
            // Update batch pagination controls
            if (this.batchPrevButton) {
                this.batchPrevButton.hidden = !showBatchPagination;
            }
            if (this.batchNextButton) {
                this.batchNextButton.hidden = !showBatchPagination;
            }
            if (this.batchPageInfoWidget) {
                this.batchPageInfoWidget.hidden = !showBatchPagination;
                if (showBatchPagination && this.comparerWidget) {
                    this.batchPageInfoWidget.value = `Page ${this.comparerWidget.currentBatchPage + 1} / ${this.comparerWidget.maxBatchPages}`;
                }
            }
            
            // Update Onion Skin opacity slider
            if (this.onionSkinOpacitySlider) {
                this.onionSkinOpacitySlider.hidden = !showOnionSkinSlider;
                if (showOnionSkinSlider) {
                    this.onionSkinOpacitySlider.value = this.properties.onionSkinOpacity;
                }
            }
        };

        // Override computeSize to account for the widget
        const originalComputeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function(out) {
            const size = originalComputeSize ? originalComputeSize.apply(this, arguments) : [200, 100]; // Reduced defaults
            if (this.comparerWidget) {
                const widgetSize = this.comparerWidget.computeSize(size[0]);
                
                // Calculate additional space needed for controls - reduced padding
                let extraHeight = 40; // Base padding for layout widget + title bar
                
                const mode = this.properties.comparer_mode;
                const hasMultiplePairs = this.comparerWidget.maxPairs > 1;
                
                if (mode === "Carousel" && hasMultiplePairs) {
                    extraHeight += 90; // Reduced from 120 for carousel controls
                } else if (["Slider", "Click", "Side-by-Side", "Stacked"].includes(mode) && hasMultiplePairs) {
                    extraHeight += 25; // Reduced from 30 for batch selector
                } else if (mode === "Batch" && this.comparerWidget.maxBatchPages > 1) {
                    extraHeight += 70; // Reduced from 90 for batch pagination
                } else if (mode === "Onion Skin") {
                    extraHeight += 25; // Reduced from 30 for opacity slider
                    if (hasMultiplePairs) {
                        extraHeight += 25; // Reduced from 30 for batch selector
                    }
                }
                
                size[1] = Math.max(size[1], widgetSize[1] + extraHeight);
            }
            return size;
        };

        // Override onExecuted to handle image data - this should be called when the node executes
        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            // Call the original onExecuted first (this handles the standard PreviewImage functionality)
            let result;
            if (originalOnExecuted) {
                result = originalOnExecuted.apply(this, arguments);
            }
            
            // Now handle our custom logic
            if (message && typeof message === 'object') {
                
                // Check for images in different possible locations
                let images = null;
                if (message.ui && message.ui.images && Array.isArray(message.ui.images)) {
                    images = message.ui.images;
                } else if (message.images && Array.isArray(message.images)) {
                    images = message.images;
                }
                
                if (images && images.length > 0) {
                    console.log(`[AdvancedImageComparer] Received ${images.length} images`);
                    
                    if (this.comparerWidget) {
                        this.comparerWidget.value = { images: images };
                        this.setDirtyCanvas(true, false);
                    } else {
                        console.error("[AdvancedImageComparer] No comparerWidget found on node!");
                    }
                } else {
                    console.log("[AdvancedImageComparer] No images received in message");
                }
            }
            
            return result || message;
        };
    
        // Mouse event handlers
        nodeType.prototype.setIsPointerDown = function(down = this.isPointerDown) {
            const newIsDown = down && !!app.canvas.pointer_is_down;
            if (this.isPointerDown !== newIsDown) {
                this.isPointerDown = newIsDown;
                this.setDirtyCanvas(true, false);
            }
            this.imageIndex = this.isPointerDown ? 1 : 0;
            
            if (this.isPointerDown) {
                requestAnimationFrame(() => {
                    this.setIsPointerDown();
                });
            }
        };

        nodeType.prototype.onMouseDown = function(event, pos, canvas) {
            this.setIsPointerDown(true);
            return false;
        };

        nodeType.prototype.onMouseEnter = function(event) {
            this.setIsPointerDown(!!app.canvas.pointer_is_down);
            this.isPointerOver = true;
            this.setDirtyCanvas(true, false);
        };
    
        nodeType.prototype.onMouseLeave = function(event) {
            this.setIsPointerDown(false);
            this.isPointerOver = false;
            this.setDirtyCanvas(true, false);
        };

        nodeType.prototype.onMouseMove = function(event, pos, canvas) {
            this.pointerOverPos = [...pos];
            
            const mode = this.properties.comparer_mode || "Slider";
            
            switch (mode) {
                case "Slider":
                    this.setDirtyCanvas(true, false);
                    break;
                case "Click":
                    this.imageIndex = this.pointerOverPos[0] > this.size[0] / 2 ? 1 : 0;
                    break;
                case "Side-by-Side":
                case "Stacked":
                    // No special mouse handling needed for these modes
                    break;
            }
            return true;
        };

        // Add context menu options
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            if (getExtraMenuOptions) {
                getExtraMenuOptions.apply(this, arguments);
            }
            
            const layoutModes = ["Slider", "Click", "Side-by-Side", "Stacked", "Grid", "Carousel", "Batch", "Onion Skin"];
            const currentMode = this.properties.comparer_mode || "Slider";
            
            // Add separator for layout modes section
            options.push(null);
            
            // Add auto-fill toggle
            options.push({
                content: "Toggle Auto Fill Empty Slot",
                callback: () => {
                    // Toggle the auto_fill widget value by finding it in widgets
                    const autoFillWidget = this.widgets?.find(w => w.name === "auto_fill");
                    if (autoFillWidget) {
                        autoFillWidget.value = !autoFillWidget.value;
                        console.log("[AdvancedImageComparer] Auto-fill toggled via menu:", autoFillWidget.value);
                    }
                }
            });
            
            options.push(null); // separator
            
            // Add main layout modes submenu
            const layoutSubmenu = [];
            
            layoutModes.forEach(mode => {
                layoutSubmenu.push({
                    content: `${mode === currentMode ? "✓ " : ""}${mode}`,
                    callback: () => {
                        this.properties.comparer_mode = mode;
                        if (this.layoutWidget) {
                            this.layoutWidget.value = mode;
                        }
                        this.updateControlsVisibility();
                        this.setDirtyCanvas(true, false);
                    }
                });
            });
            
            options.push({
                content: "Layout Mode",
                submenu: {
                    options: layoutSubmenu
                }
            });
            
            // Add quick access to most common modes
            options.push(null); // separator
            
            const quickModes = ["Slider", "Side-by-Side", "Grid", "Carousel"];
            quickModes.forEach(mode => {
                if (mode !== currentMode) {
                    options.push({
                        content: `Switch to ${mode}`,
                        callback: () => {
                            this.properties.comparer_mode = mode;
                            if (this.layoutWidget) {
                                this.layoutWidget.value = mode;
                            }
                            this.updateControlsVisibility();
                            this.setDirtyCanvas(true, false);
                        }
                    });
                }
            });
            
            // Add Select Pair submenu if applicable
            if (this.comparerWidget && this.comparerWidget.maxPairs > 1 &&
                ["Slider", "Click", "Side-by-Side", "Stacked", "Onion Skin"].includes(currentMode)) {
                
                options.push(null); // separator
                const pairSubmenu = [];
                for (let i = 0; i < this.comparerWidget.maxPairs; i++) {
                    pairSubmenu.push({
                        content: `${i === this.comparerWidget.currentPairIndex ? "✓ " : ""}Pair ${i + 1}`,
                        callback: () => {
                            if (this.comparerWidget) {
                                this.comparerWidget.currentPairIndex = i;
                                this.comparerWidget.updateSelectedPair();
                                this.setDirtyCanvas(true, false);
                            }
                        }
                    });
                }
                options.push({
                    content: "Select Pair",
                    submenu: {
                        options: pairSubmenu
                    }
                });
            }
            
            options.push(null); // separator
            
            options.push(
                {
                    content: "Reset to Default Size",
                    callback: () => {
                        this.setSize(this.computeSize());
                        this.setDirtyCanvas(true, false);
                    }
                }
            );
        };
    
        console.log("AdvancedImageComparer node setup complete with auto-fill functionality");
    }
});