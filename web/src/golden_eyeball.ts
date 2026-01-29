/**
 * golden_eyeball.ts
 * Shared rendering utilities for golden eyeball and gradient title animations
 * Used by both advanced_comparer and video_comparer widgets
 */

import type { LGraphNode } from "../types/litegraph";

// Type for the animation cache that each module must provide
export interface AnimationCache {
    frameCount: number;
    frameSkip: number;
    lastTime: number;
}

// === Shared Animation Utilities ===

/** Standard background gradient colors used across all animated titles */
const TITLE_GRADIENT_COLORS = {
    top: "#000000",
    transition: "#101010",
    bottom: "#101010",
};

/** Standard corner radius for rounded title backgrounds */
export const TITLE_CORNER_RADIUS = 8;

/**
 * Calculates shimmer position for text animations
 * @param widthFactor - Multiplier for the shimmer width (default 1.0)
 * @returns Current shimmer position (0 to widthFactor)
 */
export function calculateShimmerPosition(widthFactor: number = 1.0): number {
    const time = Date.now() / 3000;
    return ((Math.sin(time) + 1) / 2) * widthFactor;
}

/**
 * Advances the animation frame counter and returns whether to update
 * @param cache - Animation cache to update
 * @returns true if animation should update this frame
 */
export function shouldUpdateFrame(cache: AnimationCache): boolean {
    cache.frameCount = (cache.frameCount + 1) % (cache.frameSkip + 1);
    return cache.frameCount === 0;
}

/**
 * Resets context shadow properties for clean gradient rendering
 * @param ctx - Canvas context to reset
 */
export function resetShadowContext(ctx: CanvasRenderingContext2D): void {
    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
}

/**
 * Creates the standard vertical background gradient
 * @param ctx - Canvas context
 * @param fullHeight - Height of the gradient
 * @returns The gradient object
 */
export function createTitleGradient(ctx: CanvasRenderingContext2D, fullHeight: number): CanvasGradient {
    const gradient = ctx.createLinearGradient(0, 0, 0, fullHeight);
    gradient.addColorStop(0, TITLE_GRADIENT_COLORS.top);
    gradient.addColorStop(0.2, TITLE_GRADIENT_COLORS.transition);
    gradient.addColorStop(1, TITLE_GRADIENT_COLORS.bottom);
    return gradient;
}

/**
 * Draws a golden eyeball with shimmer animation
 * @param ctx - Canvas rendering context
 * @param centerX - X center of the eyeball
 * @param centerY - Y center of the eyeball
 * @param size - Base size of the eyeball
 * @param shimmerPosition - Animation position (0-1) for shimmer effect
 * @param rayCount - Number of rays around the eyeball (default: 8)
 */
function drawGoldenEyeball(
    ctx: CanvasRenderingContext2D,
    centerX: number,
    centerY: number,
    size: number,
    shimmerPosition: number,
    rayCount: number = 8
): void {
    const eyeWidth = size * 1.6;
    const eyeHeight = size * 1.0;
    const irisRadius = size * 0.35;
    const pupilRadius = size * 0.15;

    ctx.save();

    const baseGradient = ctx.createLinearGradient(0, centerY - size * 0.7, 0, centerY + size * 0.7);
    baseGradient.addColorStop(0, "#B8860B");
    baseGradient.addColorStop(0.5, "#FFD700");
    baseGradient.addColorStop(1, "#B8860B");

    const highlightWidth = eyeWidth * 0.4;
    const highlightX = -highlightWidth + (eyeWidth + highlightWidth) * shimmerPosition;

    const shimmerGradient = ctx.createLinearGradient(
        centerX + highlightX - highlightWidth / 2, 0,
        centerX + highlightX + highlightWidth / 2, 0
    );

    shimmerGradient.addColorStop(0, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(0.1, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(0.5, "rgba(255, 255, 200, 0.3)");
    shimmerGradient.addColorStop(0.9, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(1, "rgba(255, 255, 200, 0)");

    // Draw shadows
    ctx.strokeStyle = "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1.5;
    ctx.lineCap = "round";

    ctx.beginPath();
    ctx.ellipse(centerX + 2, centerY + 2, eyeWidth / 2, eyeHeight / 2, 0, 0, Math.PI * 2);
    ctx.stroke();

    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX + 2, centerY + 2, irisRadius, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(centerX + 2, centerY + 2, pupilRadius, 0, Math.PI * 2);
    ctx.stroke();

    // Draw rays
    const rayLength = size * 0.7;

    for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        const startX = centerX + Math.cos(angle) * (eyeWidth / 2 + 1);
        const startY = centerY + Math.sin(angle) * (eyeHeight / 2 + 1);
        const endX = centerX + Math.cos(angle) * (eyeWidth / 2 + rayLength);
        const endY = centerY + Math.sin(angle) * (eyeHeight / 2 + rayLength);

        ctx.beginPath();
        ctx.moveTo(startX + 2, startY + 2);
        ctx.lineTo(endX + 2, endY + 2);
        ctx.stroke();
    }

    // Draw golden outlines
    ctx.strokeStyle = baseGradient;
    ctx.lineWidth = 1.5;

    ctx.beginPath();
    ctx.ellipse(centerX, centerY, eyeWidth / 2, eyeHeight / 2, 0, 0, Math.PI * 2);
    ctx.stroke();

    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX, centerY, irisRadius, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(centerX, centerY, pupilRadius, 0, Math.PI * 2);
    ctx.stroke();

    for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        const startX = centerX + Math.cos(angle) * (eyeWidth / 2 + 1);
        const startY = centerY + Math.sin(angle) * (eyeHeight / 2 + 1);
        const endX = centerX + Math.cos(angle) * (eyeWidth / 2 + rayLength);
        const endY = centerY + Math.sin(angle) * (eyeHeight / 2 + rayLength);

        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
    }

    // Iris texture - use rayCount for consistency
    ctx.lineWidth = 0.5;
    for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(centerX + Math.cos(angle) * pupilRadius * 1.1, centerY + Math.sin(angle) * pupilRadius * 1.1);
        ctx.lineTo(centerX + Math.cos(angle) * irisRadius * 0.9, centerY + Math.sin(angle) * irisRadius * 0.9);
        ctx.stroke();
    }

    // Shimmer effect
    ctx.strokeStyle = shimmerGradient;
    ctx.lineWidth = 1.5;

    ctx.beginPath();
    ctx.ellipse(centerX, centerY, eyeWidth / 2, eyeHeight / 2, 0, 0, Math.PI * 2);
    ctx.stroke();

    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX, centerY, irisRadius, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(centerX, centerY, pupilRadius, 0, Math.PI * 2);
    ctx.stroke();

    for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        const startX = centerX + Math.cos(angle) * (eyeWidth / 2 + 1);
        const startY = centerY + Math.sin(angle) * (eyeHeight / 2 + 1);
        const endX = centerX + Math.cos(angle) * (eyeWidth / 2 + rayLength);
        const endY = centerY + Math.sin(angle) * (eyeHeight / 2 + rayLength);

        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
    }

    // Glow effect
    const glowIntensity = Math.max(0, 1 - Math.abs(centerX - (centerX + highlightX)) / (eyeWidth / 4));
    ctx.shadowColor = `rgba(255, 255, 200, ${glowIntensity * 0.3})`;
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;

    ctx.strokeStyle = baseGradient;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, eyeWidth / 2, eyeHeight / 2, 0, 0, Math.PI * 2);
    ctx.stroke();

    ctx.restore();
}

/**
 * Draws a gradient title background with golden eyeball animation
 * @param node - The LiteGraph node to draw on
 * @param ctx - Canvas rendering context
 * @param cache - Animation cache for frame counting
 * @param rayCount - Number of rays for the eyeball (default: 8)
 */
export function drawGradientTitle(
    node: LGraphNode,
    ctx: CanvasRenderingContext2D,
    cache: AnimationCache,
    rayCount: number = 8
): void {
    const titleHeight = node.flags.collapsed ? 20 : 30;
    const width = node.flags.collapsed ? 190 : node.size[0];
    const fullHeight = node.size[1];
    const eyeballY = node.flags.collapsed ? titleHeight / 2 : 25;
    const eyeballSize = node.flags.collapsed ? 6 : 10;

    const shouldUpdateAnimation = shouldUpdateFrame(cache);

    ctx.save();
    resetShadowContext(ctx);

    const gradient = createTitleGradient(ctx, fullHeight);

    // Calculate shimmer position (always compute for smooth animation)
    const shimmerPosition = calculateShimmerPosition(1.0);
    if (shouldUpdateAnimation) {
        cache.lastTime = Date.now() / 3000;
    }

    if (node.flags.collapsed) {
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, titleHeight);
        drawGoldenEyeball(ctx, width / 2, titleHeight / 2, eyeballSize, shimmerPosition, rayCount);
        ctx.restore();
        return;
    }

    ctx.fillStyle = gradient;

    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(width, 0);
    ctx.lineTo(width, fullHeight - TITLE_CORNER_RADIUS);
    ctx.arcTo(width, fullHeight, width - TITLE_CORNER_RADIUS, fullHeight, TITLE_CORNER_RADIUS);
    ctx.lineTo(TITLE_CORNER_RADIUS, fullHeight);
    ctx.arcTo(0, fullHeight, 0, fullHeight - TITLE_CORNER_RADIUS, TITLE_CORNER_RADIUS);
    ctx.lineTo(0, 0);
    ctx.closePath();
    ctx.fill();

    drawGoldenEyeball(ctx, width / 2, eyeballY, eyeballSize, shimmerPosition, rayCount);
    ctx.restore();
}
