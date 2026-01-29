/**
 * Unit tests for NoiseVisualizer module
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Import the module to populate window.NoiseVisualizer
import '../src/noise_visualizer';

describe('NoiseVisualizer', () => {
    let canvas: HTMLCanvasElement;
    let ctx: CanvasRenderingContext2D;

    beforeEach(() => {
        // Create a fresh canvas for each test
        canvas = document.createElement('canvas');
        canvas.width = 130;
        canvas.height = 130;
        ctx = canvas.getContext('2d')!;
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    describe('Initialization', () => {
        it('should expose NoiseVisualizer on window object', () => {
            expect(window.NoiseVisualizer).toBeDefined();
        });

        it('should have renderAllInModal method', () => {
            expect(typeof window.NoiseVisualizer?.renderAllInModal).toBe('function');
        });

        it('should have kofi image properties', () => {
            expect(window.NoiseVisualizer).toHaveProperty('kofiImageLoaded');
            expect(window.NoiseVisualizer).toHaveProperty('kofiImageLoadAttempted');
        });
    });

    describe('Canvas Rendering Methods', () => {
        it('should have renderPlaceholder method', () => {
            expect(typeof window.NoiseVisualizer?.renderPlaceholder).toBe('function');
        });

        it('should render placeholder without errors', () => {
            expect(() => {
                window.NoiseVisualizer?.renderPlaceholder(canvas, 'Test Noise');
            }).not.toThrow();
        });

        it('should have renderTensorField method', () => {
            expect(typeof window.NoiseVisualizer?.renderTensorField).toBe('function');
        });

        it('should render tensor field without errors', () => {
            expect(() => {
                window.NoiseVisualizer?.renderTensorField(canvas);
            }).not.toThrow();
        });

        it('should have renderCellular method', () => {
            expect(typeof window.NoiseVisualizer?.renderCellular).toBe('function');
        });

        it('should render cellular noise without errors', () => {
            expect(() => {
                window.NoiseVisualizer?.renderCellular(canvas);
            }).not.toThrow();
        });

        it('should have renderPerlin method', () => {
            expect(typeof window.NoiseVisualizer?.renderPerlin).toBe('function');
        });

        it('should render Perlin noise without errors', () => {
            expect(() => {
                window.NoiseVisualizer?.renderPerlin(canvas);
            }).not.toThrow();
        });

        it('should have renderFractal method', () => {
            expect(typeof window.NoiseVisualizer?.renderFractal).toBe('function');
        });

        it('should render fractal noise without errors', () => {
            expect(() => {
                window.NoiseVisualizer?.renderFractal(canvas);
            }).not.toThrow();
        });
    });

    describe('Mask Rendering Methods', () => {
        it('should have renderMaskRadial method', () => {
            expect(typeof window.NoiseVisualizer?.renderMaskRadial).toBe('function');
        });

        it('should render radial mask without errors', () => {
            expect(() => {
                window.NoiseVisualizer?.renderMaskRadial(canvas);
            }).not.toThrow();
        });

        it('should have renderMaskLinear method', () => {
            expect(typeof window.NoiseVisualizer?.renderMaskLinear).toBe('function');
        });

        it('should render linear mask without errors', () => {
            expect(() => {
                window.NoiseVisualizer?.renderMaskLinear(canvas);
            }).not.toThrow();
        });

        it('should have renderMaskGrid method', () => {
            expect(typeof window.NoiseVisualizer?.renderMaskGrid).toBe('function');
        });

        it('should have renderMaskSpiral method', () => {
            expect(typeof window.NoiseVisualizer?.renderMaskSpiral).toBe('function');
        });

        it('should have renderMaskHexgrid method', () => {
            expect(typeof window.NoiseVisualizer?.renderMaskHexgrid).toBe('function');
        });
    });

    describe('Helper Methods', () => {
        it('should have _clearCanvas method', () => {
            expect(typeof window.NoiseVisualizer?._clearCanvas).toBe('function');
        });

        it('should clear canvas and return context', () => {
            const result = (window.NoiseVisualizer as any)?._clearCanvas(canvas);
            expect(result).toBeDefined();
        });

        it('should have _drawKofiIcon method', () => {
            expect(typeof window.NoiseVisualizer?._drawKofiIcon).toBe('function');
        });

        it('should draw Ko-fi icon without errors', () => {
            expect(() => {
                (window.NoiseVisualizer as any)?._drawKofiIcon(ctx);
            }).not.toThrow();
        });

        it('should have _drawManualKofiCup method', () => {
            expect(typeof window.NoiseVisualizer?._drawManualKofiCup).toBe('function');
        });

        it('should draw manual Ko-fi cup without errors', () => {
            expect(() => {
                (window.NoiseVisualizer as any)?._drawManualKofiCup(ctx, 10, 10, 18);
            }).not.toThrow();
        });
    });

    describe('renderAllInModal', () => {
        it('should handle empty container', async () => {
            const container = document.createElement('div');
            await expect(
                window.NoiseVisualizer?.renderAllInModal(container)
            ).resolves.not.toThrow();
        });

        it('should process noise canvases', async () => {
            const container = document.createElement('div');

            // Add a noise canvas div
            const noiseDiv = document.createElement('div');
            noiseDiv.className = 'noise-canvas';
            noiseDiv.id = 'noise-canvas-perlin';
            container.appendChild(noiseDiv);

            await window.NoiseVisualizer?.renderAllInModal(container);

            // Should have created a canvas inside
            const createdCanvas = noiseDiv.querySelector('canvas');
            expect(createdCanvas).toBeDefined();
        });

        it('should process mask canvases', async () => {
            const container = document.createElement('div');

            // Add a mask canvas div
            const maskDiv = document.createElement('div');
            maskDiv.className = 'mask-canvas';
            maskDiv.id = 'mask-canvas-radial';
            container.appendChild(maskDiv);

            await window.NoiseVisualizer?.renderAllInModal(container);

            // Should have created a canvas inside
            const createdCanvas = maskDiv.querySelector('canvas');
            expect(createdCanvas).toBeDefined();
        });
    });
});
