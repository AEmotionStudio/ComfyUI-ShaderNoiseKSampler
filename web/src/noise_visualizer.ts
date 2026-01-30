/**
 * noise_visualizer.ts - Noise pattern visualization for ComfyUI modal
 */

// Make this file a module to allow global augmentation
export { };

// ============================
// Type Definitions
// ============================

interface Point {
    x: number;
    y: number;
}

interface WaveSource extends Point {
    phase?: number;
}

interface Particle extends Point {
    vx: number;
    vy: number;
    life: number;
    initialLife: number;
    radius: number;
    trail: Point[];
    isPrimordial: boolean;
    primordialTime: number;
}

interface BackgroundPattern {
    type: PatternType | null;
    variantSeed: number;
    alpha: number;
    startTime: number;
}

type PatternType = 'tensor_field' | 'cellular' | 'domain_warp' | 'perlin' | 'curl_noise' | 'waves_interference';

interface NoiseVisualizerInterface {
    kofiCupImageBitmap: HTMLImageElement | ImageBitmap | null;
    kofiImageLoaded: boolean;
    kofiImageLoadAttempted: boolean;
    _preloadKofiImage: () => Promise<void>;
    renderAllInModal: (modalContentElement: HTMLElement) => Promise<void>;
    _clearCanvas: (canvas: HTMLCanvasElement, backgroundColor?: string) => CanvasRenderingContext2D;
    _drawKofiIcon: (ctx: CanvasRenderingContext2D) => void;
    _drawManualKofiCup: (ctx: CanvasRenderingContext2D, x: number, y: number, iconSize: number) => void;
    renderPlaceholder: (canvas: HTMLCanvasElement, noiseName: string) => void;
    // Noise visualizations
    renderTensorField: (canvas: HTMLCanvasElement) => void;
    renderCellular: (canvas: HTMLCanvasElement) => void;
    renderDomainWarp: (canvas: HTMLCanvasElement) => void;
    renderFractal: (canvas: HTMLCanvasElement) => void;
    renderPerlin: (canvas: HTMLCanvasElement) => void;
    renderWaves: (canvas: HTMLCanvasElement) => void;
    renderGaussian: (canvas: HTMLCanvasElement) => void;
    renderHeterogeneousFBM: (canvas: HTMLCanvasElement) => void;
    renderInterference: (canvas: HTMLCanvasElement) => void;
    renderSpectral: (canvas: HTMLCanvasElement) => void;
    renderProjection3D: (canvas: HTMLCanvasElement) => void;
    renderCurlNoise: (canvas: HTMLCanvasElement) => void;
    // Mask visualizations
    renderMaskPlaceholder: (canvas: HTMLCanvasElement, maskName: string) => void;
    renderMaskRadial: (canvas: HTMLCanvasElement) => void;
    renderMaskLinear: (canvas: HTMLCanvasElement) => void;
    renderMaskGrid: (canvas: HTMLCanvasElement) => void;
    renderMaskVignette: (canvas: HTMLCanvasElement) => void;
    renderMaskSpiral: (canvas: HTMLCanvasElement) => void;
    renderMaskHexgrid: (canvas: HTMLCanvasElement) => void;
    renderMaskWavy: (canvas: HTMLCanvasElement) => void;
    renderMaskConcentricRings: (canvas: HTMLCanvasElement) => void;
    // Color scheme renders
    renderColorInferno: (swatchDiv: HTMLElement) => void;
    renderColorMagma: (swatchDiv: HTMLElement) => void;
    renderColorPlasma: (swatchDiv: HTMLElement) => void;
    renderColorViridis: (swatchDiv: HTMLElement) => void;
    renderColorTurbo: (swatchDiv: HTMLElement) => void;
    renderColorJet: (swatchDiv: HTMLElement) => void;
    renderColorParula: (swatchDiv: HTMLElement) => void;
    renderColorRainbow: (swatchDiv: HTMLElement) => void;
    renderColorHot: (swatchDiv: HTMLElement) => void;
    renderColorBlueRed: (swatchDiv: HTMLElement) => void;
    renderColorCool: (swatchDiv: HTMLElement) => void;
    renderColorHsv: (swatchDiv: HTMLElement) => void;
    renderColorAutumn: (swatchDiv: HTMLElement) => void;
    renderColorWinter: (swatchDiv: HTMLElement) => void;
    renderColorSpring: (swatchDiv: HTMLElement) => void;
    renderColorSummer: (swatchDiv: HTMLElement) => void;
    renderColorCopper: (swatchDiv: HTMLElement) => void;
    renderColorPink: (swatchDiv: HTMLElement) => void;
    renderColorBone: (swatchDiv: HTMLElement) => void;
    renderColorOcean: (swatchDiv: HTMLElement) => void;
    renderColorTerrain: (swatchDiv: HTMLElement) => void;
    renderColorNeon: (swatchDiv: HTMLElement) => void;
    renderColorFire: (swatchDiv: HTMLElement) => void;
    // Animation renders
    renderTemporalAnimation: (canvas: HTMLCanvasElement) => void;
    renderIntroNoiseDemo: (canvas: HTMLCanvasElement) => void;
    renderColorPlaceholder: (swatchDiv: HTMLElement, schemeName: string) => void;
    // Index signature for dynamic method access
    [key: string]: unknown;
}

// Extend Window interface
declare global {
    interface Window {
        NoiseVisualizer?: NoiseVisualizerInterface;
    }
}

// ============================
// Implementation
// ============================

if (!window.NoiseVisualizer) {
    const NoiseVisualizer: NoiseVisualizerInterface = {
        kofiCupImageBitmap: null,
        kofiImageLoaded: false,
        kofiImageLoadAttempted: false,

        _preloadKofiImage: async function (): Promise<void> {
            if (this.kofiImageLoadAttempted) return;
            this.kofiImageLoadAttempted = true;

            const absolutePath = '/extensions/comfyui-shadernoiseksampler/images/kofi_symbol.svg';

            // Attempt 1: Direct Image load
            try {
                console.log('Attempting Ko-fi SVG load using new Image() with direct path (Primary Attempt)...');
                const img = new Image();
                await new Promise<void>((resolve, reject) => {
                    img.onload = (): void => {
                        this.kofiCupImageBitmap = img;
                        this.kofiImageLoaded = true;
                        console.log('Ko-fi symbol SVG loaded via Image() successfully (Primary).');
                        resolve();
                    };
                    img.onerror = (e): void => {
                        console.warn('Primary Ko-fi symbol SVG load via Image() failed. Proceeding to secondary attempt.', e);
                        reject(e);
                    };
                    img.src = absolutePath;
                });
                return;
            } catch {
                this.kofiImageLoaded = false;
                this.kofiCupImageBitmap = null;
            }

            // Attempt 2: Fetch -> Blob -> Intermediate Image -> ImageBitmap
            let objectURL: string | null = null;
            try {
                console.log('Attempting Ko-fi SVG load via Blob -> Image -> ImageBitmap (Secondary Attempt)...');
                const response = await fetch(absolutePath);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status} for ${absolutePath}`);
                }
                const svgText = await response.text();
                const blob = new Blob([svgText], { type: 'image/svg+xml' });
                objectURL = URL.createObjectURL(blob);

                const intermediateImg = new Image();
                await new Promise<void>((resolve, reject) => {
                    intermediateImg.onload = (): void => resolve();
                    intermediateImg.onerror = (): void => reject(new Error('Intermediate Image() load failed for SVG blob (Secondary).'));
                    intermediateImg.src = objectURL!;
                });

                this.kofiCupImageBitmap = await createImageBitmap(intermediateImg);
                this.kofiImageLoaded = true;
                console.log('Ko-fi symbol SVG processed to ImageBitmap successfully (Secondary).');
            } catch (error) {
                this.kofiImageLoaded = false;
                this.kofiCupImageBitmap = null;
                console.error('All Ko-fi SVG load attempts failed: ', error, '. Will use fallback drawing.');
            } finally {
                if (objectURL) {
                    URL.revokeObjectURL(objectURL);
                }
            }
        },

        renderAllInModal: async function (modalContentElement: HTMLElement): Promise<void> {
            if (!this.kofiImageLoadAttempted) {
                await this._preloadKofiImage();
            }

            const noiseCanvases = modalContentElement.querySelectorAll('.noise-canvas');
            noiseCanvases.forEach(canvasDiv => {
                const canvasId = canvasDiv.id;
                if (!canvasId || !canvasId.startsWith('noise-canvas-')) return;
                const noiseType = canvasId.substring('noise-canvas-'.length);
                const noiseName = noiseType.split('_').map(word => {
                    if (word.toLowerCase() === 'fbm') return 'FBM';
                    if (word.toLowerCase() === '3d') return '3D';
                    return word.charAt(0).toUpperCase() + word.slice(1);
                }).join(' ');

                let canvas = canvasDiv.querySelector('canvas');
                if (!canvas) {
                    canvas = document.createElement('canvas');
                    canvas.width = 130;
                    canvas.height = 130;
                    canvas.setAttribute('role', 'img');
                    canvas.setAttribute('aria-label', `Visualization of ${noiseName} noise pattern`);
                    canvasDiv.innerHTML = '';
                    canvasDiv.appendChild(canvas);
                }

                const functionNameSuffix = noiseType.split('_').map(word => {
                    if (word.toLowerCase() === 'fbm') return 'FBM';
                    return word.charAt(0).toUpperCase() + word.slice(1);
                }).join('');
                const renderFunctionName = `render${functionNameSuffix.replace('3d', '3D')}`;

                const renderMethod = this[renderFunctionName];
                if (typeof renderMethod === 'function') {
                    (renderMethod as (canvas: HTMLCanvasElement) => void).call(this, canvas);
                } else {
                    console.warn('No renderer function found:', renderFunctionName, 'for noise type:', noiseType);
                    this.renderPlaceholder(canvas, noiseName);
                }
            });

            // Mask canvases
            const maskCanvases = modalContentElement.querySelectorAll('.mask-canvas');
            maskCanvases.forEach(canvasDiv => {
                const canvasId = canvasDiv.id;
                if (!canvasId || !canvasId.startsWith('mask-canvas-')) return;
                const maskType = canvasId.substring('mask-canvas-'.length);
                const maskName = maskType.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');

                let canvas = canvasDiv.querySelector('canvas');
                if (!canvas) {
                    canvas = document.createElement('canvas');
                    canvas.width = 100;
                    canvas.height = 70;
                    canvas.setAttribute('role', 'img');
                    canvas.setAttribute('aria-label', `Visualization of ${maskName} shape mask`);
                    canvasDiv.innerHTML = '';
                    canvasDiv.appendChild(canvas);
                }

                const functionNameSuffix = maskType.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join('');
                const renderFunctionName = `renderMask${functionNameSuffix}`;

                const renderMethod = this[renderFunctionName];
                if (typeof renderMethod === 'function') {
                    (renderMethod as (canvas: HTMLCanvasElement) => void).call(this, canvas);
                } else {
                    console.warn('No renderer function found:', renderFunctionName, 'for mask type:', maskType);
                    this.renderMaskPlaceholder(canvas, maskName);
                }
            });

            // Color scheme swatches
            const colorSwatches = modalContentElement.querySelectorAll('.color-swatch');
            colorSwatches.forEach(swatchDiv => {
                const element = swatchDiv as HTMLElement;
                if (element.style.background) return;

                const schemeName = element.textContent?.trim().toLowerCase() ?? '';
                const renderFunctionName = `renderColor${schemeName.charAt(0).toUpperCase() + schemeName.slice(1)}`;

                const renderMethod = this[renderFunctionName];
                if (typeof renderMethod === 'function') {
                    (renderMethod as (swatchDiv: HTMLElement) => void).call(this, element);
                } else {
                    console.log('Using existing style for color scheme:', schemeName);
                }
            });

            // Temporal animation demo
            const animationDemoContainer = modalContentElement.querySelector('#animation-demo-placeholder');
            if (animationDemoContainer) {
                let canvas = animationDemoContainer.querySelector('canvas');
                if (!canvas) {
                    canvas = document.createElement('canvas');
                    const containerStyle = getComputedStyle(animationDemoContainer);
                    canvas.width = parseInt(containerStyle.width) || 300;
                    canvas.height = parseInt(containerStyle.height) || 200;
                    canvas.setAttribute('role', 'img');
                    canvas.setAttribute('aria-label', 'Interactive animation demonstrating temporal coherence');
                    animationDemoContainer.innerHTML = '';
                    animationDemoContainer.appendChild(canvas);
                    (animationDemoContainer as HTMLElement).style.display = 'block';
                }
                if (typeof this.renderTemporalAnimation === 'function') {
                    this.renderTemporalAnimation(canvas);
                } else {
                    console.warn('renderTemporalAnimation function not found in NoiseVisualizer.');
                    this.renderMaskPlaceholder(canvas, 'Animation Demo');
                }
            }

            // Intro noise demo
            const introNoiseDemoContainer = modalContentElement.querySelector('#intro-noise-demo');
            if (introNoiseDemoContainer) {
                let canvas = introNoiseDemoContainer.querySelector('canvas');
                if (!canvas) {
                    canvas = document.createElement('canvas');
                    const containerStyle = getComputedStyle(introNoiseDemoContainer);
                    canvas.width = parseInt(containerStyle.width) || 300;
                    canvas.height = parseInt(containerStyle.height) || 350;
                    canvas.setAttribute('role', 'img');
                    canvas.setAttribute('aria-label', 'Interactive visualization of noise patterns');
                    introNoiseDemoContainer.innerHTML = '';
                    introNoiseDemoContainer.appendChild(canvas);
                    (introNoiseDemoContainer as HTMLElement).style.display = 'block';
                }
                if (typeof this.renderIntroNoiseDemo === 'function') {
                    this.renderIntroNoiseDemo(canvas);
                } else {
                    console.warn('renderIntroNoiseDemo function not found in NoiseVisualizer.');
                    this.renderMaskPlaceholder(canvas, 'Intro Noise Demo');
                }
            }
        },

        _clearCanvas: function (canvas: HTMLCanvasElement, backgroundColor = '#1a1a2e'): CanvasRenderingContext2D {
            const ctx = canvas.getContext('2d')!;
            ctx.fillStyle = backgroundColor;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            return ctx;
        },

        _drawKofiIcon: function (ctx: CanvasRenderingContext2D): void {
            const iconSize = 18;
            const padding = 3;
            const x = ctx.canvas.width - iconSize - padding;
            const y = ctx.canvas.height - iconSize - padding;

            if (this.kofiImageLoaded && this.kofiCupImageBitmap) {
                try {
                    ctx.drawImage(this.kofiCupImageBitmap, x, y, iconSize, iconSize);
                } catch (e) {
                    console.error('Error drawing local Ko-fi SVG ImageBitmap, falling back to manual draw:', e);
                    this._drawManualKofiCup(ctx, x, y, iconSize);
                }
            } else {
                this._drawManualKofiCup(ctx, x, y, iconSize);
            }
        },

        _drawManualKofiCup: function (ctx: CanvasRenderingContext2D, x: number, y: number, iconSize: number): void {
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
            ctx.arc(x + iconSize * 0.9, y + iconSize * 0.5, iconSize * 0.25, -Math.PI / 2, Math.PI / 2);
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

        renderPlaceholder: function (canvas: HTMLCanvasElement, noiseName: string): void {
            const ctx = this._clearCanvas(canvas, '#2c2c34');
            const nameToShow = noiseName.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');

            ctx.fillStyle = '#e0e0e8';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.font = 'bold 12px "Segoe UI", Tahoma, Geneva, Verdana, sans-serif';

            const words = nameToShow.split(' ');
            if (words.length > 2) {
                ctx.fillText(words.slice(0, 2).join(' '), canvas.width / 2, canvas.height / 2 - 7);
                ctx.fillText(words.slice(2).join(' '), canvas.width / 2, canvas.height / 2 + 7);
            } else {
                ctx.fillText(nameToShow, canvas.width / 2, canvas.height / 2);
            }
            ctx.strokeStyle = '#555';
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },

        // --- Visualization Functions ---
        renderTensorField: function (canvas: HTMLCanvasElement): void {
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
                    ctx.moveTo(x, y);
                }
                ctx.stroke();
            }
        },

        renderCellular: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);
            const numCells = 8;
            const points: Point[] = [];
            for (let i = 0; i < numCells; i++) {
                points.push({ x: Math.random() * canvas.width, y: Math.random() * canvas.height });
            }
            for (let x = 0; x < canvas.width; x += 4) {
                for (let y = 0; y < canvas.height; y += 4) {
                    let minDist = Infinity;
                    points.forEach(p => {
                        minDist = Math.min(minDist, Math.hypot(p.x - x, p.y - y));
                    });
                    const intensity = Math.min(255, minDist * 2);
                    ctx.fillStyle = `rgb(${intensity},${intensity},${intensity + 50})`;
                    ctx.fillRect(x, y, 4, 4);
                }
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d')!);
        },

        renderDomainWarp: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);
            ctx.lineWidth = 1.5;
            for (let i = 0; i < 20; i++) {
                ctx.beginPath();
                ctx.moveTo(Math.random() * canvas.width, Math.random() * canvas.height);
                ctx.strokeStyle = `rgba(${100 + Math.random() * 155}, ${100 + Math.random() * 155}, ${200 + Math.random() * 55}, 0.6)`;
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

        renderFractal: function (canvas: HTMLCanvasElement): void {
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
                        const noiseVal = (Math.sin(x * scale * frequency + y * scale * frequency * 0.7) +
                            Math.cos(y * scale * frequency - x * scale * frequency * 0.3)) / 2;
                        total += noiseVal * amplitude;
                        maxValue += amplitude;
                        amplitude *= persistence;
                        frequency *= lacunarity;
                    }

                    const normalizedTotal = (total / maxValue + 1) / 2;
                    const colorVal = Math.floor(normalizedTotal * 200) + 55;
                    ctx.fillStyle = `rgb(${colorVal}, ${colorVal * 0.95}, ${colorVal * 0.9})`;
                    ctx.fillRect(x, y, 2, 2);
                }
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d')!);
        },

        renderPerlin: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);
            for (let x = 0; x < canvas.width; x += 3) {
                for (let y = 0; y < canvas.height; y += 3) {
                    const noiseVal = (Math.sin(x * 0.05 + Math.cos(y * 0.08)) + Math.cos(y * 0.06)) / 2;
                    const intensity = (noiseVal + 1) / 2 * 200 + 55;
                    ctx.fillStyle = `rgba(${intensity * 0.8}, ${intensity * 0.9}, ${intensity}, 1)`;
                    ctx.fillRect(x, y, 3, 3);
                }
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d')!);
        },

        renderWaves: function (canvas: HTMLCanvasElement): void {
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
            this._drawKofiIcon(ctx.canvas.getContext('2d')!);
        },

        renderGaussian: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);
            for (let i = 0; i < 10000; i++) {
                const x = Math.random() * canvas.width;
                const y = Math.random() * canvas.height;
                const intensity = Math.floor(Math.random() * 100) + 100;
                ctx.fillStyle = `rgba(${intensity * 0.8}, ${intensity * 0.9}, ${intensity}, ${Math.random() * 0.5 + 0.1})`;
                ctx.fillRect(x - 1, y - 1, 2, 2);
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d')!);
        },

        renderHeterogeneousFBM: function (canvas: HTMLCanvasElement): void {
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
            this._drawKofiIcon(ctx.canvas.getContext('2d')!);
        },

        renderInterference: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);
            ctx.lineWidth = 0.5;
            const sources: WaveSource[] = [
                { x: canvas.width * 0.2, y: canvas.height * 0.3, phase: 0 },
                { x: canvas.width * 0.8, y: canvas.height * 0.7, phase: Math.PI / 2 }
            ];
            for (let x = 0; x < canvas.width; x += 3) {
                for (let y = 0; y < canvas.height; y += 3) {
                    let sum = 0;
                    sources.forEach(s => {
                        const dist = Math.hypot(s.x - x, s.y - y);
                        sum += Math.sin(dist * 0.1 + (s.phase ?? 0));
                    });
                    const intensity = (sum / sources.length + 1) / 2 * 255;
                    ctx.fillStyle = `rgb(${intensity * 0.7}, ${intensity}, ${intensity * 0.8})`;
                    ctx.fillRect(x, y, 3, 3);
                }
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d')!);
        },

        renderSpectral: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);
            for (let i = 0; i < canvas.height; i += 4) {
                const freqComponent = Math.sin(i * 0.1) * 0.3 + Math.cos(i * 0.05) * 0.3 + Math.random() * 0.4;
                const intensity = (freqComponent + 1) / 2 * 255;
                ctx.fillStyle = `rgb(${intensity}, ${intensity * 0.8}, ${intensity * 0.6})`;
                ctx.fillRect(0, i, canvas.width, 4);
                ctx.strokeStyle = 'rgba(0,0,0,0.2)';
                ctx.strokeRect(0, i, canvas.width, 4);
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d')!);
        },

        renderProjection3D: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);
            for (let i = 0; i < 50; i++) {
                const z = Math.random();
                const x = (Math.random() - 0.5) * canvas.width * (1 + z) + canvas.width / 2;
                const y = (Math.random() - 0.5) * canvas.height * (1 + z) + canvas.height / 2;
                const size = (1 - z) * 10 + 2;
                const opacity = (1 - z) * 0.7 + 0.1;
                ctx.fillStyle = `rgba(${150 + z * 105}, ${150 + z * 105}, ${200 + z * 55}, ${opacity})`;
                ctx.beginPath();
                ctx.arc(x, y, size, 0, Math.PI * 2);
                ctx.fill();
            }
            this._drawKofiIcon(ctx.canvas.getContext('2d')!);
        },

        renderCurlNoise: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);
            ctx.lineWidth = 1;
            for (let i = 0; i < 50; i++) {
                let x = Math.random() * canvas.width;
                let y = Math.random() * canvas.height;
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.strokeStyle = `rgba(${100 + Math.random() * 100}, ${150 + Math.random() * 105}, ${200 + Math.random() * 55}, 0.5)`;
                for (let step = 0; step < 20; step++) {
                    const angle = Math.sin(x * 0.02 + y * 0.03) * Math.PI + Math.cos(y * 0.02 - x * 0.01) * Math.PI;
                    x += Math.cos(angle) * 5;
                    y += Math.sin(angle) * 5;
                    if (x < 0 || x > canvas.width || y < 0 || y > canvas.height) break;
                    ctx.lineTo(x, y);
                }
                ctx.stroke();
            }
        },

        // Mask Placeholder
        renderMaskPlaceholder: function (canvas: HTMLCanvasElement, maskName: string): void {
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
        renderMaskRadial: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const maxRadius = Math.min(canvas.width, canvas.height) / 2;

            const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, maxRadius);
            gradient.addColorStop(0, 'rgba(255, 255, 255, 1.0)');
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0.0)');

            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },

        renderMaskLinear: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);

            const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
            gradient.addColorStop(0, 'rgba(255, 255, 255, 1.0)');
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0.0)');

            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },

        renderMaskGrid: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);
            const cellSize = 20;

            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';

            for (let x = 0; x < canvas.width; x += cellSize) {
                for (let y = 0; y < canvas.height; y += cellSize) {
                    if ((Math.floor(x / cellSize) + Math.floor(y / cellSize)) % 2 === 0) {
                        ctx.fillRect(x, y, cellSize, cellSize);
                    }
                }
            }

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },

        renderMaskVignette: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const maxRadius = Math.sqrt(centerX * centerX + centerY * centerY);

            const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, maxRadius);
            gradient.addColorStop(0, 'rgba(255, 255, 255, 0.0)');
            gradient.addColorStop(0.6, 'rgba(255, 255, 255, 0.3)');
            gradient.addColorStop(1, 'rgba(255, 255, 255, 1.0)');

            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },

        renderMaskSpiral: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const maxRadius = Math.min(canvas.width, canvas.height) / 2;

            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.strokeStyle = 'rgba(255, 255, 255, 1.0)';
            ctx.lineWidth = 5;

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

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1;
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },

        renderMaskHexgrid: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);
            const hexSize = 15;
            const hexHeight = hexSize * Math.sqrt(3);

            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1;

            for (let row = 0; row < canvas.height / hexHeight + 1; row++) {
                for (let col = 0; col < canvas.width / (hexSize * 3) + 1; col++) {
                    const offsetX = (row % 2) * hexSize * 1.5;
                    const x = col * hexSize * 3 + offsetX;
                    const y = row * hexHeight;

                    ctx.beginPath();
                    for (let i = 0; i < 6; i++) {
                        const angle = i * Math.PI / 3;
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

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },

        renderMaskWavy: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);

            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.strokeStyle = 'rgba(255, 255, 255, 1.0)';
            ctx.lineWidth = 3;

            const amplitude = 10;
            const frequency = 0.05;

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

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1;
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },

        renderMaskConcentricRings: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas);
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;

            ctx.strokeStyle = 'rgba(255, 255, 255, 1.0)';
            ctx.lineWidth = 2;

            const ringCount = 6;
            const maxRadius = Math.min(canvas.width, canvas.height) / 2;
            const step = maxRadius / ringCount;

            for (let i = 1; i <= ringCount; i++) {
                const radius = i * step;
                ctx.beginPath();
                ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
                ctx.stroke();

                if (i % 2 === 0) {
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
                    ctx.beginPath();
                    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
                    ctx.fill();
                }
            }

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1;
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        },

        // Color scheme rendering functions
        renderColorInferno: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #000004, #3b0f70, #8c2981, #de4968, #fe9f6d, #fcfdbf)';
        },

        renderColorMagma: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #000004, #3b0f70, #8c2981, #de4968, #fe9f6d, #fcfdbf)';
        },

        renderColorPlasma: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #0d0887, #7e03a8, #cc4678, #f89441, #f0f921)';
        },

        renderColorViridis: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #440154, #30678d, #35b778, #fde724)';
        },

        renderColorTurbo: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #30123b, #4669db, #26bf8c, #d4ff50, #fab74c, #ba0100)';
        },

        renderColorJet: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #00007f, #0000ff, #00ffff, #ffff00, #ff0000, #7f0000)';
        },

        renderColorParula: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #352a87, #0f5cdd, #00b5a6, #ffc337, #fcfea4)';
        },

        renderColorRainbow: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000)';
        },

        renderColorHot: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #000000, #ff0000, #ffff00, #ffffff)';
        },

        renderColorBlueRed: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #0000ff, #7777ff, #ffffff, #ff7777, #ff0000)';
        },

        renderColorCool: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #00ffff, #77aaff, #aa77ff, #ff00ff)';
        },

        renderColorHsv: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff, #ff00ff, #ff0000)';
        },

        renderColorAutumn: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #ff0000, #ff7700, #ffaa00, #ffdd00, #ffff00)';
        },

        renderColorWinter: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #0000ff, #0077cc, #00aabb, #00ddaa)';
        },

        renderColorSpring: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #ff00ff, #ff33cc, #ff77aa, #ffaa77, #ffdd44, #ffff00)';
        },

        renderColorSummer: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #004433, #008855, #44aa66, #88cc77, #ccee88, #ffff66)';
        },

        renderColorCopper: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #000000, #331100, #662200, #993300, #cc6644, #ff9966)';
        },

        renderColorPink: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #0a0a0a, #550055, #aa0066, #ff44aa, #ffaadd, #ffffff)';
        },

        renderColorBone: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #000000, #2a2a3a, #5a748a, #9ebacb, #dfdfef, #ffffff)';
        },

        renderColorOcean: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #000000, #000066, #0000bb, #0066cc, #00ccff, #99ffff)';
        },

        renderColorTerrain: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #333399, #009966, #66cc33, #cccc33, #ff9933, #ffffff)';
        },

        renderColorNeon: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #ff00ff, #aa00ff, #5500ff, #0000ff, #00aaff, #00ffff, #00ff00, #aaff00, #ffff00)';
        },

        renderColorFire: function (swatchDiv: HTMLElement): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #000000, #330000, #660000, #bb0000, #ff0000, #ff7700, #ffdd00, #ffffff)';
        },

        // Temporal animation
        renderTemporalAnimation: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas, '#0a0a0a');
            let time = 0;

            function drawPattern(currentTime: number): void {
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
                        const yOffset = Math.sin(x * 0.02 + i * 0.3 + currentTime * 0.05) * maxOffset * Math.sin(currentTime * 0.02 + i * 0.1);
                        const actualY = startY + yOffset;
                        ctx.lineTo(x, actualY);
                    }
                    ctx.stroke();
                }
            }

            function animate(): void {
                drawPattern(time);
                time += 0.1;
                if (canvas.isConnected) {
                    requestAnimationFrame(animate);
                }
            }

            if (!canvas.dataset.animationRunning) {
                canvas.dataset.animationRunning = 'true';
                animate();
            }
        },

        // Intro noise demo - complex particle animation
        renderIntroNoiseDemo: function (canvas: HTMLCanvasElement): void {
            const ctx = this._clearCanvas(canvas, '#0f0f12');
            let time = 0;
            const particles: Particle[] = [];
            const numParticles = 120;
            const particleColorBase = [138, 43, 226];
            const particleColorHighlight = [224, 224, 232];
            const loopCycleDuration = 420;
            let lastLoopInstance = -1;
            const currentBackgroundPattern: BackgroundPattern = { type: null, variantSeed: 0, alpha: 0, startTime: 0 };
            const primordialDuration = 60;

            // Initialize particles
            for (let i = 0; i < numParticles; i++) {
                particles.push({
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    vx: 0,
                    vy: 0,
                    life: 0,
                    initialLife: Math.random() * 100 + 90,
                    radius: Math.random() * 1.7 + 0.6,
                    trail: [],
                    isPrimordial: true,
                    primordialTime: 0
                });
            }

            function noiseField(x: number, y: number, t: number): number {
                const baseScale = 0.009;
                const timeScale = 0.0023;
                const loopTime = t % loopCycleDuration;
                const currentLoopInst = Math.floor(t / loopCycleDuration);

                const globalRotationAngle = currentLoopInst * 0.15;
                const cosA = Math.cos(globalRotationAngle);
                const sinA = Math.sin(globalRotationAngle);

                const rotatedX = x * cosA - y * sinA;
                const rotatedY = x * sinA + y * cosA;

                let baseChaosVal = 0;
                const scale1 = baseScale * 1.8;
                baseChaosVal += Math.sin(rotatedX * scale1 + loopTime * timeScale * 1.1) *
                    Math.cos(rotatedY * scale1 * 0.8 - loopTime * timeScale * 0.9);
                const scale2 = baseScale * 0.9;
                baseChaosVal += Math.sin(rotatedY * scale2 * 1.1 - loopTime * timeScale * 1.3) *
                    Math.cos(rotatedX * scale2 * 0.9 + loopTime * timeScale * 1.0) * 0.8;
                const scale3 = baseScale * 1.2;
                const innerAngleOffset = Math.sin(loopTime * 0.0008 + currentLoopInst * 0.05) * 0.5;
                const N_x_chaos = rotatedX * Math.cos(innerAngleOffset) - rotatedY * Math.sin(innerAngleOffset);
                const N_y_chaos = rotatedX * Math.sin(innerAngleOffset) + rotatedY * Math.cos(innerAngleOffset);
                baseChaosVal += Math.sin(N_x_chaos * scale3 * 0.6 + loopTime * timeScale * 0.7) *
                    Math.cos(N_y_chaos * scale3 * 0.7 - loopTime * timeScale * 0.5) * 0.9;

                return (baseChaosVal / 2.7) * Math.PI * 4.0;
            }

            function drawIntroPattern(): void {
                const currentLoopInst = Math.floor(time / loopCycleDuration);
                const isNewLoopStart = (time % loopCycleDuration) < (60 / 2.5);

                if (isNewLoopStart && currentLoopInst !== lastLoopInstance) {
                    ctx.fillStyle = 'rgba(15, 15, 18, 0.45)';
                    lastLoopInstance = currentLoopInst;

                    const shaderPatternTypes: PatternType[] = ['tensor_field', 'cellular', 'domain_warp', 'perlin', 'curl_noise', 'waves_interference'];
                    currentBackgroundPattern.type = shaderPatternTypes[currentLoopInst % shaderPatternTypes.length];
                    currentBackgroundPattern.variantSeed = currentLoopInst;
                    currentBackgroundPattern.alpha = 1.0;
                    currentBackgroundPattern.startTime = time;

                    particles.forEach(p => {
                        p.x = Math.random() * canvas.width;
                        p.y = Math.random() * canvas.height;
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

                // Update and draw particles
                particles.forEach(p => {
                    if (p.isPrimordial) {
                        p.primordialTime++;
                        if (p.primordialTime >= primordialDuration) {
                            p.isPrimordial = false;
                        }
                    }

                    const finalAngle = noiseField(p.x, p.y, time);
                    p.vx += Math.cos(finalAngle) * 0.10;
                    p.vy += Math.sin(finalAngle) * 0.10;
                    p.vx *= 0.94;
                    p.vy *= 0.94;
                    p.x += p.vx;
                    p.y += p.vy;

                    p.trail.push({ x: p.x, y: p.y });
                    if (p.trail.length > 25) {
                        p.trail.shift();
                    }

                    let trailBaseAlpha = 0.8;
                    let particleHeadBaseAlpha = 0.65;
                    let currentParticleRadius = p.radius;
                    let r = particleColorBase[0], g = particleColorBase[1], b = particleColorBase[2];

                    if (p.isPrimordial) {
                        const primordialRatio = p.primordialTime / primordialDuration;
                        trailBaseAlpha *= primordialRatio * 0.5;
                        particleHeadBaseAlpha = primordialRatio * 0.5;
                        currentParticleRadius = p.radius * (0.3 + primordialRatio * 0.7);
                        r = (particleColorBase[0] * primordialRatio) + (70 * (1 - primordialRatio));
                        g = (particleColorBase[1] * primordialRatio) + (70 * (1 - primordialRatio));
                        b = (particleColorBase[2] * primordialRatio) + (90 * (1 - primordialRatio));
                    }

                    if (p.trail.length > 1) {
                        ctx.beginPath();
                        ctx.moveTo(p.trail[0].x, p.trail[0].y);
                        for (let i = 1; i < p.trail.length; i++) {
                            const trailSegmentAlpha = (i / p.trail.length) * trailBaseAlpha;
                            ctx.strokeStyle = `rgba(200, 200, 200, ${trailSegmentAlpha})`;
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
                    if (!p.isPrimordial) {
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

                    if (p.life <= 0 || p.x < -finalRadius * 2 || p.x > canvas.width + finalRadius * 2 || p.y < -finalRadius * 2 || p.y > canvas.height + finalRadius * 2) {
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

            function animateIntro(): void {
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

        renderColorPlaceholder: function (swatchDiv: HTMLElement, schemeName: string): void {
            swatchDiv.style.background = 'linear-gradient(to bottom, #333333, #666666, #999999, #cccccc)';

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

    window.NoiseVisualizer = NoiseVisualizer;

    // Use local reference for type safety
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        (async (): Promise<void> => { await NoiseVisualizer._preloadKofiImage(); })();
    } else {
        document.addEventListener('DOMContentLoaded', async () => {
            await NoiseVisualizer._preloadKofiImage();
        });
    }
}
