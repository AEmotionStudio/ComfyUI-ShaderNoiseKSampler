/**
 * Test setup file for Vitest
 */
import { vi } from 'vitest';

// Mock window.requestAnimationFrame
global.requestAnimationFrame = vi.fn((callback: FrameRequestCallback) => {
    return setTimeout(() => callback(performance.now()), 16) as unknown as number;
});

global.cancelAnimationFrame = vi.fn((id: number) => {
    clearTimeout(id);
});

// Mock WebGL context (basic stub)
HTMLCanvasElement.prototype.getContext = vi.fn(function (
    this: HTMLCanvasElement,
    contextId: string
) {
    if (contextId === '2d') {
        return {
            canvas: this,
            fillRect: vi.fn(),
            strokeRect: vi.fn(),
            clearRect: vi.fn(),
            getImageData: vi.fn(() => ({ data: new Uint8ClampedArray(4) })),
            putImageData: vi.fn(),
            createImageData: vi.fn(() => ({ data: new Uint8ClampedArray(4) })),
            setTransform: vi.fn(),
            resetTransform: vi.fn(),
            drawImage: vi.fn(),
            save: vi.fn(),
            restore: vi.fn(),
            scale: vi.fn(),
            rotate: vi.fn(),
            translate: vi.fn(),
            transform: vi.fn(),
            beginPath: vi.fn(),
            closePath: vi.fn(),
            moveTo: vi.fn(),
            lineTo: vi.fn(),
            bezierCurveTo: vi.fn(),
            quadraticCurveTo: vi.fn(),
            arc: vi.fn(),
            arcTo: vi.fn(),
            ellipse: vi.fn(),
            rect: vi.fn(),
            fill: vi.fn(),
            stroke: vi.fn(),
            clip: vi.fn(),
            isPointInPath: vi.fn(),
            isPointInStroke: vi.fn(),
            measureText: vi.fn(() => ({ width: 100 })),
            fillText: vi.fn(),
            strokeText: vi.fn(),
            createLinearGradient: vi.fn(() => ({
                addColorStop: vi.fn(),
            })),
            createRadialGradient: vi.fn(() => ({
                addColorStop: vi.fn(),
            })),
            createPattern: vi.fn(),
            globalAlpha: 1,
            globalCompositeOperation: 'source-over',
            fillStyle: '#000000',
            strokeStyle: '#000000',
            lineWidth: 1,
            lineCap: 'butt',
            lineJoin: 'miter',
            miterLimit: 10,
            lineDashOffset: 0,
            shadowOffsetX: 0,
            shadowOffsetY: 0,
            shadowBlur: 0,
            shadowColor: 'rgba(0, 0, 0, 0)',
            font: '10px sans-serif',
            textAlign: 'start',
            textBaseline: 'alphabetic',
            direction: 'ltr',
            imageSmoothingEnabled: true,
        } as unknown as CanvasRenderingContext2D;
    }

    if (contextId === 'webgl' || contextId === 'webgl2') {
        return {
            canvas: this,
            createShader: vi.fn(() => ({})),
            shaderSource: vi.fn(),
            compileShader: vi.fn(),
            getShaderParameter: vi.fn(() => true),
            createProgram: vi.fn(() => ({})),
            attachShader: vi.fn(),
            linkProgram: vi.fn(),
            getProgramParameter: vi.fn(() => true),
            useProgram: vi.fn(),
            getUniformLocation: vi.fn(() => ({})),
            getAttribLocation: vi.fn(() => 0),
            uniform1f: vi.fn(),
            uniform2f: vi.fn(),
            uniform3f: vi.fn(),
            uniform4f: vi.fn(),
            uniform1i: vi.fn(),
            createBuffer: vi.fn(() => ({})),
            bindBuffer: vi.fn(),
            bufferData: vi.fn(),
            enableVertexAttribArray: vi.fn(),
            vertexAttribPointer: vi.fn(),
            drawArrays: vi.fn(),
            viewport: vi.fn(),
            clearColor: vi.fn(),
            clear: vi.fn(),
            enable: vi.fn(),
            disable: vi.fn(),
            blendFunc: vi.fn(),
            deleteShader: vi.fn(),
            deleteProgram: vi.fn(),
            deleteBuffer: vi.fn(),
            getShaderInfoLog: vi.fn(() => ''),
            getProgramInfoLog: vi.fn(() => ''),
            VERTEX_SHADER: 35633,
            FRAGMENT_SHADER: 35632,
            COMPILE_STATUS: 35713,
            LINK_STATUS: 35714,
            ARRAY_BUFFER: 34962,
            STATIC_DRAW: 35044,
            FLOAT: 5126,
            TRIANGLES: 4,
            COLOR_BUFFER_BIT: 16384,
            BLEND: 3042,
            SRC_ALPHA: 770,
            ONE_MINUS_SRC_ALPHA: 771,
        } as unknown as WebGLRenderingContext;
    }

    return null;
}) as typeof HTMLCanvasElement.prototype.getContext;
