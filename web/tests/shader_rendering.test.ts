/**
 * Integration tests for shader rendering pipeline
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { createMockNode, resetComfyMocks, app } from './mocks/comfyui';

// Import shader renderer
import '../src/shader_renderer';

describe('Shader Rendering Pipeline', () => {
    let canvas: HTMLCanvasElement;

    beforeEach(() => {
        resetComfyMocks();
        canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    describe('WebGL Context', () => {
        it('should create WebGL context from canvas', () => {
            const gl = canvas.getContext('webgl');
            expect(gl).toBeDefined();
        });

        it('should create WebGL2 context from canvas', () => {
            const gl = canvas.getContext('webgl2');
            expect(gl).toBeDefined();
        });

        it('should have shader creation methods', () => {
            const gl = canvas.getContext('webgl')!;
            expect(gl.createShader).toBeDefined();
            expect(gl.compileShader).toBeDefined();
            expect(gl.shaderSource).toBeDefined();
        });

        it('should have program creation methods', () => {
            const gl = canvas.getContext('webgl')!;
            expect(gl.createProgram).toBeDefined();
            expect(gl.linkProgram).toBeDefined();
            expect(gl.useProgram).toBeDefined();
        });

        it('should have uniform methods', () => {
            const gl = canvas.getContext('webgl')!;
            expect(gl.getUniformLocation).toBeDefined();
            expect(gl.uniform1f).toBeDefined();
            expect(gl.uniform2f).toBeDefined();
            expect(gl.uniform3f).toBeDefined();
            expect(gl.uniform4f).toBeDefined();
        });
    });

    describe('Shader Compilation', () => {
        it('should create vertex shader', () => {
            const gl = canvas.getContext('webgl')!;
            const shader = gl.createShader(gl.VERTEX_SHADER);
            expect(shader).toBeDefined();
        });

        it('should create fragment shader', () => {
            const gl = canvas.getContext('webgl')!;
            const shader = gl.createShader(gl.FRAGMENT_SHADER);
            expect(shader).toBeDefined();
        });

        it('should get compile status', () => {
            const gl = canvas.getContext('webgl')!;
            const shader = gl.createShader(gl.VERTEX_SHADER)!;
            gl.shaderSource(shader, 'void main() {}');
            gl.compileShader(shader);
            const status = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
            expect(status).toBe(true);
        });
    });

    describe('Buffer Operations', () => {
        it('should create buffer', () => {
            const gl = canvas.getContext('webgl')!;
            const buffer = gl.createBuffer();
            expect(buffer).toBeDefined();
        });

        it('should bind buffer', () => {
            const gl = canvas.getContext('webgl')!;
            const buffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
            expect(gl.bindBuffer).toHaveBeenCalled();
        });

        it('should upload buffer data', () => {
            const gl = canvas.getContext('webgl')!;
            const buffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1]), gl.STATIC_DRAW);
            expect(gl.bufferData).toHaveBeenCalled();
        });
    });

    describe('Drawing Operations', () => {
        it('should set viewport', () => {
            const gl = canvas.getContext('webgl')!;
            gl.viewport(0, 0, 512, 512);
            expect(gl.viewport).toHaveBeenCalledWith(0, 0, 512, 512);
        });

        it('should clear color buffer', () => {
            const gl = canvas.getContext('webgl')!;
            gl.clearColor(0, 0, 0, 1);
            gl.clear(gl.COLOR_BUFFER_BIT);
            expect(gl.clearColor).toHaveBeenCalled();
            expect(gl.clear).toHaveBeenCalled();
        });

        it('should draw arrays', () => {
            const gl = canvas.getContext('webgl')!;
            gl.drawArrays(gl.TRIANGLES, 0, 6);
            expect(gl.drawArrays).toHaveBeenCalledWith(gl.TRIANGLES, 0, 6);
        });
    });

    describe('Blending', () => {
        it('should enable blending', () => {
            const gl = canvas.getContext('webgl')!;
            gl.enable(gl.BLEND);
            expect(gl.enable).toHaveBeenCalledWith(gl.BLEND);
        });

        it('should set blend function', () => {
            const gl = canvas.getContext('webgl')!;
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
            expect(gl.blendFunc).toHaveBeenCalledWith(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        });
    });

    describe('Shader Node Properties', () => {
        it('should create node with shader properties', () => {
            const node = createMockNode({
                type: 'ShaderNoiseKSampler',
                properties: {
                    noiseType: 'perlin',
                    scale: 1.0,
                    octaves: 4,
                    persistence: 0.5,
                    lacunarity: 2.0,
                },
            });

            expect(node.properties.noiseType).toBe('perlin');
            expect(node.properties.scale).toBe(1.0);
            expect(node.properties.octaves).toBe(4);
        });

        it('should update shader properties', () => {
            const node = createMockNode({
                properties: { scale: 1.0 },
            });

            node.properties.scale = 2.5;

            expect(node.properties.scale).toBe(2.5);
        });
    });

    describe('Resource Cleanup', () => {
        it('should delete shader', () => {
            const gl = canvas.getContext('webgl')!;
            const shader = gl.createShader(gl.VERTEX_SHADER);
            gl.deleteShader(shader);
            expect(gl.deleteShader).toHaveBeenCalledWith(shader);
        });

        it('should delete program', () => {
            const gl = canvas.getContext('webgl')!;
            const program = gl.createProgram();
            gl.deleteProgram(program);
            expect(gl.deleteProgram).toHaveBeenCalledWith(program);
        });

        it('should delete buffer', () => {
            const gl = canvas.getContext('webgl')!;
            const buffer = gl.createBuffer();
            gl.deleteBuffer(buffer);
            expect(gl.deleteBuffer).toHaveBeenCalledWith(buffer);
        });
    });
});
