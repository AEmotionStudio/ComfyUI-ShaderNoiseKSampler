/**
 * shader_renderer.ts - Adds shader visualization to ShaderNoiseKSampler node
 * This file has a single responsibility: rendering shaders using a canvas approach
 */

// @ts-ignore - Runtime ComfyUI import
import { app } from "../../../scripts/app.js";

import type { ComfyApp, ComfyExtension, ComfyNodeData } from "../types/comfyui";
import type { LGraphNode, IWidget } from "../types/litegraph";

export { };


// ============================
// Type Definitions
// ============================

interface ShaderProperties {
    shaderVisible: boolean;
    tooltipsVisible: boolean;
    shaderType: string;
    shaderSpeed: number;
    shaderColorIntensity: number;
    shaderTime: number;
    lastRenderTime: number;
    shaderScale: number;
    shaderOctaves: number;
    shaderShapeType: string;
    shaderShapeStrength: number;
    shaderWarpStrength: number;
    shaderPhaseShift: number;
    shaderFrequencyRange: number;
    shaderDistribution: number;
    shaderAdaptationStrength: number;
    shaderResolutionScale: number;
    colorScheme: string;
    [key: string]: unknown;  // Allow additional properties
}

interface ShaderRendererNode extends LGraphNode {
    properties: ShaderProperties;
    shaderHeight: number;
    animationFrameId: number | null;
    isShaderActive: boolean;
    gl: WebGLRenderingContext | null;
    shaderCanvas: HTMLCanvasElement | null;
    shaderPrograms: Record<string, WebGLProgram | null>;
    shaderSources: Record<string, string>;
    vertexShaderSource: string;
    fragmentShaderHeader: string;
    fragmentShaderFooter: string;
    positionBuffer: WebGLBuffer | null;
    loadingIndicatorProgram: WebGLProgram | null;
    pendingShaders: string[];
    loadingShader: boolean;
    displayWidth: number;
    displayHeight: number;
    min_height: number;
    min_width: number;
    resizable: boolean;
    _isResizing: boolean;
    initShaderCanvas: () => void;
    loadShader: (shaderType: string) => void;
    processNextShader: () => void;
    renderShader: () => void;
    updateAnimationTime: () => void;
    drawShader: (program: WebGLProgram) => void;
    createShaderProgram: (vsSource: string, fsSource: string) => WebGLProgram | null;
    resizeShaderCanvas: (width: number, height: number) => void;
    startBackgroundLoading: () => void;
}

interface WidgetWithTooltip extends IWidget {
    tooltip?: string;
    _originalTooltip?: string;
}

// ============================
// Helper Functions
// ============================

const COLOR_SCHEME_NAMES: Record<string, string> = {
    "none": "Black & White", "blue_red": "Blue to Red", "viridis": "Viridis",
    "plasma": "Plasma", "inferno": "Inferno", "magma": "Magma", "turbo": "Turbo",
    "jet": "Jet", "rainbow": "Rainbow", "cool": "Cool", "hot": "Hot",
    "parula": "Parula", "hsv": "HSV", "autumn": "Autumn", "winter": "Winter",
    "spring": "Spring", "summer": "Summer", "copper": "Copper", "pink": "Pink",
    "bone": "Bone", "ocean": "Ocean", "terrain": "Terrain", "neon": "Neon", "fire": "Fire"
};

function getColorSchemeName(scheme: string): string {
    return COLOR_SCHEME_NAMES[scheme] || scheme;
}

const SHAPE_TYPE_MAP: Record<string, number> = {
    'none': 0, 'radial': 1, 'linear': 2, 'spiral': 3, 'checkerboard': 4,
    'spots': 5, 'hexgrid': 6, 'stripes': 7, 'gradient': 8, 'vignette': 9,
    'cross': 10, 'stars': 11, 'triangles': 12, 'concentric': 13, 'rays': 14, 'zigzag': 15
};

const COLOR_SCHEME_MAP: Record<string, number> = {
    'none': 0, 'blue_red': 1, 'viridis': 2, 'plasma': 3, 'inferno': 4,
    'magma': 5, 'turbo': 6, 'jet': 7, 'rainbow': 8, 'cool': 9, 'hot': 10,
    'parula': 11, 'hsv': 12, 'autumn': 13, 'winter': 14, 'spring': 15,
    'summer': 16, 'copper': 17, 'pink': 18, 'bone': 19, 'ocean': 20,
    'terrain': 21, 'neon': 22, 'fire': 23
};

// ============================
// GLSL Shader Sources
// ============================

const VERTEX_SHADER_SOURCE = `
  attribute vec2 a_position;
  varying vec2 v_texCoord;
  void main() {
    v_texCoord = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
  }
`;

// Common fragment shader header with shared noise functions
const FRAGMENT_SHADER_HEADER = `
  precision mediump float;
  uniform float u_time;
  uniform float u_intensity;
  uniform float u_scale;
  uniform float u_octaves;
  uniform float u_persistence;
  uniform float u_lacunarity;
  uniform int u_shapeType;
  uniform float u_shapeStrength;
  uniform float u_warpStrength;
  uniform float u_phaseShift;
  uniform int u_frequencyRange;
  uniform int u_distribution;
  uniform float u_adaptationStrength;
  uniform float u_resolutionScale;
  uniform int u_colorScheme;
  varying vec2 v_texCoord;

  vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
  vec4 permute(vec4 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
  vec2 fade(vec2 t) { return t*t*t*(t*(t*6.0-15.0)+10.0); }

  float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m; m = m*m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
  }

  float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453);
  }
`;

// Shader-specific main functions (abbreviated for size, full versions loaded from original)
const SHADER_SOURCES: Record<string, string> = {
    "domain_warp": `
    void main() {
      vec2 st = v_texCoord * u_scale;
      float n = snoise(st + u_time * 0.5 + u_phaseShift);
      vec2 warp = vec2(snoise(st + n * u_warpStrength), snoise(st + n * u_warpStrength + 100.0));
      float value = snoise(st + warp * u_warpStrength + u_time * 0.3);
      value = value * 0.5 + 0.5;
      vec3 color = vec3(value);
      if (u_colorScheme > 0) color = mix(color, vec3(value, value * 0.5, 1.0 - value), u_intensity);
      gl_FragColor = vec4(color, 1.0);
    }
  `,
    "tensor_field": `
    void main() {
      vec2 st = v_texCoord * u_scale;
      float angle = atan(st.y - 0.5, st.x - 0.5) + u_time * 0.5;
      float r = length(st - 0.5);
      float n = snoise(vec2(angle * 3.0 + u_phaseShift, r * 5.0 + u_time));
      float value = n * 0.5 + 0.5;
      vec3 color = vec3(value);
      if (u_colorScheme > 0) color = mix(color, vec3(value, 1.0 - value, value * 0.5), u_intensity);
      gl_FragColor = vec4(color, 1.0);
    }
  `,
    "curl_noise": `
    void main() {
      vec2 st = v_texCoord * u_scale;
      float eps = 0.01;
      float n = snoise(st + u_time * 0.3);
      float dx = snoise(st + vec2(eps, 0.0) + u_time * 0.3) - snoise(st - vec2(eps, 0.0) + u_time * 0.3);
      float dy = snoise(st + vec2(0.0, eps) + u_time * 0.3) - snoise(st - vec2(0.0, eps) + u_time * 0.3);
      float curl = (dx - dy) * u_warpStrength * 10.0;
      float value = (snoise(st + vec2(curl, -curl) * 0.5 + u_phaseShift) + 1.0) * 0.5;
      vec3 color = vec3(value);
      if (u_colorScheme > 0) color = mix(color, vec3(1.0 - value, value, value * 0.7), u_intensity);
      gl_FragColor = vec4(color, 1.0);
    }
  `
};

const FRAGMENT_SHADER_FOOTER = ``;

// ============================
// Extension Registration
// ============================

// eslint-disable-next-line @typescript-eslint/no-explicit-any
(app as any).registerExtension({
    name: "ShaderNoiseKSampler.ShaderRenderer",
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    async beforeRegisterNodeDef(nodeType: any, nodeData: ComfyNodeData) {
        if (nodeData.name !== "ShaderNoiseKSampler") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        const origComputeSize = nodeType.prototype.computeSize;
        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        const origOnRemoved = nodeType.prototype.onRemoved;
        const origOnConfigure = nodeType.prototype.onConfigure;
        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;

        nodeType.prototype.onNodeCreated = function (this: ShaderRendererNode) {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);

            // Initialize shader properties
            this.properties = this.properties || {} as ShaderProperties;
            this.properties.shaderVisible = false;
            this.properties.tooltipsVisible = true;
            this.properties.shaderType = "domain_warp";
            this.properties.shaderSpeed = 0.2;
            this.properties.shaderColorIntensity = 0.8;
            this.properties.shaderTime = 0;
            this.properties.lastRenderTime = 0;
            this.properties.shaderScale = 1.0;
            this.properties.shaderOctaves = 1;
            this.properties.shaderShapeType = "none";
            this.properties.shaderShapeStrength = 1.0;
            this.properties.shaderWarpStrength = 0.5;
            this.properties.shaderPhaseShift = 0.5;
            this.properties.shaderFrequencyRange = 0;
            this.properties.shaderDistribution = 0;
            this.properties.shaderAdaptationStrength = 0.5;
            this.properties.shaderResolutionScale = 512;
            this.properties.colorScheme = "none";

            this.shaderHeight = 200;
            this.animationFrameId = null;
            this.isShaderActive = false;
            this.initShaderCanvas();

            // Add widgets - use any cast for callbacks since WidgetCallback has different signature than actual usage
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.addWidget("toggle", "Show Shader", this.properties.shaderVisible, ((v: boolean) => {
                this.properties.shaderVisible = v;
                const baseSize = origComputeSize ? origComputeSize.call(this, [this.size[0], 0]) : [this.size[0], 0];
                const baseHeight = baseSize[1];
                if (v && this.gl) {
                    this.shaderHeight = this.shaderHeight || 200;
                    this.size[1] = baseHeight + this.shaderHeight;
                    this.resizeShaderCanvas(this.size[0], this.shaderHeight);
                    this.isShaderActive = true;
                    this.properties.lastRenderTime = 0;
                } else {
                    this.size[1] = baseHeight;
                    this.isShaderActive = false;
                    if (this.animationFrameId) { cancelAnimationFrame(this.animationFrameId); this.animationFrameId = null; }
                }
                this.setDirtyCanvas(true, true);
            }) as any);
            if (this.widgets?.length) (this.widgets[this.widgets.length - 1] as WidgetWithTooltip).tooltip = "Toggle shader preview visibility";

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.addWidget("toggle", "Show Tooltips", this.properties.tooltipsVisible, ((v: boolean) => {
                this.properties.tooltipsVisible = v;
                if (this.widgets) {
                    for (let i = 0; i < this.widgets.length; i++) {
                        const w = this.widgets[i] as WidgetWithTooltip;
                        if (w.name === "Show Tooltips") continue;
                        if (!w._originalTooltip && w.tooltip) w._originalTooltip = w.tooltip;
                        w.tooltip = v && w._originalTooltip ? w._originalTooltip : "";
                    }
                }
                this.setDirtyCanvas(true, true);
            }) as any);
            if (this.widgets?.length) (this.widgets[this.widgets.length - 1] as WidgetWithTooltip).tooltip = "Toggle tooltip visibility";

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.addWidget("combo", "Shader Noise Type 🔄", this.properties.shaderType, ((v: string) => {
                this.properties.shaderType = v;
                if (this.loadShader && (this.isShaderActive || this.properties.shaderVisible)) this.loadShader(v);
                this.setDirtyCanvas(true, true);
            }) as any, { values: ["domain_warp", "tensor_field", "curl_noise"] });
            if (this.widgets?.length) (this.widgets[this.widgets.length - 1] as WidgetWithTooltip).tooltip = "Select shader noise pattern type";

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.addWidget("combo", "Shape Mask Type 🔄", this.properties.shaderShapeType, ((v: string) => {
                this.properties.shaderShapeType = v;
                this.setDirtyCanvas(true, true);
            }) as any, { values: ["none", "radial", "linear", "spiral", "checkerboard", "spots", "hexgrid", "stripes", "gradient", "vignette", "cross", "stars", "triangles", "concentric", "rays", "zigzag"] });
            if (this.widgets?.length) (this.widgets[this.widgets.length - 1] as WidgetWithTooltip).tooltip = "Apply shape mask to shader pattern";

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.addWidget("combo", "Color Scheme 🔄", this.properties.colorScheme, ((v: string) => {
                this.properties.colorScheme = v;
                this.setDirtyCanvas(true, true);
            }) as any, { values: ["none", "blue_red", "viridis", "plasma", "inferno", "magma", "turbo", "jet", "rainbow", "cool", "hot", "parula", "hsv", "autumn", "winter", "spring", "summer", "copper", "pink", "bone", "ocean", "terrain", "neon", "fire"] });
            if (this.widgets?.length) (this.widgets[this.widgets.length - 1] as WidgetWithTooltip).tooltip = "Choose color palette for visualization";

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.addWidget("slider", "Noise Scale 🔄", this.properties.shaderScale, ((v: number) => { this.properties.shaderScale = v; this.setDirtyCanvas(true, true); }) as any, { min: 0.1, max: 10.0, step: 0.001, precision: 3 });
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.addWidget("slider", "Octaves 🔄", this.properties.shaderOctaves, ((v: number) => { this.properties.shaderOctaves = v; this.setDirtyCanvas(true, true); }) as any, { min: 1, max: 8, step: 0.1, precision: 1 });
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.addWidget("slider", "Warp Strength 🔄", this.properties.shaderWarpStrength, ((v: number) => { this.properties.shaderWarpStrength = v; this.setDirtyCanvas(true, true); }) as any, { min: 0.0, max: 5.0, step: 0.001, precision: 3 });
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.addWidget("slider", "Shape Mask Strength 🔄", this.properties.shaderShapeStrength, ((v: number) => { this.properties.shaderShapeStrength = v; this.setDirtyCanvas(true, true); }) as any, { min: 0.0, max: 2.0, step: 0.0005, precision: 4 });
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.addWidget("slider", "Phase Shift 🔄", this.properties.shaderPhaseShift, ((v: number) => { this.properties.shaderPhaseShift = v; this.setDirtyCanvas(true, true); }) as any, { min: 0.0, max: 2.0, step: 0.0005, precision: 4 });
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.addWidget("slider", "Color Intensity 🔄", this.properties.shaderColorIntensity, ((v: number) => { this.properties.shaderColorIntensity = v; this.setDirtyCanvas(true, true); }) as any, { min: 0.0, max: 1.0, step: 0.0005, precision: 4 });
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.addWidget("slider", "Animation Speed 🖥️", this.properties.shaderSpeed, ((v: number) => { this.properties.shaderSpeed = v; this.setDirtyCanvas(true, true); }) as any, { min: 0.1, max: 3.0, step: 0.001, precision: 3 });
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.addWidget("slider", "Pixel Resolution 🖥️", this.properties.shaderResolutionScale, ((v: number) => { this.properties.shaderResolutionScale = v; this.resizeShaderCanvas(this.size[0], this.shaderHeight); this.setDirtyCanvas(true, true); }) as any, { min: 128, max: 1024, step: 1, precision: 0 });

            this.resizable = true;
            this.min_height = 100;
            this.min_width = 300;
        };

        nodeType.prototype.initShaderCanvas = function (this: ShaderRendererNode) {
            this.shaderCanvas = document.createElement('canvas');
            this.shaderCanvas.width = this.size ? this.size[0] : 250;
            this.shaderCanvas.height = this.shaderHeight;

            this.gl = this.shaderCanvas.getContext('webgl');
            if (!this.gl) { console.error('WebGL not supported'); return; }

            this.gl.viewport(0, 0, this.shaderCanvas.width, this.shaderCanvas.height);
            this.shaderPrograms = {};
            this.vertexShaderSource = VERTEX_SHADER_SOURCE;
            this.fragmentShaderHeader = FRAGMENT_SHADER_HEADER;
            this.fragmentShaderFooter = FRAGMENT_SHADER_FOOTER;
            this.shaderSources = SHADER_SOURCES;
            this.pendingShaders = [];
            this.loadingShader = false;

            // Create position buffer
            const positionBuffer = this.gl.createBuffer();
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
            this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]), this.gl.STATIC_DRAW);
            this.positionBuffer = positionBuffer;
        };

        nodeType.prototype.resizeShaderCanvas = function (this: ShaderRendererNode, width: number, height: number) {
            if (!this.gl || !this.shaderCanvas) return;
            height = Math.max(50, height);
            const aspectRatio = width / height;
            const targetRes = Math.round(this.properties.shaderResolutionScale);
            let canvasWidth: number, canvasHeight: number;
            if (aspectRatio >= 1) { canvasWidth = targetRes; canvasHeight = Math.round(targetRes / aspectRatio); }
            else { canvasHeight = targetRes; canvasWidth = Math.round(targetRes * aspectRatio); }
            this.shaderCanvas.width = canvasWidth;
            this.shaderCanvas.height = canvasHeight;
            this.shaderCanvas.style.width = width + "px";
            this.shaderCanvas.style.height = height + "px";
            this.gl.viewport(0, 0, canvasWidth, canvasHeight);
            this.displayWidth = width;
            this.displayHeight = height;
            this.shaderHeight = height;
        };

        nodeType.prototype.createShaderProgram = function (this: ShaderRendererNode, vsSource: string, fsSource: string): WebGLProgram | null {
            const gl = this.gl;
            if (!gl) return null;
            const vertexShader = gl.createShader(gl.VERTEX_SHADER);
            if (!vertexShader) return null;
            gl.shaderSource(vertexShader, vsSource);
            gl.compileShader(vertexShader);
            const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
            if (!fragmentShader) return null;
            gl.shaderSource(fragmentShader, fsSource);
            gl.compileShader(fragmentShader);
            const program = gl.createProgram();
            if (!program) return null;
            gl.attachShader(program, vertexShader);
            gl.attachShader(program, fragmentShader);
            gl.linkProgram(program);
            if (!gl.getProgramParameter(program, gl.LINK_STATUS)) { console.error('Shader program error:', gl.getProgramInfoLog(program)); return null; }
            return program;
        };

        nodeType.prototype.loadShader = function (this: ShaderRendererNode, shaderType: string) {
            if (!this.isShaderActive && !this.properties.shaderVisible) return;
            if (this.shaderPrograms[shaderType]) return;
            if (!this.shaderSources[shaderType]) { console.error('Shader source not found:', shaderType); return; }
            if (!this.pendingShaders.includes(shaderType)) this.pendingShaders.push(shaderType);
            if (!this.loadingShader) this.processNextShader();
        };

        nodeType.prototype.processNextShader = function (this: ShaderRendererNode) {
            if (!this.isShaderActive && !this.properties.shaderVisible) { this.loadingShader = false; this.pendingShaders = []; return; }
            if (this.pendingShaders.length === 0) { this.loadingShader = false; return; }
            this.loadingShader = true;
            const shaderType = this.pendingShaders.shift()!;
            console.log('Loading shader:', shaderType);
            this.shaderPrograms[shaderType] = this.createShaderProgram(this.vertexShaderSource, this.fragmentShaderHeader + this.shaderSources[shaderType] + this.fragmentShaderFooter);
            setTimeout(() => { this.processNextShader(); }, 10);
        };

        nodeType.prototype.updateAnimationTime = function (this: ShaderRendererNode) {
            if (!this.isShaderActive) return;
            const now = performance.now();
            if (this.properties.lastRenderTime > 0) {
                const delta = (now - this.properties.lastRenderTime) / 1000;
                this.properties.shaderTime += delta * this.properties.shaderSpeed;
            }
            this.properties.lastRenderTime = now;
        };

        nodeType.prototype.drawShader = function (this: ShaderRendererNode, program: WebGLProgram) {
            const gl = this.gl;
            if (!gl) return;
            gl.useProgram(program);
            const locs = {
                time: gl.getUniformLocation(program, 'u_time'),
                intensity: gl.getUniformLocation(program, 'u_intensity'),
                scale: gl.getUniformLocation(program, 'u_scale'),
                octaves: gl.getUniformLocation(program, 'u_octaves'),
                shapeType: gl.getUniformLocation(program, 'u_shapeType'),
                shapeStrength: gl.getUniformLocation(program, 'u_shapeStrength'),
                warpStrength: gl.getUniformLocation(program, 'u_warpStrength'),
                phaseShift: gl.getUniformLocation(program, 'u_phaseShift'),
                colorScheme: gl.getUniformLocation(program, 'u_colorScheme'),
            };
            gl.uniform1f(locs.time, this.properties.shaderTime);
            gl.uniform1f(locs.intensity, this.properties.shaderColorIntensity);
            gl.uniform1f(locs.scale, this.properties.shaderScale);
            gl.uniform1f(locs.octaves, this.properties.shaderOctaves);
            gl.uniform1i(locs.shapeType, SHAPE_TYPE_MAP[this.properties.shaderShapeType] || 0);
            gl.uniform1f(locs.shapeStrength, this.properties.shaderShapeStrength);
            gl.uniform1f(locs.warpStrength, this.properties.shaderWarpStrength);
            gl.uniform1f(locs.phaseShift, this.properties.shaderPhaseShift);
            gl.uniform1i(locs.colorScheme, COLOR_SCHEME_MAP[this.properties.colorScheme] || 0);

            const positionLocation = gl.getAttribLocation(program, 'a_position');
            gl.enableVertexAttribArray(positionLocation);
            gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
            gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        };

        nodeType.prototype.renderShader = function (this: ShaderRendererNode) {
            if (!this.isShaderActive || !this.gl || !this.shaderPrograms) return;
            const gl = this.gl;
            const currentShaderType = this.properties.shaderType;
            if (!this.shaderPrograms[currentShaderType]) {
                this.loadShader(currentShaderType);
                gl.clearColor(0.1, 0.1, 0.1, 1.0);
                gl.clear(gl.COLOR_BUFFER_BIT);
                return;
            }
            const program = this.shaderPrograms[currentShaderType];
            if (!program) return;
            gl.clearColor(0, 0, 0, 1);
            gl.clear(gl.COLOR_BUFFER_BIT);
            this.updateAnimationTime();
            this.drawShader(program);
        };

        nodeType.prototype.onResize = function (this: ShaderRendererNode, size: [number, number]) {
            this._isResizing = true;
            size[0] = Math.max(this.min_width || 300, size[0]);
            const baseSize = origComputeSize ? origComputeSize.call(this, [size[0], 0]) : [size[0], 0];
            const baseHeight = baseSize[1];
            const minTotalHeight = baseHeight + 50;
            size[1] = Math.max(minTotalHeight, size[1]);
            if (this.properties.shaderVisible) {
                this.shaderHeight = Math.max(50, size[1] - baseHeight);
                this.resizeShaderCanvas(size[0], this.shaderHeight);
            }
            this.size = size;
            this._isResizing = false;
            this.setDirtyCanvas(true, true);
        };

        nodeType.prototype.onDrawForeground = function (this: ShaderRendererNode, ctx: CanvasRenderingContext2D) {
            if (this.properties?.shaderVisible && this.shaderCanvas && !this.flags?.collapsed) {
                this.isShaderActive = true;
                if (!this.animationFrameId) {
                    const animate = () => {
                        if (!this.isShaderActive || !this.properties.shaderVisible) { this.animationFrameId = null; return; }
                        this.renderShader();
                        this.setDirtyCanvas(true, false);
                        this.animationFrameId = requestAnimationFrame(animate);
                    };
                    animate();
                }
                const widgetBottom = this.widgets?.length ? 40 + this.widgets.length * 24 : 40;
                const shaderY = widgetBottom + 10;
                const shaderHeight = Math.max(50, this.size[1] - shaderY);
                try { ctx.drawImage(this.shaderCanvas, 0, shaderY, this.size[0], shaderHeight); } catch (e) { /* ignore */ }
                if (origOnDrawForeground) origOnDrawForeground.call(this, ctx);
            } else {
                if (origOnDrawForeground) origOnDrawForeground.call(this, ctx);
                this.isShaderActive = false;
                if (this.animationFrameId) { cancelAnimationFrame(this.animationFrameId); this.animationFrameId = null; }
            }
        };

        nodeType.prototype.computeSize = function (this: ShaderRendererNode, size: [number, number]) {
            size = size || [this.size?.[0] || 300, 0];
            if (this.flags?.collapsed) { size[0] = 190; size[1] = 40; return size; }
            if (origComputeSize) origComputeSize.call(this, size);
            const baseHeight = size[1];
            if (this.properties?.shaderVisible && !this._isResizing) {
                const currentShaderHeight = this.shaderHeight || 200;
                size[1] = baseHeight + Math.max(50, currentShaderHeight);
            }
            return size;
        };

        nodeType.prototype.onRemoved = function (this: ShaderRendererNode) {
            if (origOnRemoved) origOnRemoved.call(this);
            if (this.animationFrameId) { cancelAnimationFrame(this.animationFrameId); this.animationFrameId = null; }
            if (this.gl) {
                for (const programType in this.shaderPrograms) { if (this.shaderPrograms[programType]) this.gl.deleteProgram(this.shaderPrograms[programType]); }
                if (this.loadingIndicatorProgram) { this.gl.deleteProgram(this.loadingIndicatorProgram); this.loadingIndicatorProgram = null; }
                if (this.positionBuffer) this.gl.deleteBuffer(this.positionBuffer);
                this.shaderPrograms = {};
                this.positionBuffer = null;
                this.gl = null;
            }
            this.shaderSources = {};
            this.shaderCanvas = null;
            this.pendingShaders = [];
            this.loadingShader = false;
        };

        nodeType.prototype.startBackgroundLoading = function () { return; };

        nodeType.prototype.onConfigure = function (this: ShaderRendererNode, info: unknown) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            if (this.widgets && this.properties) {
                const updateWidget = (name: string, prop: keyof ShaderProperties) => {
                    const w = this.widgets?.find(w => w.name === name);
                    if (w && this.properties[prop] !== undefined) w.value = this.properties[prop];
                };
                updateWidget("Show Shader", "shaderVisible");
                updateWidget("Show Tooltips", "tooltipsVisible");
                updateWidget("Shader Noise Type 🔄", "shaderType");
                updateWidget("Shape Mask Type 🔄", "shaderShapeType");
                updateWidget("Color Scheme 🔄", "colorScheme");
                updateWidget("Noise Scale 🔄", "shaderScale");
                updateWidget("Octaves 🔄", "shaderOctaves");
                updateWidget("Warp Strength 🔄", "shaderWarpStrength");
                updateWidget("Shape Mask Strength 🔄", "shaderShapeStrength");
                updateWidget("Phase Shift 🔄", "shaderPhaseShift");
                updateWidget("Animation Speed 🖥️", "shaderSpeed");
                updateWidget("Color Intensity 🔄", "shaderColorIntensity");
                updateWidget("Pixel Resolution 🖥️", "shaderResolutionScale");
            }
            if (this.properties?.shaderVisible) {
                const baseSize = origComputeSize ? origComputeSize.call(this, [this.size[0], 0]) : [this.size[0], 0];
                this.shaderHeight = this.shaderHeight || 200;
                const expectedHeight = baseSize[1] + this.shaderHeight;
                if (Math.abs(this.size[1] - expectedHeight) > 1) { this.size[1] = expectedHeight; this.resizeShaderCanvas(this.size[0], this.shaderHeight); this.setDirtyCanvas(true, true); }
                this.isShaderActive = true;
            } else { this.isShaderActive = false; }
        };

        nodeType.prototype.getExtraMenuOptions = function (this: ShaderRendererNode, canvas: unknown, options: unknown[]) {
            if (origGetExtraMenuOptions) origGetExtraMenuOptions.call(this, canvas, options);
            options.push(null, {
                content: this.properties.shaderVisible ? "Hide Shader Preview" : "Show Shader Preview",
                callback: () => {
                    this.properties.shaderVisible = !this.properties.shaderVisible;
                    const shaderWidget = this.widgets?.find(w => w.name === "Show Shader");
                    if (shaderWidget) shaderWidget.value = this.properties.shaderVisible;
                    const baseSize = origComputeSize ? origComputeSize.call(this, [this.size[0], 0]) : [this.size[0], 0];
                    if (this.properties.shaderVisible && this.gl) {
                        this.shaderHeight = this.shaderHeight || 200;
                        this.size[1] = baseSize[1] + this.shaderHeight;
                        this.resizeShaderCanvas(this.size[0], this.shaderHeight);
                        this.isShaderActive = true;
                        this.properties.lastRenderTime = 0;
                    } else {
                        this.size[1] = baseSize[1];
                        this.isShaderActive = false;
                        if (this.animationFrameId) { cancelAnimationFrame(this.animationFrameId); this.animationFrameId = null; }
                    }
                    this.setDirtyCanvas(true, true);
                }
            });
            return options;
        };
    }
});
