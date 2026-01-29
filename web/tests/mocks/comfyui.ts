/**
 * Mock implementation of ComfyUI APIs for testing
 */
import { vi } from 'vitest';

// Store registered extensions for testing
export const registeredExtensions: ComfyExtension[] = [];

export interface ComfyExtension {
    name: string;
    beforeRegisterNodeDef?: (
        nodeType: unknown,
        nodeData: unknown,
        app: ComfyApp
    ) => void;
    setup?: (app: ComfyApp) => void;
    nodeCreated?: (node: unknown) => void;
}

export interface ComfyApp {
    registerExtension: (extension: ComfyExtension) => void;
    canvas: unknown;
    graph: unknown;
    queuePrompt: () => Promise<void>;
}

export const app: ComfyApp = {
    registerExtension: vi.fn((extension: ComfyExtension) => {
        registeredExtensions.push(extension);
    }),
    canvas: {
        draw: vi.fn(),
    },
    graph: {
        nodes: [],
        setDirtyCanvas: vi.fn(),
    },
    queuePrompt: vi.fn(() => Promise.resolve()),
};

export interface ComfyApi {
    fetchApi: (route: string, options?: RequestInit) => Promise<Response>;
    addEventListener: (type: string, callback: (event: CustomEvent) => void) => void;
    removeEventListener: (type: string, callback: (event: CustomEvent) => void) => void;
    apiURL: (route: string) => string;
}

export const api: ComfyApi = {
    fetchApi: vi.fn(() =>
        Promise.resolve(new Response(JSON.stringify({}), { status: 200 }))
    ),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    apiURL: vi.fn((route: string) => `http://localhost:8188${route}`),
};

/**
 * Reset all mocks and clear registered extensions
 */
export function resetComfyMocks(): void {
    registeredExtensions.length = 0;
    vi.clearAllMocks();
}

/**
 * Create a mock LGraphNode for testing
 */
export function createMockNode(overrides: Partial<MockLGraphNode> = {}): MockLGraphNode {
    return {
        id: Math.floor(Math.random() * 10000),
        type: 'TestNode',
        title: 'Test Node',
        pos: [100, 100],
        size: [200, 150],
        flags: {},
        properties: {},
        widgets: [],
        widgets_values: [],
        inputs: [],
        outputs: [],
        graph: app.graph,
        setDirtyCanvas: vi.fn(),
        triggerSlot: vi.fn(),
        addWidget: vi.fn((_type, name, value) => ({
            name,
            type: _type,
            value,
            y: 0,
        })),
        addCustomWidget: vi.fn((widget) => widget),
        computeSize: vi.fn(() => [200, 150] as [number, number]),
        ...overrides,
    };
}

export interface MockLGraphNode {
    id: number;
    type: string;
    title: string;
    pos: [number, number];
    size: [number, number];
    flags: Record<string, boolean>;
    properties: Record<string, unknown>;
    widgets: unknown[];
    widgets_values: unknown[];
    inputs: unknown[];
    outputs: unknown[];
    graph: unknown;
    setDirtyCanvas: ReturnType<typeof vi.fn>;
    triggerSlot: ReturnType<typeof vi.fn>;
    addWidget: ReturnType<typeof vi.fn<[type: string, name: string, value: unknown], { name: string; type: string; value: unknown; y: number }>>;
    addCustomWidget: ReturnType<typeof vi.fn<[widget: unknown], unknown>>;
    computeSize: ReturnType<typeof vi.fn<[], [number, number]>>;
    onNodeCreated?: () => void;
    onRemoved?: () => void;
    onConfigure?: (info: unknown) => void;
    onDrawForeground?: (ctx: CanvasRenderingContext2D) => void;
    onResize?: (size: [number, number]) => void;
}
