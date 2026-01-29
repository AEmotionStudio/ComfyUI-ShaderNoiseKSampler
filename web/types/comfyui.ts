/**
 * Type declarations for ComfyUI frontend APIs
 * These are exported as a module to be used via path mapping
 */

import type { LGraphNode, LGraph, LGraphCanvas, IWidget } from './litegraph';

export interface ComfyApp {
    /**
     * Register an extension that hooks into ComfyUI's lifecycle
     */
    registerExtension(extension: ComfyExtension): void;

    /**
     * The LiteGraph canvas instance
     */
    canvas: LGraphCanvas;

    /**
     * The LiteGraph graph instance
     */
    graph: LGraph;

    /**
     * Queue a prompt for execution
     */
    queuePrompt(number?: number, batchCount?: number): Promise<void>;
}

export interface ComfyExtension {
    /**
     * Unique name for the extension
     */
    name: string;

    /**
     * Called before a node type is registered
     */
    beforeRegisterNodeDef?(
        nodeType: NodeTypeConstructor,
        nodeData: ComfyNodeData,
        app: ComfyApp
    ): void | Promise<void>;

    /**
     * Called when the app is set up
     */
    setup?(app: ComfyApp): void | Promise<void>;

    /**
     * Called when a node is created
     */
    nodeCreated?(node: LGraphNode): void;
}

/**
 * Node type constructor with prototype
 */
export interface NodeTypeConstructor {
    new(): LGraphNode;
    prototype: LGraphNode & {
        onNodeCreated?: () => void;
        onRemoved?: () => void;
        onDrawForeground?: (ctx: CanvasRenderingContext2D) => void;
        onDrawBackground?: (ctx: CanvasRenderingContext2D) => void;
        onConfigure?: (info: unknown) => void;
        onResize?: (size: [number, number]) => void;
    };
}

export interface ComfyNodeData {
    name: string;
    display_name?: string;
    description?: string;
    category?: string;
    input?: {
        required?: Record<string, ComfyInputSpec>;
        optional?: Record<string, ComfyInputSpec>;
    };
    output?: string[];
    output_name?: string[];
}

export type ComfyInputSpec = [string | string[], Record<string, unknown>?];

export interface ComfyApi {
    /**
     * Fetch from the ComfyUI API
     */
    fetchApi(route: string, options?: RequestInit): Promise<Response>;

    /**
     * Add an event listener for ComfyUI events
     */
    addEventListener(type: string, callback: (event: CustomEvent) => void): void;

    /**
     * Remove an event listener
     */
    removeEventListener(
        type: string,
        callback: (event: CustomEvent) => void
    ): void;

    /**
     * Get the current API host URL
     */
    apiURL(route: string): string;
}

/**
 * The global app instance - will be the actual ComfyUI app at runtime
 */
export declare const app: ComfyApp;

/**
 * The global api instance - will be the actual ComfyUI api at runtime
 */
export declare const api: ComfyApi;

// Re-export LiteGraph types for convenience
export type { LGraphNode, LGraph, LGraphCanvas, IWidget };
