/**
 * Type declarations for LiteGraph.js
 * Based on LiteGraph library used by ComfyUI
 */

/**
 * Base node class in LiteGraph
 */
export interface LGraphNode {
    id: number;
    type: string;
    title: string;
    pos: [number, number];
    size: [number, number];
    flags: Record<string, boolean>;
    properties: Record<string, unknown>;
    widgets?: IWidget[];
    widgets_values?: unknown[];
    inputs?: INodeSlot[];
    outputs?: INodeSlot[];
    graph?: LGraph;

    /**
     * Called when the node is added to the graph
     */
    onAdded?(graph: LGraph): void;

    /**
     * Called when the node is removed from the graph
     */
    onRemoved?(): void;

    /**
     * Called when the node is created
     */
    onNodeCreated?(): void;

    /**
     * Called to configure the node from serialized data
     */
    onConfigure?(info: SerializedLGraphNode): void;

    /**
     * Called when drawing the node foreground
     */
    onDrawForeground?(ctx: CanvasRenderingContext2D): void;

    /**
     * Called when drawing the node background
     */
    onDrawBackground?(ctx: CanvasRenderingContext2D): void;

    /**
     * Called when the node is resized
     */
    onResize?(size: [number, number]): void;

    /**
     * Called to compute the node size
     */
    computeSize?(size?: [number, number]): [number, number];

    /**
     * Add a widget to the node
     */
    addWidget(
        type: string,
        name: string,
        value: unknown,
        callback?: WidgetCallback,
        options?: WidgetOptions
    ): IWidget;

    /**
     * Add a custom widget to the node
     */
    addCustomWidget(widget: IWidget): IWidget;

    /**
     * Get extra menu options for context menu
     */
    getExtraMenuOptions?(
        canvas: LGraphCanvas,
        options: ContextMenuItem[]
    ): ContextMenuItem[] | void;

    /**
     * Set the node's dirty state
     */
    setDirtyCanvas(fg: boolean, bg?: boolean): void;

    /**
     * Trigger a slot
     */
    triggerSlot(slot: number, param?: unknown): void;
}

/**
 * Serialized node data
 */
export interface SerializedLGraphNode {
    id: number;
    type: string;
    pos: [number, number];
    size: [number, number];
    flags?: Record<string, boolean>;
    properties?: Record<string, unknown>;
    widgets_values?: unknown[];
}

/**
 * Node input/output slot
 */
export interface INodeSlot {
    name: string;
    type: string | number;
    link?: number | null;
    links?: number[];
}

/**
 * Widget in a node
 */
export interface IWidget {
    name: string;
    type: string;
    value: unknown;
    options?: WidgetOptions;
    y?: number;
    last_y?: number;
    parent?: LGraphNode;
    tooltip?: string;

    /**
     * Draw the widget
     */
    draw?(
        ctx: CanvasRenderingContext2D,
        node: LGraphNode,
        width: number,
        y: number,
        height: number
    ): void;

    /**
     * Handle mouse events
     */
    mouse?(
        event: MouseEvent,
        pos: [number, number],
        node: LGraphNode
    ): boolean | void;

    /**
     * Compute widget size
     */
    computeSize?(width: number): [number, number];

    /**
     * Called when widget is removed
     */
    onRemoved?(): void;

    /**
     * Callback when value changes
     */
    callback?: WidgetCallback;
}

export type WidgetCallback = (
    value: unknown,
    canvas: LGraphCanvas,
    node: LGraphNode,
    pos: [number, number],
    event: Event
) => void;

export interface WidgetOptions {
    min?: number;
    max?: number;
    step?: number;
    precision?: number;
    values?: string[] | (() => string[]);
    multiline?: boolean;
    serialize?: boolean;
    className?: string;
    [key: string]: unknown;
}

/**
 * Context menu item
 */
export interface ContextMenuItem {
    content: string;
    callback?: () => void;
    has_submenu?: boolean;
    submenu?: {
        options: ContextMenuItem[];
    };
    disabled?: boolean;
    title?: string;
}

/**
 * The graph container
 */
export interface LGraph {
    nodes: LGraphNode[];
    links: Record<number, LLink>;

    /**
     * Add a node to the graph
     */
    add(node: LGraphNode): void;

    /**
     * Remove a node from the graph
     */
    remove(node: LGraphNode): void;

    /**
     * Get a node by ID
     */
    getNodeById(id: number): LGraphNode | null;

    /**
     * Configure the graph from serialized data
     */
    configure(data: unknown): void;

    /**
     * Serialize the graph
     */
    serialize(): unknown;

    /**
     * Set the graph to dirty state
     */
    setDirtyCanvas(fg: boolean, bg?: boolean): void;
}

/**
 * Link between nodes
 */
export interface LLink {
    id: number;
    origin_id: number;
    origin_slot: number;
    target_id: number;
    target_slot: number;
    type: string | number;
}

/**
 * The canvas renderer
 */
export interface LGraphCanvas {
    canvas: HTMLCanvasElement;
    ctx: CanvasRenderingContext2D;
    graph: LGraph;
    scale: number;
    offset: [number, number];
    selected_nodes: Record<number, LGraphNode>;
    current_node?: LGraphNode;

    /**
     * Draw the canvas
     */
    draw(force_fg?: boolean, force_bg?: boolean): void;

    /**
     * Convert canvas position to graph position
     */
    convertEventToCanvasOffset(event: MouseEvent): [number, number];

    /**
     * Center the view on a node
     */
    centerOnNode(node: LGraphNode): void;

    /**
     * Prompt for a value
     */
    prompt(
        title: string,
        value: string,
        callback: (value: string) => void,
        event: Event
    ): void;

    /**
     * Draw node widgets
     */
    drawNodeWidgets?(
        node: LGraphNode,
        pos: [number, number],
        ctx: CanvasRenderingContext2D,
        active_widget: IWidget | null
    ): void;
}

/**
 * Global LiteGraph namespace
 */
export interface LiteGraphStatic {
    registered_node_types: Record<string, new () => LGraphNode>;
    registerNodeType(type: string, nodeClass: new () => LGraphNode): void;
    createNode(type: string): LGraphNode | null;
    NODE_TITLE_HEIGHT: number;
    NODE_SLOT_HEIGHT: number;
    NODE_WIDGET_HEIGHT: number;
    NODE_TITLE_TEXT_Y: number;
}

// Make types available globally
declare global {
    interface Window {
        LiteGraph: LiteGraphStatic;
        LGraph: new () => LGraph;
        LGraphNode: new () => LGraphNode;
        LGraphCanvas: new () => LGraphCanvas;
        showComfyToast?: (message: string, type?: string) => void;
        storageOptimizer?: {
            forceCleanup(): void;
        };
    }

    const LGraphCanvas: {
        prototype: LGraphCanvas;
    };
}
