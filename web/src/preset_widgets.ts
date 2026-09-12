/**
 * preset_widgets.ts - make choosing a preset visibly set the widgets it controls.
 *
 * A preset bundles settings that only mean something together. The node applies
 * it at execution time (core/presets.py::apply_preset), which left the panel
 * lying: pick "explore" and shader_strength still reads whatever it read before,
 * while the run uses 0.30. This writes the preset's values into those widgets
 * when it is chosen, and drops back to "custom" as soon as one of them is edited
 * to something else, so what the panel shows is what the run does.
 *
 * The Python override stays the authority. A prompt submitted through the API
 * never runs this, the browser's copy of the table can be older than the
 * server's, and the Walk node calls the sampler directly for every point of its
 * ramp. With this in place the override is a no-op for anyone using the panel,
 * which is not a reason to remove it.
 *
 * The table is fetched from the server rather than copied here: preset values get
 * recalibrated against real runs, and a second copy would drift the first time.
 * A saved workflow is never rewritten on load -- that would be a silent edit of
 * the user's graph, and the run is already correct without it.
 */
import type {
    ComfyApp,
    ComfyNodeData,
    ComfyExtension,
    NodeTypeConstructor,
} from '../types/comfyui';

// @ts-ignore - ComfyUI provides this at runtime
import { app } from '../../scripts/app.js';
// @ts-ignore - ComfyUI provides this at runtime
import { api } from '../../scripts/api.js';

const NODE_NAMES = new Set(['ShaderNoiseKSamplerDirect', 'ShaderNoiseWalk']);
const PRESET_WIDGET = 'preset';
const WALK_PARAMETER_WIDGET = 'walk_parameter';
const CUSTOM = 'custom';
const ROUTE = '/shader_noise_ksampler/presets';

export type PresetTable = Record<string, Record<string, unknown>>;

export interface LoadedPresets {
    table: PresetTable;
    keys: string[];
}

export interface PresetWidget {
    name?: string;
    value: unknown;
    callback?: (...args: unknown[]) => unknown;
}

export interface PresetNode {
    widgets?: PresetWidget[];
    setDirtyCanvas?: (foreground: boolean, background: boolean) => void;
}

// Set while a preset is being written, so the widgets it writes do not read
// their own change as a hand edit and reset the preset they came from.
let applying = false;

function findWidget(node: PresetNode, name: string): PresetWidget | undefined {
    return node.widgets?.find((w) => w.name === name);
}

/** The values a preset sets, minus the inputs excluded. "custom" and unknown names set nothing. */
export function presetValues(
    table: PresetTable,
    preset: unknown,
    exclude: ReadonlySet<string> = new Set()
): Record<string, unknown> {
    if (typeof preset !== 'string' || preset === CUSTOM) return {};
    const bundle = table[preset];
    if (!bundle) return {};
    return Object.fromEntries(Object.entries(bundle).filter(([name]) => !exclude.has(name)));
}

/** Inputs the node drives itself, which a preset must leave alone: the Walk node's ramped parameter. */
export function excludedInputs(node: PresetNode): Set<string> {
    const walked = findWidget(node, WALK_PARAMETER_WIDGET)?.value;
    return new Set(typeof walked === 'string' ? [walked] : []);
}

/** Write a preset's values into the node's widgets. Returns the names written. */
export function applyPreset(node: PresetNode, table: PresetTable, preset: unknown): string[] {
    const written: string[] = [];
    applying = true;
    try {
        for (const [name, value] of Object.entries(presetValues(table, preset, excludedInputs(node)))) {
            const widget = findWidget(node, name);
            if (!widget) continue;
            widget.value = value;
            // Run the widget's own callback so whatever follows it notices: the
            // shader preview redraws when shader_type changes.
            widget.callback?.(value);
            written.push(name);
        }
    } finally {
        applying = false;
    }
    return written;
}

/**
 * A widget the preset controls was edited to a value the preset does not hold,
 * so the preset no longer describes the panel. Returns true if it was reset.
 */
export function releasePreset(
    node: PresetNode,
    loaded: LoadedPresets,
    changed: string,
    value: unknown
): boolean {
    if (applying || !loaded.keys.includes(changed) || excludedInputs(node).has(changed)) return false;
    const presetWidget = findWidget(node, PRESET_WIDGET);
    if (!presetWidget || presetWidget.value === CUSTOM) return false;
    if (loaded.table[presetWidget.value as string]?.[changed] === value) return false;
    presetWidget.value = CUSTOM;
    return true;
}

/** Wrap the node's widget callbacks: the preset writes the others, and the others release the preset. */
export function wirePresetWidgets(node: PresetNode, source: () => LoadedPresets | null): void {
    for (const widget of node.widgets ?? []) {
        const name = widget.name;
        if (!name) continue;
        const original = widget.callback;
        widget.callback = function (this: unknown, value: unknown, ...rest: unknown[]) {
            const loaded = source();
            if (loaded) {
                const changed = name === PRESET_WIDGET
                    ? applyPreset(node, loaded.table, value).length > 0
                    : releasePreset(node, loaded, name, value);
                if (changed) node.setDirtyCanvas?.(true, true);
            }
            return original?.call(this, value, ...rest);
        };
    }
}

let loaded: LoadedPresets | null = null;
let loading: Promise<void> | null = null;

/** The table, once the server has answered. Until then presets apply at run time only. */
export function currentPresets(): LoadedPresets | null {
    return loaded;
}

/** Fetch the preset table once. A failure leaves the panel as it was; the run is still correct. */
export function loadPresets(): Promise<void> {
    if (!loading) {
        loading = api.fetchApi(ROUTE)
            .then(async (response: Response) => {
                if (!response.ok) return;
                const body = await response.json();
                if (body && typeof body.presets === 'object' && Array.isArray(body.keys)) {
                    loaded = { table: body.presets, keys: body.keys };
                }
            })
            .catch(() => undefined);
    }
    return loading as Promise<void>;
}

const extension: ComfyExtension = {
    name: 'ShaderNoiseKSampler.PresetWidgets',
    async setup(_app: ComfyApp): Promise<void> {
        await loadPresets();
    },
    async beforeRegisterNodeDef(
        nodeType: NodeTypeConstructor,
        nodeData: ComfyNodeData,
        _app: ComfyApp
    ): Promise<void> {
        if (!NODE_NAMES.has(nodeData.name)) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (this: unknown): void {
            if (origOnNodeCreated) origOnNodeCreated.call(this);
            wirePresetWidgets(this as PresetNode, currentPresets);
        };
    },
};

app.registerExtension(extension);
