// @ts-ignore - ComfyUI provides this at runtime
import { app } from '../../scripts/app.js';
// @ts-ignore - ComfyUI provides this at runtime
import { api } from '../../scripts/api.js';
const NODE_NAMES = new Set(['ShaderNoiseKSamplerDirect', 'ShaderNoiseWalk']);
const PRESET_WIDGET = 'preset';
const WALK_PARAMETER_WIDGET = 'walk_parameter';
const CUSTOM = 'custom';
const ROUTE = '/shader_noise_ksampler/presets';
// Set while a preset is being written, so the widgets it writes do not read
// their own change as a hand edit and reset the preset they came from.
let applying = false;
function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}
/** The values a preset sets, minus the inputs excluded. "custom" and unknown names set nothing. */
export function presetValues(table, preset, exclude = new Set()) {
    if (typeof preset !== 'string' || preset === CUSTOM)
        return {};
    const bundle = table[preset];
    if (!bundle)
        return {};
    return Object.fromEntries(Object.entries(bundle).filter(([name]) => !exclude.has(name)));
}
/** Inputs the node drives itself, which a preset must leave alone: the Walk node's ramped parameter. */
export function excludedInputs(node) {
    const walked = findWidget(node, WALK_PARAMETER_WIDGET)?.value;
    return new Set(typeof walked === 'string' ? [walked] : []);
}
/** Write a preset's values into the node's widgets. Returns the names written. */
export function applyPreset(node, table, preset) {
    const written = [];
    applying = true;
    try {
        for (const [name, value] of Object.entries(presetValues(table, preset, excludedInputs(node)))) {
            const widget = findWidget(node, name);
            if (!widget)
                continue;
            widget.value = value;
            // Run the widget's own callback so whatever follows it notices: the
            // shader preview redraws when shader_type changes.
            widget.callback?.(value);
            written.push(name);
        }
    }
    finally {
        applying = false;
    }
    return written;
}
/**
 * A widget the preset controls was edited to a value the preset does not hold,
 * so the preset no longer describes the panel. Returns true if it was reset.
 */
export function releasePreset(node, loaded, changed, value) {
    if (applying || !loaded.keys.includes(changed) || excludedInputs(node).has(changed))
        return false;
    const presetWidget = findWidget(node, PRESET_WIDGET);
    if (!presetWidget || presetWidget.value === CUSTOM)
        return false;
    if (loaded.table[presetWidget.value]?.[changed] === value)
        return false;
    presetWidget.value = CUSTOM;
    return true;
}
/** Wrap the node's widget callbacks: the preset writes the others, and the others release the preset. */
export function wirePresetWidgets(node, source) {
    for (const widget of node.widgets ?? []) {
        const name = widget.name;
        if (!name)
            continue;
        const original = widget.callback;
        widget.callback = function (value, ...rest) {
            const loaded = source();
            if (loaded) {
                const changed = name === PRESET_WIDGET
                    ? applyPreset(node, loaded.table, value).length > 0
                    : releasePreset(node, loaded, name, value);
                if (changed)
                    node.setDirtyCanvas?.(true, true);
            }
            return original?.call(this, value, ...rest);
        };
    }
}
let loaded = null;
let loading = null;
/** The table, once the server has answered. Until then presets apply at run time only. */
export function currentPresets() {
    return loaded;
}
/** Fetch the preset table once. A failure leaves the panel as it was; the run is still correct. */
export function loadPresets() {
    if (!loading) {
        loading = api.fetchApi(ROUTE)
            .then(async (response) => {
            if (!response.ok)
                return;
            const body = await response.json();
            if (body && typeof body.presets === 'object' && Array.isArray(body.keys)) {
                loaded = { table: body.presets, keys: body.keys };
            }
        })
            .catch(() => undefined);
    }
    return loading;
}
const extension = {
    name: 'ShaderNoiseKSampler.PresetWidgets',
    async setup(_app) {
        await loadPresets();
    },
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (!NODE_NAMES.has(nodeData.name))
            return;
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated)
                origOnNodeCreated.call(this);
            wirePresetWidgets(this, currentPresets);
        };
    },
};
app.registerExtension(extension);
//# sourceMappingURL=preset_widgets.js.map