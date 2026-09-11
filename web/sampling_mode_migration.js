// Import app from ComfyUI at runtime (this import is resolved by the browser)
// @ts-ignore - ComfyUI provides this at runtime
import { app } from '../../scripts/app.js';
const NODE_NAME = 'ShaderNoiseKSamplerDirect';
const SAMPLING_MODE_WIDGET = 'sampling_mode';
const VERSION_PROPERTY = 'snk_version';
const CURRENT_VERSION = 2;
/** True when this serialized node predates the sampling modes. */
export function needsLegacySampling(info) {
    const properties = info?.properties;
    return properties?.[VERSION_PROPERTY] === undefined;
}
/** Set the sampling_mode widget, if the node has one. */
export function setSamplingMode(node, mode) {
    const widget = node.widgets?.find((w) => w.name === SAMPLING_MODE_WIDGET);
    if (widget)
        widget.value = mode;
}
/** Record that this node was written by a version that understands sampling modes. */
function stampVersion(node) {
    node.properties = node.properties || {};
    node.properties[VERSION_PROPERTY] = CURRENT_VERSION;
}
const extension = {
    name: 'ShaderNoiseKSampler.SamplingModeMigration',
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_NAME)
            return;
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated)
                origOnNodeCreated.call(this);
            stampVersion(this);
        };
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origOnConfigure)
                origOnConfigure.call(this, info);
            // Read the marker from the saved data, not from this.properties:
            // onNodeCreated has already stamped the live node by this point.
            if (needsLegacySampling(info)) {
                setSamplingMode(this, 'legacy');
            }
            stampVersion(this);
        };
    },
};
app.registerExtension(extension);
//# sourceMappingURL=sampling_mode_migration.js.map