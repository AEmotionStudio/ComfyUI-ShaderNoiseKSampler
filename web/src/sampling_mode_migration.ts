/**
 * sampling_mode_migration.ts - keeps workflows saved before 2.0 on legacy sampling.
 *
 * 2.0 changed how stages sample: they are now segments of one run rather than
 * independent restarts, denoise and custom sigmas take effect, and blended noise
 * keeps the distribution the model expects. That changes what an existing seed
 * produces, so a node loaded from a workflow saved before 2.0 is switched to
 * "legacy" and reproduces its original output. Newly added nodes keep the
 * default, "standard".
 *
 * Saved nodes are recognised by the absence of the snk_version property, which
 * only 2.0 and later write.
 */
// Type imports from our local type definitions
import type {
    ComfyApp,
    ComfyNodeData,
    ComfyExtension,
    NodeTypeConstructor,
    LGraphNode,
} from '../types/comfyui';

// Import app from ComfyUI at runtime (this import is resolved by the browser)
// @ts-ignore - ComfyUI provides this at runtime
import { app } from '../../scripts/app.js';

const NODE_NAME = 'ShaderNoiseKSamplerDirect';
const SAMPLING_MODE_WIDGET = 'sampling_mode';
const VERSION_PROPERTY = 'snk_version';
const CURRENT_VERSION = 2;

/** Node carrying the properties this migration stamps */
interface MigratableNode extends LGraphNode {
    properties: Record<string, unknown>;
}

/** True when this serialized node predates the sampling modes. */
export function needsLegacySampling(info: unknown): boolean {
    const properties = (info as { properties?: Record<string, unknown> } | null | undefined)?.properties;
    return properties?.[VERSION_PROPERTY] === undefined;
}

/** Set the sampling_mode widget, if the node has one. */
export function setSamplingMode(node: MigratableNode, mode: string): void {
    const widget = node.widgets?.find((w) => w.name === SAMPLING_MODE_WIDGET);
    if (widget) widget.value = mode;
}

/** Record that this node was written by a version that understands sampling modes. */
function stampVersion(node: MigratableNode): void {
    node.properties = node.properties || {};
    node.properties[VERSION_PROPERTY] = CURRENT_VERSION;
}

const extension: ComfyExtension = {
    name: 'ShaderNoiseKSampler.SamplingModeMigration',
    async beforeRegisterNodeDef(
        nodeType: NodeTypeConstructor,
        nodeData: ComfyNodeData,
        _app: ComfyApp
    ): Promise<void> {
        if (nodeData.name !== NODE_NAME) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (this: MigratableNode): void {
            if (origOnNodeCreated) origOnNodeCreated.call(this);
            stampVersion(this);
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (this: MigratableNode, info: unknown): void {
            if (origOnConfigure) origOnConfigure.call(this, info);

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
