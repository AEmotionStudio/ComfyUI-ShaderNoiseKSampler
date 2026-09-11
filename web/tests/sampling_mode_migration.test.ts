/**
 * Workflows saved before 2.0 must keep sampling the way they were tuned.
 *
 * 2.0 changes what a seed produces (stages became segments of one run, denoise
 * and custom sigmas take effect, blended noise keeps its distribution), so a
 * node loaded without the snk_version marker is switched to "legacy".
 */
import { describe, it, expect } from 'vitest';

import { needsLegacySampling, setSamplingMode } from '../src/sampling_mode_migration.js';

interface TestWidget {
    name?: string;
    value?: unknown;
}

function nodeWithWidgets(widgets: TestWidget[]): any {
    return { properties: {}, widgets };
}

describe('sampling mode migration', () => {
    describe('needsLegacySampling', () => {
        it('treats a node saved before 2.0 as legacy', () => {
            expect(needsLegacySampling({ properties: { 'Node name for S&R': 'ShaderNoiseKSamplerDirect' } })).toBe(true);
        });

        it('treats missing or empty serialized data as legacy', () => {
            expect(needsLegacySampling(undefined)).toBe(true);
            expect(needsLegacySampling(null)).toBe(true);
            expect(needsLegacySampling({})).toBe(true);
            expect(needsLegacySampling({ properties: {} })).toBe(true);
        });

        it('leaves a node saved by 2.0 alone', () => {
            expect(needsLegacySampling({ properties: { snk_version: 2 } })).toBe(false);
        });
    });

    describe('setSamplingMode', () => {
        it('sets the sampling_mode widget', () => {
            const node = nodeWithWidgets([
                { name: 'seed', value: 8888 },
                { name: 'sampling_mode', value: 'standard' },
            ]);

            setSamplingMode(node, 'legacy');

            expect(node.widgets[1].value).toBe('legacy');
            expect(node.widgets[0].value).toBe(8888);
        });

        it('does nothing when the widget is absent', () => {
            const node = nodeWithWidgets([{ name: 'seed', value: 8888 }]);
            expect(() => setSamplingMode(node, 'legacy')).not.toThrow();
        });

        it('does nothing when the node has no widgets', () => {
            expect(() => setSamplingMode({ properties: {} } as any, 'legacy')).not.toThrow();
        });
    });
});
