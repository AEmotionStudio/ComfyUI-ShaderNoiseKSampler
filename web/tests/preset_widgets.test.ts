/**
 * Choosing a preset must show its values in the panel, and editing one of those
 * values must stop the panel claiming the preset still applies.
 */
import { describe, it, expect, vi } from 'vitest';

import {
    applyPreset,
    currentPresets,
    loadPresets,
    presetValues,
    releasePreset,
    wirePresetWidgets,
    type LoadedPresets,
    type PresetNode,
} from '../src/preset_widgets.js';
import { api } from './mocks/comfyui';

const TABLE = {
    custom: {},
    explore: { shader_type: 'domain_warp', shader_strength: 0.3, travel_mode: 'walk' },
    jump: { shader_type: 'domain_warp', shader_strength: 0.7, travel_mode: 'jump' },
};
const LOADED: LoadedPresets = { table: TABLE, keys: ['shader_type', 'shader_strength', 'travel_mode'] };

function directNode(preset = 'custom'): PresetNode & { widgets: any[] } {
    return {
        widgets: [
            { name: 'seed', value: 8888 },
            { name: 'shader_type', value: 'curl_noise' },
            { name: 'shader_strength', value: 0.9 },
            { name: 'travel_mode', value: 'drift' },
            { name: 'preset', value: preset },
        ],
        setDirtyCanvas: vi.fn(),
    };
}

function value(node: { widgets: any[] }, name: string): unknown {
    return node.widgets.find((w) => w.name === name)?.value;
}

describe('preset widgets', () => {
    describe('presetValues', () => {
        it('sets nothing for custom or an unknown preset', () => {
            expect(presetValues(TABLE, 'custom')).toEqual({});
            expect(presetValues(TABLE, 'hyperdrive')).toEqual({});
            expect(presetValues(TABLE, undefined)).toEqual({});
        });

        it('returns the bundle, minus excluded inputs', () => {
            expect(presetValues(TABLE, 'explore')).toEqual(TABLE.explore);
            expect(presetValues(TABLE, 'explore', new Set(['shader_strength']))).toEqual({
                shader_type: 'domain_warp', travel_mode: 'walk',
            });
        });
    });

    describe('applyPreset', () => {
        it('writes exactly the values the preset names', () => {
            const node = directNode();
            expect(applyPreset(node, TABLE, 'explore').sort()).toEqual(['shader_strength', 'shader_type', 'travel_mode']);
            expect(value(node, 'shader_strength')).toBe(0.3);
            expect(value(node, 'travel_mode')).toBe('walk');
            expect(value(node, 'seed')).toBe(8888);
        });

        it('leaves the parameter a Walk node is ramping alone', () => {
            const node = directNode();
            node.widgets.push({ name: 'walk_parameter', value: 'shader_strength' });
            applyPreset(node, TABLE, 'explore');
            expect(value(node, 'shader_strength')).toBe(0.9);
            expect(value(node, 'shader_type')).toBe('domain_warp');
        });

        it('runs each written widget\'s own callback', () => {
            const node = directNode();
            const callback = vi.fn();
            node.widgets[1].callback = callback;
            applyPreset(node, TABLE, 'jump');
            expect(callback).toHaveBeenCalledWith('domain_warp');
        });
    });

    describe('releasePreset', () => {
        it('drops back to custom when a controlled widget is edited to another value', () => {
            const node = directNode('explore');
            expect(releasePreset(node, LOADED, 'shader_strength', 0.5)).toBe(true);
            expect(value(node, 'preset')).toBe('custom');
        });

        it('keeps the preset when the edit matches what the preset holds', () => {
            const node = directNode('explore');
            expect(releasePreset(node, LOADED, 'shader_strength', 0.3)).toBe(false);
            expect(value(node, 'preset')).toBe('explore');
        });

        it('ignores widgets no preset controls, and the walked parameter', () => {
            const node = directNode('explore');
            expect(releasePreset(node, LOADED, 'seed', 1)).toBe(false);
            node.widgets.push({ name: 'walk_parameter', value: 'shader_strength' });
            expect(releasePreset(node, LOADED, 'shader_strength', 0.5)).toBe(false);
            expect(value(node, 'preset')).toBe('explore');
        });
    });

    describe('wirePresetWidgets', () => {
        it('applies a chosen preset without the writes releasing it again', () => {
            const node = directNode();
            wirePresetWidgets(node, () => LOADED);
            node.widgets[4].value = 'explore';
            node.widgets[4].callback('explore');
            expect(value(node, 'shader_strength')).toBe(0.3);
            expect(value(node, 'preset')).toBe('explore');
            expect(node.setDirtyCanvas).toHaveBeenCalled();
        });

        it('releases the preset on a hand edit and still runs the original callback', () => {
            const node = directNode('explore');
            const original = vi.fn();
            node.widgets[2].callback = original;
            wirePresetWidgets(node, () => LOADED);
            node.widgets[2].callback(0.55);
            expect(value(node, 'preset')).toBe('custom');
            expect(original).toHaveBeenCalledWith(0.55);
        });

        it('does nothing until the table has loaded', () => {
            const node = directNode();
            wirePresetWidgets(node, () => null);
            node.widgets[4].callback('explore');
            expect(value(node, 'shader_strength')).toBe(0.9);
        });
    });

    describe('loadPresets', () => {
        it('takes the table from the server', async () => {
            vi.mocked(api.fetchApi).mockResolvedValueOnce(
                new Response(JSON.stringify({ presets: TABLE, keys: LOADED.keys }), { status: 200 })
            );
            await loadPresets();
            expect(api.fetchApi).toHaveBeenCalledWith('/shader_noise_ksampler/presets');
            expect(currentPresets()).toEqual(LOADED);
        });
    });
});
