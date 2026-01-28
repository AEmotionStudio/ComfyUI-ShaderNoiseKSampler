import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
    test: {
        globals: true,
        environment: 'jsdom',
        include: ['web/tests/**/*.test.ts'],
        coverage: {
            provider: 'v8',
            reporter: ['text', 'json', 'html'],
            include: ['web/src/**/*.ts'],
            exclude: ['web/types/**', 'web/tests/**'],
        },
        setupFiles: ['web/tests/setup.ts'],
    },
    resolve: {
        alias: {
            // Mock ComfyUI imports during testing - use absolute paths
            '../../scripts/app.js': path.resolve(__dirname, 'web/tests/mocks/comfyui.ts'),
            '../../../scripts/app.js': path.resolve(__dirname, 'web/tests/mocks/comfyui.ts'),
            '../../scripts/api.js': path.resolve(__dirname, 'web/tests/mocks/comfyui.ts'),
            '../../../scripts/api.js': path.resolve(__dirname, 'web/tests/mocks/comfyui.ts'),
        },
    },
});
