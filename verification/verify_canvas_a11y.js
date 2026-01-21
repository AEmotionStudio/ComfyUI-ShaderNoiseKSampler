const fs = require('fs');
const path = require('path');

// Mock DOM environment
const dom = {
    _elements: [],
    createElement: (tag) => {
        const el = {
            tagName: tag.toUpperCase(),
            attributes: {},
            children: [],
            style: {},
            classList: {
                add: () => {},
                remove: () => {},
                contains: () => false
            },
            setAttribute: function(k, v) { this.attributes[k] = v; },
            getAttribute: function(k) { return this.attributes[k]; },
            appendChild: function(c) { this.children.push(c); c.parentElement = this; },
            querySelector: function(s) { return null; }, // Default empty
            querySelectorAll: function(s) { return []; },
            getContext: function() { // Define getContext here
                 // Circular ref for context canvas is needed in the visualizer code: ctx.canvas.getContext('2d')
                 const ctx = {
                    fillStyle: '',
                    fillRect: () => {},
                    beginPath: () => {},
                    moveTo: () => {},
                    lineTo: () => {},
                    stroke: () => {},
                    fill: () => {},
                    arc: () => {},
                    strokeRect: () => {},
                    fillText: () => {},
                    measureText: () => ({ width: 0 }),
                    save: () => {},
                    restore: () => {},
                    translate: () => {},
                    rotate: () => {},
                    createRadialGradient: () => ({ addColorStop: () => {} }),
                    createLinearGradient: () => ({ addColorStop: () => {} }),
                    canvas: el, // Reference back to element
                    drawImage: () => {}
                };
                return ctx;
            }
        };
        return el;
    },
    querySelectorAll: () => [],
    body: { appendChild: () => {} },
    addEventListener: () => {},
    readyState: 'loading'
};

// Global mocks
global.document = dom;
global.window = {};
global.Image = class {
    constructor() {
        this.onload = null;
        this.onerror = null;
        setTimeout(() => {
            if (this.onload) this.onload();
        }, 10);
    }
};
global.createImageBitmap = async () => ({});
global.fetch = async () => ({
    ok: true,
    text: async () => "<svg></svg>"
});
global.URL = {
    createObjectURL: () => "blob:test",
    revokeObjectURL: () => {}
};
global.Blob = class {};
global.getComputedStyle = () => ({ width: '100px', height: '100px' });
global.requestAnimationFrame = (cb) => setTimeout(cb, 16);

// Load the file content
const visualizerPath = path.join(__dirname, '../web/noise_visualizer.js');
const visualizerCode = fs.readFileSync(visualizerPath, 'utf8');

// Evaluate the code
eval(visualizerCode);

async function runTest() {
    console.log("Running Accessibility Verification...");

    // Setup mock modal content
    const modalContent = dom.createElement('div');
    modalContent.id = 'mock-modal';

    // Helper to create mock noise container
    function createNoiseContainer(id) {
        const div = dom.createElement('div');
        div.className = 'noise-canvas'; // Class looked for by visualizer
        div.id = id;
        // Mock querySelector to return null initially (no canvas yet)
        div.querySelector = (sel) => {
            if (sel === 'canvas') return div.children.find(c => c.tagName === 'CANVAS');
            return null;
        };
        return div;
    }

    const testContainers = [
        createNoiseContainer('noise-canvas-perlin'),
        createNoiseContainer('noise-canvas-tensor_field')
    ];

    testContainers.forEach(c => modalContent.appendChild(c));

    // Mock querySelectorAll on modalContent
    modalContent.querySelectorAll = (sel) => {
        if (sel === '.noise-canvas') return testContainers;
        if (sel === '.mask-canvas') return [];
        if (sel === '.color-swatch') return [];
        if (sel === '#animation-demo-placeholder') return null;
        if (sel === '#intro-noise-demo') return null;
        return [];
    };
    modalContent.querySelector = (sel) => null;

    // Run the visualizer
    await window.NoiseVisualizer.renderAllInModal(modalContent);

    // Verify attributes
    let allPassed = true;
    testContainers.forEach(container => {
        const canvas = container.children.find(c => c.tagName === 'CANVAS');
        if (!canvas) {
            console.error(`FAILED: Canvas not created for ${container.id}`);
            allPassed = false;
            return;
        }

        const role = canvas.getAttribute('role');
        const ariaLabel = canvas.getAttribute('aria-label');

        console.log(`Checking ${container.id}:`);
        console.log(`  Role: ${role}`);
        console.log(`  Aria-Label: ${ariaLabel}`);

        if (role !== 'img') {
            console.error(`  FAILED: Missing role="img"`);
            allPassed = false;
        }
        if (!ariaLabel) {
            console.error(`  FAILED: Missing aria-label`);
            allPassed = false;
        } else {
            console.log(`  SUCCESS: Has aria-label "${ariaLabel}"`);
        }
    });

    if (allPassed) {
        console.log("VERIFICATION PASSED: All canvases have accessibility attributes.");
        process.exit(0);
    } else {
        console.log("VERIFICATION FAILED: Missing accessibility attributes.");
        process.exit(1);
    }
}

runTest().catch(e => {
    console.error(e);
    process.exit(1);
});
