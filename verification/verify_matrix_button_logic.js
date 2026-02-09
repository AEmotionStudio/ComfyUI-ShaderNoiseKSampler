
// Mock environment
const window = {
    showComfyToast: (msg) => console.log('Toast:', msg),
    setTimeout: setTimeout,
    clearTimeout: clearTimeout,
};

const navigator = {
    clipboard: {
        writeText: (text) => Promise.resolve(),
    },
};

class HTMLElement {
    constructor() {
        this.classList = {
            add: (cls) => this.classes.add(cls),
            remove: (cls) => this.classes.delete(cls),
            contains: (cls) => this.classes.has(cls),
        };
        this.classes = new Set();
        this.dataset = {};
        this.attributes = {};
        this.style = {};
        this.textContent = '';
    }

    getAttribute(name) {
        return this.attributes[name];
    }

    setAttribute(name, value) {
        this.attributes[name] = value;
    }

    focus() {
        console.log('Element focused');
        this.isFocused = true;
    }
}

const app = {
    canvas: {
        canvas: new HTMLElement(),
    },
};

// --- Proposed Implementation ---

// Helper to reset copy button state after timeout
const resetButtonTimeout = (button) => {
    if (button.dataset.timeoutId) {
        clearTimeout(parseInt(button.dataset.timeoutId));
    }
    const timeoutId = setTimeout(() => {
        button.textContent = "Copy";
        button.classList.remove('copied');

        // Restore aria-label
        const originalLabel = button.dataset.originalAriaLabel;
        if (originalLabel) {
             button.setAttribute('aria-label', originalLabel);
             delete button.dataset.originalAriaLabel;
        } else {
             button.setAttribute('aria-label', 'Copy code'); // Fallback
        }

        delete button.dataset.timeoutId;
    }, 100); // Reduced timeout for test speed
    button.dataset.timeoutId = String(timeoutId);
};

window.copyCodeSection = function (buttonElement) {
    // ... setup ...
    const codeText = 'some code';

    navigator.clipboard.writeText(codeText).then(() => {
        if (window.showComfyToast) {
            window.showComfyToast("Code copied to clipboard!", "success");
        }

        // Store original label if not already stored
        if (!buttonElement.dataset.originalAriaLabel) {
            buttonElement.dataset.originalAriaLabel = buttonElement.getAttribute('aria-label') || 'Copy code';
        }

        buttonElement.textContent = "Copied!";
        buttonElement.classList.add('copied');
        buttonElement.setAttribute('aria-label', 'Copied'); // Update for screen readers

        resetButtonTimeout(buttonElement);
    });
};

const closeModalCleanup = () => {
    // ... existing cleanup logic ...

    // UX Enhancement: Restore focus to the main canvas
    const appCanvas = app.canvas?.canvas; // Access the DOM <canvas> element
    if (appCanvas && appCanvas.focus) {
        appCanvas.focus();
    }
};


// --- Tests ---

async function runTests() {
    console.log('Running tests...');

    // Test 1: Copy Button Logic
    const button = new HTMLElement();
    button.setAttribute('aria-label', 'Copy snippet');
    button.textContent = 'Copy';

    await window.copyCodeSection(button);

    // Verify immediate state
    if (button.getAttribute('aria-label') !== 'Copied') {
        throw new Error('Test 1 Failed: aria-label should be "Copied" immediately after click');
    }
    if (button.textContent !== 'Copied!') {
        throw new Error('Test 1 Failed: textContent should be "Copied!" immediately after click');
    }
    if (button.dataset.originalAriaLabel !== 'Copy snippet') {
         throw new Error('Test 1 Failed: original aria-label not stored correctly');
    }

    console.log('Test 1 Passed: Immediate update correct');

    // Verify timeout restore
    await new Promise(resolve => setTimeout(resolve, 150)); // Wait for timeout

    if (button.getAttribute('aria-label') !== 'Copy snippet') {
        throw new Error(`Test 1 Failed: aria-label should be restored to "Copy snippet", got "${button.getAttribute('aria-label')}"`);
    }
    if (button.textContent !== 'Copy') {
        throw new Error('Test 1 Failed: textContent should be restored to "Copy"');
    }
    if (button.dataset.originalAriaLabel) {
         throw new Error('Test 1 Failed: original aria-label dataset should be cleared');
    }

    console.log('Test 1 Passed: Restore correct');


    // Test 2: Focus Restoration
    app.canvas.canvas.isFocused = false;
    closeModalCleanup();

    if (!app.canvas.canvas.isFocused) {
        throw new Error('Test 2 Failed: Canvas should be focused after closeModalCleanup');
    }

    console.log('Test 2 Passed: Canvas focused');
}

runTests().catch(e => {
    console.error(e);
    process.exit(1);
});
