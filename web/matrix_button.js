import { app } from "../../scripts/app.js";

/**
 * Adds a "Show Matrix" button to the ShaderDisplay and ShaderNoiseKSampler nodes
 */
(function() {
    // Define utility functions on window object to be accessible by treatiseHTML
    window.scrollToSection = function(sectionId) {
        const modalContent = document.querySelector('.shader-matrix-treatise'); // Scroll within the modal
        if (!modalContent) return;
        const section = modalContent.querySelector('#' + sectionId);
        if (section) {
            section.scrollIntoView({ behavior: 'smooth' });
        }
    };

    window.showTab = function(tabIdToActivate, clickedTabElement) {
        let tabsContainer;
        let contentScope;
        let tabSelector;
        let tabContentSelector;

        if (clickedTabElement) { // Prioritize if element is passed
             tabsContainer = clickedTabElement.closest('.tabs') || clickedTabElement.closest('.sacred-tabs');
        } else { // Fallback to querying globally within the modal if no element
            const modalDiv = document.querySelector('.shader-matrix-treatise');
            if (!modalDiv) return;
            tabsContainer = modalDiv.querySelector('.tabs') || modalDiv.querySelector('.sacred-tabs');
        }

        if (!tabsContainer) {
            console.error("showTab: Could not find '.tabs' or '.sacred-tabs' container.");
            return;
        }

        if (tabsContainer.classList.contains('tabs')) {
            tabSelector = '.tab';
            tabContentSelector = '.tab-content';
            contentScope = tabsContainer.parentNode; // Assumes content is sibling to .tabs div
        } else if (tabsContainer.classList.contains('sacred-tabs')) {
            tabSelector = '.sacred-tab';
            tabContentSelector = '.sacred-tab-content';
            contentScope = tabsContainer.closest('.sacred-section') || tabsContainer.parentNode;
        } else {
            return; // Unknown tab structure
        }

        tabsContainer.querySelectorAll(tabSelector).forEach(tab => tab.classList.remove('active'));
        if (clickedTabElement) {
            clickedTabElement.classList.add('active');
        } else {
            // If no clicked element, try to find the tab by tabId (less robust)
            const targetTab = Array.from(tabsContainer.querySelectorAll(tabSelector)).find(
                t => t.getAttribute('onclick') && t.getAttribute('onclick').includes(tabIdToActivate)
            );
            if (targetTab) targetTab.classList.add('active');
        }

        if (contentScope) {
            contentScope.querySelectorAll(tabContentSelector).forEach(content => {
                content.style.display = 'none';
                content.classList.remove('active');
            });
            const activeContent = contentScope.querySelector('#' + tabIdToActivate);
            if (activeContent) {
                activeContent.style.display = 'block';
                activeContent.classList.add('active');
            }
        }
    };

    window.setupScrollTop = function(modalContentElement) { // Added modalContentElement parameter
        if (!modalContentElement) return; // Check the passed element
        const scrollTopButton = modalContentElement.querySelector('#scroll-top'); // Use passed element

        if (scrollTopButton) {
            modalContentElement.addEventListener('scroll', () => { // Use passed element
                if (modalContentElement.scrollTop > 200) { // Use passed element
                    scrollTopButton.classList.add('visible');
                } else {
                    scrollTopButton.classList.remove('visible');
                }
            });
            scrollTopButton.addEventListener('click', (e) => {
                e.stopPropagation();
                modalContentElement.scrollTo({ top: 0, behavior: 'smooth' });
            });
        }
    };

    window.toggleCodeSection = function(buttonElement) {
        const headerElement = buttonElement.closest('.code-block-header');
        if (!headerElement) return;
        const codeBlockContainer = headerElement.parentNode;
        if (!codeBlockContainer) return;

        const preElement = codeBlockContainer.querySelector('pre.foldable-content');
        if (!preElement) return;

        const isHidden = preElement.style.display === 'none' || preElement.style.display === '';

        if (isHidden) {
            preElement.style.display = 'block';
            buttonElement.textContent = 'Hide';
        } else {
            preElement.style.display = 'none';
            buttonElement.textContent = 'Show';
        }
    };

    // Register the extension for ShaderDisplay and ShaderNoiseKSampler nodes
    app.registerExtension({
        name: "ComfyUI.ShaderNoise.MatrixButton",
        
        beforeRegisterNodeDef(nodeType, nodeData) {
            // Modify ShaderDisplay, ShaderNoiseKSampler, and ShaderNoiseKSamplerDirect nodes
            if (nodeData.name !== "ShaderDisplay" && nodeData.name !== "ShaderNoiseKSampler" && nodeData.name !== "ShaderNoiseKSamplerDirect") {
                return;
            }
            
            // Store the original methods
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            
            // Add our button to the node
            nodeType.prototype.onNodeCreated = function() {
                // Call the original onNodeCreated method first
                const self = this; // Node instance

                if (originalOnNodeCreated) {
                    originalOnNodeCreated.apply(self, arguments);
                }
                
                // Add the "Show Matrix" button widget
                // const self = this; // self is already defined above
                
                // Function to add the matrix button
                const addMatrixButton = () => {
                    // MODIFIED: HTML content from Treatise.js
                    // window.showTab_SacredCodex = function(tabId, clickedTabElement) { ... }; 
                    // This ^ is now replaced by the global window.showTab for new HTML,
                    // but can be kept if old structures outside this modal might use it.
                    // For this modal, the new HTML will use window.showTab.

                    const button = self.addWidget("button", "📊 Show Shader Matrix", null, function() {
                        // Create modal container
                        const modal = document.createElement("div");
                        modal.style.cssText = `
                            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                            background: linear-gradient(135deg, rgba(0,0,0,0.9), rgba(26,13,52,0.95));
                            display: flex; justify-content: center; align-items: center; z-index: 10000;
                            backdrop-filter: blur(5px);
                        `;
                        
                        let handleEscPress; 

                        const closeModalCleanup = () => {
                            if (modal && modal.parentNode) {
                                document.body.removeChild(modal);
                            }
                            if (handleEscPress) {
                                document.removeEventListener('keydown', handleEscPress);
                            }
                        };

                        handleEscPress = (e) => {
                            if (e.key === "Escape") {
                                closeModalCleanup();
                            }
                        };

                        document.addEventListener('keydown', handleEscPress);
                        
                        modal.onclick = (e) => {
                            if (e.target === modal) {
                                closeModalCleanup();
                            }
                        };
                        
                        const content = document.createElement("div");
                        content.className = "shader-matrix-treatise"; // This is the main scrollable container
                        content.style.cssText = `
                            position: relative; /* For positioning of child elements like scroll-to-top */
                            background: linear-gradient(145deg, #0f0f12, #1a1a2e); /* Match new body bg */
                            border-radius: 16px; padding: 0; /* Remove padding, let content manage it */
                            max-width: 95vw; width: 1400px; /* Wider for new content */
                            max-height: 90vh; overflow-y: auto; color: #e0e0e8; /* Match new text color */
                            box-shadow: 0 0 60px rgba(138, 43, 226, 0.5);
                            border: 1px solid rgba(138, 43, 226, 0.4);
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Base font */
                            line-height: 1.6;
                        `;
                        
                        // MODIFIED: HTML content from Treatise.js
                        const treatiseHTML = `
                            <style>
        :root {
            --golden-ratio: 1.618;
            --background-color: #0f0f12; /* Used by modal content style */
            --text-color: #e0e0e8;  /* Used by modal content style */
            --accent-color: #8a2be2;
            --secondary-color: #3498db;
            --tertiary-color: #f39c12;
            --quaternary-color: #2ecc71;
            --quaternary-color-rgb: 46, 204, 113; /* For rgba() usage */
            --section-padding: calc(1rem * var(--golden-ratio));
            --header-font: 'Georgia', serif;
            --body-font: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            --code-font: 'Courier New', monospace;
        }

        /* Reset some global styles that might be inherited if body styles are applied directly */
        .shader-matrix-treatise header, .shader-matrix-treatise main, .shader-matrix-treatise footer, .shader-matrix-treatise section {
            display: block; /* Ensure block display for semantic elements */
        }
        .shader-matrix-treatise * { /* Scoped reset */
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: all 0.3s ease;
        }
         .shader-matrix-treatise { /* Styles for the main container, replacing body */
            /* background-color: var(--background-color); Already set on content div */
            /* color: var(--text-color); Already set on content div */
            /* font-family: var(--body-font); Already set on content div */
            /* line-height: 1.6; Already set on content div */
            overflow-x: hidden; /* Keep from treatise body */
         }


        .shader-matrix-treatise #canvas-container { /* Scoped and adjusted for modal */
            position: absolute; /* Relative to .shader-matrix-treatise */
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0; /* Behind content but inside modal */
            opacity: 0.05; /* More subtle */
            pointer-events: none;
        }

        .shader-matrix-treatise header {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: calc(var(--section-padding) * 1.5) var(--section-padding); /* Reduced padding */
            text-align: center;
            position: relative; /* For z-indexing if needed */
            background: linear-gradient(135deg, rgba(138, 43, 226, 0.15), rgba(52, 152, 219, 0.08));
            z-index: 1;
        }

        .shader-matrix-treatise h1 {
            font-family: var(--header-font);
            font-size: calc(1.8rem * var(--golden-ratio)); /* Adjusted size */
            margin-bottom: 0.8rem;
            background: linear-gradient(45deg, var(--accent-color), var(--secondary-color));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-shadow: 0 0 10px rgba(138, 43, 226, 0.3);
        }

        .shader-matrix-treatise h2 {
            font-family: var(--header-font);
            font-size: calc(1.3rem * var(--golden-ratio)); /* Adjusted size */
            margin: calc(var(--section-padding) * 0.7) 0 calc(var(--section-padding) * 0.4);
            color: var(--secondary-color);
            border-bottom: 1px solid rgba(52, 152, 219, 0.3);
            padding-bottom: 0.4rem;
            text-align: center; /* Center the h2 headers */
        }

        .shader-matrix-treatise h3 {
            font-family: var(--header-font);
            font-size: calc(0.9rem * var(--golden-ratio)); /* Adjusted size */
            margin: calc(var(--section-padding) * 0.5) 0 calc(var(--section-padding) * 0.25);
            color: var(--tertiary-color);
        }

        .shader-matrix-treatise .tagline {
            font-style: italic;
            margin-bottom: 1.5rem;
            opacity: 0.8;
            max-width: 700px; /* Adjusted */
            font-size: 0.95rem;
        }

        .shader-matrix-treatise main {
            max-width: calc(750px * var(--golden-ratio)); /* Adjusted */
            margin: 0 auto;
            padding: var(--section-padding);
            /* background-color: rgba(15, 15, 18, 0.7); No inner background, modal has it */
            /* backdrop-filter: blur(10px); */
            border-radius: 8px;
            /* box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5); */
            position: relative; /* For z-indexing if needed */
            z-index: 1;
        }

        .shader-matrix-treatise section {
            margin-bottom: calc(var(--section-padding) * 1.3); /* Adjusted */
        }

        .shader-matrix-treatise p {
            margin-bottom: 0.9rem;
            font-size: 0.9rem;
        }
        
        .shader-matrix-treatise li {
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }


        .shader-matrix-treatise .interactive-demo {
            width: 100%;
            height: 350px; /* Adjusted */
            margin: calc(var(--section-padding) * 0.4) 0;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
            background: #0a0a0a;
            border: 1px solid rgba(var(--accent-color), 0.2);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }

        .shader-matrix-treatise .parameter-section {
            background-color: rgba(30, 30, 40, 0.5); /* More transparent */
            padding: calc(var(--section-padding) * 0.6);
            border-radius: 8px;
            margin: calc(var(--section-padding) * 0.4) 0;
        }

        .shader-matrix-treatise .parameter-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); /* Adjusted */
            gap: 0.9rem;
        }

        .shader-matrix-treatise .parameter-item {
            background-color: rgba(45, 45, 60, 0.5); /* More transparent */
            padding: 0.9rem;
            border-radius: 8px;
            border-left: 3px solid var(--accent-color);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        .shader-matrix-treatise .parameter-item p {
            font-size: 0.85rem;
        }

        .shader-matrix-treatise .parameter-name {
            font-weight: bold;
            color: var(--tertiary-color);
            margin-bottom: 0.4rem;
            font-size: 0.95rem;
        }

        .shader-matrix-treatise code, .shader-matrix-treatise pre {
            font-family: var(--code-font);
            background-color: rgba(20, 22, 34, 0.8); /* Darker, less blue */
            border-radius: 4px;
            font-size: 0.85rem; /* Smaller code font */
        }

        .shader-matrix-treatise code { /* Inline code */
            padding: 2px 5px;
            color: var(--quaternary-color);
        }

        .shader-matrix-treatise pre { /* Block code */
            padding: 0.8rem 1rem; /* Adjusted padding */
            overflow-x: auto;
            margin: 0.8rem 0;
            border: 1px solid rgba(var(--accent-color),0.2);
            border-left: 3px solid var(--accent-color);
            color: #d0d0d0; /* Lighter text for pre */
        }

        .shader-matrix-treatise .architecture-diagram {
            width: 100%;
            min-height: 450px; /* Adjusted */
            margin: calc(var(--section-padding) * 0.4) 0;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
            background: #0a0a0a;
            border: 1px solid rgba(var(--accent-color), 0.2);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }

        .shader-matrix-treatise .noise-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); /* Adjusted */
            gap: 0.9rem;
            margin: calc(var(--section-padding) * 0.4) 0;
        }

        .shader-matrix-treatise .noise-item {
            background-color: rgba(30, 30, 40, 0.5);
            padding: 0.9rem;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            transform: translateY(0);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
            border: 1px solid transparent;
        }
         .shader-matrix-treatise .noise-item:hover {
            transform: translateY(-4px); /* Adjusted */
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4); /* Adjusted */
            border-color: rgba(var(--accent-color), 0.5);
        }

        .shader-matrix-treatise .noise-canvas {
            width: 130px; /* Adjusted */
            height: 130px; /* Adjusted */
            margin-bottom: 0.4rem;
            border-radius: 4px;
            background: #111;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }

        .shader-matrix-treatise .noise-name {
            font-weight: bold;
            color: var(--secondary-color);
            font-size: 0.9rem;
        }
         .shader-matrix-treatise .noise-description {
            font-size: 0.8rem; /* Adjusted */
            margin-top: 0.4rem;
            color: rgba(224, 224, 232, 0.7); /* Adjusted */
            /* height: 5em; Removed */
            /* overflow: hidden; Removed */
        }

        .shader-matrix-treatise .sacred-symbol {
            display: inline-block;
            margin: 0 0.4rem;
            opacity: 0.7;
            font-size: 1.2em;
        }

        .shader-matrix-treatise .implementation-steps {
            list-style: none;
            counter-reset: step-counter;
            padding: 0;
            margin: calc(var(--section-padding) * 0.4) 0;
        }

        .shader-matrix-treatise .implementation-steps li {
            position: relative;
            padding-left: 2.2rem; /* Adjusted */
            margin-bottom: 0.8rem;
            counter-increment: step-counter;
            font-size: 0.9rem;
        }

        .shader-matrix-treatise .implementation-steps li::before {
            content: counter(step-counter);
            position: absolute;
            left: 0;
            top: 0;
            width: 1.6rem; /* Adjusted */
            height: 1.6rem; /* Adjusted */
            background-color: var(--accent-color);
            color: var(--text-color);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.85rem;
        }

        .shader-matrix-treatise .card {
            background-color: rgba(30, 30, 40, 0.5);
            padding: calc(var(--section-padding) * 0.6);
            border-radius: 8px;
            margin: calc(var(--section-padding) * 0.4) 0;
            border-left: 3px solid var(--secondary-color);
            box-shadow: 0 3px 12px rgba(0, 0, 0, 0.3);
        }
         .shader-matrix-treatise .card h3 { margin-top: 0; }


        .shader-matrix-treatise .warning {
            background-color: rgba(243, 156, 18, 0.1);
            padding: calc(var(--section-padding) * 0.6);
            border-radius: 8px;
            margin: calc(var(--section-padding) * 0.4) 0;
            border-left: 3px solid var(--tertiary-color);
            box-shadow: 0 3px 12px rgba(0, 0, 0, 0.2);
        }
         .shader-matrix-treatise .warning p { font-size: 0.85rem; }

        .shader-matrix-treatise .warning-title {
            color: var(--tertiary-color);
            font-weight: bold;
            display: flex;
            align-items: center;
            margin-bottom: 0.4rem;
            font-size: 1rem;
        }

        .shader-matrix-treatise .warning-title::before {
            content: "⚠️";
            margin-right: 0.4rem;
        }

        .shader-matrix-treatise footer {
            text-align: center;
            padding: calc(var(--section-padding) * 1.3) var(--section-padding);
            opacity: 0.7;
            font-size: 0.8rem; /* Adjusted */
            background: linear-gradient(135deg, rgba(138, 43, 226, 0.08), rgba(52, 152, 219, 0.03));
            margin-top: var(--section-padding);
            position: relative; /* For z-indexing if needed */
            z-index: 1;
        }

        .shader-matrix-treatise .golden-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--secondary-color), transparent);
            margin: calc(var(--section-padding) * 0.7) 0; /* Adjusted */
            opacity: 0.5;
        }
        
        .shader-matrix-treatise .math-section {
            background-color: rgba(30, 40, 60, 0.5);
            padding: calc(var(--section-padding) * 0.6);
            border-radius: 8px;
            margin: calc(var(--section-padding) * 0.4) 0;
            border-left: 3px solid var(--quaternary-color);
        }
        
        .shader-matrix-treatise .math-title {
            color: var(--quaternary-color);
            font-weight: bold;
            display: flex;
            align-items: center;
            margin-bottom: 0.4rem;
            font-size: 1rem;
        }
        
        .shader-matrix-treatise .math-title::before {
            content: "📊";
            margin-right: 0.4rem;
        }
        
        .shader-matrix-treatise .math-formula {
            font-family: var(--code-font); /* Or specific math font */
            padding: 0.8em;
            background-color: rgba(20, 22, 34, 0.8);
            border-radius: 4px;
            text-align: center;
            margin: 0.8em 0;
            overflow-x: auto;
            font-size: 1rem; /* Larger for formulas */
            color: var(--tertiary-color);
        }
         .shader-matrix-treatise .tabs {
            display: flex;
            flex-wrap: wrap;
            margin-top: 0.8em;
            margin-bottom: -1px; /* To make active tab border connect */
            position: relative;
            z-index: 1;
        }
         .shader-matrix-treatise .tab {
            padding: 0.6em 1.2em; /* Adjusted */
            background-color: rgba(30, 30, 40, 0.6);
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            cursor: pointer;
            margin-right: 0.3em;
            color: var(--text-color);
            opacity: 0.7;
            border: 1px solid transparent;
            border-bottom: 1px solid rgba(var(--accent-color),0.3);
            font-size: 0.85rem;
        }
         .shader-matrix-treatise .tab.active {
            background-color: rgba(45, 45, 60, 0.8); /* Darker for modal */
            font-weight: bold;
            opacity: 1;
            border-color: rgba(var(--accent-color),0.3);
            border-bottom-color: transparent; /* Or match tab content bg */
            transform: translateY(1px);
        }
         .shader-matrix-treatise .tab:hover {
            opacity: 1;
            background-color: rgba(40,40,55,0.7);
         }
        
        .shader-matrix-treatise .tab-content {
            display: none; /* Handled by JS */
            padding: 1.2em; /* Adjusted */
            background-color: rgba(45, 45, 60, 0.8); /* Match active tab */
            border-radius: 0 6px 6px 6px;
            border: 1px solid rgba(var(--accent-color),0.3);
            border-top-color: transparent; /* If tabs overlap */
            clear: both; /* If tabs were floated */
            margin-top: 0;
        }
         .shader-matrix-treatise .tab-content.active {
            display: block; /* Handled by JS */
        }
         .shader-matrix-treatise .color-example {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8em; /* Adjusted */
            justify-content: center;
            margin: 1.5em 0; /* Adjusted */
        }
         .shader-matrix-treatise .color-swatch {
            width: 100px; /* Adjusted */
            height: 70px; /* Adjusted */
            border-radius: 6px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-end; /* Text at bottom */
            padding-bottom: 5px;
            color: white;
            font-size: 0.75rem; /* Adjusted */
            text-shadow: 0 1px 2px rgba(0,0,0,0.8);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4); /* Adjusted */
            transition: transform 0.2s;
            border: 1px solid rgba(255,255,255,0.1);
        }
         .shader-matrix-treatise .color-swatch:hover {
            transform: scale(1.08); /* Adjusted */
        }
         .shader-matrix-treatise .animation-container {
            height: 220px; /* Adjusted */
            background-color: rgba(30, 30, 40, 0.5);
            border-radius: 8px;
            overflow: hidden;
            position: relative;
            margin: 1.5em 0;
            border: 1px solid rgba(var(--accent-color), 0.2);
        }
         .shader-matrix-treatise .navigation { /* For in-page nav buttons */
            /* position: sticky; */ /* Sticky might not work well in a scrolled modal, let it scroll */
            /* top: 0; */ /* Remove sticky positioning */
            background-color: rgba(15, 15, 18, 0.8); /* Slightly transparent */
            padding: 0.6rem; /* Adjusted */
            border-radius: 0 0 8px 8px;
            margin-bottom: 1.5rem;
            z-index: 10; /* Above canvas container, below modal controls if any */
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem; /* Adjusted */
            justify-content: center;
            border-bottom: 1px solid rgba(var(--accent-color), 0.2);
        }
         .shader-matrix-treatise .nav-button {
            padding: 0.4rem 0.8rem; /* Adjusted */
            background-color: rgba(45, 45, 60, 0.7);
            border-radius: 4px;
            cursor: pointer;
            color: var(--text-color);
            border: 1px solid transparent;
            transition: all 0.2s;
            font-size: 0.8rem;
        }
         .shader-matrix-treatise .nav-button:hover {
            background-color: var(--accent-color);
            color: white;
            border-color: rgba(255,255,255,0.3);
            transform: scale(1.1);
        }
         .shader-matrix-treatise #scroll-top { /* Scroll-to-top button within the modal */
            position: sticky;
            bottom: 20px;
            left: 100%; /* Position left edge at the far right */
            transform: translateX(calc(-2.5rem - 0px)); /* Pull back by its width + 0px offset */
            background-color: var(--accent-color);
            color: white;
            width: 2.5rem; /* Adjusted */
            height: 2.5rem; /* Adjusted */
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem; /* Adjusted */
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.3);
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.3s, transform 0.3s;
            z-index: 20; /* High z-index within modal */
            border: none;
        }
         .shader-matrix-treatise #scroll-top.visible {
            opacity: 0.7;
        }
         .shader-matrix-treatise #scroll-top:hover {
            opacity: 1;
            transform: translateX(calc(-2.5rem - 0px)) scale(1.1); /* Combined transforms with 0px offset */
        }
        
        /* Responsive layouts */
        @media (max-width: 768px) {
            .shader-matrix-treatise h1 {
                font-size: calc(1.3rem * var(--golden-ratio));
            }
            
            .shader-matrix-treatise h2 {
                font-size: calc(1.0rem * var(--golden-ratio));
            }
            
            .shader-matrix-treatise .parameter-grid {
                grid-template-columns: 1fr;
            }
            
            .shader-matrix-treatise .noise-grid {
                grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            }
            
            .shader-matrix-treatise .interactive-demo, .shader-matrix-treatise .architecture-diagram {
                height: 250px;
            }
            .shader-matrix-treatise .navigation {
                justify-content: flex-start; /* Align buttons to left on small screens */
            }
        }

        /* Animation keyframes */
        @keyframes pulse {
            0% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(1.05); opacity: 1; }
            100% { transform: scale(1); opacity: 0.8; }
        }
         .shader-matrix-treatise .pulse {
            animation: pulse 2s infinite ease-in-out;
        }
        
        /* Code highlighting styles (Highlight.js default or theme) */
        /* Base styles from highlight.js default.css, can be overridden by a theme */
        .shader-matrix-treatise .hljs {
            display: block;
            overflow-x: auto;
            padding: 0.5em;
            color: #abb2bf; /* Default text color for Monokai-like */
            background: #282c34; /* Default background for Monokai-like */
        }
        .shader-matrix-treatise .hljs-comment,
        .shader-matrix-treatise .hljs-quote {
            color: #5c6370;
            font-style: italic;
        }
        .shader-matrix-treatise .hljs-doctag,
        .shader-matrix-treatise .hljs-keyword,
        .shader-matrix-treatise .hljs-formula {
            color: #c678dd; /* Magenta */
        }
        .shader-matrix-treatise .hljs-section,
        .shader-matrix-treatise .hljs-name,
        .shader-matrix-treatise .hljs-selector-tag,
        .shader-matrix-treatise .hljs-deletion,
        .shader-matrix-treatise .hljs-subst {
            color: #e06c75; /* Red */
        }
        .shader-matrix-treatise .hljs-literal {
            color: #56b6c2; /* Cyan */
        }
        .shader-matrix-treatise .hljs-string,
        .shader-matrix-treatise .hljs-regexp,
        .shader-matrix-treatise .hljs-addition,
        .shader-matrix-treatise .hljs-attribute,
        .shader-matrix-treatise .hljs-meta-string {
            color: #98c379; /* Green */
        }
        .shader-matrix-treatise .hljs-built_in,
        .shader-matrix-treatise .hljs-class .hljs-title {
            color: #e6c07b; /* Yellow */
        }
        .shader-matrix-treatise .hljs-attr,
        .shader-matrix-treatise .hljs-variable,
        .shader-matrix-treatise .hljs-template-variable,
        .shader-matrix-treatise .hljs-type,
        .shader-matrix-treatise .hljs-selector-class,
        .shader-matrix-treatise .hljs-selector-attr,
        .shader-matrix-treatise .hljs-selector-pseudo,
        .shader-matrix-treatise .hljs-number {
            color: #d19a66; /* Orange */
        }
        .shader-matrix-treatise .hljs-symbol,
        .shader-matrix-treatise .hljs-bullet,
        .shader-matrix-treatise .hljs-link,
        .shader-matrix-treatise .hljs-meta,
        .shader-matrix-treatise .hljs-selector-id,
        .shader-matrix-treatise .hljs-title {
            color: #61afef; /* Blue */
        }
        .shader-matrix-treatise .hljs-emphasis {
            font-style: italic;
        }
        .shader-matrix-treatise .hljs-strong {
            font-weight: bold;
        }
        .shader-matrix-treatise .hljs-link {
            text-decoration: underline;
        }
        /* Custom HLJS for the provided python blocks if needed */
        .shader-matrix-treatise pre code.hljs.language-python .hljs-keyword { color: #ff79c6; }
        .shader-matrix-treatise pre code.hljs.language-python .hljs-string { color: #f1fa8c; }
        .shader-matrix-treatise pre code.hljs.language-python .hljs-comment { color: #6272a4; }
        .shader-matrix-treatise pre code.hljs.language-python .hljs-number { color: #bd93f9; }
        .shader-matrix-treatise pre code.hljs.language-python .hljs-function > .hljs-title { color: #50fa7b; }
        /* End of HLJS custom theme */

        /* Ensure button inside treatise has pointer */
        .shader-matrix-treatise button {
            cursor: pointer;
        }

        /* CSS for Foldable Code Blocks */
        .shader-matrix-treatise .code-block-container {
            background-color: transparent; /* Let child pre handle its background */
            border-radius: 6px;
            margin: 0.8rem 0; /* Keep original pre margin */
            border: 1px solid rgba(var(--accent-color),0.2);
            border-left: 3px solid var(--accent-color); /* Keep original pre left border */
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .shader-matrix-treatise .code-block-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0.8rem;
            background-color: rgba(30, 33, 45, 0.9); /* Header background */
            border-top-left-radius: 4px; /* Align with container rounding */
            border-top-right-radius: 4px;
            cursor: default;
        }

        .shader-matrix-treatise .code-block-header span {
            font-family: var(--code-font);
            color: var(--tertiary-color);
            font-size: 0.9rem;
            font-weight: bold;
        }

        .shader-matrix-treatise .toggle-code-button {
            background-color: var(--accent-color);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.3rem 0.7rem;
            font-size: 0.75rem;
            cursor: pointer;
            transition: background-color 0.2s;
        }

        .shader-matrix-treatise .toggle-code-button:hover {
            background-color: var(--secondary-color);
        }

        .shader-matrix-treatise .code-block-container > pre.foldable-content {
            /* background-color, color, font-family, font-size, padding, overflow-x are inherited or set by original pre styles */
            margin: 0 !important; /* Remove margin from pre itself */
            border: none !important; /* Container handles borders */
            border-radius: 0 0 4px 4px !important; /* Bottom corners rounded */
            /* Original pre styles like background-color: rgba(20, 22, 34, 0.8); will apply */
        }

        .parameter-item {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }

        .compatibility-chart {
            grid-column: 1 / -1;
            margin-top: 20px;
        }

        .compatibility-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 2fr;
            gap: 10px;
            margin: 15px 0;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
            padding: 10px;
        }

        .compatibility-header {
            font-weight: bold;
            color: #4a9eff;
            padding: 8px;
            border-bottom: 1px solid rgba(74, 158, 255, 0.3);
        }

        .compatibility-item {
            padding: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .compatibility-level {
            padding: 8px;
            border-radius: 4px;
            text-align: center;
            font-weight: 500;
        }

        .compatibility-level.high {
            background: rgba(46, 160, 67, 0.2);
            color: #2ea043;
        }

        .compatibility-level.medium {
            background: rgba(218, 165, 32, 0.2);
            color: #daa520;
        }

        .compatibility-note {
            padding: 8px;
            font-size: 0.9em;
            color: rgba(255, 255, 255, 0.8);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .scheduler-compatibility {
            margin-top: 20px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
        }

        .scheduler-compatibility h4 {
            color: #4a9eff;
            margin-bottom: 10px;
        }

        .scheduler-compatibility ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .scheduler-compatibility li {
            margin-bottom: 8px;
            padding-left: 20px;
            position: relative;
        }

        .scheduler-compatibility li:before {
            content: "•";
            color: #4a9eff;
            position: absolute;
            left: 0;
        }
                            </style>
                            
                            <!-- This button is handled by the closeModalCleanup logic if class matches -->
                            <button class="close-button" title="Close" style="position: absolute; top: 15px; right: 15px; width: 30px; height: 30px; border-radius: 50%; background: rgba(var(--accent-color), 0.7); color: white; border: none; font-size: 1.2rem; z-index: 100; display: flex; align-items:center; justify-content:center; line-height:1;">×</button>
                            
                            <div id="canvas-container"></div>
    
                            <header>
                                <h1>The Shader Matrix</h1>
                                <p class="tagline">Harnessing Noise with Shader Algorithms for Image Generation</p>
                            </header>
                            
                            <div class="navigation">
                                <button class="nav-button" onclick="window.scrollToSection('introduction')">Introduction</button>
                                <button class="nav-button" onclick="window.scrollToSection('genesis')">Genesis</button>
                                <button class="nav-button" onclick="window.scrollToSection('core-concept')">Core Concept</button>
                                <button class="nav-button" onclick="window.scrollToSection('sacred-patterns')">Shader Patterns</button>
                                <button class="nav-button" onclick="window.scrollToSection('noise-math')">Noise Mathematics</button>
                                <button class="nav-button" onclick="window.scrollToSection('blend-modes')">Blend Modes</button>
                                <button class="nav-button" onclick="window.scrollToSection('noise-transforms')">Noise Transformations</button>
                                <button class="nav-button" onclick="window.scrollToSection('animation')">Temporal Coherence</button>
                                <button class="nav-button" onclick="window.scrollToSection('shape-mask-alchemy')">Shape Masks</button>
                                <button class="nav-button" onclick="window.scrollToSection('color-schemes')">Color Schemes</button>
                                <button class="nav-button" onclick="window.scrollToSection('usage')">Usage</button>
                                <button class="nav-button" onclick="window.scrollToSection('navigating-latent-space')">Navigating Latent Space</button>
                                <button class="nav-button" onclick="window.scrollToSection('conclusion')">Conclusion</button>
                            </div>
                            
                            <main>
                                <section id="introduction">
                                    <h2>🔮 Introduction <span class="sacred-symbol">✨</span> The Essence of Noise</h2>
                                    <p>ShaderNoiseKSampler is a ComfyUI node that uses shader algorithms to generate and control procedural noise. This noise, refeered to as shader noise, can then be used to influence the AI image creation process in latent diffusion models. The aim is to provide a way to introduce structured mathematical patterns into the diffusion process, offering a different approach to guiding image generation.</p>
                                    <p>This document outlines the features, technical details, and underlying ideas of ShaderNoiseKSampler, showing how different shader noise types can create various visual characteristics that influence and drive the diffusion process.</p>
                                    
                                    <div class="interactive-demo" id="intro-noise-demo">
                                        <!-- Three.js intro noise visualization will be rendered here -->
                                    </div>
                                    
                                    <div class="card">
                                        <h3>💠 The Path of Noise</h3>
                                        <p>Noise is fundamentally a mathematical expression of ordered chaos—a seemingly random pattern that nonetheless follows precise mathematical rules. In the context of image generation, these noise patterns serve as the seed from which creation emerges. ShaderNoiseKSampler allows artists to shape this primordial mathematical chaos according to sacred geometric principles, guiding the diffusion model\'s sampling process along specific aesthetic trajectories.</p>
                                        <p>Just as ancient civilizations recognized patterns in nature and encoded them in their art and architecture, ShaderNoiseKSampler enables digital artists to encode mathematical archetypes into the generative process, creating a bridge between abstract mathematics and visual expression.</p>
                                    </div>
                                </section>
                        
                                <div class="golden-divider"></div>
                                
                                <section id="genesis">
                                    <h2>🐜 Project Genesis <span class="sacred-symbol">∞</span>The Birth of Order from Chaos</h2>
                                    
                                    <div class="card">
                                        <h3>👁️ The Vision Unveiled</h3>
                                        <p>The genesis of ShaderNoiseKSampler lies in a contemplation of nature\'s fundamental paradox: the inherent order within apparent chaos. Observing the natural world reveals that seemingly random phenomena—the flicker of fire, the branching of lightning, the flow of water, the structure of terrestrial forms—all exhibit underlying fractal patterns. This ubiquitous principle, that profound order underpins apparent disorder, became a guiding inspiration.</p>
                                        <p>The initial conceptual spark was <em>hyperspace</em>—an unseen realm between dimensions. While seemingly chaotic in its construct, closer examination reveals a perfect, almost unimaginable mathematical precision and scope. It is this notion that the unseen can be rendered observable with the appropriate tools which led to a pivotal question: <strong>Could a method be devised to introduce deliberate order into random noise? Could one inject structure into the latent space itself, thereby crystallizing defined pathways for creative generation?</strong></p>
                                    </div>
                                    
                                    <div class="parameter-section">
                                        <h3>🏗️ The Ant Colony Revelation</h3>
                                        <p>A compelling metaphor emerged from an unexpected source: the art of ant sculpture. This process, which captures the hidden architecture of ant tunnels by filling them with molten metal, offered a tangible analogy. Once cooled and excavated, the metal cast reveals a previously unseen, complex structure—a physical manifestation of the colony\'s hidden pathways. The question then became: how could one create a virtual equivalent of this molten metal? The answer was clear: Shaders.</p>
                                        
                                        <div class="implementation-steps">
                                            <li><strong>Virtual Molten Metal (Shaders) = Shader Noise</strong><br/>Shader patterns, like molten metal, flow into and define the pathways within latent space.</li>
                                            <li><strong>Solidification = Sampling Process</strong><br/>The K-sampling process allows this structured noise to effectively \'solidify\' within the latent dimensions.</li>
                                            <li><strong>Excavation = Denoising</strong><br/>As the model denoises, it progressively reveals the crystallized patterns embedded by the shader.</li>
                                            <li><strong>Revealed Sculpture = Final Image</strong><br/>The output is a manifestation of a previously unseen space, a navigable pathway forged through the realm of possibility.</li>
                                        </div>
                                    </div>
                                         <div class="parameter-section">
                                        <h3>⚗️ The Alchemical Process: From Theory to Practice</h3>
                                        <p>The journey from concept to a functional implementation involved extensive experimentation, iterative debugging, and countless visualization tests. The challenge was multifaceted, encompassing not only technical hurdles but also philosophical considerations: How could the abstract vision of controlled chaos be translated into robust code? Critically, how could novel structures be introduced without disrupting the model\'s foundational training?</p>
                                        <p>A crucial breakthrough was the understanding that models cannot be expected to produce coherent results when fed noise types entirely alien to their training data. This led to the principle of <strong>augmentation rather than replacement</strong>. The final noise input to the KSampler is therefore not a wholesale substitution but a careful modification of the standard base noise. The shader noise augments and sculpts this base noise, and the influence of these shader-defined patterns began to yield compelling and controllable results in the generated outputs.</p>
                                        <p>This hybrid methodology—a delicate equilibrium between the familiar (the model\'s learned representations) and the novel (the intentionally introduced structure)—fosters both generative stability and exploratory capacity.</p>
                                        <div class="warning">
                                            <div class="warning-title">The Blend: A Cornerstone Principle</div>
                                            <p>Respecting a model\'s training by blending shader-generated noise with the base noise—as opposed to outright replacement—is a cornerstone of ShaderNoiseKSampler. This approach preserves generative coherence while enabling nuanced, controllable structural influence.</p>
                                        </div>
                                        <div style="text-align: center; margin: 2rem 0; font-style: italic; color: var(--tertiary-color);">
                                            "Not to replace the chaos, but to give it direction<br/>
                                            Not to eliminate randomness, but to make it purposeful<br/>
                                            To transform noise into navigation, and navigation into art"<br>
                                            <span style="opacity: 0.7;">— The Genesis Vision</span>
                                        </div>
                                    </div>
                                    
                                    <div class="card">                                        
                                        <h3>🗺️ Exploring Latent Space: A Developing Idea</h3>
                                        <p>The exploration extended to the nature of latent space. It became apparent that by adjusting shader noise parameters with a consistent seed, one could systematically alter the generative output. This suggested a way to modify not just noise characteristics, but potentially the "topology" of the latent space being explored.</p>
                                        <p>This led to an interesting mode of exploration. It was observed that different seeds seemed to contain unique thematic elements—recurring characters, color palettes, objects, and styles—that could be traced and explored. ShaderNoiseKSampler evolved from a noise tool into a means of <strong>navigating within a model's latent space with more specific control</strong>. This helped clarify the tool's potential.</p>
                                        <p>Each parameter adjustment can be seen as a step along conceptual pathways, each blend mode a different way to perceive and traverse the possibilities.</p>
                                    
                                    </div>
                                    
                                </section>

                                <div class="golden-divider"></div>
                                
                                <section id="core-concept">
                                    <h2>🌌 Core Concept <span class="sacred-symbol">△</span> The Alchemical Transformation</h2>
                                    <p>ShaderNoiseKSampler implements the formula <code>Lt=Sα(N)∘Kβ(t)</code>—a mathematical invocation where structured noise Sα transforms the base noise N before it enters the diffusion sampling process Kβ. This elegant formula captures the essence of guided chaos, where randomness is structured according to sacred patterns before manifesting in the final image.</p>
                                    
                                    <div class="math-section">
                                        <div class="math-title">The Mathematics of Transformation</div>
                                        <p>The core equation <code>Lt=Sα(N)∘Kβ(t)</code> can be understood as follows:</p>
                                        <div class="math-formula">
                                            L<sub>t</sub> = S<sub>α</sub>(N) ∘ K<sub>β</sub>(t)
                                        </div>
                                        <p>Where:</p>
                                        <ul>
                                            <li><strong>L<sub>t</sub></strong>: The final latent representation at timestep t</li>
                                            <li><strong>S<sub>α</sub>(N)</strong>: The shader transformation with parameters α applied to base noise N</li>
                                            <li><strong>K<sub>β</sub>(t)</strong>: The diffusion sampling process with parameters β at timestep t</li>
                                            <li><strong>∘</strong>: Function composition operator (application of one function to the result of another)</li>
                                        </ul>
                                        <p>This mathematical formulation allows for more precise control over the noise structures that guide the generative process, creating a harmony between randomness and order.</p>
                                    </div>

                                    <div class="card" style="margin-top: calc(var(--section-padding) * 0.8); background: rgba(var(--quaternary-color-rgb), 0.05); border-left-color: var(--quaternary-color);">
                                        <h3 style="color: var(--quaternary-color); margin-top:0;"><span class="sacred-symbol">⚙️</span> The Role of ShaderToTensor: Bridging Code and Concept</h3>
                                        <p>The <code>ShaderToTensor</code> class, implemented in <code>shader_to_tensor.py</code>, is the cornerstone of this system\'s ability to translate abstract shader concepts into concrete PyTorch tensors. It acts as the alchemical crucible where mathematical descriptions of noise are transmuted into the actual structured noise (<code>Sα</code> in our formula) that the KSampler can utilize.</p>
                                        <p>Key functions of <code>ShaderToTensor</code> include:</p>
                                        <ul style="list-style-type: disc; padding-left: 20px; margin-top: 0.5rem;">
                                            <li><strong>Noise Synthesis:</strong> It contains the Python and PyTorch implementations of various noise algorithms (Perlin, Cellular, Curl, etc.). When you select a \'Shader Noise Type\' in the UI, you\'re choosing a specific method within this class.</li>
                                            <li><strong>Parameter Interpretation:</strong> It takes the parameters you set (Scale, Octaves, Warp Strength, etc.) and uses them to control the generation of these noise tensors.</li>
                                            <li><strong>Tensor Formatting:</strong> It ensures the generated noise is in the correct format (shape, data type, device) to be compatible with the ComfyUI latent workflow and the KSampler. This includes handling channel expansion (e.g., from 1-channel noise to 4 or 9 channels for the latent space).</li>
                                            <li><strong>Temporal Coherence Logic:</strong> For animated noise, <code>ShaderToTensor</code> (often in conjunction with specialized generator classes) implements the logic to produce evolving noise patterns over a time dimension.</li>
                                        </ul>
                                        <p>Essentially, <code>ShaderToTensor</code> is the engine that powers the S<sub>α</sub>(N) part of the equation. Without it, the conceptual shader patterns would remain abstract; this class makes them tangible and usable by the diffusion model.</p>
                                    </div>
                                    
                                    <div class="parameter-section">
                                        <h3>📊 Essential Parameters</h3>
                                        <div class="parameter-grid">
                                            <div class="parameter-item">
                                                <div class="parameter-name">Shader Noise Type</div>
                                                <p>The fundamental pattern archetype (tensor_field, curl_noise, domain_warp, etc.)</p>
                                            </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Noise Scale</div>
                                                <p>Controls the frequency of pattern repetition (1.0 is the harmonic baseline)</p>
                                            </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Noise Octaves</div>
                                                <p>The number of recursive self-similar layers (follows Fibonacci principles)</p>
                                            </div>
                                              <div class="parameter-item">
                                                <div class="parameter-name">Noise Warp Strength</div>
                                                <p>Controls the amount of displacement or distortion applied to the noise coordinates, altering the pattern's structure.</p>
                                            </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Noise Phase Shift</div>
                                                <p>Adjusts the input to the noise function, effectively shifting the generated pattern. This can be used to explore variations of the noise.</p>
                                            </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Shape Mask Type</div>
                                                <p>The geometric form overlaid upon the noise (radial, spiral, hexgrid, etc.)</p>
                                            </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Shape Mask Strength</div>
                                                <p>Controls the intensity of the shape mask, determining how much of the noise is affected by the mask.</p>
                                            </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Noise Transformation</div>
                                                <p>Mathematical operations (e.g., absolute, sin, sqrt) applied to the shader noise before blending. <strong>[Recommended: Experiment freely. See 'Noise Transformations' section for details]</strong>.</p>
                                            </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Blend Mode</div>
                                                <p>The alchemical method of combining shader noise with base noise (multiply, add, etc.). <strong>[Recommended: Multiply, Normal, Screen, Overlay]</strong>.</p>
                                            </div>
                                               <div class="parameter-item">
                                                <div class="parameter-name">Color Intensity</div>
                                                <p>Adjusts the impact of the selected color scheme on the noise pattern.</p>
                                            </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Sequential Stages</div>
                                                <p>Number of sequential shader stages to apply before injection stages. Each stage can have varied strength and apply noise over a specific portion of the diffusion steps. <strong>[Recommended: 1 - 3]</strong>.</p>
                                            </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Injection Stages</div>
                                                <p>Number of injection shader stages to apply after sequential stages. These stages typically inject noise at specific steps within the diffusion process, allowing for targeted interventions. <strong>[Recommended: 0 - 2]</strong>.</p>
                                            </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Shader Strength</div>
                                                <p>Overall strength of the shader noise influence for all stage types. Set to 0.0 to disable shader noise and use only the base noise. This acts as a global multiplier for stage-specific strengths. <strong>[Recommended: 0.1 - 0.4]</strong>.</p>
                                            </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">CFG</div>
                                                <p>Classifier-Free Guidance scale. <strong>[Recommended: 6 - 9]</strong>.</p>
                                            </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Steps</div>
                                                <p>Number of sampling iterations. <strong>[Recommended: 20 - 60]</strong>.</p>
                                            </div>
                                            <div class="parameter-item compatibility-chart">
                                                <div class="parameter-name">Sampler & Scheduler Compatibility</div>
                                                <div class="compatibility-grid">
                                                    <div class="compatibility-header">Sampler</div>
                                                    <div class="compatibility-header">Compatibility</div>
                                                    <div class="compatibility-header">Notes</div>
                                                    
                                                    <div class="compatibility-item">Euler_Ancestral</div>
                                                    <div class="compatibility-level high">High</div>
                                                    <div class="compatibility-note">Excellent across all model types</div>
                                                    
                                                    <div class="compatibility-item">dpm_2_ancestral</div>
                                                    <div class="compatibility-level high">High</div>
                                                    <div class="compatibility-note">Strong performance with all models</div>
                                                    
                                                    <div class="compatibility-item">dpmppm_2_ancestral</div>
                                                    <div class="compatibility-level high">High</div>
                                                    <div class="compatibility-note">Reliable for all model variants</div>
                                                    
                                                    <div class="compatibility-item">LCM</div>
                                                    <div class="compatibility-level high">High</div>
                                                    <div class="compatibility-note">Consistent results across models</div>
                                                    
                                                    <div class="compatibility-header scheduler">Scheduler</div>
                                                    <div class="compatibility-header">Compatibility</div>
                                                    <div class="compatibility-header">Notes</div>
                                                    
                                                    <div class="compatibility-item">Beta</div>
                                                    <div class="compatibility-level high">High</div>
                                                    <div class="compatibility-note">Recommended scheduler</div>
                                                    
                                                    <div class="compatibility-item">Normal</div>
                                                    <div class="compatibility-level high">High</div>
                                                    <div class="compatibility-note">Good general performance</div>
                                                    
                                                    <div class="compatibility-item">Simple</div>
                                                    <div class="compatibility-level high">High</div>
                                                    <div class="compatibility-note">Effective for most cases</div>
                                                    
                                                    <div class="compatibility-item">kl_optimal</div>
                                                    <div class="compatibility-level high">High</div>
                                                    <div class="compatibility-note">Optimized performance</div>

                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </section>
                        
                                <div class="golden-divider"></div>
                                
                                <section id="sacred-patterns">
                                    <h2>🏵️ Shader Noise Patterns <span class="sacred-symbol">✧</span> The Twelve Noise Archetypes</h2>
                                    <p>ShaderNoiseKSampler harnesses twelve fundamental shader noise archetypes, each channeling a different aspect of mathematical reality. These patterns are the building blocks of visual coherence, from the flowing whorls of Perlin noise to the cosmic web-like structures of Tensor Fields. Each archetype generates its own distinct style of noise, offering a diverse palette for visual expression.</p>
                                    
                                    <div class="noise-grid" id="noise-type-grid">
                                        <div class="noise-item">
                                            <div class="noise-canvas" id="noise-canvas-tensor_field"></div>
                                            <div class="noise-name">Tensor Field</div>
                                            <div class="noise-description">Manifests flow patterns based on tensor mathematics, revealing the underlying force vectors of the mathematical space.</div>
                                        </div>
                                        <div class="noise-item">
                                            <div class="noise-canvas" id="noise-canvas-cellular"></div>
                                            <div class="noise-name">Cellular</div>
                                            <div class="noise-description">Creates organic, cell-like structures through Voronoi diagrams, mimicking natural growth patterns.</div>
                                            <div style="margin-top: 5px; text-align: center;"><a href="https://ko-fi.com/aemotionstudio" target="_blank" rel="noopener noreferrer"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" style="height: 22px; border:0px;"/></a></div>
                                        </div>
                                        <div class="noise-item">
                                            <div class="noise-canvas" id="noise-canvas-domain_warp"></div>
                                            <div class="noise-name">Domain Warp</div>
                                            <div class="noise-description">Applies non-linear distortions to space itself, creating fluid-like transformations of the underlying pattern.</div>
                                        </div>
                                        <div class="noise-item">
                                            <div class="noise-canvas" id="noise-canvas-fractal"></div>
                                            <div class="noise-name">Fractal</div>
                                            <div class="noise-description">Embodies self-similarity across scales, reflecting the infinite recursive patterns found throughout nature.</div>
                                            <div style="margin-top: 5px; text-align: center;"><a href="https://ko-fi.com/aemotionstudio" target="_blank" rel="noopener noreferrer"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" style="height: 22px; border:0px;"/></a></div>
                                        </div>
                                        <div class="noise-item">
                                            <div class="noise-canvas" id="noise-canvas-perlin"></div>
                                            <div class="noise-name">Perlin</div>
                                            <div class="noise-description">The classic gradient noise that creates smooth, natural-looking transitions and flowing textures.</div>
                                            <div style="margin-top: 5px; text-align: center;"><a href="https://ko-fi.com/aemotionstudio" target="_blank" rel="noopener noreferrer"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" style="height: 22px; border:0px;"/></a></div>
                                        </div>
                                        <div class="noise-item">
                                            <div class="noise-canvas" id="noise-canvas-waves"></div>
                                            <div class="noise-name">Waves</div>
                                            <div class="noise-description">Generates harmonic oscillations that combine to form complex interference patterns.</div>
                                            <div style="margin-top: 5px; text-align: center;"><a href="https://ko-fi.com/aemotionstudio" target="_blank" rel="noopener noreferrer"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" style="height: 22px; border:0px;"/></a></div>
                                        </div>
                                        <div class="noise-item">
                                            <div class="noise-canvas" id="noise-canvas-gaussian"></div>
                                            <div class="noise-name">Gaussian</div>
                                            <div class="noise-description">Pure probabilistic noise based on the normal distribution, the foundation of natural randomness.</div>
                                            <div style="margin-top: 5px; text-align: center;"><a href="https://ko-fi.com/aemotionstudio" target="_blank" rel="noopener noreferrer"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" style="height: 22px; border:0px;"/></a></div>
                                        </div>
                                        <div class="noise-item">
                                            <div class="noise-canvas" id="noise-canvas-heterogeneous_fbm"></div>
                                            <div class="noise-name">Heterogeneous FBM</div>
                                            <div class="noise-description">Varies the fractal dimension across space, creating regions of different turbulence and detail.</div>
                                            <div style="margin-top: 5px; text-align: center;"><a href="https://ko-fi.com/aemotionstudio" target="_blank" rel="noopener noreferrer"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" style="height: 22px; border:0px;"/></a></div>
                                        </div>
                                        <div class="noise-item">
                                            <div class="noise-canvas" id="noise-canvas-interference"></div>
                                            <div class="noise-name">Interference</div>
                                            <div class="noise-description">Simulates wave interaction patterns, creating complex nodal structures through phase relationships.</div>
                                            <div style="margin-top: 5px; text-align: center;"><a href="https://ko-fi.com/aemotionstudio" target="_blank" rel="noopener noreferrer"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" style="height: 22px; border:0px;"/></a></div>
                                        </div>
                                        <div class="noise-item">
                                            <div class="noise-canvas" id="noise-canvas-spectral"></div>
                                            <div class="noise-name">Spectral</div>
                                            <div class="noise-description">Controls the frequency spectrum directly, allowing precise frequency band manipulation.</div>
                                            <div style="margin-top: 5px; text-align: center;"><a href="https://ko-fi.com/aemotionstudio" target="_blank" rel="noopener noreferrer"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" style="height: 22px; border:0px;"/></a></div>
                                        </div>
                                        <div class="noise-item">
                                            <div class="noise-canvas" id="noise-canvas-projection_3d"></div>
                                            <div class="noise-name">3D Projection</div>
                                            <div class="noise-description">Projects three-dimensional noise onto a 2D plane, creating depth and volumetric effects.</div>
                                            <div style="margin-top: 5px; text-align: center;"><a href="https://ko-fi.com/aemotionstudio" target="_blank" rel="noopener noreferrer"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" style="height: 22px; border:0px;"/></a></div>
                                        </div>
                                        <div class="noise-item">
                                            <div class="noise-canvas" id="noise-canvas-curl_noise"></div>
                                            <div class="noise-name">Curl Noise</div>
                                            <div class="noise-description">Generates divergence-free vector fields that create perfect fluid-like flow patterns.</div>
                                        </div>
                                    </div>
                                </section>

                                <div class="card" style="text-align: center; margin-top: calc(var(--section-padding) * 1.2); padding: calc(var(--section-padding) * 0.9); background: linear-gradient(145deg, rgba(var(--accent-color-rgb, 138, 43, 226), 0.1), rgba(var(--secondary-color-rgb, 52, 152, 219), 0.05)); border: 1px solid var(--accent-color); border-radius: 12px; box-shadow: 0 4px 20px rgba(var(--accent-color-rgb, 138, 43, 226), 0.2);">
                                        <h3 style="color: var(--accent-color); text-shadow: 0 0 8px rgba(var(--accent-color-rgb, 138, 43, 226), 0.5); margin-bottom: 0.7rem; font-size: 1.4rem;">🌟 Unlock Exclusive Shader Noise Palettes!</h3>
                                        <p style="font-size: 0.95rem; margin-bottom: 0rem; opacity: 0.9; line-height: 1.5;">Become a valued member to access an expanded library of unique shader noise archetypes, advanced shader features, and custom ComfyUI tools. Your support helps fuel further development!</p>
                                    </div>

                                </section>

                                <div class="golden-divider"></div>
                                
                                <section id="noise-math">
                                    <h2>⚛️ Noise Mathematics <span class="sacred-symbol">📐</span> The Underlying Formulations</h2>
                                    <p>Each shader noise archetype in ShaderNoiseKSampler is built upon rigorous mathematical foundations. Understanding these mathematical principles reveals the sacred geometry inherent in these patterns and allows for more intentional application in the creative process.</p>
                                    
                                    <div class="tabs">
                                        <div class="tab active" onclick="window.showTab('tab-perlin', this)">Perlin</div>
                                        <div class="tab" onclick="window.showTab('tab-cellular', this)">Cellular</div>
                                        <div class="tab" onclick="window.showTab('tab-tensor', this)">Tensor Field</div>
                                        <div class="tab" onclick="window.showTab('tab-curl', this)">Curl Noise</div>
                                        <div class="tab" onclick="window.showTab('tab-domain-warp', this)">Domain Warp</div>
                                        <div class="tab" onclick="window.showTab('tab-fractal', this)">Fractal</div>
                                        <div class="tab" onclick="window.showTab('tab-waves', this)">Waves</div>
                                        <div class="tab" onclick="window.showTab('tab-gaussian', this)">Gaussian</div>
                                        <div class="tab" onclick="window.showTab('tab-heterogeneous-fbm', this)">Heterogeneous FBM</div>
                                        <div class="tab" onclick="window.showTab('tab-interference', this)">Interference</div>
                                        <div class="tab" onclick="window.showTab('tab-spectral', this)">Spectral</div>
                                        <div class="tab" onclick="window.showTab('tab-projection-3d', this)">3D Projection</div>
                                    </div>
                                    
                                    <div id="tab-perlin" class="tab-content active">
                                        <h3>Perlin Noise Mathematics</h3>
                                        <p>Perlin noise, developed by Ken Perlin in 1983, uses a grid of random gradient vectors with interpolation to create smooth, natural-looking noise. The core computation involves:</p>
                                        <div class="math-formula">
                                            n(x,y) = ∑ ω<sub>i</sub> · g<sub>i</sub> · ((x,y) - (x<sub>i</sub>,y<sub>i</sub>))
                                        </div>
                                        <p>Where g<sub>i</sub> are random gradient vectors at grid points, and ω<sub>i</sub> are interpolation weights. The smoothstep function provides the basis for the interpolation:</p>
                                        <div class="math-formula">
                                            smoothstep(t) = t<sup>2</sup>(3 - 2t)
                                        </div>
                                        <p>This smooth interpolation creates the characteristic flowing appearance of Perlin noise, making it ideal for natural phenomena like terrain, clouds, and flowing water.</p>
                                        <div class="code-block-container">
                                            <div class="code-block-header">
                                                <span>Perlin Noise Implementation</span>
                                                <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                            </div>
                                            <pre class="foldable-content" style="display: none;"><code class="language-python">def perlin_noise_with_params(cls, batch_size, height, width, shader_params, time, device, seed):
    # Extract parameters for noise generation
    scale = shader_params.get("shaderScale", 1.0)
    octaves = shader_params.get("shaderOctaves", 3)
    warp_strength = shader_params.get("shaderWarpStrength", 0.5)
    phase_shift = shader_params.get("shaderPhaseShift", 0.5)
    
    # Create coordinate grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )
    
    # Combine into coordinate tensor [batch, height, width, 2]
    p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # Apply Perlin noise with multiple octaves
    result = torch.zeros(batch_size, height, width, 1, device=device)
    amplitude = 0.5
    frequency = 1.0
    
    for i in range(int(octaves)):
        # Apply domain warping if warp_strength > 0
        if i > 0 and warp_strength > 0.0:
            warp = compute_warp_field(p, frequency, time, seed + i)
            noise = sample_perlin(p * frequency + warp * warp_strength, seed + i)
        else:
            noise = sample_perlin(p * frequency, seed + i)
        
        # Add to result with decreasing amplitude
        result += amplitude * noise
        
        # Prepare for next octave
        frequency *= 2.0
        amplitude *= 0.5
    
    # Apply phase shift (contrast)
    result *= (1.0 + phase_shift)
    
    return result</code></pre>
                                        </div>
                                    </div>
                                    
                                    <div id="tab-cellular" class="tab-content">
                                        <h3>Cellular Noise Mathematics</h3>
                                        <p>Cellular noise (also known as Worley noise) creates Voronoi-like patterns based on distance metrics to feature points:</p>
                                        <div class="math-formula">
                                            F<sub>n</sub>(x) = n<sup>th</sup> min{dist(x, x<sub>i</sub>) | i ∈ feature points}
                                        </div>
                                        <p>Where dist() is a distance function (typically Euclidean) and F<sub>n</sub> returns the distance to the n<sup>th</sup> closest feature point. Various cellular patterns emerge by combining these distances:</p>
                                        <div class="math-formula">
                                            F<sub>2</sub> - F<sub>1</sub> : cell edges<br/>
                                            F<sub>1</sub> : cell interiors<br/>
                                            2F<sub>1</sub> - F<sub>2</sub> : cracks
                                        </div>
                                        <p>These combinations create patterns reminiscent of cellular structures, bubbles, and organic tissues.</p>
                                        <div class="code-block-container">
                                            <div class="code-block-header">
                                                <span>Cellular Noise Implementation</span>
                                                <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                            </div>
                                            <pre class="foldable-content" style="display: none;"><code class="language-python">def cellular_noise_with_params(cls, batch_size, height, width, shader_params, time, device, seed):
    # Extract parameters
    scale = shader_params.get("shaderScale", 1.0)
    octaves = shader_params.get("shaderOctaves", 1)
    # warp_strength = shader_params.get("shaderWarpStrength", 0.5) # Not used in this snippet
    
    # Create coordinate grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )
    p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # Apply cellular noise
    result = torch.zeros(batch_size, height, width, 1, device=device)
    
    # Generate cellular pattern based on octaves mode
    pattern_type = int(octaves) % 4  # Use octaves to select pattern type
    
    # Calculate F1 and F2 distances (assuming compute_feature_distances exists)
    f1, f2 = compute_feature_distances(p * scale, seed) 
    
    # Select pattern based on type
    if pattern_type == 0:
        result = f1  # Basic Worley/cellular noise
    elif pattern_type == 1:
        result = f2  # Second closest point
    elif pattern_type == 2:
        result = f2 - f1  # Classic cellular look with ridges
    else:
        result = f1 * f2  # Different look with smaller features
    
    return result</code></pre>
                                        </div>
                                    </div>
                                    
                                    <div id="tab-tensor" class="tab-content">
                                        <h3>Tensor Field Mathematics</h3>
                                        <p>Tensor fields represent directional information at every point in space, creating flow-like patterns. The core mathematics involves computing the eigenvalues λ and eigenvectors v of tensor matrices:</p>
                                        <div class="math-formula">
                                            T(x,y) = 
                                            \\begin{pmatrix}
                                            T_{xx} & T_{xy} \\\\
                                            T_{xy} & T_{yy}
                                            \\end{pmatrix}
                                        </div>
                                        <p>The eigenvalues and eigenvectors are found by solving:</p>
                                        <div class="math-formula">
                                            T·v = λ·v
                                        </div>
                                        <p>Different visualizations emerge based on how we render these tensor properties:</p>
                                        <ul>
                                            <li><strong>Eigenvalue visualization</strong>: Shows magnitude of deformation</li>
                                            <li><strong>Streamlines</strong>: Shows direction of principal stress</li>
                                            <li><strong>Hyperstreamlines</strong>: Combines both with weighted influence</li>
                                            <li><strong>Ellipses</strong>: Represents the tensor as oriented ellipses</li>
                                        </ul>
                                        <div class="code-block-container">
                                            <div class="code-block-header">
                                                <span>Tensor Field Implementation</span>
                                                <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                            </div>
                                            <pre class="foldable-content" style="display: none;"><code class="language-python">def tensor_field_with_params(cls, batch_size, height, width, shader_params, time, device, seed):
    # Extract parameters
    scale = shader_params.get("shaderScale", 1.0)
    viz_type = int(shader_params.get("shaderOctaves", 3)) % 4 # Using octaves to pick viz_type
    
    # Create coordinate grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )
    p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # Compute tensor field components (assuming compute_tensor_field exists)
    tensor_xx, tensor_xy, tensor_yy = compute_tensor_field(p * scale, time, seed)
    
    # Calculate eigenvalues and eigenvectors
    trace = tensor_xx + tensor_yy
    det = tensor_xx * tensor_yy - tensor_xy * tensor_xy
    # Add epsilon to prevent sqrt of negative numbers due to precision
    sqrt_discriminant = torch.sqrt(torch.relu(trace * trace - 4 * det)) 
    lambda1 = (trace + sqrt_discriminant) / 2
    lambda2 = (trace - sqrt_discriminant) / 2
    
    result = torch.zeros_like(lambda1) # Ensure result is initialized

    # Select visualization based on type
    if viz_type == 0:  # Eigenvalue visualization
        result = torch.max(torch.abs(lambda1), torch.abs(lambda2))
    elif viz_type == 1:  # Streamlines (placeholder for actual streamline generation)
        # result = create_streamlines(p, lambda1, lambda2, tensor_xx, tensor_xy, tensor_yy)
        result = (torch.sin(p[..., 0] * lambda1) + torch.cos(p[..., 1] * lambda2)) / 2 # Example
    else:  # Ellipses by default (placeholder)
        # result = create_ellipses(p, lambda1, lambda2, tensor_xx, tensor_xy, tensor_yy)
        result = (lambda1 + lambda2) / 2 # Example magnitude
    
    return result.unsqueeze(-1) # Ensure channel dim</code></pre>
                                        </div>
                                    </div>
                                    
                                    <div id="tab-curl" class="tab-content">
                                        <h3>Curl Noise Mathematics</h3>
                                        <p>Curl noise generates divergence-free vector fields, perfect for fluid-like motions. It\'s based on the curl operator from vector calculus:</p>
                                        <div class="math-formula">
                                            ∇ × Ψ = (∂Ψ<sub>z</sub>/∂y - ∂Ψ<sub>y</sub>/∂z, ∂Ψ<sub>x</sub>/∂z - ∂Ψ<sub>z</sub>/∂x, ∂Ψ<sub>y</sub>/∂x - ∂Ψ<sub>x</sub>/∂y)
                                        </div>
                                        <p>In 2D, this simplifies to a scalar field where the curl is perpendicular to the plane:</p>
                                        <div class="math-formula">
                                            curl(Ψ)(x,y) = ∂Ψ<sub>y</sub>/∂x - ∂Ψ<sub>x</sub>/∂y
                                        </div>
                                        <p>This mathematical property ensures the resulting vector field has zero divergence (∇·v = 0), creating perfect flow patterns without sources or sinks.</p>
                                        <div class="code-block-container">
                                            <div class="code-block-header">
                                                <span>Curl Noise Implementation</span>
                                                <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                            </div>
                                            <pre class="foldable-content" style="display: none;"><code class="language-python">def curl_noise_with_params(cls, batch_size, height, width, shader_params, time, device, seed):
    # Extract parameters
    scale = shader_params.get("shaderScale", 1.0)
    # warp_strength = shader_params.get("shaderWarpStrength", 0.5) # Not used directly here for simple curl value
    
    # Create coordinate grid
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )
    # p = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1) # Not used

    # Compute potential fields (scalar fields to take curl of, assuming compute_noise_field)
    # For 2D curl, we need two scalar potential fields or components of a vector potential field
    # Let's assume Ψ = (ψx(x,y,t), ψy(x,y,t)) and we want curl_z = d(ψy)/dx - d(ψx)/dy
    
    # Simplified: generate two noise fields for potential components
    # These would typically be generated by a noise function like Perlin or Simplex
    potential_y_coords = torch.stack([x_coords * scale, y_coords * scale + time], dim=-1)
    potential_x_coords = torch.stack([x_coords * scale + time, y_coords * scale], dim=-1)

    # Placeholder for actual noise generation (e.g., simplex_noise(potential_coords, seed))
    potential_y = torch.sin(potential_y_coords[...,0] * 5) * torch.cos(potential_y_coords[...,1] * 5) 
    potential_x = torch.cos(potential_x_coords[...,0] * 5) * torch.sin(potential_x_coords[...,1] * 5)
    
    potential_y = potential_y.unsqueeze(0).unsqueeze(-1).repeat(batch_size,1,1,1)
    potential_x = potential_x.unsqueeze(0).unsqueeze(-1).repeat(batch_size,1,1,1)


    # Calculate curl using finite differences
    epsilon = 0.01 # step for derivative
    
    # Partial derivative of potential_y with respect to x
    # potential_y(x + eps) - potential_y(x - eps) / (2*eps)
    # We'd need to sample noise at shifted coords or use analytical derivatives if possible
    # Simplified finite difference on the grid:
    dpy_dx = (torch.roll(potential_y, shifts=-1, dims=2) - torch.roll(potential_y, shifts=1, dims=2)) / (2 * (2.0/width) ) # dx = 2.0/width for range -1 to 1

    # Partial derivative of potential_x with respect to y
    # potential_x(y + eps) - potential_x(y - eps) / (2*eps)
    dpx_dy = (torch.roll(potential_x, shifts=-1, dims=1) - torch.roll(potential_x, shifts=1, dims=1)) / (2 * (2.0/height) ) # dy = 2.0/height

    # Curl in 2D (z-component)
    curl_z = dpy_dx - dpx_dy
    
    return curl_z # This is a scalar field representing curl strength
</code></pre>
                                        </div>
                                    </div>

                                    <div id="tab-domain-warp" class="tab-content">
                                        <h3>Domain Warp Mathematics</h3>
                                        <p>Domain warping is a technique where the input coordinates (the domain) of a noise function are displaced or distorted by another noise function. This creates swirling, turbulent, or flowing effects in the final pattern. Instead of sampling noise at point P, we sample at P + offset(P), where offset(P) is itself a noise function.</p>
                                        <div class="math-formula">
                                            Noise<sub>final</sub>(P) = Noise<sub>base</sub>(P + S<sub>warp</sub> × Noise<sub>displacement</sub>(P × F<sub>warp</sub>))
                                        </div>
                                        <p>Where:</p>
                                        <ul>
                                            <li><strong>P</strong>: The input coordinate (e.g., (x,y)).</li>
                                            <li><strong>Noise<sub>base</sub></strong>: The primary noise function (e.g., Perlin, Simplex).</li>
                                            <li><strong>Noise<sub>displacement</sub></strong>: A noise function generating the offset vectors.</li>
                                            <li><strong>S<sub>warp</sub></strong>: Warp strength, controlling the magnitude of distortion.</li>
                                            <li><strong>F<sub>warp</sub></strong>: Warp frequency, controlling the detail of the distortion.</li>
                                        </ul>
                                        <p>By applying this recursively (warping the domain of the displacement noise itself), more complex fractal warping effects can be achieved.</p>
                                        <div class="code-block-container">
                                            <div class="code-block-header">
                                                <span>Domain Warp Implementation</span>
                                                <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                            </div>
                                            <pre class="foldable-content" style="display: none;"><code class="language-python">def domain_warp_noise_with_params(cls, batch_size, height, width, shader_params, time, device, seed):
    # Extract parameters
    scale = shader_params.get("shaderScale", 1.0)
    octaves = shader_params.get("shaderOctaves", 3)) # Used for the base noise often
    warp_strength = shader_params.get("shaderWarpStrength", 0.5)
    warp_frequency_mult = shader_params.get("shaderWarpFrequency", 0.5) # Relative frequency for warp noise

    # Create coordinate grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )
    p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Time component for animation
    t_offset = time * 0.1

    # Generate displacement fields (two noise channels for 2D offset)
    # It's common to use a simpler noise like Perlin for displacement
    # Assuming a 'sample_perlin_octaves' function for simplicity
    
    # Coordinates for displacement noise
    p_warp = p * scale * warp_frequency_mult
    
    # Simplified Perlin sampling for displacement
    # In a real scenario, you'd use an actual Perlin/Simplex implementation
    q_x = sample_perlin_like(p_warp + torch.tensor([10.3 + t_offset, 20.7], device=device), seed + 100)
    q_y = sample_perlin_like(p_warp + torch.tensor([-5.1, -15.9 + t_offset], device=device), seed + 200)
    
    displacement = torch.cat([q_x, q_y], dim=-1) * warp_strength

    # Apply displacement to original coordinates
    p_displaced = p * scale + displacement

    # Generate base noise with displaced coordinates
    # Again, assuming a base noise function (e.g., Perlin with octaves)
    # result = sample_perlin_octaves(p_displaced, octaves, seed)
    result = sample_perlin_like(p_displaced, seed) # Simplified

    return result

# Helper placeholder for a Perlin-like noise generation
def sample_perlin_like(coords, seed):
    # This is a highly simplified stand-in for actual Perlin noise
    # A real implementation would involve gradients, lattice points, interpolation etc.
    torch.manual_seed(seed)
    # Create some pseudo-randomness based on coords and seed
    # This is NOT Perlin noise but serves as a placeholder structure
    noise = torch.sin(coords[..., 0] * 5.0 + coords[..., 1] * 3.0 + seed * 0.1) * \
            torch.cos(coords[..., 0] * 2.0 - coords[..., 1] * 6.0 + seed * 0.05)
    return noise.unsqueeze(-1) # Add channel dimension
                                            </code></pre>
                                        </div>
                                    </div>

                                    <div id="tab-fractal" class="tab-content">
                                        <h3>Fractal Noise (FBM) Mathematics</h3>
                                        <p>Fractal noise, often implemented as Fractal Brownian Motion (FBM), is a fundamental technique for generating natural-looking textures. It is constructed by summing multiple layers (octaves) of a base noise function (like Perlin or Simplex). Each successive octave has a higher frequency and a lower amplitude.</p>
                                        <div class="math-formula">
                                            FBM(P) = ∑<sup>N-1</sup><sub>i=0</sub> A<sup>i</sup> × Noise(F<sup>i</sup> × P)
                                        </div>
                                        <p>Where:</p>
                                        <ul>
                                            <li><strong>P</strong>: The input coordinate.</li>
                                            <li><strong>N</strong>: The number of octaves (layers of detail).</li>
                                            <li><strong>Noise</strong>: The base coherent noise function (e.g., Perlin).</li>
                                            <li><strong>A</strong>: Amplitude factor (persistence, typically < 1, e.g., 0.5). Controls how much amplitude decreases per octave.</li>
                                            <li><strong>F</strong>: Frequency factor (lacunarity, typically > 1, e.g., 2.0). Controls how much frequency increases per octave.</li>
                                        </ul>
                                        <p>The summation creates a pattern that exhibits self-similarity across different scales, characteristic of many natural phenomena.</p>
                                        <div class="code-block-container">
                                            <div class="code-block-header">
                                                <span>Fractal Noise (FBM) Implementation</span>
                                                <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                            </div>
                                            <pre class="foldable-content" style="display: none;"><code class="language-python">def fractal_noise_with_params(cls, batch_size, height, width, shader_params, time, device, seed):
    # Extract parameters
    scale = shader_params.get("shaderScale", 1.0)
    octaves = int(shader_params.get("shaderOctaves", 4))
    persistence = shader_params.get("shaderPersistence", 0.5) # Amplitude scaling
    lacunarity = shader_params.get("shaderLacunarity", 2.0)  # Frequency scaling
    phase_shift = shader_params.get("shaderPhaseShift", 0.0) # For adding variation

    # Create coordinate grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )
    p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    total_noise = torch.zeros(batch_size, height, width, 1, device=device)
    current_amplitude = 1.0
    current_frequency = 1.0
    normalization_factor = 0.0

    for i in range(octaves):
        # Apply scale and current frequency to coordinates
        # Add time and phase_shift for variation / animation
        coords_at_octave = p * scale * current_frequency + time * 0.05 * current_frequency + phase_shift * i
        
        # Generate base noise (e.g., Perlin or Simplex)
        # Assuming 'sample_perlin_like' as a placeholder for a proper noise function
        noise_val = sample_perlin_like(coords_at_octave, seed + i) 
        
        total_noise += noise_val * current_amplitude
        normalization_factor += current_amplitude
        
        current_amplitude *= persistence
        current_frequency *= lacunarity
        
    # Normalize the result to roughly [-1, 1] range
    if normalization_factor > 0:
        total_noise /= normalization_factor
        
    return total_noise
                                            </code></pre>
                                        </div>
                                    </div>

                                    <div id="tab-waves" class="tab-content">
                                        <h3>Waves Noise Mathematics</h3>
                                        <p>Waves noise is typically generated by summing multiple sine or cosine wave functions. Each wave can have its own amplitude, frequency, phase, and direction. The superposition of these waves can create a wide variety_of_patterns, from simple ripples to complex interference effects.</p>
                                        <div class="math-formula">
                                            Waves(P) = ∑<sub>i</sub> A<sub>i</sub> × sin(k<sub>i</sub> ⋅ P + ω<sub>i</sub>t + φ<sub>i</sub>)
                                        </div>
                                        <p>Where for each wave i:</p>
                                        <ul>
                                            <li><strong>P</strong>: The input coordinate (e.g., (x,y)).</li>
                                            <li><strong>A<sub>i</sub></strong>: Amplitude of the wave.</li>
                                            <li><strong>k<sub>i</sub></strong>: Wave vector (determines direction and spatial frequency/wavelength). Its magnitude |k<sub>i</sub>| = 2π / λ<sub>i</sub>.</li>
                                            <li><strong>ω<sub>i</sub></strong>: Angular frequency (determines temporal oscillation speed).</li>
                                            <li><strong>t</strong>: Time.</li>
                                            <li><strong>φ<sub>i</sub></strong>: Phase offset.</li>
                                            <li><strong>⋅</strong>: Dot product.</li>
                                        </ul>
                                        <p>Simpler forms might fix directions or use scalar frequencies if directionality is not complex.</p>
                                        <div class="code-block-container">
                                            <div class="code-block-header">
                                                <span>Waves Noise Implementation</span>
                                                <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                            </div>
                                            <pre class="foldable-content" style="display: none;"><code class="language-python">def waves_noise_with_params(cls, batch_size, height, width, shader_params, time, device, seed):
    # Extract parameters
    scale = shader_params.get("shaderScale", 1.0)       # General scale for coordinates
    num_waves = int(shader_params.get("shaderOctaves", 3)) # Use 'octaves' as number of waves
    base_frequency = shader_params.get("shaderBaseFrequency", 5.0)
    amplitude_variation = shader_params.get("shaderAmplitudeVariation", 0.5)
    phase_shift_speed = shader_params.get("shaderPhaseShift", 0.2) # Use for time-based phase shift

    # Create coordinate grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )
    p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    p = p * scale # Apply general scale

    total_waves = torch.zeros(batch_size, height, width, 1, device=device)
    
    torch.manual_seed(seed) # For reproducible wave directions/phases

    for i in range(num_waves):
        # Generate random direction for each wave
        angle = torch.rand(1, device=device) * 2.0 * 3.14159265 # math.pi
        direction = torch.tensor([torch.cos(angle), torch.sin(angle)], device=device).view(1, 1, 1, 2)
        
        frequency = base_frequency * (1.0 + (torch.rand(1, device=device) - 0.5) * 0.5 * i) # Vary frequency slightly
        amplitude = 1.0 / num_waves * (1.0 + (torch.rand(1, device=device) - 0.5) * amplitude_variation)
        phase = torch.rand(1, device=device) * 2.0 * 3.14159265 + time * phase_shift_speed * (i + 1)

        # Project coordinates onto wave direction: dot(p, direction)
        wave_input = (p * direction).sum(dim=-1, keepdim=True)
        
        wave_val = torch.sin(wave_input * frequency + phase) * amplitude
        total_waves += wave_val
        
    # Normalize to be roughly in [-1, 1], though sum of sines can exceed this
    # A more robust normalization might be needed depending on num_waves and amplitudes
    # total_waves = torch.clamp(total_waves, -1.0, 1.0) 
    
    return total_waves
                                            </code></pre>
                                        </div>
                                    </div>

                                    <div id="tab-gaussian" class="tab-content">
                                        <h3>Gaussian Noise Mathematics</h3>
                                        <p>Gaussian noise is a statistical noise characterized by a probability density function (PDF) that follows the Gaussian (or normal) distribution. Its values are typically clustered around a mean (μ), with a spread determined by the standard deviation (σ). In image processing, it's often used to simulate random sensor noise or as a basis for other effects.</p>
                                        <div class="math-formula">
                                            PDF: f(x | μ, σ<sup>2</sup>) = (1 / (σ√(2π))) × e<sup>-((x-μ)<sup>2</sup> / (2σ<sup>2</sup>))</sup>
                                        </div>
                                        <p>For generating Gaussian noise, random numbers are drawn from this distribution. In practice, libraries provide functions to directly generate such noise (e.g., <code>torch.randn</code> which samples from N(0,1)).</p>
                                        <ul>
                                            <li><strong>μ (Mean)</strong>: The average value, often 0 for noise.</li>
                                            <li><strong>σ (Standard Deviation)</strong>: Controls the "spread" or intensity of the noise. σ<sup>2</sup> is the variance.</li>
                                        </ul>
                                        <p>Gaussian noise is "white" if its values are statistically independent and identically distributed at each point.</p>
                                        <div class="code-block-container">
                                            <div class="code-block-header">
                                                <span>Gaussian Noise Implementation</span>
                                                <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                            </div>
                                            <pre class="foldable-content" style="display: none;"><code class="language-python">def gaussian_noise_with_params(cls, batch_size, height, width, shader_params, time, device, seed):
    # Extract parameters
    mean = shader_params.get("shaderMean", 0.0)
    std_dev = shader_params.get("shaderStdDev", 1.0) # This is 'scale' in some contexts
    # 'scale' from shader_params usually means spatial scale for other noises,
    # for Gaussian, it often refers to standard deviation or amplitude.
    # Let's use a specific param like shaderStdDev.
    
    # Ensure seed is used for reproducibility
    torch.manual_seed(seed)

    # Generate Gaussian noise
    # torch.randn generates noise from a standard normal distribution (mean 0, std 1)
    noise_tensor = torch.randn(batch_size, 1, height, width, device=device) 
    
    # Adjust mean and standard deviation
    # Noise = Mean + StdDev * StandardNormalNoise
    noise_tensor = mean + std_dev * noise_tensor
    
    # Unlike coherent noise (Perlin, etc.), Gaussian noise typically doesn't use a coordinate grid
    # for its generation, as each pixel's value is independent (for white Gaussian noise).
    # Parameters like 'scale', 'octaves' are less relevant in the typical sense.
    # 'time' could be used to modulate mean or std_dev for animated effects if desired.
    # For example: animated_std_dev = std_dev * (1.0 + 0.5 * torch.sin(torch.tensor(time)))
    # noise_tensor = mean + animated_std_dev * noise_tensor

    # Output is typically 1 channel, but can be expanded if needed for coloring or blending.
    # The KSampler expects 4 channels if blending with latent.
    # This function would typically output raw noise, channel expansion happens later.
    
    return noise_tensor # Shape: [batch_size, 1, height, width]
                                            </code></pre>
                                        </div>
                                    </div>

                                    <div id="tab-heterogeneous-fbm" class="tab-content">
                                        <h3>Heterogeneous FBM Mathematics</h3>
                                        <p>Heterogeneous Fractal Brownian Motion (Hetero FBM) is an extension of standard FBM. While standard FBM uses constant parameters (like persistence/amplitude falloff and lacunarity/frequency gain) across all octaves and spatial locations, Hetero FBM allows these parameters, or the base noise characteristics, to vary spatially. This creates textures with non-uniform complexity, where some areas might be smoother and others rougher or more detailed.</p>
                                        <div class="math-formula">
                                            HeteroFBM(P) = ∑<sup>N-1</sup><sub>i=0</sub> A(P, i) × Noise(F(P, i) × P, params(P,i))
                                        </div>
                                        <p>Where:</p>
                                        <ul>
                                            <li><strong>P</strong>: The input coordinate.</li>
                                            <li><strong>N</strong>: The number of octaves.</li>
                                            <li><strong>A(P, i)</strong>: Spatially varying amplitude for octave i at point P.</li>
                                            <li><strong>F(P, i)</strong>: Spatially varying frequency for octave i at point P.</li>
                                            <li><strong>Noise(..., params(P,i))</strong>: Base noise function whose own internal parameters might also vary spatially.</li>
                                        </ul>
                                        <p>The spatial variation itself is often controlled by another noise function or a predefined map. For example, the Hurst exponent (H), which relates to persistence, could be made to vary across the domain.</p>
                                        <div class="code-block-container">
                                            <div class="code-block-header">
                                                <span>Heterogeneous FBM Implementation</span>
                                                <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                            </div>
                                            <pre class="foldable-content" style="display: none;"><code class="language-python">def heterogeneous_fbm_noise_with_params(cls, batch_size, height, width, shader_params, time, device, seed):
    # Extract base parameters
    scale = shader_params.get("shaderScale", 1.0)
    octaves = int(shader_params.get("shaderOctaves", 4))
    
    # Parameters for heterogeneity - these could be controlled by other noise fields
    base_persistence = shader_params.get("shaderPersistence", 0.5)
    base_lacunarity = shader_params.get("shaderLacunarity", 2.0)
    heterogeneity_strength = shader_params.get("shaderHeteroStrength", 0.3)

    # Create coordinate grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )
    p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Generate a control noise field for varying persistence/lacunarity
    # This control noise determines how parameters change spatially
    control_coords = p * scale * 0.3 + time * 0.02 # Slower varying control
    control_noise = sample_perlin_like(control_coords, seed + 500) # Placeholder

    total_noise = torch.zeros(batch_size, height, width, 1, device=device)
    current_amplitude = torch.ones(batch_size, height, width, 1, device=device)
    current_frequency = torch.ones(batch_size, height, width, 1, device=device) * scale
    normalization_factor = torch.zeros(batch_size, height, width, 1, device=device)

    for i in range(octaves):
        # Modulate persistence and lacunarity spatially using control_noise
        # This is a simplified modulation
        spatial_persistence_mod = (control_noise * heterogeneity_strength) # Range e.g. [-0.3, 0.3]
        spatial_lacunarity_mod = (control_noise * heterogeneity_strength * 0.5)

        current_persistence = torch.clamp(base_persistence + spatial_persistence_mod, 0.1, 0.9)
        current_lacunarity = torch.clamp(base_lacunarity + spatial_lacunarity_mod, 1.1, 3.0)
        
        coords_at_octave = p * current_frequency + time * 0.05 * current_frequency.mean() # Use mean freq for time anim
        
        noise_val = sample_perlin_like(coords_at_octave, seed + i)
        
        total_noise += noise_val * current_amplitude
        normalization_factor += current_amplitude
        
        current_amplitude *= current_persistence
        current_frequency *= current_lacunarity
        
    if normalization_factor.min() > 1e-5: # Avoid division by zero
        total_noise /= normalization_factor
    else: # Fallback if normalization factor is too small everywhere
        total_noise = torch.clamp(total_noise, -1.0, 1.0)

    return total_noise
                                            </code></pre>
                                        </div>
                                    </div>

                                    <div id="tab-interference" class="tab-content">
                                        <h3>Interference Noise Mathematics</h3>
                                        <p>Interference patterns arise from the superposition of two or more waves. When waves meet, they can reinforce each other (constructive interference) or cancel each other out (destructive interference), depending on their relative phases and amplitudes. This principle is fundamental in physics (e.g., light and sound waves) and can be used to generate complex visual patterns.</p>
                                        <div class="math-formula">
                                            Pattern(P) = f(Wave<sub>1</sub>(P), Wave<sub>2</sub>(P), ..., Wave<sub>N</sub>(P))
                                        </div>
                                        <p>A common way to generate interference is by summing or multiplying wave functions:</p>
                                        <p>Example with two sine waves:</p>
                                        <div class="math-formula">
                                          V(P) = A<sub>1</sub>sin(k<sub>1</sub>⋅P + φ<sub>1</sub>) + A<sub>2</sub>sin(k<sub>2</sub>⋅P + φ<sub>2</sub>)
                                        </div>
                                        <p>Or using noise functions as sources:</p>
                                        <div class="math-formula">
                                          V(P) = Noise<sub>1</sub>(P) + Noise<sub>2</sub>(P × S + O)
                                        </div>
                                        <p>Where S is a scale and O is an offset for Noise<sub>2</sub> to make it different from Noise<sub>1</sub>. The visual character depends heavily on the frequencies and relative phases of the interfering sources.</p>
                                        <div class="code-block-container">
                                            <div class="code-block-header">
                                                <span>Interference Noise Implementation</span>
                                                <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                            </div>
                                            <pre class="foldable-content" style="display: none;"><code class="language-python">def interference_noise_with_params(cls, batch_size, height, width, shader_params, time, device, seed):
    # Extract parameters
    scale1 = shader_params.get("shaderScale", 1.0)
    scale2 = shader_params.get("shaderScale2", 1.5) # Scale for the second noise source
    octaves1 = int(shader_params.get("shaderOctaves", 3))
    octaves2 = int(shader_params.get("shaderOctaves2", 2))
    phase_offset = shader_params.get("shaderPhaseShift", 0.5) # Controls relative phase or timing

    # Create coordinate grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )
    p = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Generate two separate noise fields
    # Using placeholder 'sample_fbm_like' for simplicity
    noise_field1_coords = p * scale1 + time * 0.03
    noise_field1 = sample_fbm_like(noise_field1_coords, octaves1, seed, persistence=0.5, lacunarity=2.0)

    noise_field2_coords = p * scale2 + time * 0.05 + phase_offset # Add phase_offset to vary interaction
    noise_field2 = sample_fbm_like(noise_field2_coords, octaves2, seed + 10, persistence=0.6, lacunarity=1.8)

    # Combine the noise fields to create interference
    # Different combination methods yield different patterns:
    # 1. Addition: result = noise_field1 + noise_field2
    # 2. Multiplication: result = noise_field1 * noise_field2
    # 3. Difference: result = noise_field1 - noise_field2
    # 4. Using trigonometric functions: result = torch.cos(noise_field1 * math.pi) + torch.sin(noise_field2 * math.pi)
    
    # Example: Cosine/Sine interference for more distinct patterns
    pi = 3.14159265
    interference_pattern = torch.cos(noise_field1 * pi) + torch.sin(noise_field2 * pi * (1.0 + phase_offset*0.2))
    
    # Normalize the result (sum of two sines/cosines can range from -2 to 2)
    result = interference_pattern / 2.0 
    result = torch.clamp(result, -1.0, 1.0)
    
    return result

# Placeholder for FBM-like noise generation
def sample_fbm_like(coords, octaves, seed, persistence, lacunarity):
    total_noise = torch.zeros_like(coords[..., 0:1]) # Ensure output has 1 channel
    current_amplitude = 1.0
    current_frequency = 1.0
    temp_coords = coords # Avoid modifying original coords if it's used elsewhere

    for i in range(int(octaves)):
        noise_val = sample_perlin_like(temp_coords * current_frequency, seed + i)
        total_noise += noise_val * current_amplitude
        current_amplitude *= persistence
        current_frequency *= lacunarity
    
    # Basic normalization attempt
    max_possible_amp = sum([persistence**i for i in range(int(octaves))])
    if max_possible_amp > 0:
      total_noise /= max_possible_amp
      
    return total_noise
                                            </code></pre>
                                        </div>
                                    </div>

                                    <div id="tab-spectral" class="tab-content">
                                        <h3>Spectral Noise Mathematics</h3>
                                        <p>Spectral noise generation involves directly defining or manipulating the noise's properties in the frequency domain (its spectrum) using techniques like the Fourier Transform. By controlling the amplitude and phase of different frequencies, a wide variety_of_textures can be created, from smooth to rough, or with specific directional biases.</p>
                                        <p>The general process is:</p>
                                        <ol>
                                            <li>Start with white noise in the spatial domain or directly create a spectrum.</li>
                                            <li>Compute its Fast Fourier Transform (FFT) to get the frequency domain representation (spectrum).</li>
                                            <li>Modify the spectrum: Apply a filter (e.g., 1/f<sup>β</sup> for pink/brown/blue noise, band-pass, directional filters).</li>
                                            <li>Compute the Inverse Fast Fourier Transform (IFFT) to convert the modified spectrum back to the spatial domain.</li>
                                        </ol>
                                        <div class="math-formula">
                                            SpatialNoise = IFFT( Filter(Spectrum) × FFT(InitialNoise) )
                                        </div>
                                        <p>The filter often takes the form 1/f<sup>β</sup>, where f is frequency and β controls the "color" of the noise (e.g., β=0 for white, β=1 for pink, β=2 for brown/red).</p>
                                        <div class="code-block-container">
                                            <div class="code-block-header">
                                                <span>Spectral Noise Implementation</span>
                                                <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                            </div>
                                            <pre class="foldable-content" style="display: none;"><code class="language-python">def spectral_noise_with_params(cls, batch_size, height, width, shader_params, time, device, seed):
    # Extract parameters
    beta = shader_params.get("shaderSpectralBeta", 1.0) # Exponent for 1/f^beta filter
    # A low-pass/high-pass cutoff might also be a parameter
    low_freq_boost = shader_params.get("shaderLowFreqBoost", 1.0)
    high_freq_attenuation = shader_params.get("shaderHighFreqAtten", 1.0)

    torch.manual_seed(seed)

    # 1. Start with white noise in spatial domain
    white_noise_spatial = torch.rand(batch_size, 1, height, width, device=device) * 2.0 - 1.0
    
    # 2. Compute FFT
    # torch.fft.fft2 performs 2D FFT. For real input, rfft2 is more efficient.
    spectrum = torch.fft.rfft2(white_noise_spatial, norm="ortho") # norm="ortho" is often good

    # Create frequency coordinates (kx, ky)
    # Frequencies for rfft2: ky goes from 0 to H//2. kx goes from 0 to W-1, then wraps around.
    ky = torch.fft.rfftfreq(height, d=1.0/height, device=device) # Normalized frequencies
    kx = torch.fft.fftfreq(width, d=1.0/width, device=device)   # Frequencies for full FFT
    
    # For rfft2, kx used will be kx[:W//2 + 1]
    kx_r = kx[:width//2 + 1]

    # Create a 2D grid of frequencies (squared magnitude for filter)
    # Broadcasting ky.unsqueeze(1) and kx_r.unsqueeze(0)
    freq_sq = ky.unsqueeze(1)**2 + kx_r.unsqueeze(0)**2
    freq_sq = freq_sq.unsqueeze(0).unsqueeze(0) # Add batch and channel dims

    # Avoid division by zero at DC component (freq=0)
    freq_sq[freq_sq < 1e-6] = 1e-6 
    
    # 3. Create and apply the 1/f^beta filter
    # Power spectrum is proportional to 1 / (freq_magnitude ^ beta)
    # Amplitude is proportional to 1 / (freq_magnitude ^ (beta/2))
    filter_amplitude = (1.0 / (torch.sqrt(freq_sq)**(beta/2.0)))

    # Apply additional boosts/attenuations
    # Example: Boost low frequencies, attenuate high frequencies
    # This is a very simplified frequency band control
    freq_magnitude = torch.sqrt(freq_sq)
    filter_amplitude *= torch.exp(- (freq_magnitude / (height*0.5)) * (1.0-low_freq_boost) ) # Boost low (smaller effect here)
    filter_amplitude *= torch.exp(- (freq_magnitude / (height*0.1)) * (high_freq_attenuation-1.0) ) # Attenuate high

    # Ensure DC component (mean) is not overly amplified if beta is high
    # This depends on specific needs. Sometimes DC is zeroed out or handled separately.
    # filter_amplitude[..., 0, 0] = 1.0 # Or some other controlled value

    modified_spectrum = spectrum * filter_amplitude
    
    # Add some phase shift based on time for animation (subtle effect)
    if time > 0:
        phase_shift_val = torch.exp(1j * freq_magnitude * time * 0.01)
        modified_spectrum *= phase_shift_val

    # 4. Compute IFFT
    spatial_noise_filtered = torch.fft.irfft2(modified_spectrum, s=(height, width), norm="ortho")
    
    # Normalize the result to roughly [-1, 1]
    std_val = torch.std(spatial_noise_filtered)
    if std_val > 1e-5:
        spatial_noise_filtered = (spatial_noise_filtered - torch.mean(spatial_noise_filtered)) / std_val
    
    return torch.clamp(spatial_noise_filtered, -1.0, 1.0)
                                            </code></pre>
                                        </div>
                                    </div>

                                    <div id="tab-projection-3d" class="tab-content">
                                        <h3>3D Projection Noise Mathematics</h3>
                                        <p>3D Projection noise involves generating a 3D noise field (e.g., 3D Perlin, Simplex, or FBM) and then sampling a 2D slice from it. This technique creates 2D patterns that appear to have depth, volume, or temporal evolution if the slice position changes over time.</p>
                                        <div class="math-formula">
                                            Noise<sub>2D</sub>(x, y) = Noise<sub>3D</sub>(x × S<sub>xy</sub>, y × S<sub>xy</sub>, z<sub>slice</sub> × S<sub>z</sub> + T × V<sub>t</sub>)
                                        </div>
                                        <p>Where:</p>
                                        <ul>
                                            <li><strong>(x, y)</strong>: 2D coordinates for the output noise.</li>
                                            <li><strong>Noise<sub>3D</sub></strong>: A 3D coherent noise function.</li>
                                            <li><strong>S<sub>xy</sub></strong>: Spatial scaling factor for x and y axes.</li>
                                            <li><strong>z<sub>slice</sub></strong>: The depth or position of the 2D slice along the third dimension (often z).</li>
                                            <li><strong>S<sub>z</sub></strong>: Scaling factor for the z-dimension, controlling "thickness" or detail along z.</li>
                                            <li><strong>T</strong>: Time parameter for animation.</li>
                                            <li><strong>V<sub>t</sub></strong>: Velocity of slicing through the 3D noise field over time.</li>
                                        </ul>
                                        <p>By animating z<sub>slice</sub> or rotating the sampling plane, dynamic evolving textures can be created from a static 3D noise volume.</p>
                                        <div class="code-block-container">
                                            <div class="code-block-header">
                                                <span>3D Projection Noise Implementation</span>
                                                <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                            </div>
                                            <pre class="foldable-content" style="display: none;"><code class="language-python">def projection_3d_noise_with_params(cls, batch_size, height, width, shader_params, time, device, seed):
    # Extract parameters
    scale_xy = shader_params.get("shaderScale", 1.0)  # Spatial scale for X and Y
    scale_z = shader_params.get("shaderZScale", 1.0)  # Scale for the Z dimension of the 3D noise
    octaves = int(shader_params.get("shaderOctaves", 3))
    z_slice_pos = shader_params.get("shaderZSlice", 0.0) # Static Z position for the slice
    time_travel_speed = shader_params.get("shaderTimeSpeed", 0.1) # How fast 'time' moves the slice in Z

    # Create 2D coordinate grid for x, y
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )
    
    # Prepare 3D coordinates for sampling 3D noise
    # X and Y are scaled by scale_xy
    # Z is determined by z_slice_pos, animated by time, and scaled by scale_z
    
    x_3d = x_coords.unsqueeze(0).repeat(batch_size, 1, 1) * scale_xy
    y_3d = y_coords.unsqueeze(0).repeat(batch_size, 1, 1) * scale_xy
    
    # Calculate Z coordinate for the slice
    # Static part from z_slice_pos, dynamic part from time
    z_val_static = z_slice_pos
    z_val_dynamic = time * time_travel_speed
    z_3d_scalar = (z_val_static + z_val_dynamic) * scale_z
    
    # Expand z_scalar to match dimensions of x_3d, y_3d for stacking
    z_3d = torch.full_like(x_3d, z_3d_scalar)

    # Stack to form [Batch, Height, Width, 3] coordinates
    p_3d = torch.stack([x_3d, y_3d, z_3d], dim=-1)

    # Generate 3D noise (e.g., 3D Perlin or Simplex FBM)
    # Assuming a placeholder 'sample_3d_fbm_like'
    # This function would take 3D coordinates and produce a scalar noise value.
    result_3d_noise = sample_3d_fbm_like(p_3d, octaves, seed, persistence=0.5, lacunarity=2.0)
    
    return result_3d_noise # Expected shape: [Batch, Height, Width, 1]

# Placeholder for 3D FBM-like noise generation
def sample_3d_fbm_like(coords_3d, octaves, seed, persistence, lacunarity):
    # coords_3d shape: [B, H, W, 3]
    total_noise = torch.zeros_like(coords_3d[..., 0:1]) # Output [B,H,W,1]
    current_amplitude = 1.0
    current_frequency = 1.0
    
    for i in range(int(octaves)):
        # In a real 3D noise, this would use a 3D hash/gradient function
        # Simplified pseudo-3D noise for placeholder:
        temp_coords = coords_3d * current_frequency
        noise_val = (torch.sin(temp_coords[..., 0] + temp_coords[..., 2] * 0.5 + seed * 0.1 * (i+1)) *
                     torch.cos(temp_coords[..., 1] - temp_coords[..., 2] * 0.3 + seed * 0.05 * (i+1)))
        noise_val = noise_val.unsqueeze(-1) # Add channel dim
        
        total_noise += noise_val * current_amplitude
        current_amplitude *= persistence
        current_frequency *= lacunarity
        
    max_possible_amp = sum([persistence**i for i in range(int(octaves))])
    if max_possible_amp > 0:
      total_noise /= max_possible_amp
      
    return torch.clamp(total_noise, -1.0, 1.0)

                                            </code></pre>
                                        </div>
                                    </div>
                                </section>

                                <div class="golden-divider"></div>
       
        <section id="blend-modes">
            <h2>🎭 Blend Modes <span class="sacred-symbol">⚖️</span> The Alchemical Combinations</h2>
            <p>ShaderNoiseKSampler offers various blend modes to combine shader noise with base noise, each creating distinct alchemical transformations:</p>
            
            <div class="parameter-section">
                <h3>🧩 Blend Operations</h3>
                <div class="parameter-grid">
                    <div class="parameter-item">
                        <div class="parameter-name">Normal</div>
                        <p>Simple linear interpolation between base and shader noise, controlled by shader_strength</p>
                        <div class="math-formula">result = base * (1.0 - α) + shader * α</div>
                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Add</div>
                        <p>Adds shader noise to base noise, creating brightened areas where patterns align</p>
                        <div class="math-formula">result = base + shader * α</div>
                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Multiply</div>
                        <p>Multiplies base and shader noise, darkening the overall pattern</p>
                        <div class="math-formula">result = base * (shader * α + (1.0 - α))</div>
                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Screen</div>
                        <p>Inverts, multiplies, then inverts again, brightening patterns</p>
                        <div class="math-formula">result = 1.0 - (1.0 - base) * (1.0 - shader * α)</div>
                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Overlay</div>
                        <p>Combines Multiply and Screen modes for enhanced contrast</p>
                        <div class="math-formula">result = base < 0.5 ? 2.0 * base * shader : 1.0 - 2.0 * (1.0 - base) * (1.0 - shader)</div>
                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Soft Light</div>
                        <p>Darkens or lightens colors depending on shader noise</p>
                        <div class="math-formula">result = (1.0 - base) * base * shader + base * (1.0 - (1.0 - base) * (1.0 - shader))</div>
                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Hard Light</div>
                        <p>More intense version of Overlay with sharper contrast</p>
                        <div class="math-formula">result = shader > 0.5 ? 1.0 - (1.0 - base) * (1.0 - 2.0 * (shader - 0.5)) : 2.0 * base * shader</div>
                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Difference</div>
                        <p>Subtracts darker color from lighter color, creating distinctive edges</p>
                        <div class="math-formula">result = base * (1.0 - α) + abs(base - shader) * α</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>🧿 Blend Mode Harmonics</h3>
                <p>The choice of blend mode significantly impacts the generated image. Different blend modes resonate with different shader noise types and prompt themes:</p>
                <ul>
                    <li><strong>Multiply</strong> + Cellular Noise ⟶ Organic, tissue-like structures</li>
                    <li><strong>Add</strong> + Curl Noise ⟶ Ethereal, flowing energy patterns</li>
                    <li><strong>Overlay</strong> + Tensor Field ⟶ Crystalline, structured patterns</li>
                    <li><strong>Soft Light</strong> + Domain Warp ⟶ Dream-like, fluid transformations</li>
                    <li><strong>Difference</strong> + Spectral Noise ⟶ Boundary-focused, edge-highlighting patterns</li>
                </ul>
                <p>Experiment with these combinations to find the blend harmonics that resonate with your artistic vision.</p>
            </div>
        </section>

        <div class="golden-divider"></div>
        
        <section id="noise-transforms">
            <h2>🌀 Noise Transformations <span class="sacred-symbol">💎</span> Mathematical Metamorphosis</h2>
            <p>Beyond the core shader noise patterns, ShaderNoiseKSampler offers additional mathematical transformations that can dramatically alter the character of the generated shader noise:</p>
            
            <div class="parameter-section">
                <h3>🧮 Mathematical Operators</h3>
                <div class="parameter-grid">
                    <div class="parameter-item">
                        <div class="parameter-name">Reverse</div>
                        <p>Inverts the sign of the noise, turning peaks into valleys and vice versa</p>
                        <div class="math-formula">T(noise) = -noise</div>
                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Inverse</div>
                        <p>Reverses the values within the 0-1 range, preserving the overall pattern</p>
                        <div class="math-formula">T(noise) = 1.0 - noise</div>
                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Absolute</div>
                        <p>Takes the absolute value, creating sharp ridges at zero-crossings</p>
                        <div class="math-formula">T(noise) = |noise|</div>
                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Square</div>
                        <p>Squares the values, enhancing high values and diminishing low values</p>
                        <div class="math-formula">T(noise) = noise<sup>2</sup></div>
                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Square Root</div>
                        <p>Takes the square root, enhancing low values and compressing high values</p>
                        <div class="math-formula">T(noise) = √noise</div>
                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Log</div>
                        <p>Takes the logarithm, greatly enhancing low values</p>
                        <div class="math-formula">T(noise) = log(noise + ε)</div>
                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Sin</div>
                        <p>Applies the sine function, creating oscillating patterns</p>
                        <div class="math-formula">T(noise) = sin(noise × π)</div>
                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Cos</div>
                        <p>Applies the cosine function, creating alternating bands</p>
                        <div class="math-formula">T(noise) = cos(noise × π)</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>🧪 Transformation Alchemy</h3>
                <p>These transformations can dramatically alter the character of shader noise patterns, creating new structural properties:</p>
                <ul>
                    <li><strong>Absolute</strong> transform creates ridge-like features along zero-crossings, perfect for geological formations</li>
                    <li><strong>Square</strong> transform enhances high-intensity regions while subduing low-intensity areas, ideal for creating focused areas of interest</li>
                    <li><strong>Sin/Cos</strong> transforms create banded patterns, excellent for stratified or layered structures</li>
                    <li><strong>Log</strong> transform magnifies subtle details in low-intensity regions, revealing hidden structures</li>
                </ul>
                <p>These transformations can be combined with different shader noise types and blend modes to create an almost infinite variety of structural guidance patterns.</p>
            </div>
        </section>

        <div class="golden-divider"></div>
        
                                <section id="animation">
                                    <h2>🎬 Temporal Coherence <span class="sacred-symbol">⏳</span> Evolving Noise for Animations</h2>
                                    <p>ShaderNoiseKSampler's temporal coherence feature ensures consistent noise patterns. For sequences like video frames, it helps maintain frame-to-frame consistency by evolving noise naturally over time. For single image generations, it ensures the base noise is derived consistently from the main seed, leading to predictable noise structures when parameters are tweaked.</p>

                                    <div class="parameter-section">
                                        <h3>⚙️ Enabling Evolving Animations</h3>
                                        <p>Temporal coherence fundamentally changes how shader noise is generated:</p>
                                        <ul>
                                            <li><strong><code>use_temporal_coherence</code> (Boolean):</strong> This is the primary toggle. When enabled, ShaderNoiseKSampler ensures noise is generated consistently. For videos, it uses techniques (like the <code>TemporalCoherentNoiseGenerator</code>) that treat time as an evolving dimension. For images, it ensures the same base noise is used if the main seed is unchanged.</li>
                                            <li><strong>Consistent Base Seed:</strong> For temporal coherence to be effective (for both images and videos), the main <code>seed</code> input must remain constant. For videos, variation then comes from the noise evolving over an internal 'time' parameter. For images, this ensures that if other shader parameters are changed, the underlying base noise structure remains the same.</li>
                                            <li><strong>Consistent/Evolving Patterns:</strong> For videos, this mode generates patterns that transform coherently from one frame to the next, leading to more stable animations. For single images, it means the generated noise pattern will be the same for a given seed, even if other parameters (like shader strength or blend mode) are changed, allowing for more predictable exploration.</li>
                                        </ul>
                                        <div class="animation-container" id="animation-demo-placeholder" style="display:flex; align-items:center; justify-content:center; text-align:center;">
                                            <p style="opacity:0.7;">Conceptual animation preview placeholder.</p>
                                        </div>
                                    </div>
                                    
                                    <div class="code-block-container">
                                        <div class="code-block-header">
                                            <span>Conceptual Temporal Noise (Python)</span>
                                            <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                        </div>
                                        <pre class="foldable-content" style="display: none;"><code class="language-python">
import torch

# Placeholder for a 3D simplex noise function
def simplex_noise_3d_placeholder(coords_xyt, seed):
    # coords_xyt is expected to be [B, H, W, 3] (x, y, time)
    torch.manual_seed(seed)
    noise = torch.sin(coords_xyt[..., 0] * 2.0 + coords_xyt[..., 2] * 1.5) * \
            torch.cos(coords_xyt[..., 1] * 2.0 - coords_xyt[..., 2] * 1.0)
    return noise.unsqueeze(-1) # Output [B, H, W, 1]

def generate_animated_noise_frame(batch_size, height, width, scale, current_time, device, base_seed):
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )
    spatial_coords = torch.stack([x_coords, y_coords], dim=-1) * scale
    time_coord_val = torch.full((height, width, 1), current_time, device=device)
    p_temporal = torch.cat([spatial_coords, time_coord_val], dim=-1)
    p_temporal_batch = p_temporal.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    frame_noise = simplex_noise_3d_placeholder(p_temporal_batch, base_seed)
    return frame_noise.permute(0, 3, 1, 2)

# --- Example Conceptual Usage for an Animation ---
# total_frames = 100; animation_seed = 12345 
# for frame_idx in range(total_frames):
#     time_for_frame = frame_idx / total_frames
#     noise = generate_animated_noise_frame(1, 64, 64, 1.0, time_for_frame, 'cpu', animation_seed)
    # noise would then be used by KSampler for this frame.
                                        </code></pre>
                                    </div>
                                </section>
                                
                                <div class="golden-divider"></div>

                                <section id="shape-mask-alchemy">
                                    <h2>🎭 Shape Masks <span class="sacred-symbol">💠</span> Geometric Modulation</h2>
                                    <p>Shape masks provide spatial control by modulating shader noise patterns with procedurally generated geometric overlays. These masks, typically grayscale tensors (0.0 for no effect, 1.0 for full effect), guide the generative process, influencing compositions and textural details.</p>

                                    <div class="parameter-section">
                                        <h3>Mask Generation &amp; Application</h3>
                                        <p>Shape masks are procedurally generated within the node as fixed grayscale patterns where values range from 0.0 (no effect area) to 1.0 (full effect area). The generated mask \( M \) is then used to modulate an input field (e.g., a noise or velocity field \( V \)) based on the <code>shape_mask_strength</code> \( \\alpha_s \). The application is a linear interpolation:</p>
                                        <div class="math-formula">
                                            V_{masked} = V \\times (1 - \\alpha_s) + (V \\times M) \\times \\alpha_s
                                        </div>
                                        <p>This means if \( \\alpha_s = 1 \), the field is fully multiplied by the mask pattern (\( V \\times M \)). If \( \\alpha_s = 0 \), the mask has no effect. Intermediate values blend between the original and the mask-modulated field. Key characteristics include:</p>
                                        <ul>
                                            <li><strong>Mask Type:</strong> Defines the base geometry from a selection including Radial, Linear, and various Geometric or Procedural patterns.</li>
                                            <li><strong>Mask Strength:</strong> The \( \\alpha_s \) parameter, controlling the blend intensity of the mask\'s effect.</li>
                                            <li><strong>Edge Handling:</strong> Mask types inherently produce different edge styles, some with soft, feathered falloffs (often using <code>smoothstep</code> internally) and others with hard, thresholded edges.</li>
                                        </ul>
                                        <p>Note that while masks might appear animated in visualization previews, in the backend implementation they are static patterns applied at generation time. Any animation effects in the final output would come from changing parameters between frames.</p>
                                        
                                        <style>
                                        .mask-example {
                                            display: flex;
                                            flex-wrap: wrap;
                                            gap: 0.8em;
                                            justify-content: center;
                                            margin: 1.5em 0;
                                            background: rgba(0, 0, 0, 0.2);
                                            padding: 1.2em;
                                            border-radius: 8px;
                                        }
                                        .mask-swatch {
                                            width: 100px;
                                            height: 70px;
                                            border-radius: 6px;
                                            display: flex;
                                            flex-direction: column;
                                            align-items: center;
                                            justify-content: flex-end;
                                            background-color: #1e1e26;
                                            padding-bottom: 5px;
                                            color: white;
                                            font-size: 0.75rem;
                                            text-shadow: 0 1px 2px rgba(0,0,0,0.8);
                                            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
                                            transition: transform 0.2s;
                                            border: 1px solid rgba(255,255,255,0.1);
                                            position: relative;
                                            overflow: hidden;
                                        }
                                        .mask-swatch:hover {
                                            transform: scale(1.08);
                                        }
                                        .mask-canvas {
                                            position: absolute;
                                            top: 0;
                                            left: 0;
                                            width: 100%;
                                            height: calc(100% - 20px);
                                        }
                                        .mask-swatch .mask-name {
                                            position: relative;
                                            z-index: 2;
                                            margin-top: auto;
                                            font-weight: bold;
                                            background-color: rgba(0,0,0,0.6);
                                            padding: 2px 6px;
                                            border-radius: 3px;
                                        }
                                        </style>
                                        
                                        <div class="mask-example">
                                            <div class="mask-swatch">
                                                <div id="mask-canvas-radial" class="mask-canvas"></div>
                                                <div class="mask-name">Radial</div>
                                            </div>
                                            <div class="mask-swatch">
                                                <div id="mask-canvas-linear" class="mask-canvas"></div>
                                                <div class="mask-name">Linear</div>
                                            </div>
                                            <div class="mask-swatch">
                                                <div id="mask-canvas-grid" class="mask-canvas"></div>
                                                <div class="mask-name">Grid</div>
                                            </div>
                                            <div class="mask-swatch">
                                                <div id="mask-canvas-vignette" class="mask-canvas"></div>
                                                <div class="mask-name">Vignette</div>
                                            </div>
                                            <div class="mask-swatch">
                                                <div id="mask-canvas-spiral" class="mask-canvas"></div>
                                                <div class="mask-name">Spiral</div>
                                            </div>
                                            <div class="mask-swatch">
                                                <div id="mask-canvas-hexgrid" class="mask-canvas"></div>
                                                <div class="mask-name">Hexgrid</div>
                                            </div>
                                            <div class="mask-swatch">
                                                <div id="mask-canvas-wavy" class="mask-canvas"></div>
                                                <div class="mask-name">Wavy</div>
                                            </div>
                                            <div class="mask-swatch">
                                                <div id="mask-canvas-concentric_rings" class="mask-canvas"></div>
                                                <div class="mask-name">Concentric</div>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="code-block-container">
                                        <div class="code-block-header">
                                            <span>Illustrative Shape Mask Application (Python)</span>
                                            <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                        </div>
                                        <pre class="foldable-content" style="display: none;"><code class="language-python">import torch

def apply_shape_mask_illustrative(input_field, coords_bhwc, mask_type="radial", shape_mask_strength=0.7):
    '''Illustrative: Generate and apply a shape mask to an input field.
    Args:
        input_field (torch.Tensor): Field to be masked [B, H, W, C].
        coords_bhwc (torch.Tensor): Coordinates for mask generation (0-1 range) [B, H, W, 2].
        mask_type (str): Type of mask to generate.
        shape_mask_strength (float): Strength of the mask application.
    Returns:
        torch.Tensor: Masked input_field.
    '''
    batch_size, height, width, _ = coords_bhwc.shape
    device = coords_bhwc.device
    
    # --- Mask Generation (Simplified Radial Example) ---
    # Initialize shape_mask with ones (no effect by default if mask_type is unrecognized)
    shape_mask = torch.ones_like(coords_bhwc[..., 0]) 

    if mask_type == "radial":
        center_x, center_y = 0.5, 0.5
        
        # Calculate distance from the center
        y_diff = coords_bhwc[..., 1] - center_y 
        x_diff = coords_bhwc[..., 0] - center_x
        # Normalized distance, similar to GLSL masks (stretching to fill 0-1 range)
        dist = torch.sqrt(x_diff**2 + y_diff**2) * 2.0 
        current_mask_values = torch.clamp(1.0 - dist, 0.0, 1.0)
        shape_mask = current_mask_values
    # ... (other mask_type implementations would go here)

    # Ensure mask is [B, H, W, 1] for broadcasting with input_field [B, H, W, C]
    shape_mask = shape_mask.unsqueeze(-1)

    # --- Mask Application ---
    # Modulate the input field with the generated mask pattern
    field_modulated_by_mask_pattern = input_field * shape_mask 
    
    # Apply shape_mask_strength via linear interpolation (lerp)
    # output_field = base_value + (target_value - base_value) * weight
    # output_field = input_field + (field_modulated_by_mask_pattern - input_field) * shape_mask_strength
    # This is equivalent to PyTorch's torch.lerp:
    output_field = torch.lerp(input_field, field_modulated_by_mask_pattern, shape_mask_strength)
    
    return output_field

# --- Example Conceptual Usage ---
# B, H, W, C_noise = 1, 64, 64, 2 # Example: 2-channel velocity field
# example_field = torch.rand(B, H, W, C_noise, device='cpu') # Random field

# # Create normalized coordinates (0 to 1 range)
# y_coords_tensor = torch.linspace(0, 1, H, device='cpu').view(1, H, 1, 1).expand(B, H, W, 1)
# x_coords_tensor = torch.linspace(0, 1, W, device='cpu').view(1, 1, W, 1).expand(B, H, W, 1)
# # Coordinates are expected as [x, y] in the last dimension for mask generation
# coordinates_tensor = torch.cat((x_coords_tensor, y_coords_tensor), dim=-1) 

# masked_field = apply_shape_mask_illustrative(example_field, coordinates_tensor, mask_type="radial", shape_mask_strength=0.8)
# print(f"Masked field tensor shape: {masked_field.shape}")

                                    </code></pre>
                                    </div>
                                    
                                    <div class="card">
                                        <h3>🎨 Creative Applications</h3>
                                        <p>The interplay between different mask types, the noise they modulate, and various parameter settings (like <code>shape_mask_strength</code>, <code>noise_scale</code>, etc.) allows for a wide range of artistic effects. Consider these combinations:</p>
                                        <ul>
                                            <li><strong>Vignettes & Focus:</strong> Soft Radial or Vignette masks to draw attention centrally.</li>
                                            <li><strong>Structured Organics:</strong> Grid Masks combined with Cellular Noise for bio-mechanical effects.</li>
                                            <li><strong>Atmospheric Depth:</strong> Linear Gradient Masks to simulate fog or distance with flowing noise.</li>
                                            <li><strong>Selective Detailing:</strong> Geometric Masks to apply high-frequency noise to specific regions.</li>
                                        </ul>
                                        <p>Experiment with mask types and their strengths in conjunction with noise parameters to guide the generative process.</p>
                                    </div>
                                </section>

                                <div class="golden-divider"></div>

                                <section id="color-schemes">
                                    <h2>🌈 Color Schemes <span class="sacred-symbol">⚙️</span> Chromatic Adjustments</h2>
                                    <p>ShaderNoiseKSampler allows for the application of color transformations directly to the generated shader noise patterns. It's important to understand that this is not a traditional post-processing color adjustment. Instead, the color information is integrated into the shader noise <em>before</em> it influences the diffusion model during the sampling process. Consequently, these 'colors' can function as an integral part of the shader noise's texture and structure, subtly guiding the model's interpretation and shaping the features, patterns, and overall aesthetic of the final image. This technique offers a unique way to enhance visual impact and steer the generative outcome.</p>
                                    
                                    <div class="parameter-section">
                                        <h3>Color Transformation Process</h3>
                                        <p>The color mapping function transforms a normalized noise value t ∈ [0,1] into RGB color space through gradient functions. Each color scheme defines specific transfer functions for each channel, creating distinctive visual characteristics that can alter how the diffusion model interprets the noise structure.</p>
                                        
                                        <div class="math-formula">
                                            C(t) = (R(t), G(t), B(t))
                                        </div>
                                        
                                        <p>Key characteristics of the coloring process include:</p>
                                        <ul>
                                            <li><strong>Color Scheme Selection:</strong> Different schemes provide various perceptual encodings of the noise data.</li>
                                            <li><strong>Color Intensity:</strong> Controls how strongly the coloring affects the final noise blend, typically between 0.0 (no effect) and 1.0 (full effect).</li>
                                            <li><strong>Channel Blending:</strong> The colored information is blended into the noise channels before they influence the diffusion process.</li>
                                        </ul>
                                        
                                        <style>
                                        .color-scheme-section {
                                            display: flex;
                                            flex-direction: column;
                                            gap: 1.2em;
                                            background: rgba(0, 0, 0, 0.2);
                                            padding: 1.2em;
                                            border-radius: 8px;
                                            margin: 1.5em 0;
                                        }
                                        .color-scheme-description {
                                            font-size: 0.9em;
                                            margin-bottom: 0.8em;
                                            line-height: 1.5;
                                        }
                                        .color-example {
                                            display: flex;
                                            flex-wrap: wrap;
                                            gap: 0.8em;
                                            justify-content: center;
                                            margin: 0;
                                        }
                                        .color-swatch {
                                            width: 100px;
                                            height: 70px;
                                            border-radius: 6px;
                                            display: flex;
                                            flex-direction: column;
                                            align-items: center;
                                            justify-content: flex-end;
                                            padding-bottom: 5px;
                                            color: white;
                                            font-size: 0.75rem;
                                            text-shadow: 0 1px 2px rgba(0,0,0,0.8);
                                            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
                                            transition: transform 0.2s;
                                            border: 1px solid rgba(255,255,255,0.1);
                                            position: relative;
                                        }
                                        .color-swatch:hover {
                                            transform: scale(1.08);
                                            z-index: 2;
                                        }
                                        .color-category {
                                            font-weight: bold;
                                            margin-bottom: 0.5em;
                                            color: var(--secondary-color);
                                            font-size: 0.95em;
                                            border-bottom: 1px solid rgba(52, 152, 219, 0.3);
                                            padding-bottom: 0.2em;
                                        }
                                        </style>
                                        
                                        <div class="color-scheme-section">
                                            <div class="color-scheme-description">
                                                The choice of color scheme significantly impacts how shader noise patterns guide the generative process. Different schemes create various perceptual encodings that can emphasize different aspects of the shader noise structure, potentially influencing the final image's mood, texture, and compositional elements.
                                            </div>
                                            
                                            <div class="color-category">Spectral Schemes - Emphasizing Transitions</div>
                                            <div class="color-example">
                                                <div class="color-swatch" style="background: linear-gradient(to bottom, #000004, #3b0f70, #8c2981, #de4968, #fe9f6d, #fcfdbf);">
                                                    Inferno
                                                </div>
                                                <div class="color-swatch" style="background: linear-gradient(to bottom, #000004, #3b0f70, #8c2981, #de4968, #fe9f6d, #fcfdbf);">
                                                    Magma
                                                </div>
                                                <div class="color-swatch" style="background: linear-gradient(to bottom, #0d0887, #7e03a8, #cc4678, #f89441, #f0f921);">
                                                    Plasma
                                                </div>
                                                <div class="color-swatch" style="background: linear-gradient(to bottom, #440154, #30678d, #35b778, #fde724);">
                                                    Viridis
                                                </div>
                                            </div>
                                            
                                            <div class="color-category">Technical Schemes - Enhancing Perception</div>
                                            <div class="color-example">
                                                <div class="color-swatch" style="background: linear-gradient(to bottom, #30123b, #4669db, #26bf8c, #d4ff50, #fab74c, #ba0100);">
                                                    Turbo
                                                </div>
                                                <div class="color-swatch" style="background: linear-gradient(to bottom, #00007f, #0000ff, #00ffff, #ffff00, #ff0000, #7f0000);">
                                                    Jet
                                                </div>
                                                <div class="color-swatch" style="background: linear-gradient(to bottom, #352a87, #0f5cdd, #00b5a6, #ffc337, #fcfea4);">
                                                    Parula
                                                </div>
                                                <div class="color-swatch" style="background: linear-gradient(to bottom, #0000ff, #00ffff, #ffff00);">
                                                    Rainbow
                                                </div>
                                                <div class="color-swatch" style="background: linear-gradient(to bottom, #000000, #ff0000, #ffff00, #ffffff);">
                                                    Hot
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="card">
                                        <h3>Applying Color Schemes</h3>
                                        <p>ShaderNoiseKSampler allows for the selection of various color schemes that are applied to the generated noise. The intensity of this coloration can also be controlled, determining how strongly the chosen color scheme influences the noise that guides the diffusion model.</p>
                                        
                                        <div class="parameter-section" style="margin-top: 1em; background: rgba(20, 20, 30, 0.4);">
                                            <h4 style="color: var(--quaternary-color); margin-bottom: 0.7em;">Color Scheme Effects on Generation</h4>
                                            <ul style="list-style-type: disc; padding-left: 1.2em; margin-bottom: 0.6em;">
                                                <li><strong>Contrast Enhancement:</strong> Color schemes like Inferno and Jet can increase the perceived contrast in the noise pattern, potentially leading to more defined boundaries in the generated image.</li>
                                                <li><strong>Perceptual Organization:</strong> Schemes like Viridis and Turbo help to organize the noise data perceptually, which can influence how the model interprets spatial relationships.</li>
                                                <li><strong>Mood and Tone:</strong> The predominant hues of a color scheme can subtly influence the mood or tonal quality of the generated image, even when the colors themselves aren't directly visible.</li>
                                                <li><strong>Detail Emphasis:</strong> Some schemes are better at emphasizing small details within the noise pattern, which can lead to enhanced textural complexity in the final output.</li>
                                            </ul>
                                        </div>
                                        
                                        <div class="code-block-container">
                                            <div class="code-block-header">
                                                <span>Color Scheme Implementation</span>
                                                <button class="toggle-code-button" onclick="window.toggleCodeSection(this)">Show</button>
                                            </div>
                                            <pre class="foldable-content" style="display: none;"><code class="language-python">
# How color schemes are applied in the backend (simplified from curl_noise.py)
def apply_color_scheme(normalized_value, color_scheme, device):
    """
    Apply a color scheme to normalized noise values (0-1 range).
    
    Args:
        normalized_value: Tensor of shape [batch, 1, height, width] with values in [0,1]
        color_scheme: String identifier for the chosen color scheme
        device: Computation device (cuda or cpu)
        
    Returns:
        Tuple of (r, g, b) channel tensors
    """
    # Helper function for lerping colors
    def lerp(a, b, t):
        return a + (b - a) * t
    
    # Helper for color stops interpolation
    def interpolate_colors(stops, t):
        # stops: list of [value, color_tensor]
        # Find which segment t falls into and interpolate
        idx = torch.zeros_like(t, dtype=torch.long)
        for i in range(len(stops) - 1):
            idx = torch.where((t >= stops[i][0]) & (t < stops[i+1][0]), 
                             torch.full_like(idx, i), idx)
        
        # Handle edge case for t >= last stop
        idx = torch.where(t >= stops[-1][0], 
                         torch.full_like(idx, len(stops) - 2), idx)
                         
        # Initialize output tensor
        final_color = torch.zeros_like(stops[0][1].expand(-1, -1, t.shape[2], t.shape[3]))
        
        # Apply interpolation for each segment
        for i in range(len(stops) - 1):
            mask = (idx == i)
            t0, c0 = stops[i]
            t1, c1 = stops[i+1]
            
            local_t = torch.clamp((t - t0) / (t1 - t0 + 1e-8), 0.0, 1.0)
            segment_color = lerp(c0, c1, local_t)
            final_color = torch.where(mask.expand_as(segment_color), 
                                     segment_color, final_color)
                                     
        return final_color[:, 0:1], final_color[:, 1:2], final_color[:, 2:3]
    
    # Different color schemes using the normalized value tensor
    if color_scheme == "viridis":
        stops = [
            (0.0, torch.tensor([0.267, 0.005, 0.329], device=device).view(1, 3, 1, 1)),  # #440154
            (0.33, torch.tensor([0.188, 0.407, 0.553], device=device).view(1, 3, 1, 1)), # #30678D
            (0.66, torch.tensor([0.208, 0.718, 0.471], device=device).view(1, 3, 1, 1)), # #35B778
            (1.0, torch.tensor([0.992, 0.906, 0.143], device=device).view(1, 3, 1, 1))   # #FDE724
        ]
        r, g, b = interpolate_colors(stops, normalized_value)
    
    elif color_scheme == "magma":
        stops = [
            (0.0, torch.tensor([0.001, 0.001, 0.016], device=device).view(1, 3, 1, 1)),   # #000004
            (0.25, torch.tensor([0.231, 0.059, 0.439], device=device).view(1, 3, 1, 1)),  # #3B0F70
            (0.5, torch.tensor([0.549, 0.161, 0.506], device=device).view(1, 3, 1, 1)),   # #8C2981
            (0.75, torch.tensor([0.871, 0.288, 0.408], device=device).view(1, 3, 1, 1)),  # #DE4968
            (0.85, torch.tensor([0.996, 0.624, 0.427], device=device).view(1, 3, 1, 1)),  # #FE9F6D
            (1.0, torch.tensor([0.988, 0.992, 0.749], device=device).view(1, 3, 1, 1))    # #FCFDBF
        ]
        r, g, b = interpolate_colors(stops, normalized_value)
    
    elif color_scheme == "rainbow":
        # Use HSV color space, with hue from the normalized value
        h = normalized_value  # Hue from normalized value [0,1]
        s = torch.ones_like(h) * 0.8  # Fixed saturation
        v = torch.ones_like(h) * 0.9  # Fixed value/brightness
        r, g, b = hsv_to_rgb(h, s, v)  # Convert to RGB
    
    # ... other color schemes ...
    
    else:
        # Default grayscale if color scheme unknown
        r = g = b = normalized_value
    
    return r, g, b

# Usage example in ShaderNoiseKSampler
def apply_color_to_noise(noise_tensor, shader_params):
    """Apply color scheme to noise tensor before blending with base noise"""
    color_scheme = shader_params.get("color_scheme", "none")
    color_intensity = shader_params.get("color_intensity", 0.8)
    
    if color_scheme == "none" or color_intensity <= 0.0:
        return noise_tensor  # Skip coloring if disabled
    
    # Normalize a channel (usually magnitude) for color mapping
    channel_for_color = noise_tensor[:, 0:1]  # Use first channel
    normalized = torch.clamp((channel_for_color + 1.0) / 2.0, 0.0, 1.0)  # Map [-1,1] to [0,1]
    
    # Apply the color scheme
    r, g, b = apply_color_scheme(normalized, color_scheme, noise_tensor.device)
    
    # Create colored noise (first 3 channels get RGB, 4th unchanged or derived from magnitude)
    colored_tensor = torch.cat([r, g, b, noise_tensor[:, 3:4]], dim=1)
    
    # Blend with original noise based on intensity
    if color_intensity < 1.0:
        # Linear interpolation between original and colored noise
        result = lerp(noise_tensor, colored_tensor, color_intensity)
        return result
    else:
        return colored_tensor
                                            </code></pre>
                                        </div>
                                    </div>
                                </section>

                                <div class="golden-divider"></div>

                                        <section id="usage">
            <h2>🛠️ Usage <span class="sacred-symbol">🔧</span> Working with this Tool</h2>
            <p>ShaderNoiseKSampler integrates advanced noise generation directly into the ComfyUI sampling process. It replaces the standard KSampler's noise generation with a sophisticated system that allows for multi-stage shader application, diverse noise types, transformations, and blending, offering fine-grained control over the creative output.</p>
            
            <div class="card">
                <h3>Core Workflow Integration</h3>
                <ol>
                    <li><strong>Connect Inputs:</strong>
                        <ul>
                            <li><code>model</code>: The primary AI model for generation.</li>
                            <li><code>positive</code> & <code>negative</code>: Conditioning prompts.</li>
                            <li><code>latent_image</code>: The input latent to be processed.</li>
                        </ul>
                    </li>
                    <li><strong>Basic Sampling Parameters:</strong>
                        <ul>
                            <li><code>seed</code>: For reproducibility use a fixed seed value for consistent results.</li>
                            <li><code>steps</code>: Number of sampling iterations.</li>
                            <li><code>cfg</code>: Classifier-Free Guidance scale.</li>
                            <li><code>sampler_name</code> & <code>scheduler</code>: Standard KSampler choices.</li>
                            <li><code>denoise</code>: Controls the extent of denoising.</li>
                        </ul>
                    </li>
                    <li><strong>Shader Noise Configuration:</strong> This is where ShaderNoiseKSampler shines:
                        <ul>
                            <li><code>sequential_stages</code>: Number of shader stages applied sequentially over defined step ranges.</li>
                            <li><code>injection_stages</code>: Number of shader stages applied at specific injection points during sampling.</li>
                            <li><code>shader_strength</code>: Global strength of the shader noise influence. Set to 0.0 to use only base noise.</li>
                            <li><code>blend_mode</code>: How shader noise combines with base noise (e.g., multiply, add, overlay).</li>
                            <li><code>noise_transform</code>: Mathematical operations applied to the generated noise (e.g., absolute, sin, sqrt).</li>
                            <li><code>use_temporal_coherence</code>: For generating frame-consistent noise, useful in animations.</li>
                        </ul>
                    </li>
                    <li><strong>(Optional) Advanced Control:</strong>
                        <ul>
                            <li><code>custom_sigmas</code>: Provide a custom sigma schedule to override the model's default.</li>
                        </ul>
                    </li>
                    <li><strong>Run Workflow:</strong> Execute the ComfyUI graph. The ShaderNoiseKSampler will dynamically generate and blend noise according to your settings throughout the sampling process.</li>
                </ol>
</div>

                                <div class="parameter-section">
                <h3>⚙️ Key Parameter Explanations</h3>
                <div class="parameter-grid">
                    <div class="parameter-item">
                        <div class="parameter-name">Sequential vs. Injection Stages</div>
                        <p><code>sequential_stages</code> apply shader noise over segments of the diffusion process. The total steps are divided among these stages.
                           <code>injection_stages</code> apply shader noise at specific, discrete steps.
                            Sequential stages don't seem to add time to the generation but injection stages will add some time to the generation.
  </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Shader Strength & Distributions</div>
                        <p><code>shader_strength</code> is a global multiplier. Higher shader strength can lead to increased artifacting or generational oddities, but not always and not with all models or shader noise types. Small variations in shader strength is a good way to maintain the sum concepts of an image and not heavily vary its degree of difference.</p>
                                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Noise Transformation</div>
                        <p>The <code>noise_transform</code> (e.g., absolute, sin, sqrt) is a global setting applied to the generated shader noise within each stage before it's blended with the base noise. The chosen transformation is used consistently across all stages.</p>
</div>
                     <div class="parameter-item">
                        <div class="parameter-name">Temporal Coherence</div>
                        <p>When <code>use_temporal_coherence</code> is enabled, the node aims to generate consistent noise. For videos, this means noise evolves smoothly over a "time" dimension (often using 3D/4D noise with one dimension as time), with the base seed consistent across frames. For images, it ensures the noise pattern is consistently derived from the main seed, even with other parameter changes.</p>
                                    </div>
                    <div class="parameter-item">
                        <div class="parameter-name">Custom Sigmas</div>
                        <p><code>custom_sigmas</code> allows advanced users to define their own noise schedule (sigmas), giving more control over how noise is added and removed at each step. This can significantly alter the sampling dynamics.</p>
</div>
                     <div class="parameter-item">
                        <div class="parameter-name">Blend Mode</div>
                        <p>The <code>blend_mode</code> (e.g., multiply, add, overlay) is a global setting that determines how the shader noise combines with the base noise. This selected blend mode is used for all stages.</p>
                                    </div>
</div>
                                    </div>
            
            <div class="card">
                <h3>🧠 Creative Techniques & Advanced Usage</h3>
                <ul>
                    <li><strong>Shader Visualizer as a Guide:</strong> Using the shader visualizer in connection to the generation process is crucial for gaining insights into your parameter adjustments. The shader display was designed as a visual aid, not just for presentation. It helps you understand the shader noise\'s impact on the output image and serves as a navigational tool for latent space, mirroring backend shader parameters.</li>
                    <li><strong>Detail through Noise Scale & Octaves:</strong> Higher noise scale and octaves can contribute to more detailed images. Essentially, a more "noisy" input increases the likelihood of observing finer-grained detail in the output.</li>
                    <li><strong>Parameter Interdependence:</strong> Parameters are not always independent. If phase shift adjustments seem insufficient, try incorporating a bit of warp strength. Conversely, if your noise scale and octaves are already high, parameters like warp strength and phase shift might appear less impactful.</li>
                    <li><strong>Color Intensity with Shape Masks:</strong> Color intensity is <em>only</em> effective when a shape mask is active. Once active, it can be used to introduce subtle to significant variations in your image, depending on the input values.</li>
                    <li><strong>Shape Mask Strength:</strong> Similar to color intensity, shape mask strength is <em>only</em> usable when a shape mask is active. When enabled, it allows for small to large image variations based on your input values.</li>
                    <li><strong>Warp & Phase with Active Masks/Colors:</strong> You might notice that when a color scheme or a shape mask (especially with higher strength/intensity) is active, the perceptual impact of <code>warp_strength</code> and <code>phase_shift</code> can seem diminished. This is likely because the structural influence of the mask or the channel modifications from coloring are significantly shaping the noise, potentially overshadowing the more subtle distortions or shifts from warp and phase. Experiment with moderating mask/color influence if you need to bring out the warp/phase effects more clearly.</li>
                    <li><strong>Quick Ideation:</strong> For rapid idea generation without the need to constantly save parameters, consider using the (direct) version of this node. As is rapid parameter changes cannot be made to queue in succession, due to the way the node is currently implemented.</li>
                    <li><strong>Saving Parameters:</strong> The "Save Shader Parameters" button (found on the ShaderNoiseKSampler node) allows you to save your current shader settings. These are saved to <code>custom_nodes/ComfyUI-ShaderNoiseKsampler/data/shader_params.json</code>. Note: this file must be named exactly <code>shader_params.json</code> in this specific directory and will be overwritten on each save. This is crucial for features that load these parameters. Use the (direct) version of this node to avoid saving parameters.</li>
                    <li><strong>Shader Noise Palettes & Performance:</strong> Each Shader Noise Type (or palette) offers a distinct lens through which to explore the latent space, revealing different facets and possibilities within a seed. Most shader noise types process at roughly comparable speeds, typically adding only about 3-4 seconds to the generation time compared to a standard KSampler. While some may occasionally take a moment longer to initialize or load, the overall impact on generation time is generally minimal.</li>
                </ul>
                                    </div>
                                </section>

                                <div class="golden-divider"></div>

                                <section id="navigating-latent-space">
                                    <h2>🗺️ Navigating Latent Space <span class="sacred-symbol">🧭</span> Shader Controls for Exploration</h2>
                                    <p>ShaderNoiseKSampler offers a way to explore latent space with more deliberate control compared to relying solely on random seeds. Shader parameters act as navigational tools, allowing users to chart the 'territory' within a given seed. This approach can lead to a more nuanced manipulation of the generative process, turning image generation into a more methodical exploration.</p>

                                <div class="parameter-section">
                                        <h3>🕹️ Parameter Deep Dive: Your Exploration Toolkit</h3>
                                        <p>Understanding how each core shader parameter influences the noise provides a map for your creative journey:</p>
                                        <div class="parameter-grid">
                                            <div class="parameter-item">
                                                <div class="parameter-name">Noise Scale: The Zoom Control</div>
                                                <p>Think of Noise Scale as the 'zoom lens' for your exploration within the latent space.</p>
                                                <ul>
                                                    <li><strong>Low scale values:</strong> You're examining fine details and subtle variations within a small, localized region of the seed's potential. Ideal for nuanced adjustments.</li>
                                                    <li><strong>High scale values:</strong> You're taking broader steps, observing more sweeping changes across a wider expanse. This can lead to more dramatic shifts in the output.</li>
                                                </ul>
                                                <p><em>Effect on Generation:</em> Controls how 'localized' or 'expansive' your exploration is. Smaller scales keep you closer to the seed's core characteristics, while larger scales venture further afield.</p>
</div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Octaves: The Detail Slider</div>
                                                <p>Octaves determine the level of detail and complexity embedded within the noise pattern, and consequently, in the generated variations.</p>
                                                <ul>
                                                    <li><strong>Lower octaves:</strong> Produce simpler, smoother noise, leading to more foundational or gentle variations of the theme.</li>
                                                    <li><strong>Higher octaves:</strong> Introduce more intricate, layered patterns with multiple frequencies of detail, revealing complex elaborations and richer textures from the seed's inherent elements.</li>
                                                </ul>
                                                <p><em>Effect on Generation:</em> As you increase octaves, you're effectively adding finer and finer layers of detail, uncovering more complex permutations of the same underlying concepts.</p>
                                    </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Warp Strength: The Non-Linear Navigator</div>
                                                <p>Warp Strength introduces non-linear distortions to the noise pattern, carving unconventional and often surprising paths through the latent space.</p>
                                                <ul>
                                                    <li><strong>Low warp:</strong> Results in more predictable, somewhat linear modifications to the noise structure.</li>
                                                    <li><strong>High warp:</strong> Creates significant distortions, 'folding,' and 'twisting' in the noise pattern. This allows you to unearth unusual or 'hidden' variations that might reside between the more straightforward pathways.</li>
                                                </ul>
                                                <p><em>Effect on Generation:</em> Enables the discovery of unique aesthetics by navigating the latent space in a less constrained, more fluid manner.</p>
</div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Phase Shift: The Perspective Shifter</div>
                                                <p>Phase Shift functions like adjusting your viewpoint or the 'timing' of the noise pattern, subtly altering its manifestation in latent space.</p>
                                                <ul>
                                                    <li><strong>Different phase values:</strong> Shift the entire noise pattern while generally maintaining its internal structural characteristics. It's akin to viewing the same intricate object from slightly different angles or under slightly different lighting conditions.</li>
                                                </ul>
                                                <p><em>Effect on Generation:</em> Reveals different 'facets' or 'expressions' of the same core elements within the seed's potential, offering a way to cycle through related variations smoothly.</p>
                                    </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Shape Mask Strength: The Selective Amplifier</div>
                                                <p>Shape Mask Strength controls the intensity of geometric masks that modulate the noise pattern, allowing for spatial control.</p>
                                                <ul>
                                                    <li><strong>Note:</strong> This parameter only has effect when a mask type is actively selected.</li>
                                                    <li><strong>Small value changes:</strong> Even minor adjustments (0.01-0.05) can produce subtle yet noticeable variations in the final image, as they alter how the mask influences the noise structure.</li>
                                                </ul>
                                                <p><em>Effect on Generation:</em> Acts as a fine-tuning dial for how strongly the selected geometric pattern guides the noise formation, enabling controlled exploration of variations with spatial emphasis.</p>
                                    </div>
                                            <div class="parameter-item">
                                                <div class="parameter-name">Color Intensity: The Chromatic Influence</div>
                                                <p>Color Intensity determines how strongly the selected color scheme affects the noise pattern's structure and influence.</p>
                                                <ul>
                                                    <li><strong>Note:</strong> Similar to Shape Mask Strength, this parameter only takes effect when a color scheme is enabled.</li>
                                                    <li><strong>Small adjustments:</strong> Like shape mask strength, minor tweaks can subtly alter the generated outcome by changing how color information is integrated into the noise structure.</li>
                                                </ul>
                                                <p><em>Effect on Generation:</em> Provides nuanced control over how color information within the noise influences the diffusion process, allowing for exploration of variations with different color-driven structural emphasis.</p>
                                    </div>
</div>
                                    </div>

                                <div class="golden-divider"></div>

                                    <div class="card">
                                        <h3>🌠 Potential of Guided Exploration</h3>
                                        <p>This methodical approach to navigating latent space using shader parameters can offer interesting possibilities:</p>
                                        <ul style="list-style-type: disc; padding-left: 20px;">
                                            <li><strong>Latent Space Cartography:</strong> You're not just generating images; you're effectively creating a map of the conceptual territory surrounding your chosen seed. Different parameter combinations become 'landmarks,' helping you develop an intuition for navigating towards specific aesthetic effects or thematic elements.</li>
                                            <li><strong>Persistent Identities in Variation:</strong> The remarkable tendency for similar elements (characters, objects, styles) to persist across various parameter adjustments suggests that certain semantic features are robustly encoded within the latent space neighborhood of a seed. This offers glimpses into how the model 'conceptualizes' and relates visual information.</li>
                                            <li><strong>Discovering "Hidden Gems":</strong> Traditional random seed exploration, by its nature, can easily miss nuanced or specific variations that exist in the 'gaps' between broadly different seeds. Deliberate navigation with shader parameters allows for the systematic discovery of these unique possibilities that random sampling might statistically overlook.</li>
                                            <li><strong>Creative Control Meets Serendipity:</strong> This methodology provides a powerful balance between intentional artistic direction and the joy of unexpected discovery. You're not rigidly dictating the output, nor are you entirely at the mercy of randomness; instead, you're skillfully steering your exploration through a rich space of possibilities.</li>
                                            <li><strong>Frequency Domain Exploration (Implicit):</strong> Given the mathematical underpinnings of shader noise (like Perlin, Fractal, etc.), you are, in essence, exploring how different frequency characteristics and patterns within the latent space map to tangible semantic features and visual styles in the generated images. This can reveal fundamental patterns in how the model encodes and interprets visual information.</li>
                                        </ul>
                                        <p style="text-align: center; font-style: italic; margin-top: 1.5rem; opacity: 0.9;">
                                            "The true power lies in shifting from 'random sampling' to 'deliberate exploration'—<br>empowering artists to develop an intuition for navigating towards desired variations,<br> rather than merely hoping to stumble upon them."
                                        </p>
                                    </div>
                                </section>

                                <div class="golden-divider"></div>

                                <section id="conclusion">
                                    <h2>🌅 Conclusion <span class="sacred-symbol">☯</span> The Path of Creation</h2>
                                    <p>ShaderNoiseKSampler is a bridge between the mathematical sublime and practical artistry. By understanding and harnessing the sacred geometry of shader noise, artists can guide the generative process with precision and intentionality, transforming pure mathematical concepts into tools of creative expression.</p>
                                    <p>It is important to note that this project represents a concept in its infancy—a minimal viable product of ongoing research. Much remains to be explored and understood in this fascinating intersection of mathematical noise patterns and generative art. The techniques and approaches presented here are foundational stepping stones in a journey that has only just begun.</p>
                                    <p>This work constitutes original, independent investigation and implementation—a product of genuine curiosity and hands-on experimentation rather than an adaptation of existing research. The path forward relies on community support, both in collaborative exploration and financial backing, to fuel further innovation. By supporting grassroots research like this, you contribute directly to the advancement of creative technology driven by unfiltered exploration and authentic discovery.</p>
                                    <p>May your journey with ShaderNoiseKSampler be one of discovery, inspiration, and creation—of digital tapestries woven from the very fabric of mathematical reality.</p>
                                    <div style="text-align: center; margin-top: 2rem; font-style: italic; color: var(--tertiary-color);">
                                        "In every shader noise pattern lies a universe of possibility<br>
                                        In every parameter, a doorway to creation"<br>
                                        - The Shader Matrix
                                    </div>
                                </section>
                            </main>
                            
                            <footer>
                                <p>&copy; 2025 Æmotion Studio. All rights reserved. May sacred geometry guide your creations.</p>
                                <div style="margin-top: 10px;">
                                    <a href="https://ko-fi.com/aemotionstudio" target="_blank" rel="noopener noreferrer"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" style="height: 36px; border:0px;"/></a>
</div>
                            </footer>
                        
                            <button id="scroll-top" title="Go to top">↑</button>
                            
                            <!-- Scripts temporarily removed for debugging linter errors -->
                            
                        `; // This line should end the template literal correctly
                        content.innerHTML = treatiseHTML;

                        // Attach listener to the close button *inside* the treatiseHTML
                        const closeButtonInTreatise = content.querySelector('.close-button');
                        if (closeButtonInTreatise) {
                            closeButtonInTreatise.onclick = (e) => {
                                e.stopPropagation();
                                closeModalCleanup();
                            };
                        }
                        
                        modal.appendChild(content);
                        document.body.appendChild(modal);
                        
                        // Call the renderer for noise visualizations
                        if (window.NoiseVisualizer && window.NoiseVisualizer.renderAllInModal) {
                            // Defer to ensure layout is complete and modal content is fully rendered
                            setTimeout(() => {
                                window.NoiseVisualizer.renderAllInModal(content); // 'content' is the div with class 'shader-matrix-treatise'
                            }, 0);
                        } else {
                            console.warn("NoiseVisualizer not found. Ensure noise_visualizer.js is loaded and available on the window object.");
                        }
                        
                        // Initial scroll to top of modal content
                        content.scrollTop = 0;

                        // Activate the scroll-to-top button functionality
                        window.setupScrollTop(content); // Pass the 'content' element

                    });

                    // Add tooltip to the button
                    button.tooltip = "Show Shader Matrix & Documentation (Alt+M)";

                    // Position the button appropriately based on node type (from matrix_button - Copy.js)
                    if (nodeData.name === "ShaderNoiseKsampler") {
                        // For KSampler node, add to a specific section or position
                        if (!button.options) { 
                            button.options = {};
                        }
                        button.options.section = "advanced";

                        // Custom button style for the KSampler node
                        button.label = "📊 Show Shader Matrix";
                    } else {
                        // Default styling for ShaderDisplay (label is already "Show Matrix")
                        // button.label = "Show Matrix"; // No change needed if initialized with "Show Matrix"
                    }

                    // Set button appearance (from matrix_button - Copy.js)
                    button.name = "📊 Show Shader Matrix";
                    button.serialize = false; // Don't include in serialization
                    
                    // For ShaderDisplay, the button will use default positioning.
                    // For ShaderNoiseKsampler, without a section, it should append after other widgets.

                };
                
                // Call addMatrixButton conditionally (from matrix_button - Copy.js)
                if (self.constructor.type_name === "ShaderNoiseKsampler") {
                    addMatrixButton();
                } else {
                    // For other nodes, add after a small delay to ensure all widgets are ready
                    setTimeout(addMatrixButton, 50);
                }

                // --- Keybinding Logic for Matrix Button ---
                const triggerMatrixButton = () => {
                    const matrixButtonWidget = self.widgets.find(w => w.name === "📊 Show Shader Matrix" && w.type === "button");
                    if (matrixButtonWidget && typeof matrixButtonWidget.callback === 'function') {
                        matrixButtonWidget.callback.call(matrixButtonWidget.value, app.canvas, self, null, null);
                    } else {
                        console.warn("Matrix button widget not found or callback is not a function for Alt+M.");
                    }
                };

                const handleMatrixKeyDown = (event) => {
                    if (event.altKey && event.key.toLowerCase() === 'm') {
                        if (app.canvas && (app.canvas.current_node === self || (app.canvas.selected_nodes && app.canvas.selected_nodes[self.id]))) {
                            if (document.activeElement && (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA' || document.activeElement.isContentEditable)) {
                                return; // Don't interfere with text input
                            }
                            console.log("Alt+M detected for current/selected node to show matrix.");
                            event.preventDefault();
                            event.stopPropagation();
                            triggerMatrixButton();
                        }
                    }
                };

                document.addEventListener('keydown', handleMatrixKeyDown);
                self.handleMatrixButtonKeyDown = handleMatrixKeyDown; // Store for removal

                const originalOnRemoved = self.onRemoved;
                self.onRemoved = function() {
                    if (self.handleMatrixButtonKeyDown) {
                        document.removeEventListener('keydown', self.handleMatrixButtonKeyDown);
                        delete self.handleMatrixButtonKeyDown;
                        console.log("Removed Alt+M keydown listener for matrix button on node:", self.id);
                    }
                    if (originalOnRemoved) {
                        originalOnRemoved.apply(self, arguments);
                    }
                };
                // --- End Keybinding Logic ---
            }
        }
    });
})(); 