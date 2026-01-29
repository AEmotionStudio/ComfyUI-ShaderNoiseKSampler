/**
 * Unit tests for Matrix Button window utilities
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Import the module to populate window utilities
import '../src/matrix_button';

describe('Matrix Button Window Utilities', () => {
    beforeEach(() => {
        // Reset DOM state
        document.body.innerHTML = '';
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    describe('window.scrollToSection', () => {
        it('should be defined on window', () => {
            expect(typeof window.scrollToSection).toBe('function');
        });

        it('should handle missing modal content gracefully', () => {
            expect(() => {
                window.scrollToSection('test-section');
            }).not.toThrow();
        });

        it('should scroll to section when modal exists', () => {
            // Create modal structure
            const modal = document.createElement('div');
            modal.className = 'shader-matrix-treatise';

            const section = document.createElement('section');
            section.id = 'test-section';
            section.scrollIntoView = vi.fn();

            const header = document.createElement('h2');
            header.focus = vi.fn();
            section.appendChild(header);

            modal.appendChild(section);
            document.body.appendChild(modal);

            window.scrollToSection('test-section');

            expect(section.scrollIntoView).toHaveBeenCalledWith({ behavior: 'smooth' });
        });
    });

    describe('window.showTab', () => {
        it('should be defined on window', () => {
            expect(typeof window.showTab).toBe('function');
        });

        it('should handle missing tabs container gracefully', () => {
            expect(() => {
                window.showTab('tab-1', null);
            }).not.toThrow();
        });

        it('should switch tabs correctly', () => {
            // Create tab structure
            const modal = document.createElement('div');
            modal.className = 'shader-matrix-treatise';

            const tabsContainer = document.createElement('div');
            tabsContainer.className = 'tabs';

            const tab1 = document.createElement('button');
            tab1.className = 'tab active';
            tab1.setAttribute('aria-selected', 'true');
            tabsContainer.appendChild(tab1);

            const tab2 = document.createElement('button');
            tab2.className = 'tab';
            tab2.setAttribute('aria-selected', 'false');
            tabsContainer.appendChild(tab2);

            modal.appendChild(tabsContainer);

            const content1 = document.createElement('div');
            content1.className = 'tab-content active';
            content1.id = 'tab-1';
            content1.style.display = 'block';
            modal.appendChild(content1);

            const content2 = document.createElement('div');
            content2.className = 'tab-content';
            content2.id = 'tab-2';
            content2.style.display = 'none';
            modal.appendChild(content2);

            document.body.appendChild(modal);

            window.showTab('tab-2', tab2);

            expect(tab2.classList.contains('active')).toBe(true);
            expect(tab2.getAttribute('aria-selected')).toBe('true');
        });
    });

    describe('window.setupScrollTop', () => {
        it('should be defined on window', () => {
            expect(typeof window.setupScrollTop).toBe('function');
        });

        it('should handle null element gracefully', () => {
            expect(() => {
                window.setupScrollTop(null as any);
            }).not.toThrow();
        });

        it('should setup scroll listener', () => {
            const modalContent = document.createElement('div');
            modalContent.className = 'shader-matrix-treatise';
            modalContent.addEventListener = vi.fn();

            const scrollButton = document.createElement('button');
            scrollButton.id = 'scroll-top';
            scrollButton.addEventListener = vi.fn();
            modalContent.appendChild(scrollButton);

            document.body.appendChild(modalContent);

            window.setupScrollTop(modalContent);

            expect(modalContent.addEventListener).toHaveBeenCalledWith('scroll', expect.any(Function));
            expect(scrollButton.addEventListener).toHaveBeenCalledWith('click', expect.any(Function));
        });
    });

    describe('window.copyCodeSection', () => {
        it('should be defined on window', () => {
            expect(typeof window.copyCodeSection).toBe('function');
        });

        it('should handle missing header gracefully', () => {
            const button = document.createElement('button');
            expect(() => {
                window.copyCodeSection(button);
            }).not.toThrow();
        });

        it('should copy code to clipboard', async () => {
            // Mock clipboard
            const mockClipboard = {
                writeText: vi.fn().mockResolvedValue(undefined),
            };
            Object.defineProperty(navigator, 'clipboard', {
                value: mockClipboard,
                configurable: true,
            });

            // Create code block structure
            const container = document.createElement('div');

            const header = document.createElement('div');
            header.className = 'code-block-header';

            const button = document.createElement('button');
            button.dataset.timeoutId = undefined as any;
            header.appendChild(button);

            container.appendChild(header);

            const pre = document.createElement('pre');
            pre.className = 'foldable-content';
            const code = document.createElement('code');
            code.textContent = 'const test = 1;';
            pre.appendChild(code);
            container.appendChild(pre);

            document.body.appendChild(container);

            window.copyCodeSection(button);

            // Wait for clipboard promise
            await new Promise(resolve => setTimeout(resolve, 10));

            expect(mockClipboard.writeText).toHaveBeenCalledWith('const test = 1;');
        });
    });

    describe('window.toggleCodeSection', () => {
        it('should be defined on window', () => {
            expect(typeof window.toggleCodeSection).toBe('function');
        });

        it('should handle missing header gracefully', () => {
            const button = document.createElement('button');
            expect(() => {
                window.toggleCodeSection(button);
            }).not.toThrow();
        });

        it('should toggle code visibility', () => {
            // Create code block structure
            const container = document.createElement('div');

            const header = document.createElement('div');
            header.className = 'code-block-header';

            const button = document.createElement('button');
            button.textContent = 'Show';
            header.appendChild(button);

            container.appendChild(header);

            const pre = document.createElement('pre');
            pre.className = 'foldable-content';
            pre.style.display = 'none';
            container.appendChild(pre);

            document.body.appendChild(container);

            window.toggleCodeSection(button);

            expect(pre.style.display).toBe('block');
            expect(button.textContent).toBe('Hide');
            expect(button.getAttribute('aria-expanded')).toBe('true');
        });
    });

    describe('window.handleTabNavigation', () => {
        it('should be defined on window', () => {
            expect(typeof window.handleTabNavigation).toBe('function');
        });

        it('should activate tab on Enter key', () => {
            const tabElement = document.createElement('button');
            tabElement.className = 'tab';
            tabElement.click = vi.fn();

            const event = new KeyboardEvent('keydown', { key: 'Enter' });
            event.preventDefault = vi.fn();

            window.handleTabNavigation(event, tabElement);

            expect(event.preventDefault).toHaveBeenCalled();
            expect(tabElement.click).toHaveBeenCalled();
        });

        it('should activate tab on Space key', () => {
            const tabElement = document.createElement('button');
            tabElement.className = 'tab';
            tabElement.click = vi.fn();

            const event = new KeyboardEvent('keydown', { key: ' ' });
            event.preventDefault = vi.fn();

            window.handleTabNavigation(event, tabElement);

            expect(event.preventDefault).toHaveBeenCalled();
            expect(tabElement.click).toHaveBeenCalled();
        });

        it('should navigate to next tab on ArrowRight', () => {
            const parent = document.createElement('div');

            const tab1 = document.createElement('button');
            tab1.className = 'tab';
            tab1.focus = vi.fn();
            tab1.click = vi.fn();
            parent.appendChild(tab1);

            const tab2 = document.createElement('button');
            tab2.className = 'tab';
            tab2.focus = vi.fn();
            tab2.click = vi.fn();
            parent.appendChild(tab2);

            const event = new KeyboardEvent('keydown', { key: 'ArrowRight' });
            event.preventDefault = vi.fn();

            window.handleTabNavigation(event, tab1);

            expect(event.preventDefault).toHaveBeenCalled();
            expect(tab2.focus).toHaveBeenCalled();
            expect(tab2.click).toHaveBeenCalled();
        });
    });
});
