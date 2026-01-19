## 2025-05-22 - ComfyUI Manual Modals
**Learning:** ComfyUI custom nodes often inject raw HTML for modals via JS, bypassing standard UI libraries. This frequently leads to missing ARIA roles and focus management.
**Action:** When working on ComfyUI extensions, always check `document.createElement` calls for missing `role="dialog"`, `aria-modal`, and manual focus handling.

## 2025-05-23 - Dynamic Content in Raw HTML Modals
**Learning:** When using raw HTML strings for modals in ComfyUI extensions, dynamic content toggles (like "Show/Hide Code") often lack state indication (`aria-expanded`) and relationship linking (`aria-controls`), as they rely on simple onclick handlers.
**Action:** Ensure all toggle buttons in template strings include unique IDs for target content and `aria-expanded`/`aria-controls` attributes, with JS handlers updating the state.

## 2025-05-24 - In-Page Navigation Focus Management
**Learning:** In single-page documentation modals (common in custom nodes), in-page navigation buttons (`scrollToSection`) often scroll content but leave focus on the button. This forces keyboard users to traverse the entire menu again to reach the content.
**Action:** Update scroll handlers to programmatically move focus to the target section's heading (setting `tabindex="-1"` if needed) to maintain logical reading flow.
