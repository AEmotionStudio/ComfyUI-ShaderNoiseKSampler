## 2025-05-22 - ComfyUI Manual Modals
**Learning:** ComfyUI custom nodes often inject raw HTML for modals via JS, bypassing standard UI libraries. This frequently leads to missing ARIA roles and focus management.
**Action:** When working on ComfyUI extensions, always check `document.createElement` calls for missing `role="dialog"`, `aria-modal`, and manual focus handling.
