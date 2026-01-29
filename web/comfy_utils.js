/**
 * comfy_utils.ts
 * Shared ComfyUI utility functions used across widgets
 */
// @ts-ignore - Runtime ComfyUI import
import { app } from "../../../scripts/app.js";
// @ts-ignore - Runtime ComfyUI import
import { api } from "../../../scripts/api.js";
/**
 * Converts image data to a ComfyUI API URL
 * @param data - Image data containing filename, type, and subfolder
 * @returns API URL for the image, or empty string if data is invalid
 */
export function imageDataToUrl(data) {
    if (!data || !data.filename) {
        return "";
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const apiObj = api;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const appObj = app;
    return apiObj.apiURL(`/view?filename=${encodeURIComponent(data.filename)}&type=${encodeURIComponent(data.type || "")}&subfolder=${encodeURIComponent(data.subfolder || "")}${appObj.getPreviewFormatParam()}${appObj.getRandParam()}`);
}
//# sourceMappingURL=comfy_utils.js.map