/**
 * comfy_utils.ts
 * Shared ComfyUI utility functions used across widgets
 */

// @ts-ignore - Runtime ComfyUI import
import { app } from "../../../scripts/app.js";
// @ts-ignore - Runtime ComfyUI import
import { api } from "../../../scripts/api.js";

/**
 * Common interface for image data with filename, type, and subfolder
 * filename is optional - the function returns empty string if missing
 */
export interface ImageData {
    filename?: string;
    type?: string;
    subfolder?: string;
}

/**
 * Converts image data to a ComfyUI API URL
 * @param data - Image data containing filename, type, and subfolder
 * @returns API URL for the image, or empty string if data is invalid
 */
export function imageDataToUrl(data: ImageData | null | undefined): string {
    if (!data || !data.filename) {
        return "";
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const apiObj = api as any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const appObj = app as any;
    return apiObj.apiURL(
        `/view?filename=${encodeURIComponent(data.filename)}&type=${encodeURIComponent(data.type || "")}&subfolder=${encodeURIComponent(data.subfolder || "")}${appObj.getPreviewFormatParam()}${appObj.getRandParam()}`
    );
}
