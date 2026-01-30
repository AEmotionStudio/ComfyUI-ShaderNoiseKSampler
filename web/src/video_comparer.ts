/**
 * VideoComparer.ts
 * Video comparison widget for ComfyUI with multiple display modes
 */

// @ts-ignore - Runtime ComfyUI import
import { app } from "../../../scripts/app.js";
import type { ComfyApp, ComfyExtension, ComfyNodeData } from "../types/comfyui";
import type { LGraphNode, IWidget, ContextMenuItem } from "../types/litegraph";
import { drawGradientTitle, AnimationCache } from "./golden_eyeball.js";
import { imageDataToUrl } from "./comfy_utils.js";

export { };


// === TYPE DEFINITIONS ===

interface FrameData {
    filename?: string;
    subfolder?: string;
    type?: string;
    frame_index: number;
    data_url?: string;
}

interface VideoData {
    name: string;
    fps: number;
    frames: FrameData[];
    index: number;
    is_video_a?: boolean;
    is_video_b?: boolean;
}

interface VideoWidgetValue {
    video_data: VideoData[];
    difference_data?: Record<string, unknown>;
    histogram_data?: unknown[];
}

interface ComparerProperties {
    comparer_mode: string;
    selected_video: string;
    onionSkinOpacity: number;
    user_resized: boolean;
    [key: string]: unknown;
}

// Custom properties added to HTMLImageElement for tracking load state
type LoadingImage = HTMLImageElement & {
    isLoading?: boolean;
    queued?: boolean;
    failed?: boolean;
};

interface LoadQueueItem {
    frameData: FrameData;
    videoId: string;
    cacheKey: string;
    retryCount: number;
}

interface ComparerNode extends LGraphNode {
    properties: ComparerProperties;
    videoComparerWidget?: VideoComparerWidget;
    layoutWidget?: IWidget;
    videoSelectorWidget?: IWidget;
    onionSkinOpacitySlider?: IWidget;
    isPointerOver?: boolean;
    pointerOverPos?: [number, number];
    selectedVideo?: string;
    updateControlsVisibility?(): void;
    setSize?(size: [number, number]): void;
}

// === CACHE FOR RENDERING OPTIMIZATION ===
const CACHE: AnimationCache = {
    lastTime: 0,
    frameCount: 0,
    frameSkip: 2,
};

// === VIDEO COMPARER WIDGET CLASS ===

class VideoComparerWidget implements IWidget {
    name: string;
    type: string = "video_comparer";
    node: ComparerNode;
    y?: number;
    last_y?: number;

    private _value: VideoWidgetValue = { video_data: [] };
    private loadedFramesA: Record<string, LoadingImage> = {};
    private loadedFramesB: Record<string, LoadingImage> = {};
    private framesA: FrameData[] = [];
    private framesB: FrameData[] = [];
    private currentFrameIndex: number = 0;
    private isPlaying: boolean = false;
    private animationFrame: number | null = null;
    private fps: number = 8;

    private loadingQueue: LoadQueueItem[] = [];
    private activeLoads: Set<string> = new Set();
    private maxConcurrentLoads: number = 3;
    private retryAttempts: number = 3;
    private loadingInProgress: boolean = false;

    private videosA: VideoData[] = [];
    private videosB: VideoData[] = [];
    private currentPairIndex: number = 0;
    private maxPairs: number = 0;

    private currentBatchPage: number = 0;
    private pairsPerPage: number = 2;
    private maxBatchPages: number = 0;

    private isInitialLoading: boolean = false;
    private loadedFrameCount: number = 0;
    private targetLoadCount: number = 0;
    private initialLoadTimer: ReturnType<typeof setTimeout> | null = null;
    private pendingCanvasUpdate: boolean = false;
    private lastFrameTime: number = 0;
    private _drawCallCount: number = 0;
    private canvasUpdateTimer: ReturnType<typeof setTimeout> | null = null;

    constructor(name: string, node: ComparerNode) {
        this.name = name;
        this.node = node;
    }

    set value(v: VideoWidgetValue) {
        this.isPlaying = false;
        this.currentFrameIndex = 0;
        this.stopPlayback();

        const videoData = v.video_data || [];
        const videosA = videoData.filter(video => video.name === "video_a" || video.is_video_a);
        const videosB = videoData.filter(video => video.name === "video_b" || video.is_video_b);

        this.videosA = videosA.map((video, index) => ({
            name: `A${index + 1}`,
            fps: video.fps || 8,
            frames: video.frames || [],
            index
        }));

        this.videosB = videosB.map((video, index) => ({
            name: `B${index + 1}`,
            fps: video.fps || 8,
            frames: video.frames || [],
            index
        }));

        this.maxPairs = Math.max(this.videosA.length, this.videosB.length);
        this.currentPairIndex = 0;
        this.maxBatchPages = Math.ceil(this.maxPairs / this.pairsPerPage);
        this.currentBatchPage = 0;

        const videoA = this.videosA[0] || null;
        const videoB = this.videosB[0] || null;
        this.framesA = videoA ? videoA.frames : [];
        this.framesB = videoB ? videoB.frames : [];
        this.fps = (videoA?.fps) || (videoB?.fps) || 8;

        this.loadedFramesA = {};
        this.loadedFramesB = {};
        this._value = v;

        if (this.framesA.length > 0 || this.framesB.length > 0) {
            this.preloadInitialFrames();
        }

        if (this.node?.updateControlsVisibility) {
            this.node.updateControlsVisibility();
        }
        this.node.setDirtyCanvas(true, false);
    }

    get value(): VideoWidgetValue {
        // Return minimal data for serialization to avoid localStorage quota issues
        // The actual frame data is stored in videosA/videosB properties, not here
        if (!this._value) return { video_data: [] };

        // Return only essential metadata without frame URLs
        const minimalVideoData = (this._value.video_data || []).map((video, idx) => ({
            name: video.name,
            fps: video.fps,
            frame_count: video.frames?.length || 0,
            index: idx,
            // Don't include frames array - it's already loaded in videosA/videosB
            frames: [] as FrameData[]
        }));

        return { video_data: minimalVideoData };
    }

    private preloadInitialFrames(): void {
        this.isInitialLoading = true;
        this.loadedFrameCount = 0;
        const bufferSize = Math.min(5, Math.max(this.framesA.length, this.framesB.length));
        this.targetLoadCount = Math.min(bufferSize, this.framesA.length) + Math.min(bufferSize, this.framesB.length);

        if (this.initialLoadTimer) clearTimeout(this.initialLoadTimer);

        for (let i = 0; i < bufferSize; i++) {
            if (this.framesA.length > i) this.loadFrame(this.framesA[i], "A");
            if (this.framesB.length > i) this.loadFrame(this.framesB[i], "B");
        }

        this.initialLoadTimer = setTimeout(() => {
            this.isInitialLoading = false;
            if (this.pendingCanvasUpdate) {
                this.node.setDirtyCanvas(true, false);
                this.pendingCanvasUpdate = false;
            }
        }, 1000);
    }

    private loadFrame(frameData: FrameData, videoId: string): LoadingImage | null {
        if (!frameData) return null;

        const cacheKey = `${videoId}_${frameData.frame_index}`;
        const cache = videoId === "A" ? this.loadedFramesA : this.loadedFramesB;

        if (cache[cacheKey]?.complete && !cache[cacheKey].failed) return cache[cacheKey];
        if (cache[cacheKey]?.isLoading || cache[cacheKey]?.queued) return cache[cacheKey];

        // Increased cache size to 100 frames to support smooth playback
        if (Object.keys(cache).length > 100) this.cleanupFrameCache(cache, frameData.frame_index);

        if (!this.loadingQueue.find(item => item.cacheKey === cacheKey)) {
            this.loadingQueue.push({ frameData, videoId, cacheKey, retryCount: 0 });
        }

        if (!cache[cacheKey]) {
            const img = new Image() as LoadingImage;
            img.queued = true;
            cache[cacheKey] = img;
        }

        this.processLoadingQueue();
        return cache[cacheKey];
    }

    private processLoadingQueue(): void {
        if (this.loadingInProgress) return;
        this.loadingInProgress = true;

        while (this.loadingQueue.length > 0 && this.activeLoads.size < this.maxConcurrentLoads) {
            const item = this.loadingQueue.shift();
            if (item) this.loadFrameImmediate(item);
        }

        this.loadingInProgress = false;

        if (this.loadingQueue.length > 0) {
            setTimeout(() => this.processLoadingQueue(), 100);
        }
    }

    private loadFrameImmediate(queueItem: LoadQueueItem): void {
        const { frameData, videoId, cacheKey, retryCount } = queueItem;
        const cache = videoId.startsWith("A") ? this.loadedFramesA : this.loadedFramesB;

        this.activeLoads.add(cacheKey);

        const img = (cache[cacheKey] || new Image()) as LoadingImage;
        img.isLoading = true;
        img.queued = false;
        img.failed = false;

        img.onload = () => {
            img.isLoading = false;
            this.activeLoads.delete(cacheKey);

            if (this.isInitialLoading) {
                this.loadedFrameCount++;
                if (this.loadedFrameCount >= this.targetLoadCount || this.loadedFrameCount >= 2) {
                    this.isInitialLoading = false;
                    if (this.initialLoadTimer) clearTimeout(this.initialLoadTimer);
                    this.node.setDirtyCanvas(true, false);
                } else {
                    this.pendingCanvasUpdate = true;
                }
            } else {
                this.debouncedCanvasUpdate();
            }

            setTimeout(() => this.processLoadingQueue(), 10);
        };

        img.onerror = () => {
            img.isLoading = false;
            img.failed = true;
            this.activeLoads.delete(cacheKey);

            if (retryCount < this.retryAttempts) {
                setTimeout(() => {
                    this.loadingQueue.unshift({ ...queueItem, retryCount: retryCount + 1 });
                    this.processLoadingQueue();
                }, 1000 * (retryCount + 1));
            } else {
                delete cache[cacheKey];
            }

            setTimeout(() => this.processLoadingQueue(), 10);
        };

        // Load immediately without delay - 50ms delay was causing blank frames during playback
        if (!img.failed) {
            img.src = frameData.data_url || imageDataToUrl(frameData);
        }

        cache[cacheKey] = img;
    }

    private cleanupFrameCache(cache: Record<string, LoadingImage>, currentFrameIndex: number = 0): void {
        const keys = Object.keys(cache);
        // Sort by distance from current frame, keeping frames near current position
        const sortedKeys = keys.sort((a, b) => {
            const frameA = parseInt(a.split('_')[1]) || 0;
            const frameB = parseInt(b.split('_')[1]) || 0;
            return Math.abs(frameA - currentFrameIndex) - Math.abs(frameB - currentFrameIndex);
        });
        // Remove frames farthest from current position (keep first 50)
        const toRemove = sortedKeys.slice(50);
        toRemove.forEach(key => {
            if (cache[key] && !cache[key].isLoading) delete cache[key];
        });
    }

    private debouncedCanvasUpdate(): void {
        if (this.canvasUpdateTimer) clearTimeout(this.canvasUpdateTimer);
        this.canvasUpdateTimer = setTimeout(() => {
            this.node.setDirtyCanvas(true, false);
        }, 16);
    }

    private getFrameImageForIndex(index: number, videoId: string): LoadingImage | null {
        const frames = videoId === "A" ? this.framesA : this.framesB;
        if (!frames.length || index < 0 || index >= frames.length) return null;
        return this.loadFrame(frames[index], videoId);
    }

    draw(ctx: CanvasRenderingContext2D, node: LGraphNode, width: number, y: number, height: number): void {
        this._drawCallCount++;
        this.y = y;
        this.last_y = y;

        const nodeHeight = node.size[1];
        const availableHeight = nodeHeight - y - 10;
        const mode = (node as ComparerNode).properties?.comparer_mode || "Playback";

        switch (mode) {
            case "Side-by-Side": this.drawSideBySideMode(ctx, y, width, availableHeight); break;
            case "Stacked": this.drawStackedMode(ctx, y, width, availableHeight); break;
            case "Slider": this.drawSliderMode(ctx, y, width, availableHeight); break;
            case "Onion Skin": this.drawOnionSkinMode(ctx, y, width, availableHeight); break;
            case "Sync Compare": this.drawSyncCompareMode(ctx, y, width, availableHeight); break;
            case "Grid": this.drawGridMode(ctx, y, width, availableHeight); break;
            case "Batch": this.drawBatchMode(ctx, y, width, availableHeight); break;
            default: this.drawPlaybackMode(ctx, y, width, availableHeight); break;
        }

        if (["Grid", "Batch"].includes(mode) && this.maxPairs > 1) {
            this.drawBatchControls(ctx, y, width, availableHeight);
        } else {
            this.drawPlaybackControls(ctx, y + availableHeight - 45, width);
        }
    }

    private drawPlaybackMode(ctx: CanvasRenderingContext2D, y: number, width: number, availableHeight: number): void {
        const videoId = this.node.properties?.selected_video || "A";
        const videoFrames = videoId === "A" ? this.framesA : this.framesB;

        if (!videoFrames.length) { this.drawNoVideoMessage(ctx, y, width, availableHeight - 45, videoId); return; }

        const currentFrame = Math.min(this.currentFrameIndex, videoFrames.length - 1);
        const frameImg = this.getFrameImageForIndex(currentFrame, videoId);

        if (this.isInitialLoading && (!frameImg || !frameImg.complete)) {
            this.drawLoadingMessage(ctx, y, width, availableHeight - 45);
        } else if (frameImg?.complete && !frameImg.failed) {
            this.drawFrame(ctx, frameImg, y, width, availableHeight - 45);
            this.drawFrameCounter(ctx, y + 10, width, currentFrame + 1, videoFrames.length, videoId);
        } else {
            this.drawLoadingMessage(ctx, y, width, availableHeight - 45);
        }
    }

    private drawSideBySideMode(ctx: CanvasRenderingContext2D, y: number, width: number, availableHeight: number): void {
        const halfWidth = width / 2;
        if (!this.framesA.length && !this.framesB.length) { this.drawNoVideoMessage(ctx, y, width, availableHeight, "both"); return; }

        const currentFrameA = Math.min(this.currentFrameIndex, this.framesA.length - 1);
        const currentFrameB = Math.min(this.currentFrameIndex, this.framesB.length - 1);

        if (this.framesA.length) {
            const frameImgA = this.getFrameImageForIndex(currentFrameA, "A");
            if (!this.isInitialLoading && frameImgA?.complete) {
                this.drawFrameInRegion(ctx, frameImgA, y, 0, halfWidth, availableHeight);
                this.drawFrameCounter(ctx, y + 10, halfWidth, currentFrameA + 1, this.framesA.length, "A");
            } else {
                this.drawLoadingMessage(ctx, y, halfWidth, availableHeight, 0);
            }
        } else {
            this.drawNoVideoMessage(ctx, y, halfWidth, availableHeight, "A", 0);
        }

        if (this.framesB.length) {
            const frameImgB = this.getFrameImageForIndex(currentFrameB, "B");
            if (!this.isInitialLoading && frameImgB?.complete) {
                this.drawFrameInRegion(ctx, frameImgB, y, halfWidth, halfWidth, availableHeight);
                this.drawFrameCounter(ctx, y + 10, halfWidth, currentFrameB + 1, this.framesB.length, "B", halfWidth);
            } else {
                this.drawLoadingMessage(ctx, y, halfWidth, availableHeight, halfWidth);
            }
        } else {
            this.drawNoVideoMessage(ctx, y, halfWidth, availableHeight, "B", halfWidth);
        }

        ctx.beginPath();
        ctx.moveTo(halfWidth, y);
        ctx.lineTo(halfWidth, y + availableHeight - 45);
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    private drawStackedMode(ctx: CanvasRenderingContext2D, y: number, width: number, availableHeight: number): void {
        const halfHeight = (availableHeight - 45) / 2;
        if (!this.framesA.length && !this.framesB.length) { this.drawNoVideoMessage(ctx, y, width, availableHeight, "both"); return; }

        const currentFrameA = Math.min(this.currentFrameIndex, this.framesA.length - 1);
        const currentFrameB = Math.min(this.currentFrameIndex, this.framesB.length - 1);

        if (this.framesA.length) {
            const frameImgA = this.getFrameImageForIndex(currentFrameA, "A");
            if (frameImgA?.complete) {
                this.drawFrameInRegion(ctx, frameImgA, y, 0, width, halfHeight);
                this.drawFrameCounter(ctx, y + 10, width, currentFrameA + 1, this.framesA.length, "A");
            } else { this.drawLoadingMessage(ctx, y, width, halfHeight); }
        } else { this.drawNoVideoMessage(ctx, y, width, halfHeight, "A"); }

        if (this.framesB.length) {
            const frameImgB = this.getFrameImageForIndex(currentFrameB, "B");
            if (frameImgB?.complete) {
                this.drawFrameInRegion(ctx, frameImgB, y + halfHeight, 0, width, halfHeight);
                this.drawFrameCounter(ctx, y + halfHeight + 10, width, currentFrameB + 1, this.framesB.length, "B");
            } else { this.drawLoadingMessage(ctx, y + halfHeight, width, halfHeight); }
        } else { this.drawNoVideoMessage(ctx, y + halfHeight, width, halfHeight, "B"); }

        ctx.beginPath();
        ctx.moveTo(0, y + halfHeight);
        ctx.lineTo(width, y + halfHeight);
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    private drawSliderMode(ctx: CanvasRenderingContext2D, y: number, width: number, availableHeight: number): void {
        if (!this.framesA.length && !this.framesB.length) { this.drawNoVideoMessage(ctx, y, width, availableHeight, "both"); return; }

        const currentFrameA = Math.min(this.currentFrameIndex, this.framesA.length - 1);
        const currentFrameB = Math.min(this.currentFrameIndex, this.framesB.length - 1);

        if (this.framesA.length) {
            const frameImgA = this.getFrameImageForIndex(currentFrameA, "A");
            if (frameImgA?.complete) {
                this.drawFrame(ctx, frameImgA, y, width, availableHeight - 45);
                this.drawFrameCounter(ctx, y + 10, width, currentFrameA + 1, this.framesA.length, "A");
            } else { this.drawLoadingMessage(ctx, y, width, availableHeight - 45); }
        }

        if (this.framesB.length && this.node.isPointerOver) {
            const frameImgB = this.getFrameImageForIndex(currentFrameB, "B");
            if (frameImgB?.complete) {
                const sliderX = this.node.pointerOverPos?.[0] || width / 2;
                ctx.save();
                ctx.beginPath();
                ctx.rect(0, y, sliderX, availableHeight - 45);
                ctx.clip();
                this.drawFrame(ctx, frameImgB, y, width, availableHeight - 45);
                ctx.restore();

                ctx.beginPath();
                ctx.moveTo(sliderX, y);
                ctx.lineTo(sliderX, y + availableHeight - 45);
                ctx.strokeStyle = "rgba(255,255,255,0.8)";
                ctx.lineWidth = 2;
                ctx.stroke();

                ctx.fillStyle = "rgba(0,0,0,0.7)";
                ctx.fillRect(sliderX + 5, y + 10, 20, 20);
                ctx.fillStyle = "white";
                ctx.font = "12px Arial";
                ctx.textAlign = "center";
                ctx.fillText("B", sliderX + 15, y + 24);
            }
        }
    }

    private drawOnionSkinMode(ctx: CanvasRenderingContext2D, y: number, width: number, availableHeight: number): void {
        if (!this.framesA.length && !this.framesB.length) { this.drawNoVideoMessage(ctx, y, width, availableHeight, "both"); return; }

        const currentFrameA = Math.min(this.currentFrameIndex, this.framesA.length - 1);
        const currentFrameB = Math.min(this.currentFrameIndex, this.framesB.length - 1);
        const opacity = this.node.properties?.onionSkinOpacity || 0.5;

        if (this.framesA.length) {
            const frameImgA = this.getFrameImageForIndex(currentFrameA, "A");
            if (frameImgA?.complete) {
                this.drawFrame(ctx, frameImgA, y, width, availableHeight - 45);
                this.drawFrameCounter(ctx, y + 10, width, currentFrameA + 1, this.framesA.length, "A");
            } else { this.drawLoadingMessage(ctx, y, width, availableHeight - 45); }
        }

        if (this.framesB.length) {
            const frameImgB = this.getFrameImageForIndex(currentFrameB, "B");
            if (frameImgB?.complete) {
                ctx.save();
                ctx.globalAlpha = opacity;
                this.drawFrame(ctx, frameImgB, y, width, availableHeight - 45);
                ctx.restore();

                ctx.fillStyle = "rgba(0,0,0,0.7)";
                ctx.fillRect(width - 30, y + 10, 20, 20);
                ctx.fillStyle = "white";
                ctx.font = "12px Arial";
                ctx.textAlign = "center";
                ctx.fillText("B", width - 20, y + 24);
            }
        }
    }

    private drawSyncCompareMode(ctx: CanvasRenderingContext2D, y: number, width: number, availableHeight: number): void {
        const selectedVideo = this.node.selectedVideo || "A";
        const frames = selectedVideo === "A" ? this.framesA : this.framesB;

        if (!frames.length) { this.drawNoVideoMessage(ctx, y, width, availableHeight, selectedVideo); return; }

        const currentFrame = Math.min(this.currentFrameIndex, frames.length - 1);
        const frameImg = this.getFrameImageForIndex(currentFrame, selectedVideo);

        if (frameImg?.complete) {
            this.drawFrame(ctx, frameImg, y, width, availableHeight - 45);
            this.drawFrameCounter(ctx, y + 10, width, currentFrame + 1, frames.length, selectedVideo);

            ctx.fillStyle = "rgba(0,0,0,0.7)";
            ctx.fillRect(width / 2 - 60, y + availableHeight - 65, 120, 24);
            ctx.fillStyle = "white";
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            ctx.fillText("Click to toggle A/B", width / 2, y + availableHeight - 49);
        } else {
            this.drawLoadingMessage(ctx, y, width, availableHeight - 45);
        }
    }

    private drawGridMode(ctx: CanvasRenderingContext2D, y: number, width: number, availableHeight: number): void {
        // Simplified grid - draw first 4 pairs in 2x2 grid
        const cols = 2, rows = 2;
        const cellWidth = width / cols, cellHeight = (availableHeight - 30) / rows;

        for (let i = 0; i < Math.min(this.maxPairs, 4); i++) {
            const col = i % cols, row = Math.floor(i / cols);
            const x = col * cellWidth, cellY = y + row * cellHeight;

            const videoA = this.videosA[i], videoB = this.videosB[i];
            if (videoA?.frames.length) {
                const frame = this.loadFrame(videoA.frames[this.currentFrameIndex % videoA.frames.length], `A_${i}`);
                if (frame?.complete) this.drawFrameInRegion(ctx, frame, cellY, x, cellWidth / 2, cellHeight);
            }
            if (videoB?.frames.length) {
                const frame = this.loadFrame(videoB.frames[this.currentFrameIndex % videoB.frames.length], `B_${i}`);
                if (frame?.complete) this.drawFrameInRegion(ctx, frame, cellY, x + cellWidth / 2, cellWidth / 2, cellHeight);
            }
        }
    }

    private drawBatchMode(ctx: CanvasRenderingContext2D, y: number, width: number, availableHeight: number): void {
        const startPair = this.currentBatchPage * this.pairsPerPage;
        const pairHeight = (availableHeight - 30) / this.pairsPerPage;

        for (let i = 0; i < this.pairsPerPage && startPair + i < this.maxPairs; i++) {
            const pairIdx = startPair + i;
            const pairY = y + i * pairHeight;

            const videoA = this.videosA[pairIdx], videoB = this.videosB[pairIdx];
            if (videoA?.frames.length) {
                const frame = this.loadFrame(videoA.frames[this.currentFrameIndex % videoA.frames.length], `A_${pairIdx}`);
                if (frame?.complete) this.drawFrameInRegion(ctx, frame, pairY, 0, width / 2, pairHeight);
            }
            if (videoB?.frames.length) {
                const frame = this.loadFrame(videoB.frames[this.currentFrameIndex % videoB.frames.length], `B_${pairIdx}`);
                if (frame?.complete) this.drawFrameInRegion(ctx, frame, pairY, width / 2, width / 2, pairHeight);
            }
        }
    }

    private drawFrame(ctx: CanvasRenderingContext2D, img: HTMLImageElement, y: number, width: number, availableHeight: number): void {
        if (!img?.complete) return;
        const imageAspect = img.naturalWidth / img.naturalHeight;
        const canvasAspect = width / availableHeight;
        let drawWidth: number, drawHeight: number, offsetX = 0, offsetY = 0;

        if (imageAspect > canvasAspect) {
            drawWidth = width;
            drawHeight = width / imageAspect;
            offsetY = (availableHeight - drawHeight) / 2;
        } else {
            drawHeight = availableHeight;
            drawWidth = availableHeight * imageAspect;
            offsetX = (width - drawWidth) / 2;
        }

        ctx.drawImage(img, offsetX, y + offsetY, drawWidth, drawHeight);
    }

    private drawFrameInRegion(ctx: CanvasRenderingContext2D, img: HTMLImageElement, y: number, x: number, regionWidth: number, regionHeight: number): void {
        if (!img?.complete) return;
        const imageAspect = img.naturalWidth / img.naturalHeight;
        const regionAspect = regionWidth / regionHeight;
        let drawWidth: number, drawHeight: number, offsetX = 0, offsetY = 0;

        if (imageAspect > regionAspect) {
            drawWidth = regionWidth;
            drawHeight = regionWidth / imageAspect;
            offsetY = (regionHeight - drawHeight) / 2;
        } else {
            drawHeight = regionHeight;
            drawWidth = regionHeight * imageAspect;
            offsetX = (regionWidth - drawWidth) / 2;
        }

        ctx.drawImage(img, x + offsetX, y + offsetY, drawWidth, drawHeight);
    }

    private drawFrameCounter(ctx: CanvasRenderingContext2D, y: number, width: number, current: number, total: number, videoId: string, offsetX = 0): void {
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(offsetX + 10, y, 80, 24);
        ctx.fillStyle = "white";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        ctx.fillText(`${videoId}: ${current}/${total}`, offsetX + 15, y + 16);
    }

    private drawNoVideoMessage(ctx: CanvasRenderingContext2D, y: number, width: number, height: number, videoId: string, offsetX = 0): void {
        ctx.fillStyle = "rgba(0,0,0,0.3)";
        ctx.fillRect(offsetX, y, width, height);
        ctx.fillStyle = "white";
        ctx.font = "14px Arial";
        ctx.textAlign = "center";
        ctx.fillText(videoId === "both" ? "No videos available" : `No video ${videoId} available`, offsetX + width / 2, y + height / 2);
    }

    private drawLoadingMessage(ctx: CanvasRenderingContext2D, y: number, width: number, height: number, offsetX = 0): void {
        ctx.fillStyle = "rgba(0,0,0,0.3)";
        ctx.fillRect(offsetX, y, width, height);
        ctx.fillStyle = "white";
        ctx.font = "14px Arial";
        ctx.textAlign = "center";
        ctx.fillText("Loading frames...", offsetX + width / 2, y + height / 2);
    }

    private drawPlaybackControls(ctx: CanvasRenderingContext2D, y: number, width: number): void {
        const controlHeight = 45;
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(0, y, width, controlHeight);

        const buttonSize = 36;
        const playPauseX = 12;
        const sliderStart = playPauseX + buttonSize + 8;
        const sliderWidth = width - sliderStart - 68;

        ctx.fillStyle = "rgba(100,100,100,0.8)";
        const buttonY = y + (controlHeight - buttonSize) / 2;
        ctx.beginPath();
        (ctx as any).roundRect(playPauseX, buttonY, buttonSize, buttonSize, 4);
        ctx.fill();

        ctx.fillStyle = "white";
        ctx.font = "20px Arial";
        ctx.textAlign = "center";
        ctx.fillText(this.isPlaying ? "⏸" : "▶", playPauseX + buttonSize / 2, y + controlHeight / 2 + 7);

        const trackHeight = 8;
        ctx.fillStyle = "rgba(60,60,60,0.8)";
        ctx.beginPath();
        (ctx as any).roundRect(sliderStart, y + controlHeight / 2 - trackHeight / 2, sliderWidth, trackHeight, trackHeight / 2);
        ctx.fill();

        const totalFrames = Math.max(this.framesA.length, this.framesB.length);
        if (totalFrames > 0) {
            const progress = this.currentFrameIndex / (totalFrames - 1);
            const scrubberPos = sliderStart + progress * sliderWidth;
            const scrubberRadius = 10;

            ctx.fillStyle = "rgba(0,0,0,0.3)";
            ctx.beginPath();
            ctx.arc(scrubberPos + 1, y + controlHeight / 2 + 1, scrubberRadius, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = "white";
            ctx.beginPath();
            ctx.arc(scrubberPos, y + controlHeight / 2, scrubberRadius, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = "white";
            ctx.font = "14px Arial";
            ctx.textAlign = "right";
            ctx.fillText(`${this.currentFrameIndex + 1}/${totalFrames}`, width - 12, y + controlHeight / 2 + 5);
        }
    }

    private drawBatchControls(ctx: CanvasRenderingContext2D, y: number, width: number, availableHeight: number): void {
        const controlY = y + availableHeight - 30;
        ctx.save();
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(0, controlY, width, 30);
        ctx.fillStyle = "white";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        ctx.fillText(`Frame: ${this.currentFrameIndex + 1}`, 10, controlY + 18);

        const buttonWidth = 60, buttonHeight = 20, buttonY = controlY + 5;
        ctx.fillStyle = this.isPlaying ? "rgba(200,100,100,0.8)" : "rgba(100,200,100,0.8)";
        ctx.fillRect(80, buttonY, buttonWidth, buttonHeight);
        ctx.fillStyle = "white";
        ctx.textAlign = "center";
        ctx.fillText(this.isPlaying ? "⏸ Pause" : "▶ Play", 80 + buttonWidth / 2, buttonY + 14);
        ctx.restore();
    }

    startPlayback(): void {
        if (this.isPlaying) return;
        this.isPlaying = true;
        this.isInitialLoading = false;
        this.lastFrameTime = performance.now();

        const playbackLoop = (): void => {
            const now = performance.now();
            if (now - this.lastFrameTime >= 1000 / this.fps) {
                this.advanceFrame();
                this.lastFrameTime = now;
            }
            if (this.isPlaying) this.animationFrame = requestAnimationFrame(playbackLoop);
        };

        this.animationFrame = requestAnimationFrame(playbackLoop);
        this.node.setDirtyCanvas(true, false);
    }

    stopPlayback(): void {
        if (!this.isPlaying) return;
        this.isPlaying = false;
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        this.node.setDirtyCanvas(true, false);
    }

    togglePlayback(): void {
        this.isPlaying ? this.stopPlayback() : this.startPlayback();
    }

    advanceFrame(): void {
        const totalFrames = Math.max(this.framesA.length, this.framesB.length);
        if (totalFrames === 0) return;
        this.currentFrameIndex = (this.currentFrameIndex + 1) % totalFrames;

        // Preload upcoming frames for smooth playback (look-ahead buffer)
        this.preloadUpcomingFrames(10);

        this.node.setDirtyCanvas(true, false);
    }

    private preloadUpcomingFrames(count: number): void {
        const totalFramesA = this.framesA.length;
        const totalFramesB = this.framesB.length;

        for (let i = 1; i <= count; i++) {
            // Preload A frames
            if (totalFramesA > 0) {
                const frameIdx = (this.currentFrameIndex + i) % totalFramesA;
                if (this.framesA[frameIdx]) {
                    this.loadFrame(this.framesA[frameIdx], "A");
                }
            }
            // Preload B frames
            if (totalFramesB > 0) {
                const frameIdx = (this.currentFrameIndex + i) % totalFramesB;
                if (this.framesB[frameIdx]) {
                    this.loadFrame(this.framesB[frameIdx], "B");
                }
            }
        }
    }

    previousFrame(): void {
        const totalFrames = Math.max(this.framesA.length, this.framesB.length);
        if (totalFrames === 0) return;
        this.currentFrameIndex = (this.currentFrameIndex - 1 + totalFrames) % totalFrames;
        this.node.setDirtyCanvas(true, false);
    }

    nextFrame(): void { this.advanceFrame(); }

    mouse(event: MouseEvent, pos: [number, number], node: LGraphNode): boolean {
        // Note: onMouseDown handler only calls this for mousedown/pointerdown events

        const mode = (node as ComparerNode).properties?.comparer_mode;
        if (mode === "Sync Compare") {
            (node as ComparerNode).selectedVideo = (node as ComparerNode).selectedVideo === "A" ? "B" : "A";
            node.setDirtyCanvas(true, false);
            return true;
        }

        // Check playback controls
        const widgetY = this.y || 0;
        const availableHeight = node.size[1] - widgetY - 10;
        const controlsY = widgetY + availableHeight - 45;

        if (pos[1] >= controlsY && pos[1] <= controlsY + 45) {
            if (pos[0] >= 12 && pos[0] <= 48) {
                this.togglePlayback();
                return true;
            }

            const sliderStart = 56, sliderWidth = node.size[0] - sliderStart - 68;
            if (pos[0] >= sliderStart && pos[0] <= sliderStart + sliderWidth) {
                const totalFrames = Math.max(this.framesA.length, this.framesB.length);
                if (totalFrames > 0) {
                    const progress = (pos[0] - sliderStart) / sliderWidth;
                    this.currentFrameIndex = Math.floor(progress * (totalFrames - 1));
                    node.setDirtyCanvas(true, false);
                    return true;
                }
            }
        }

        return false;
    }

    computeSize(width: number): [number, number] {
        return [width, 300];
    }

    onRemoved(): void {
        this.stopPlayback();
        if (this.initialLoadTimer) clearTimeout(this.initialLoadTimer);
        if (this.canvasUpdateTimer) clearTimeout(this.canvasUpdateTimer);
        this.loadedFramesA = {};
        this.loadedFramesB = {};
    }
}

// === COMFYUI EXTENSION REGISTRATION ===

(app as any).registerExtension({
    name: "VideoComparer",
    async beforeRegisterNodeDef(nodeType: any, nodeData: ComfyNodeData, _app: ComfyApp) {
        if (nodeData.name !== "Video Comparer" && nodeData.name !== "VideoComparer") return;

        nodeType.prototype.properties = nodeType.prototype.properties || {};
        nodeType.prototype.properties.comparer_mode = "Playback";
        nodeType.prototype.properties.selected_video = "A";
        nodeType.prototype.properties.onionSkinOpacity = 0.5;
        nodeType.prototype.properties.user_resized = false;

        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (this: ComparerNode, ctx: CanvasRenderingContext2D) {
            if (origOnDrawForeground) origOnDrawForeground.apply(this, arguments);
            drawGradientTitle(this, ctx, CACHE, 6);
        };

        const origOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (this: ComparerNode) {
            if (origOnRemoved) origOnRemoved.apply(this, arguments);
            if (this.videoComparerWidget) this.videoComparerWidget.onRemoved();
            // CACHE only holds animation timing state, no cleanup needed
        };

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (this: ComparerNode) {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);

            this.properties = this.properties || {} as ComparerProperties;
            this.properties.comparer_mode = this.properties.comparer_mode || "Playback";
            this.properties.selected_video = this.properties.selected_video || "A";
            this.properties.onionSkinOpacity = this.properties.onionSkinOpacity ?? 0.5;

            this.isPointerOver = false;
            this.pointerOverPos = [0, 0];
            this.selectedVideo = "A";

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.layoutWidget = this.addWidget("combo", "Comparison Mode", this.properties.comparer_mode, ((value: string) => {
                this.properties.comparer_mode = value;
                if (this.updateControlsVisibility) this.updateControlsVisibility();
                this.setDirtyCanvas(true, false);
            }) as any, { values: ["Playback", "Side-by-Side", "Stacked", "Slider", "Onion Skin", "Sync Compare"] });

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.videoSelectorWidget = this.addWidget("combo", "Video", this.properties.selected_video, ((value: string) => {
                this.properties.selected_video = value;
                this.selectedVideo = value;
                this.setDirtyCanvas(true, false);
            }) as any, { values: ["A", "B"] });

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.onionSkinOpacitySlider = this.addWidget("slider", "Opacity B", this.properties.onionSkinOpacity, ((value: number) => {
                this.properties.onionSkinOpacity = value;
                this.setDirtyCanvas(true, false);
            }) as any, { min: 0.0, max: 1.0, step: 0.01 });

            this.videoComparerWidget = this.addCustomWidget(new VideoComparerWidget("video_comparer", this)) as any as VideoComparerWidget;

            if (this.updateControlsVisibility) this.updateControlsVisibility();

            const initialSize = this.computeSize?.() || [400, 300];
            if (this.setSize) this.setSize([Math.max(400, initialSize[0]), Math.max(300, initialSize[1])]);
            this.setDirtyCanvas(true, true);
        };

        nodeType.prototype.updateControlsVisibility = function (this: ComparerNode) {
            const mode = this.properties.comparer_mode;
            if (this.videoSelectorWidget) (this.videoSelectorWidget as any).hidden = mode !== "Playback";
            if (this.onionSkinOpacitySlider) (this.onionSkinOpacitySlider as any).hidden = mode !== "Onion Skin";
        };

        const origComputeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function (this: ComparerNode, out?: [number, number]): [number, number] {
            const size = origComputeSize ? origComputeSize.apply(this, arguments) : [400, 300];
            if (this.videoComparerWidget) {
                const widgetSize = this.videoComparerWidget.computeSize(size[0]);
                let extraHeight = 60;
                if (this.properties.comparer_mode === "Playback" || this.properties.comparer_mode === "Onion Skin") extraHeight += 30;
                size[1] = Math.max(size[1], widgetSize[1] + extraHeight);
            }
            return size;
        };

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (this: ComparerNode, message: any) {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);

            if (!message || typeof message !== 'object') return message;

            let videoData = message.ui?.video_data || message.video_data;
            if (videoData && this.videoComparerWidget) {
                this.videoComparerWidget.value = {
                    video_data: videoData,
                    difference_data: message.ui?.difference_data || {},
                    histogram_data: message.ui?.histogram_data || []
                };
                this.setDirtyCanvas(true, true);
            }

            return message;
        };

        nodeType.prototype.onMouseDown = function (this: ComparerNode, event: MouseEvent, pos: [number, number]) {
            return this.videoComparerWidget?.mouse(event, pos, this) || false;
        };

        nodeType.prototype.onMouseEnter = function (this: ComparerNode) {
            this.isPointerOver = true;
            this.setDirtyCanvas(true, false);
        };

        nodeType.prototype.onMouseLeave = function (this: ComparerNode) {
            this.isPointerOver = false;
            this.setDirtyCanvas(true, false);
        };

        nodeType.prototype.onMouseMove = function (this: ComparerNode, event: MouseEvent, pos: [number, number]) {
            this.pointerOverPos = pos;
            if (this.properties.comparer_mode === "Slider") this.setDirtyCanvas(true, false);
            return false;
        };

        nodeType.prototype.onKeyDown = function (this: ComparerNode, event: KeyboardEvent) {
            if (!this.videoComparerWidget) return false;
            switch (event.key) {
                case "ArrowLeft": this.videoComparerWidget.previousFrame(); return true;
                case "ArrowRight": this.videoComparerWidget.nextFrame(); return true;
                case " ": this.videoComparerWidget.togglePlayback(); event.preventDefault(); return true;
            }
            return false;
        };

        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (this: ComparerNode, _: any, options: ContextMenuItem[]) {
            if (origGetExtraMenuOptions) origGetExtraMenuOptions.apply(this, arguments);

            options.push(null as any);
            const currentMode = this.properties.comparer_mode || "Playback";
            const modes = ["Playback", "Side-by-Side", "Stacked", "Slider", "Onion Skin", "Sync Compare"];

            options.push({
                content: "Comparison Mode",
                has_submenu: true,
                submenu: {
                    options: modes.map(mode => ({
                        content: mode === currentMode ? `✓ ${mode}` : mode,
                        callback: () => {
                            this.properties.comparer_mode = mode;
                            if (this.layoutWidget) this.layoutWidget.value = mode;
                            if (this.updateControlsVisibility) this.updateControlsVisibility();
                            this.setDirtyCanvas(true, false);
                        }
                    }))
                }
            });

            options.push({
                content: "Reset to Default Size",
                callback: () => {
                    this.properties.user_resized = false;
                    if (this.setSize && this.computeSize) this.setSize(this.computeSize());
                    this.setDirtyCanvas(true, false);
                }
            });
        };

    }
} as ComfyExtension);
