/**
 * VideoComparer.ts
 * Video comparison widget for ComfyUI with multiple display modes
 */
// @ts-ignore - Runtime ComfyUI import
import { app } from "../../../scripts/app.js";
import { drawGradientTitle } from "./golden_eyeball.js";
import { imageDataToUrl } from "./comfy_utils.js";
// === CACHE FOR RENDERING OPTIMIZATION ===
const CACHE = {
    lastTime: 0,
    frameCount: 0,
    frameSkip: 2,
};
// === VIDEO COMPARER WIDGET CLASS ===
class VideoComparerWidget {
    constructor(name, node) {
        this.type = "video_comparer";
        this._value = { video_data: [] };
        this.loadedFramesA = {};
        this.loadedFramesB = {};
        this.framesA = [];
        this.framesB = [];
        this.currentFrameIndex = 0;
        this.isPlaying = false;
        this.animationFrame = null;
        this.fps = 8;
        this.loadingQueue = [];
        this.activeLoads = new Set();
        this.maxConcurrentLoads = 3;
        this.retryAttempts = 3;
        this.loadingInProgress = false;
        this.videosA = [];
        this.videosB = [];
        this.currentPairIndex = 0;
        this.maxPairs = 0;
        this.currentBatchPage = 0;
        this.pairsPerPage = 2;
        this.maxBatchPages = 0;
        this.isInitialLoading = false;
        this.loadedFrameCount = 0;
        this.targetLoadCount = 0;
        this.initialLoadTimer = null;
        this.pendingCanvasUpdate = false;
        this.lastFrameTime = 0;
        this._drawCallCount = 0;
        this.canvasUpdateTimer = null;
        this.name = name;
        this.node = node;
    }
    set value(v) {
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
    get value() {
        // Return minimal data for serialization to avoid localStorage quota issues
        // The actual frame data is stored in videosA/videosB properties, not here
        if (!this._value)
            return { video_data: [] };
        // Return only essential metadata without frame URLs
        const minimalVideoData = (this._value.video_data || []).map((video, idx) => ({
            name: video.name,
            fps: video.fps,
            frame_count: video.frames?.length || 0,
            index: idx,
            // Don't include frames array - it's already loaded in videosA/videosB
            frames: []
        }));
        return { video_data: minimalVideoData };
    }
    preloadInitialFrames() {
        this.isInitialLoading = true;
        this.loadedFrameCount = 0;
        const bufferSize = Math.min(5, Math.max(this.framesA.length, this.framesB.length));
        this.targetLoadCount = Math.min(bufferSize, this.framesA.length) + Math.min(bufferSize, this.framesB.length);
        if (this.initialLoadTimer)
            clearTimeout(this.initialLoadTimer);
        for (let i = 0; i < bufferSize; i++) {
            if (this.framesA.length > i)
                this.loadFrame(this.framesA[i], "A");
            if (this.framesB.length > i)
                this.loadFrame(this.framesB[i], "B");
        }
        this.initialLoadTimer = setTimeout(() => {
            this.isInitialLoading = false;
            if (this.pendingCanvasUpdate) {
                this.node.setDirtyCanvas(true, false);
                this.pendingCanvasUpdate = false;
            }
        }, 1000);
    }
    loadFrame(frameData, videoId) {
        if (!frameData)
            return null;
        const cacheKey = `${videoId}_${frameData.frame_index}`;
        const cache = videoId === "A" ? this.loadedFramesA : this.loadedFramesB;
        if (cache[cacheKey]?.complete && !cache[cacheKey].failed)
            return cache[cacheKey];
        if (cache[cacheKey]?.isLoading || cache[cacheKey]?.queued)
            return cache[cacheKey];
        // Increased cache size to 100 frames to support smooth playback
        if (Object.keys(cache).length > 100)
            this.cleanupFrameCache(cache, frameData.frame_index);
        if (!this.loadingQueue.find(item => item.cacheKey === cacheKey)) {
            this.loadingQueue.push({ frameData, videoId, cacheKey, retryCount: 0 });
        }
        if (!cache[cacheKey]) {
            const img = new Image();
            img.queued = true;
            cache[cacheKey] = img;
        }
        this.processLoadingQueue();
        return cache[cacheKey];
    }
    processLoadingQueue() {
        if (this.loadingInProgress)
            return;
        this.loadingInProgress = true;
        while (this.loadingQueue.length > 0 && this.activeLoads.size < this.maxConcurrentLoads) {
            const item = this.loadingQueue.shift();
            if (item)
                this.loadFrameImmediate(item);
        }
        this.loadingInProgress = false;
        if (this.loadingQueue.length > 0) {
            setTimeout(() => this.processLoadingQueue(), 100);
        }
    }
    loadFrameImmediate(queueItem) {
        const { frameData, videoId, cacheKey, retryCount } = queueItem;
        const cache = videoId.startsWith("A") ? this.loadedFramesA : this.loadedFramesB;
        this.activeLoads.add(cacheKey);
        const img = (cache[cacheKey] || new Image());
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
                    if (this.initialLoadTimer)
                        clearTimeout(this.initialLoadTimer);
                    this.node.setDirtyCanvas(true, false);
                }
                else {
                    this.pendingCanvasUpdate = true;
                }
            }
            else {
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
            }
            else {
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
    cleanupFrameCache(cache, currentFrameIndex = 0) {
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
            if (cache[key] && !cache[key].isLoading)
                delete cache[key];
        });
    }
    debouncedCanvasUpdate() {
        if (this.canvasUpdateTimer)
            clearTimeout(this.canvasUpdateTimer);
        this.canvasUpdateTimer = setTimeout(() => {
            this.node.setDirtyCanvas(true, false);
        }, 16);
    }
    getFrameImageForIndex(index, videoId) {
        const frames = videoId === "A" ? this.framesA : this.framesB;
        if (!frames.length || index < 0 || index >= frames.length)
            return null;
        return this.loadFrame(frames[index], videoId);
    }
    draw(ctx, node, width, y, height) {
        this._drawCallCount++;
        this.y = y;
        this.last_y = y;
        const nodeHeight = node.size[1];
        const availableHeight = nodeHeight - y - 10;
        const mode = node.properties?.comparer_mode || "Playback";
        switch (mode) {
            case "Side-by-Side":
                this.drawSideBySideMode(ctx, y, width, availableHeight);
                break;
            case "Stacked":
                this.drawStackedMode(ctx, y, width, availableHeight);
                break;
            case "Slider":
                this.drawSliderMode(ctx, y, width, availableHeight);
                break;
            case "Onion Skin":
                this.drawOnionSkinMode(ctx, y, width, availableHeight);
                break;
            case "Sync Compare":
                this.drawSyncCompareMode(ctx, y, width, availableHeight);
                break;
            case "Grid":
                this.drawGridMode(ctx, y, width, availableHeight);
                break;
            case "Batch":
                this.drawBatchMode(ctx, y, width, availableHeight);
                break;
            default:
                this.drawPlaybackMode(ctx, y, width, availableHeight);
                break;
        }
        if (["Grid", "Batch"].includes(mode) && this.maxPairs > 1) {
            this.drawBatchControls(ctx, y, width, availableHeight);
        }
        else {
            this.drawPlaybackControls(ctx, y + availableHeight - 45, width);
        }
    }
    drawPlaybackMode(ctx, y, width, availableHeight) {
        const videoId = this.node.properties?.selected_video || "A";
        const videoFrames = videoId === "A" ? this.framesA : this.framesB;
        if (!videoFrames.length) {
            this.drawNoVideoMessage(ctx, y, width, availableHeight - 45, videoId);
            return;
        }
        const currentFrame = Math.min(this.currentFrameIndex, videoFrames.length - 1);
        const frameImg = this.getFrameImageForIndex(currentFrame, videoId);
        if (this.isInitialLoading && (!frameImg || !frameImg.complete)) {
            this.drawLoadingMessage(ctx, y, width, availableHeight - 45);
        }
        else if (frameImg?.complete && !frameImg.failed) {
            this.drawFrame(ctx, frameImg, y, width, availableHeight - 45);
            this.drawFrameCounter(ctx, y + 10, width, currentFrame + 1, videoFrames.length, videoId);
        }
        else {
            this.drawLoadingMessage(ctx, y, width, availableHeight - 45);
        }
    }
    drawSideBySideMode(ctx, y, width, availableHeight) {
        const halfWidth = width / 2;
        if (!this.framesA.length && !this.framesB.length) {
            this.drawNoVideoMessage(ctx, y, width, availableHeight, "both");
            return;
        }
        const currentFrameA = Math.min(this.currentFrameIndex, this.framesA.length - 1);
        const currentFrameB = Math.min(this.currentFrameIndex, this.framesB.length - 1);
        if (this.framesA.length) {
            const frameImgA = this.getFrameImageForIndex(currentFrameA, "A");
            if (!this.isInitialLoading && frameImgA?.complete) {
                this.drawFrameInRegion(ctx, frameImgA, y, 0, halfWidth, availableHeight);
                this.drawFrameCounter(ctx, y + 10, halfWidth, currentFrameA + 1, this.framesA.length, "A");
            }
            else {
                this.drawLoadingMessage(ctx, y, halfWidth, availableHeight, 0);
            }
        }
        else {
            this.drawNoVideoMessage(ctx, y, halfWidth, availableHeight, "A", 0);
        }
        if (this.framesB.length) {
            const frameImgB = this.getFrameImageForIndex(currentFrameB, "B");
            if (!this.isInitialLoading && frameImgB?.complete) {
                this.drawFrameInRegion(ctx, frameImgB, y, halfWidth, halfWidth, availableHeight);
                this.drawFrameCounter(ctx, y + 10, halfWidth, currentFrameB + 1, this.framesB.length, "B", halfWidth);
            }
            else {
                this.drawLoadingMessage(ctx, y, halfWidth, availableHeight, halfWidth);
            }
        }
        else {
            this.drawNoVideoMessage(ctx, y, halfWidth, availableHeight, "B", halfWidth);
        }
        ctx.beginPath();
        ctx.moveTo(halfWidth, y);
        ctx.lineTo(halfWidth, y + availableHeight - 45);
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    drawStackedMode(ctx, y, width, availableHeight) {
        const halfHeight = (availableHeight - 45) / 2;
        if (!this.framesA.length && !this.framesB.length) {
            this.drawNoVideoMessage(ctx, y, width, availableHeight, "both");
            return;
        }
        const currentFrameA = Math.min(this.currentFrameIndex, this.framesA.length - 1);
        const currentFrameB = Math.min(this.currentFrameIndex, this.framesB.length - 1);
        if (this.framesA.length) {
            const frameImgA = this.getFrameImageForIndex(currentFrameA, "A");
            if (frameImgA?.complete) {
                this.drawFrameInRegion(ctx, frameImgA, y, 0, width, halfHeight);
                this.drawFrameCounter(ctx, y + 10, width, currentFrameA + 1, this.framesA.length, "A");
            }
            else {
                this.drawLoadingMessage(ctx, y, width, halfHeight);
            }
        }
        else {
            this.drawNoVideoMessage(ctx, y, width, halfHeight, "A");
        }
        if (this.framesB.length) {
            const frameImgB = this.getFrameImageForIndex(currentFrameB, "B");
            if (frameImgB?.complete) {
                this.drawFrameInRegion(ctx, frameImgB, y + halfHeight, 0, width, halfHeight);
                this.drawFrameCounter(ctx, y + halfHeight + 10, width, currentFrameB + 1, this.framesB.length, "B");
            }
            else {
                this.drawLoadingMessage(ctx, y + halfHeight, width, halfHeight);
            }
        }
        else {
            this.drawNoVideoMessage(ctx, y + halfHeight, width, halfHeight, "B");
        }
        ctx.beginPath();
        ctx.moveTo(0, y + halfHeight);
        ctx.lineTo(width, y + halfHeight);
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    drawSliderMode(ctx, y, width, availableHeight) {
        if (!this.framesA.length && !this.framesB.length) {
            this.drawNoVideoMessage(ctx, y, width, availableHeight, "both");
            return;
        }
        const currentFrameA = Math.min(this.currentFrameIndex, this.framesA.length - 1);
        const currentFrameB = Math.min(this.currentFrameIndex, this.framesB.length - 1);
        if (this.framesA.length) {
            const frameImgA = this.getFrameImageForIndex(currentFrameA, "A");
            if (frameImgA?.complete) {
                this.drawFrame(ctx, frameImgA, y, width, availableHeight - 45);
                this.drawFrameCounter(ctx, y + 10, width, currentFrameA + 1, this.framesA.length, "A");
            }
            else {
                this.drawLoadingMessage(ctx, y, width, availableHeight - 45);
            }
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
    drawOnionSkinMode(ctx, y, width, availableHeight) {
        if (!this.framesA.length && !this.framesB.length) {
            this.drawNoVideoMessage(ctx, y, width, availableHeight, "both");
            return;
        }
        const currentFrameA = Math.min(this.currentFrameIndex, this.framesA.length - 1);
        const currentFrameB = Math.min(this.currentFrameIndex, this.framesB.length - 1);
        const opacity = this.node.properties?.onionSkinOpacity || 0.5;
        if (this.framesA.length) {
            const frameImgA = this.getFrameImageForIndex(currentFrameA, "A");
            if (frameImgA?.complete) {
                this.drawFrame(ctx, frameImgA, y, width, availableHeight - 45);
                this.drawFrameCounter(ctx, y + 10, width, currentFrameA + 1, this.framesA.length, "A");
            }
            else {
                this.drawLoadingMessage(ctx, y, width, availableHeight - 45);
            }
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
    drawSyncCompareMode(ctx, y, width, availableHeight) {
        const selectedVideo = this.node.selectedVideo || "A";
        const frames = selectedVideo === "A" ? this.framesA : this.framesB;
        if (!frames.length) {
            this.drawNoVideoMessage(ctx, y, width, availableHeight, selectedVideo);
            return;
        }
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
        }
        else {
            this.drawLoadingMessage(ctx, y, width, availableHeight - 45);
        }
    }
    drawGridMode(ctx, y, width, availableHeight) {
        // Simplified grid - draw first 4 pairs in 2x2 grid
        const cols = 2, rows = 2;
        const cellWidth = width / cols, cellHeight = (availableHeight - 30) / rows;
        for (let i = 0; i < Math.min(this.maxPairs, 4); i++) {
            const col = i % cols, row = Math.floor(i / cols);
            const x = col * cellWidth, cellY = y + row * cellHeight;
            const videoA = this.videosA[i], videoB = this.videosB[i];
            if (videoA?.frames.length) {
                const frame = this.loadFrame(videoA.frames[this.currentFrameIndex % videoA.frames.length], `A_${i}`);
                if (frame?.complete)
                    this.drawFrameInRegion(ctx, frame, cellY, x, cellWidth / 2, cellHeight);
            }
            if (videoB?.frames.length) {
                const frame = this.loadFrame(videoB.frames[this.currentFrameIndex % videoB.frames.length], `B_${i}`);
                if (frame?.complete)
                    this.drawFrameInRegion(ctx, frame, cellY, x + cellWidth / 2, cellWidth / 2, cellHeight);
            }
        }
    }
    drawBatchMode(ctx, y, width, availableHeight) {
        const startPair = this.currentBatchPage * this.pairsPerPage;
        const pairHeight = (availableHeight - 30) / this.pairsPerPage;
        for (let i = 0; i < this.pairsPerPage && startPair + i < this.maxPairs; i++) {
            const pairIdx = startPair + i;
            const pairY = y + i * pairHeight;
            const videoA = this.videosA[pairIdx], videoB = this.videosB[pairIdx];
            if (videoA?.frames.length) {
                const frame = this.loadFrame(videoA.frames[this.currentFrameIndex % videoA.frames.length], `A_${pairIdx}`);
                if (frame?.complete)
                    this.drawFrameInRegion(ctx, frame, pairY, 0, width / 2, pairHeight);
            }
            if (videoB?.frames.length) {
                const frame = this.loadFrame(videoB.frames[this.currentFrameIndex % videoB.frames.length], `B_${pairIdx}`);
                if (frame?.complete)
                    this.drawFrameInRegion(ctx, frame, pairY, width / 2, width / 2, pairHeight);
            }
        }
    }
    drawFrame(ctx, img, y, width, availableHeight) {
        if (!img?.complete)
            return;
        const imageAspect = img.naturalWidth / img.naturalHeight;
        const canvasAspect = width / availableHeight;
        let drawWidth, drawHeight, offsetX = 0, offsetY = 0;
        if (imageAspect > canvasAspect) {
            drawWidth = width;
            drawHeight = width / imageAspect;
            offsetY = (availableHeight - drawHeight) / 2;
        }
        else {
            drawHeight = availableHeight;
            drawWidth = availableHeight * imageAspect;
            offsetX = (width - drawWidth) / 2;
        }
        ctx.drawImage(img, offsetX, y + offsetY, drawWidth, drawHeight);
    }
    drawFrameInRegion(ctx, img, y, x, regionWidth, regionHeight) {
        if (!img?.complete)
            return;
        const imageAspect = img.naturalWidth / img.naturalHeight;
        const regionAspect = regionWidth / regionHeight;
        let drawWidth, drawHeight, offsetX = 0, offsetY = 0;
        if (imageAspect > regionAspect) {
            drawWidth = regionWidth;
            drawHeight = regionWidth / imageAspect;
            offsetY = (regionHeight - drawHeight) / 2;
        }
        else {
            drawHeight = regionHeight;
            drawWidth = regionHeight * imageAspect;
            offsetX = (regionWidth - drawWidth) / 2;
        }
        ctx.drawImage(img, x + offsetX, y + offsetY, drawWidth, drawHeight);
    }
    drawFrameCounter(ctx, y, width, current, total, videoId, offsetX = 0) {
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(offsetX + 10, y, 80, 24);
        ctx.fillStyle = "white";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        ctx.fillText(`${videoId}: ${current}/${total}`, offsetX + 15, y + 16);
    }
    drawNoVideoMessage(ctx, y, width, height, videoId, offsetX = 0) {
        ctx.fillStyle = "rgba(0,0,0,0.3)";
        ctx.fillRect(offsetX, y, width, height);
        ctx.fillStyle = "white";
        ctx.font = "14px Arial";
        ctx.textAlign = "center";
        ctx.fillText(videoId === "both" ? "No videos available" : `No video ${videoId} available`, offsetX + width / 2, y + height / 2);
    }
    drawLoadingMessage(ctx, y, width, height, offsetX = 0) {
        ctx.fillStyle = "rgba(0,0,0,0.3)";
        ctx.fillRect(offsetX, y, width, height);
        ctx.fillStyle = "white";
        ctx.font = "14px Arial";
        ctx.textAlign = "center";
        ctx.fillText("Loading frames...", offsetX + width / 2, y + height / 2);
    }
    drawPlaybackControls(ctx, y, width) {
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
        ctx.roundRect(playPauseX, buttonY, buttonSize, buttonSize, 4);
        ctx.fill();
        ctx.fillStyle = "white";
        ctx.font = "20px Arial";
        ctx.textAlign = "center";
        ctx.fillText(this.isPlaying ? "⏸" : "▶", playPauseX + buttonSize / 2, y + controlHeight / 2 + 7);
        const trackHeight = 8;
        ctx.fillStyle = "rgba(60,60,60,0.8)";
        ctx.beginPath();
        ctx.roundRect(sliderStart, y + controlHeight / 2 - trackHeight / 2, sliderWidth, trackHeight, trackHeight / 2);
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
    drawBatchControls(ctx, y, width, availableHeight) {
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
    startPlayback() {
        if (this.isPlaying)
            return;
        this.isPlaying = true;
        this.isInitialLoading = false;
        this.lastFrameTime = performance.now();
        const playbackLoop = () => {
            const now = performance.now();
            if (now - this.lastFrameTime >= 1000 / this.fps) {
                this.advanceFrame();
                this.lastFrameTime = now;
            }
            if (this.isPlaying)
                this.animationFrame = requestAnimationFrame(playbackLoop);
        };
        this.animationFrame = requestAnimationFrame(playbackLoop);
        this.node.setDirtyCanvas(true, false);
    }
    stopPlayback() {
        if (!this.isPlaying)
            return;
        this.isPlaying = false;
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        this.node.setDirtyCanvas(true, false);
    }
    togglePlayback() {
        this.isPlaying ? this.stopPlayback() : this.startPlayback();
    }
    advanceFrame() {
        const totalFrames = Math.max(this.framesA.length, this.framesB.length);
        if (totalFrames === 0)
            return;
        this.currentFrameIndex = (this.currentFrameIndex + 1) % totalFrames;
        // Preload upcoming frames for smooth playback (look-ahead buffer)
        this.preloadUpcomingFrames(10);
        this.node.setDirtyCanvas(true, false);
    }
    preloadUpcomingFrames(count) {
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
    previousFrame() {
        const totalFrames = Math.max(this.framesA.length, this.framesB.length);
        if (totalFrames === 0)
            return;
        this.currentFrameIndex = (this.currentFrameIndex - 1 + totalFrames) % totalFrames;
        this.node.setDirtyCanvas(true, false);
    }
    nextFrame() { this.advanceFrame(); }
    mouse(event, pos, node) {
        // Note: onMouseDown handler only calls this for mousedown/pointerdown events
        const mode = node.properties?.comparer_mode;
        if (mode === "Sync Compare") {
            node.selectedVideo = node.selectedVideo === "A" ? "B" : "A";
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
    computeSize(width) {
        return [width, 300];
    }
    onRemoved() {
        this.stopPlayback();
        if (this.initialLoadTimer)
            clearTimeout(this.initialLoadTimer);
        if (this.canvasUpdateTimer)
            clearTimeout(this.canvasUpdateTimer);
        this.loadedFramesA = {};
        this.loadedFramesB = {};
    }
}
// === COMFYUI EXTENSION REGISTRATION ===
app.registerExtension({
    name: "VideoComparer",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "Video Comparer" && nodeData.name !== "VideoComparer")
            return;
        nodeType.prototype.properties = nodeType.prototype.properties || {};
        nodeType.prototype.properties.comparer_mode = "Playback";
        nodeType.prototype.properties.selected_video = "A";
        nodeType.prototype.properties.onionSkinOpacity = 0.5;
        nodeType.prototype.properties.user_resized = false;
        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (origOnDrawForeground)
                origOnDrawForeground.apply(this, arguments);
            drawGradientTitle(this, ctx, CACHE, 6);
        };
        const origOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (origOnRemoved)
                origOnRemoved.apply(this, arguments);
            if (this.videoComparerWidget)
                this.videoComparerWidget.onRemoved();
            // CACHE only holds animation timing state, no cleanup needed
        };
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated)
                origOnNodeCreated.apply(this, arguments);
            this.properties = this.properties || {};
            this.properties.comparer_mode = this.properties.comparer_mode || "Playback";
            this.properties.selected_video = this.properties.selected_video || "A";
            this.properties.onionSkinOpacity = this.properties.onionSkinOpacity ?? 0.5;
            this.isPointerOver = false;
            this.pointerOverPos = [0, 0];
            this.selectedVideo = "A";
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.layoutWidget = this.addWidget("combo", "Comparison Mode", this.properties.comparer_mode, ((value) => {
                this.properties.comparer_mode = value;
                if (this.updateControlsVisibility)
                    this.updateControlsVisibility();
                this.setDirtyCanvas(true, false);
            }), { values: ["Playback", "Side-by-Side", "Stacked", "Slider", "Onion Skin", "Sync Compare"] });
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.videoSelectorWidget = this.addWidget("combo", "Video", this.properties.selected_video, ((value) => {
                this.properties.selected_video = value;
                this.selectedVideo = value;
                this.setDirtyCanvas(true, false);
            }), { values: ["A", "B"] });
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.onionSkinOpacitySlider = this.addWidget("slider", "Opacity B", this.properties.onionSkinOpacity, ((value) => {
                this.properties.onionSkinOpacity = value;
                this.setDirtyCanvas(true, false);
            }), { min: 0.0, max: 1.0, step: 0.01 });
            this.videoComparerWidget = this.addCustomWidget(new VideoComparerWidget("video_comparer", this));
            if (this.updateControlsVisibility)
                this.updateControlsVisibility();
            const initialSize = this.computeSize?.() || [400, 300];
            if (this.setSize)
                this.setSize([Math.max(400, initialSize[0]), Math.max(300, initialSize[1])]);
            this.setDirtyCanvas(true, true);
        };
        nodeType.prototype.updateControlsVisibility = function () {
            const mode = this.properties.comparer_mode;
            if (this.videoSelectorWidget)
                this.videoSelectorWidget.hidden = mode !== "Playback";
            if (this.onionSkinOpacitySlider)
                this.onionSkinOpacitySlider.hidden = mode !== "Onion Skin";
        };
        const origComputeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function (out) {
            const size = origComputeSize ? origComputeSize.apply(this, arguments) : [400, 300];
            if (this.videoComparerWidget) {
                const widgetSize = this.videoComparerWidget.computeSize(size[0]);
                let extraHeight = 60;
                if (this.properties.comparer_mode === "Playback" || this.properties.comparer_mode === "Onion Skin")
                    extraHeight += 30;
                size[1] = Math.max(size[1], widgetSize[1] + extraHeight);
            }
            return size;
        };
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (origOnExecuted)
                origOnExecuted.apply(this, arguments);
            if (!message || typeof message !== 'object')
                return message;
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
        nodeType.prototype.onMouseDown = function (event, pos) {
            return this.videoComparerWidget?.mouse(event, pos, this) || false;
        };
        nodeType.prototype.onMouseEnter = function () {
            this.isPointerOver = true;
            this.setDirtyCanvas(true, false);
        };
        nodeType.prototype.onMouseLeave = function () {
            this.isPointerOver = false;
            this.setDirtyCanvas(true, false);
        };
        nodeType.prototype.onMouseMove = function (event, pos) {
            this.pointerOverPos = pos;
            if (this.properties.comparer_mode === "Slider")
                this.setDirtyCanvas(true, false);
            return false;
        };
        nodeType.prototype.onKeyDown = function (event) {
            if (!this.videoComparerWidget)
                return false;
            switch (event.key) {
                case "ArrowLeft":
                    this.videoComparerWidget.previousFrame();
                    return true;
                case "ArrowRight":
                    this.videoComparerWidget.nextFrame();
                    return true;
                case " ":
                    this.videoComparerWidget.togglePlayback();
                    event.preventDefault();
                    return true;
            }
            return false;
        };
        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            if (origGetExtraMenuOptions)
                origGetExtraMenuOptions.apply(this, arguments);
            options.push(null);
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
                            if (this.layoutWidget)
                                this.layoutWidget.value = mode;
                            if (this.updateControlsVisibility)
                                this.updateControlsVisibility();
                            this.setDirtyCanvas(true, false);
                        }
                    }))
                }
            });
            options.push({
                content: "Reset to Default Size",
                callback: () => {
                    this.properties.user_resized = false;
                    if (this.setSize && this.computeSize)
                        this.setSize(this.computeSize());
                    this.setDirtyCanvas(true, false);
                }
            });
        };
    }
});
//# sourceMappingURL=video_comparer.js.map