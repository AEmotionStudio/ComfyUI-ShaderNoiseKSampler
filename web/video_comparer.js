// VideoComparer.js
// Implementation based on AdvancedImageComparer but adapted for video comparison

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("VideoComparer module loaded");

// Global storage management
window.comfyUIStorageManager = window.comfyUIStorageManager || {
    initialized: false,
    
    init() {
        if (this.initialized) return;
        this.initialized = true;
        
        // Add global error handlers
        window.addEventListener('error', (event) => {
            if (event.error && event.error.name === 'QuotaExceededError') {
                console.warn("[StorageManager] Global quota exceeded error caught");
                this.emergencyCleanup();
                event.preventDefault();
            }
        });
        
        window.addEventListener('unhandledrejection', (event) => {
            if (event.reason && event.reason.name === 'QuotaExceededError') {
                console.warn("[StorageManager] Global quota exceeded promise rejection caught");
                this.emergencyCleanup();
                event.preventDefault();
            }
        });
        
        console.log("[StorageManager] Initialized with emergency cleanup handlers");
    },
    
    emergencyCleanup() {
        console.log("[StorageManager] Performing emergency cleanup");
        
        try {
            // Don't clean up if video comparers are actively loading
            if (window.videoComparerInstances) {
                const activeLoading = window.videoComparerInstances.some(instance => 
                    instance.activeLoads && instance.activeLoads.size > 0
                );
                
                if (activeLoading) {
                    console.log("[StorageManager] Skipping cleanup - active loading in progress");
                    return;
                }
            }
            
            // Clear workflow persistence data
            const keys = Object.keys(localStorage);
            let cleaned = 0;
            
            keys.forEach(key => {
                if (key.startsWith('workflow') || 
                    key.startsWith('comfyui') ||
                    key.includes('temp') ||
                    key.includes('cache') ||
                    key.includes('persist')) {
                    try {
                        localStorage.removeItem(key);
                        cleaned++;
                    } catch (e) {
                        // Ignore individual removal errors
                    }
                }
            });
            
            console.log(`[StorageManager] Emergency cleanup removed ${cleaned} localStorage items`);
            
            // Only clean video frames if really necessary
            if (cleaned === 0 && window.videoComparerInstances) {
                console.log("[StorageManager] Cleaning oldest video frame caches");
                window.videoComparerInstances.forEach(instance => {
                    if (instance.cleanupOldestFrames) {
                        instance.cleanupOldestFrames();
                    }
                });
            }
            
            // Force garbage collection if available
            if (window.gc) {
                window.gc();
            }
            
        } catch (error) {
            console.error("[StorageManager] Error during emergency cleanup:", error);
        }
    },
    
    getStorageUsage() {
        let total = 0;
        try {
            for (let key in localStorage) {
                if (localStorage.hasOwnProperty(key)) {
                    total += localStorage[key].length + key.length;
                }
            }
        } catch (e) {
            console.warn("[StorageManager] Could not calculate storage usage:", e);
        }
        return total;
    }
};

// Initialize storage manager
window.comfyUIStorageManager.init();

// Track video comparer instances for cleanup
window.videoComparerInstances = window.videoComparerInstances || [];

// Cache for rendering optimization
const CACHE = {
    titleCanvas: null,
    titleCtx: null,
    lastWidth: 0,
    lastHeight: 0,
    lastTime: 0,
    frameCount: 0,
    frameSkip: 2, // Only update animation every X frames
    collapsed: {
        canvas: null,
        ctx: null,
        lastWidth: 0
    }
};

function imageDataToUrl(data) {
    if (!data || !data.filename) {
        console.error("[VideoComparer] Invalid image data", data);
        return "";
    }
    return api.apiURL(`/view?filename=${encodeURIComponent(data.filename)}&type=${data.type || ""}&subfolder=${data.subfolder || ""}${app.getPreviewFormatParam()}${app.getRandParam()}`);
}

/**
 * Draws a custom golden eyeball using canvas drawing commands
 * @param {CanvasRenderingContext2D} ctx - The canvas context
 * @param {number} centerX - X center position
 * @param {number} centerY - Y center position
 * @param {number} size - Size of the eyeball
 * @param {number} shimmerPosition - Position of the shimmer effect (0-1)
 */
function drawGoldenEyeball(ctx, centerX, centerY, size, shimmerPosition) {
    // Refined proportions for smaller, cleaner look
    const eyeWidth = size * 1.6;
    const eyeHeight = size * 1.0;
    const irisRadius = size * 0.35;
    const pupilRadius = size * 0.15;
    
    ctx.save();
    
    // Create base golden gradient exactly like the mathematical formula
    const baseGradient = ctx.createLinearGradient(0, centerY - size*0.7, 0, centerY + size*0.7);
    baseGradient.addColorStop(0, "#B8860B");    // Darker gold
    baseGradient.addColorStop(0.5, "#FFD700");  // Bright gold
    baseGradient.addColorStop(1, "#B8860B");    // Darker gold
    
    // Create moving highlight effect
    const highlightWidth = eyeWidth * 0.4; // Width of the highlight
    const highlightX = -highlightWidth + (eyeWidth + highlightWidth) * shimmerPosition;
    
    const shimmerGradient = ctx.createLinearGradient(
        centerX + highlightX - highlightWidth/2, 0,
        centerX + highlightX + highlightWidth/2, 0
    );
    
    // Create smooth highlight transition
    shimmerGradient.addColorStop(0, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(0.1, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(0.5, "rgba(255, 255, 200, 0.3)");
    shimmerGradient.addColorStop(0.9, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(1, "rgba(255, 255, 200, 0)");
    
    // Draw etched shadow for all outlines
    ctx.strokeStyle = "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1.5;
    ctx.lineCap = "round";
    
    // Shadow for main eye outline
    ctx.beginPath();
    ctx.ellipse(centerX + 2, centerY + 2, eyeWidth/2, eyeHeight/2, 0, 0, Math.PI * 2);
    ctx.stroke();
    
    // Shadow for iris
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX + 2, centerY + 2, irisRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Shadow for pupil
    ctx.beginPath();
    ctx.arc(centerX + 2, centerY + 2, pupilRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Draw 6 eyelashes/rays around the eye (representing 6 comparison modes) - shadows first
    const rayCount = 6;  // Changed from 8 to 6
    const rayLength = size * 0.7;
    ctx.lineWidth = 1;
    
    for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        const startX = centerX + Math.cos(angle) * (eyeWidth/2 + 1);
        const startY = centerY + Math.sin(angle) * (eyeHeight/2 + 1);
        const endX = centerX + Math.cos(angle) * (eyeWidth/2 + rayLength);
        const endY = centerY + Math.sin(angle) * (eyeHeight/2 + rayLength);
        
        ctx.beginPath();
        ctx.moveTo(startX + 2, startY + 2);
        ctx.lineTo(endX + 2, endY + 2);
        ctx.stroke();
    }
    
    // Now draw the golden base outlines
    ctx.strokeStyle = baseGradient;
    ctx.lineWidth = 1.5;
    
    // Main eye outline
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, eyeWidth/2, eyeHeight/2, 0, 0, Math.PI * 2);
    ctx.stroke();
    
    // Iris outline
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX, centerY, irisRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Pupil outline
    ctx.beginPath();
    ctx.arc(centerX, centerY, pupilRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Draw 6 eyelashes/rays with base golden color
    for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        const startX = centerX + Math.cos(angle) * (eyeWidth/2 + 1);
        const startY = centerY + Math.sin(angle) * (eyeHeight/2 + 1);
        const endX = centerX + Math.cos(angle) * (eyeWidth/2 + rayLength);
        const endY = centerY + Math.sin(angle) * (eyeHeight/2 + rayLength);
        
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
    }
    
    // Add subtle iris texture lines
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 6; i++) {  // Changed from 8 to 6 to match rays
        const angle = (i / 6) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(centerX + Math.cos(angle) * pupilRadius * 1.1, centerY + Math.sin(angle) * pupilRadius * 1.1);
        ctx.lineTo(centerX + Math.cos(angle) * irisRadius * 0.9, centerY + Math.sin(angle) * irisRadius * 0.9);
        ctx.stroke();
    }
    
    // Apply shimmer highlight effect to all outlines
    ctx.strokeStyle = shimmerGradient;
    ctx.lineWidth = 1.5;
    
    // Shimmer on main eye outline
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, eyeWidth/2, eyeHeight/2, 0, 0, Math.PI * 2);
    ctx.stroke();
    
    // Shimmer on iris
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX, centerY, irisRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Shimmer on pupil
    ctx.beginPath();
    ctx.arc(centerX, centerY, pupilRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Shimmer on 6 eyelashes/rays
    for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        const startX = centerX + Math.cos(angle) * (eyeWidth/2 + 1);
        const startY = centerY + Math.sin(angle) * (eyeHeight/2 + 1);
        const endX = centerX + Math.cos(angle) * (eyeWidth/2 + rayLength);
        const endY = centerY + Math.sin(angle) * (eyeHeight/2 + rayLength);
        
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
    }
    
    // Add outline glow that follows the highlight
    const glowIntensity = Math.max(0, 1 - Math.abs(centerX - (centerX + highlightX))/(eyeWidth/4));
    ctx.shadowColor = `rgba(255, 255, 200, ${glowIntensity * 0.3})`;
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    
    // Final glow pass on main outline
    ctx.strokeStyle = baseGradient;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, eyeWidth/2, eyeHeight/2, 0, 0, Math.PI * 2);
    ctx.stroke();
    
    ctx.restore();
}

/**
 * Draws a gradient background with golden eyeball title
 * @param {LGraphNode} node - The node to apply the gradient to
 * @param {CanvasRenderingContext2D} ctx - The canvas context
 */
function drawGradientTitle(node, ctx) {
    // Get title area dimensions
    const titleHeight = node.flags.collapsed ? 20 : 30; // Smaller height when collapsed
    const width = node.flags.collapsed ? 190 : node.size[0]; // Smaller width when collapsed
    const fullHeight = node.size[1]; // Get actual node height
    const eyeballY = node.flags.collapsed ? titleHeight / 2 : 25; // Much closer to top
    
    // Eyeball size based on collapsed state (made smaller)
    const eyeballSize = node.flags.collapsed ? 6 : 10;
    
    // Update animation frame counter (performance optimization)
    CACHE.frameCount = (CACHE.frameCount + 1) % (CACHE.frameSkip + 1);
    const shouldUpdateAnimation = CACHE.frameCount === 0;
    
    // Save current state
    ctx.save();
    
    // Reset shadow properties for gradient drawing
    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    
    // Create vertical black gradient for entire background that fills node height
    const gradient = ctx.createLinearGradient(0, 0, 0, fullHeight);
    gradient.addColorStop(0, "#000000");     // Pure black at top
    gradient.addColorStop(0.2, "#101010");   // Transition to very dark gray
    gradient.addColorStop(1, "#101010");     // Very dark gray at bottom
    
    // Create smooth shimmer effect for eyeball - only calculate if animation should update
    let shimmerPosition = 0.5; // Default middle position
    if (shouldUpdateAnimation) {
        const time = Date.now() / 3000; // Faster time factor
        shimmerPosition = (Math.sin(time) + 1) / 2; // Changed from cos to sin for left-to-right only
        // Store for later use if needed
        CACHE.lastTime = time;
    } else {
        // Reuse last calculation for animation frames we're skipping
        const time = CACHE.lastTime || Date.now() / 3000;
        shimmerPosition = (Math.sin(time) + 1) / 2;
    }
    
    // Add collapse button handler
    if (node.flags.collapsed) {
        // If node is collapsed, adjust the title rendering
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, titleHeight);
        
        // Draw custom golden eyeball for collapsed version
        drawGoldenEyeball(ctx, width / 2, titleHeight / 2, eyeballSize, shimmerPosition);
        
        // Skip the rest of the rendering when collapsed
        ctx.restore();
        return;
    }
    
    // Draw background that fills the entire node
    ctx.fillStyle = gradient;
    
    // Use rounded rectangle for the background with rounded corners at the bottom
    if (!node.flags.collapsed) {
        const cornerRadius = 8; // Adjust radius as needed
        
        // Create path for rounded rectangle
        ctx.beginPath();
        ctx.moveTo(0, 0); // Top-left corner (no rounding)
        ctx.lineTo(width, 0); // Top-right corner (no rounding)
        ctx.lineTo(width, fullHeight - cornerRadius); // Right edge before bottom-right corner
        ctx.arcTo(width, fullHeight, width - cornerRadius, fullHeight, cornerRadius); // Bottom-right rounded corner
        ctx.lineTo(cornerRadius, fullHeight); // Bottom edge before bottom-left corner
        ctx.arcTo(0, fullHeight, 0, fullHeight - cornerRadius, cornerRadius); // Bottom-left rounded corner
        ctx.lineTo(0, 0); // Left edge back to top
        ctx.closePath();
        ctx.fill();
    } else {
        // Keep regular rectangle for collapsed state
        ctx.fillRect(0, 0, width, fullHeight); // No extra padding
    }
    
    // Draw custom golden eyeball for expanded version
    drawGoldenEyeball(ctx, width / 2, eyeballY, eyeballSize, shimmerPosition);
    
    // Restore context state
    ctx.restore();
}

class VideoComparerWidget {
    constructor(name, node) {
        console.log("[VideoComparer] Widget constructor called with name:", name, "node:", node);
        
        this.name = name;
        this.node = node;
        this.type = "video_comparer";
        this.value = { video_data: [] };
        
        // Initialize frame caches and tracking
        this.loadedFramesA = {};
        this.loadedFramesB = {};
        this.framesA = [];
        this.framesB = [];
        this.currentFrameIndex = 0;
        this.isPlaying = false;
        this.playbackInterval = null;
        
        // Add loading queue management
        this.loadingQueue = [];
        this.activeLoads = new Set();
        this.maxConcurrentLoads = 3;
        this.retryAttempts = 3;
        this.loadingInProgress = false;
        
        // Enhanced batch handling properties (similar to advanced_comparer.js)
        this.videosA = [];
        this.videosB = [];
        this.currentPairIndex = 0;
        this.maxPairs = 0;
        this.animationFrame = null;
        this.autoPlayEnabled = false;
        this.autoPlaySpeed = 2000; // 2 seconds per pair
        
        // Batch pagination properties
        this.currentBatchPage = 0;
        this.pairsPerPage = 2; // Show 2 pairs per page in batch mode (videos are larger than images)
        this.maxBatchPages = 0;
        
        // Register this instance for global cleanup
        if (window.videoComparerInstances && !window.videoComparerInstances.includes(this)) {
            window.videoComparerInstances.push(this);
        }
        
        console.log("[VideoComparer] Widget constructor complete");
    }

    set value(v) {
        console.log("[VideoComparer] Widget value setter called with:", v);
        
        // Check storage before processing large video data
        if (window.comfyUIStorageManager) {
            const currentUsage = window.comfyUIStorageManager.getStorageUsage();
            const usageMB = (currentUsage / (1024 * 1024)).toFixed(2);
            
            if (currentUsage > 5 * 1024 * 1024) { // 5MB threshold
                console.warn(`[VideoComparer] High storage usage detected: ${usageMB}MB`);
                window.comfyUIStorageManager.emergencyCleanup();
            }
        }
        
        // Reset state
        this.isPlaying = false;
        this.currentFrameIndex = 0;
        this.stopPlayback();
        
        // Process video data (now supporting batch)
        const videoData = v.video_data || [];
        console.log("[VideoComparer] Raw video data:", videoData);
        
        // Separate videos by type for batch processing
        const videosA = videoData.filter(video => video.name === "video_a" || video.is_video_a);
        const videosB = videoData.filter(video => video.name === "video_b" || video.is_video_b);
        
        // Store all videos for batch processing
        this.videosA = videosA.map((video, index) => ({
            name: `A${index + 1}`,
            fps: video.fps || 8,
            frames: video.frames || [],
            index: index
        }));
        
        this.videosB = videosB.map((video, index) => ({
            name: `B${index + 1}`,
            fps: video.fps || 8,
            frames: video.frames || [],
            index: index
        }));
        
        // Calculate max pairs for comparison
        this.maxPairs = Math.max(this.videosA.length, this.videosB.length);
        this.currentPairIndex = 0;
        
        // Calculate batch pagination
        this.maxBatchPages = Math.ceil(this.maxPairs / this.pairsPerPage);
        this.currentBatchPage = 0;
        
        // Maintain backward compatibility - set current video pair
        this.videoA = this.videosA[0] || null;
        this.videoB = this.videosB[0] || null;
        this.framesA = this.videoA ? this.videoA.frames : [];
        this.framesB = this.videoB ? this.videoB.frames : [];
        this.fps = (this.videoA && this.videoA.fps) || (this.videoB && this.videoB.fps) || 8;
        
        console.log("[VideoComparer] Setup complete - A:", this.framesA.length, "B:", this.framesB.length, "Pairs:", this.maxPairs);
        
        // Clear caches
        this.loadedFramesA = {};
        this.loadedFramesB = {};
        
        // Set the raw value
        this._value = v;
        
        // Update selected pair for batch modes
        this.updateSelectedPair();
        
        // Preload frames
        if (this.framesA.length > 0 || this.framesB.length > 0) {
            this.preloadInitialFrames();
        }
        
        // Update controls and redraw
        if (this.node && this.node.updateControlsVisibility) {
            this.node.updateControlsVisibility();
        }
        this.node.setDirtyCanvas(true, false);
    }

    get value() {
        return this._value || { video_data: [] };
    }
    
    preloadInitialFrames() {
        console.log("[VideoComparer] Preloading initial frames with smart buffering");
        
        // Set initial loading state to prevent flicker
        this.isInitialLoading = true;
        this.loadedFrameCount = 0;
        
        // Calculate how many frames to preload for smooth playback
        const bufferSize = Math.min(5, Math.max(this.framesA.length, this.framesB.length)); // Preload up to 5 frames
        this.targetLoadCount = Math.min(bufferSize, this.framesA.length) + Math.min(bufferSize, this.framesB.length);
        
        // Clear any existing timer
        if (this.initialLoadTimer) {
            clearTimeout(this.initialLoadTimer);
        }
        
        // Load the first several frames immediately without delays
        const framesToPreload = Math.min(bufferSize, Math.max(this.framesA.length, this.framesB.length));
        
        for (let i = 0; i < framesToPreload; i++) {
            if (this.framesA.length > i) {
                // Load immediately without setTimeout
                this.loadFrame(this.framesA[i], "A");
            }
            if (this.framesB.length > i) {
                // Load immediately without setTimeout  
                this.loadFrame(this.framesB[i], "B");
            }
        }
        
        // Set a more aggressive timer to end initial loading state
        this.initialLoadTimer = setTimeout(() => {
            this.isInitialLoading = false;
            console.log("[VideoComparer] Initial loading period ended");
            if (this.pendingCanvasUpdate) {
                this.node.setDirtyCanvas(true, false);
                this.pendingCanvasUpdate = false;
            }
        }, 1000); // Reduced from 2000ms to 1000ms
    }
    
    loadFrame(frameData, videoId, retryCount = 0) {
        if (!frameData) {
            console.error(`[VideoComparer] Invalid frame data for video ${videoId}:`, frameData);
            return null;
        }
        
        const cacheKey = `${videoId}_${frameData.frame_index}`;
        const cache = videoId === "A" ? this.loadedFramesA : this.loadedFramesB;
        
        // Return from cache if already loaded
        if (cache[cacheKey] && cache[cacheKey].complete && !cache[cacheKey].failed) {
            return cache[cacheKey];
        }
        
        // If already in queue or loading, return existing image object
        if (cache[cacheKey] && (cache[cacheKey].loading || cache[cacheKey].queued)) {
            return cache[cacheKey];
        }
        
        // Clean up cache if getting too large
        if (Object.keys(cache).length > 10) {
            this.cleanupFrameCache(cache, videoId);
        }
        
        // Add to loading queue if not already there
        const queueItem = { frameData, videoId, cacheKey, retryCount };
        if (!this.loadingQueue.find(item => item.cacheKey === cacheKey)) {
            this.loadingQueue.push(queueItem);
        }
        
        // Create placeholder image
        if (!cache[cacheKey]) {
            const img = new Image();
            img.queued = true;
            cache[cacheKey] = img;
        }
        
        // Process queue
        this.processLoadingQueue();
        
        return cache[cacheKey];
    }
    
    processLoadingQueue() {
        if (this.loadingInProgress) return;
        this.loadingInProgress = true;
        
        // Process queue items up to concurrency limit
        while (this.loadingQueue.length > 0 && this.activeLoads.size < this.maxConcurrentLoads) {
            const item = this.loadingQueue.shift();
            this.loadFrameImmediate(item);
        }
        
        this.loadingInProgress = false;
        
        // Schedule next processing if queue not empty
        if (this.loadingQueue.length > 0) {
            setTimeout(() => this.processLoadingQueue(), 100);
        }
    }
    
    loadFrameImmediate(queueItem) {
        const { frameData, videoId, cacheKey, retryCount } = queueItem;
        const cache = videoId === "A" ? this.loadedFramesA : this.loadedFramesB;
        
        this.activeLoads.add(cacheKey);
        
        try {
            const img = cache[cacheKey] || new Image();
            img.loading = true;
            img.queued = false;
            img.failed = false;
            
            img.onload = () => {
                console.log(`[VideoComparer] Frame loaded: ${videoId}_${frameData.frame_index}`);
                img.loading = false;
                this.activeLoads.delete(cacheKey);
                
                // Track initial loading progress
                if (this.isInitialLoading) {
                    this.loadedFrameCount++;
                    console.log(`[VideoComparer] Initial loading progress: ${this.loadedFrameCount}/${this.targetLoadCount}`);
                    
                    // Only update canvas after sufficient frames are loaded or initial loading is complete
                    if (this.loadedFrameCount >= this.targetLoadCount || this.loadedFrameCount >= 2) {
                        this.isInitialLoading = false;
                        if (this.initialLoadTimer) {
                            clearTimeout(this.initialLoadTimer);
                            this.initialLoadTimer = null;
                        }
                        this.node.setDirtyCanvas(true, false);
                        console.log("[VideoComparer] Initial frames loaded, updating canvas");
                    } else {
                        // Store that we have a pending update
                        this.pendingCanvasUpdate = true;
                    }
                } else {
                    // Normal loading - debounce canvas updates
                    this.debouncedCanvasUpdate();
                }
                
                // Process next item in queue
                setTimeout(() => this.processLoadingQueue(), 10);
            };
            
            img.onerror = (error) => {
                console.error(`[VideoComparer] Frame failed to load: ${videoId}_${frameData.frame_index}`, error);
                img.loading = false;
                img.failed = true;
                this.activeLoads.delete(cacheKey);
                
                // Retry if we haven't exceeded retry attempts
                if (retryCount < this.retryAttempts) {
                    console.log(`[VideoComparer] Retrying frame load: ${videoId}_${frameData.frame_index} (attempt ${retryCount + 1})`);
                    setTimeout(() => {
                        const retryItem = { ...queueItem, retryCount: retryCount + 1 };
                        this.loadingQueue.unshift(retryItem); // Add to front of queue
                        this.processLoadingQueue();
                    }, 1000 * (retryCount + 1)); // Exponential backoff
                } else {
                    console.error(`[VideoComparer] Failed to load frame after ${this.retryAttempts} attempts: ${videoId}_${frameData.frame_index}`);
                    delete cache[cacheKey];
                }
                
                // Process next item in queue
                setTimeout(() => this.processLoadingQueue(), 10);
            };
            
            // Set src with small delay to prevent browser overload
            setTimeout(() => {
                if (!img.failed) {
                    img.src = frameData.data_url;
                }
            }, 50);
            
            cache[cacheKey] = img;
            
        } catch (error) {
            console.error(`[VideoComparer] Error creating image for frame: ${videoId}_${frameData.frame_index}`, error);
            this.activeLoads.delete(cacheKey);
            setTimeout(() => this.processLoadingQueue(), 10);
            return null;
        }
    }
    
    getFrameImageForIndex(index, videoId) {
        const frames = videoId === "A" ? this.framesA : this.framesB;
        
        if (!frames.length || index < 0 || index >= frames.length) {
            return null;
        }
        
        return this.loadFrame(frames[index], videoId);
    }
    
    draw(ctx, node, width, y, height) {
        // Dramatically reduce logging frequency
        if (!this._drawCallCount) this._drawCallCount = 0;
        this._drawCallCount++;
        
        // Only log every 2000th draw call
        if (this._drawCallCount % 2000 === 0) {
            console.log("[VideoComparer] Draw called - mode:", node.properties?.comparer_mode, this._drawCallCount);
        }
        
        this.y = y;
        this.last_y = y;
        
        // Calculate the actual available height for videos
        const [nodeWidth, nodeHeight] = node.size;
        const availableHeight = nodeHeight - y - 10;
        
        // Get current comparison mode
        const mode = node.properties?.comparer_mode || "Playback";
        
        // Draw videos based on mode
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
            default: // "Playback"
                this.drawPlaybackMode(ctx, y, width, availableHeight);
                break;
        }
        
        // Draw controls based on mode
        if (["Grid", "Batch"].includes(mode) && this.maxPairs > 1) {
            this.drawBatchControls(ctx, y, width, availableHeight);
        } else {
            // Draw regular playback controls
        this.drawPlaybackControls(ctx, y + availableHeight - 45, width);
        }
    }
    
    drawPlaybackMode(ctx, y, width, availableHeight) {
        // For single video playback, show only the selected video (A by default)
        const videoId = this.node.properties?.selected_video || "A";
        const videoFrames = videoId === "A" ? this.framesA : this.framesB;
        
        if (!videoFrames.length) {
            this.drawNoVideoMessage(ctx, y, width, availableHeight - 45, videoId);
            return;
        }
        
        // Get current frame to display
        const currentFrame = Math.min(this.currentFrameIndex, videoFrames.length - 1);
        const frameImg = this.getFrameImageForIndex(currentFrame, videoId);
        
        // Show loading only if we're in initial loading AND the current frame isn't ready
        // This prevents flickering during normal playback
        if (this.isInitialLoading && (!frameImg || !frameImg.complete)) {
            this.drawLoadingMessage(ctx, y, width, availableHeight - 45);
        } else if (frameImg && frameImg.complete && !frameImg.failed) {
            // Draw the frame - subtract 45 for controls area
            this.drawFrame(ctx, frameImg, y, width, availableHeight - 45);
            
            // Draw frame counter
            this.drawFrameCounter(ctx, y + 10, width, currentFrame + 1, videoFrames.length, videoId);
        } else {
            // Frame not ready but not in initial loading - show previous frame or placeholder
            this.drawLoadingMessage(ctx, y, width, availableHeight - 45);
        }
    }
    
    drawSideBySideMode(ctx, y, width, availableHeight) {
        const halfWidth = width / 2;
        const hasVideoA = this.framesA.length > 0;
        const hasVideoB = this.framesB.length > 0;
        
        if (!hasVideoA && !hasVideoB) {
            this.drawNoVideoMessage(ctx, y, width, availableHeight, "both");
            return;
        }
        
        // Calculate current frames
        const currentFrameA = Math.min(this.currentFrameIndex, this.framesA.length - 1);
        const currentFrameB = Math.min(this.currentFrameIndex, this.framesB.length - 1);
        
        // Draw video A on the left
        if (hasVideoA) {
            const frameImgA = this.getFrameImageForIndex(currentFrameA, "A");
            if (!this.isInitialLoading && frameImgA && frameImgA.complete) {
                this.drawFrameInRegion(ctx, frameImgA, y, 0, halfWidth, availableHeight);
                this.drawFrameCounter(ctx, y + 10, halfWidth, currentFrameA + 1, this.framesA.length, "A");
            } else {
                this.drawLoadingMessage(ctx, y, halfWidth, availableHeight, 0);
            }
        } else {
            this.drawNoVideoMessage(ctx, y, halfWidth, availableHeight, "A", 0);
        }
        
        // Draw video B on the right
        if (hasVideoB) {
            const frameImgB = this.getFrameImageForIndex(currentFrameB, "B");
            if (!this.isInitialLoading && frameImgB && frameImgB.complete) {
                this.drawFrameInRegion(ctx, frameImgB, y, halfWidth, halfWidth, availableHeight);
                this.drawFrameCounter(ctx, y + 10, halfWidth, currentFrameB + 1, this.framesB.length, "B", halfWidth);
            } else {
                this.drawLoadingMessage(ctx, y, halfWidth, availableHeight, halfWidth);
            }
        } else {
            this.drawNoVideoMessage(ctx, y, halfWidth, availableHeight, "B", halfWidth);
        }
        
        // Draw separator line
        ctx.beginPath();
        ctx.moveTo(halfWidth, y);
        ctx.lineTo(halfWidth, y + availableHeight - 45);
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    
    drawStackedMode(ctx, y, width, availableHeight) {
        const halfHeight = (availableHeight - 45) / 2;
        const hasVideoA = this.framesA.length > 0;
        const hasVideoB = this.framesB.length > 0;
        
        if (!hasVideoA && !hasVideoB) {
            this.drawNoVideoMessage(ctx, y, width, availableHeight, "both");
            return;
        }
        
        // Calculate current frames
        const currentFrameA = Math.min(this.currentFrameIndex, this.framesA.length - 1);
        const currentFrameB = Math.min(this.currentFrameIndex, this.framesB.length - 1);
        
        // Draw video A on top
        if (hasVideoA) {
            const frameImgA = this.getFrameImageForIndex(currentFrameA, "A");
            if (frameImgA && frameImgA.complete) {
                this.drawFrameInRegion(ctx, frameImgA, y, 0, width, halfHeight);
                this.drawFrameCounter(ctx, y + 10, width, currentFrameA + 1, this.framesA.length, "A");
            } else {
                this.drawLoadingMessage(ctx, y, width, halfHeight);
            }
        } else {
            this.drawNoVideoMessage(ctx, y, width, halfHeight, "A");
        }
        
        // Draw video B on bottom
        if (hasVideoB) {
            const frameImgB = this.getFrameImageForIndex(currentFrameB, "B");
            if (frameImgB && frameImgB.complete) {
                this.drawFrameInRegion(ctx, frameImgB, y + halfHeight, 0, width, halfHeight);
                this.drawFrameCounter(ctx, y + halfHeight + 10, width, currentFrameB + 1, this.framesB.length, "B");
            } else {
                this.drawLoadingMessage(ctx, y + halfHeight, width, halfHeight);
            }
        } else {
            this.drawNoVideoMessage(ctx, y + halfHeight, width, halfHeight, "B");
        }
        
        // Draw separator line
        ctx.beginPath();
        ctx.moveTo(0, y + halfHeight);
        ctx.lineTo(width, y + halfHeight);
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    
    drawSliderMode(ctx, y, width, availableHeight) {
        const hasVideoA = this.framesA.length > 0;
        const hasVideoB = this.framesB.length > 0;
        
        if (!hasVideoA && !hasVideoB) {
            this.drawNoVideoMessage(ctx, y, width, availableHeight, "both");
            return;
        }
        
        // Calculate current frames
        const currentFrameA = Math.min(this.currentFrameIndex, this.framesA.length - 1);
        const currentFrameB = Math.min(this.currentFrameIndex, this.framesB.length - 1);
        
        // Draw video A as the base
        if (hasVideoA) {
            const frameImgA = this.getFrameImageForIndex(currentFrameA, "A");
            if (frameImgA && frameImgA.complete) {
                this.drawFrame(ctx, frameImgA, y, width, availableHeight - 45);
                this.drawFrameCounter(ctx, y + 10, width, currentFrameA + 1, this.framesA.length, "A");
            } else {
                this.drawLoadingMessage(ctx, y, width, availableHeight - 45);
            }
        }
        
        // Draw video B with slider if mouse is over
        if (hasVideoB && this.node.isPointerOver) {
            const frameImgB = this.getFrameImageForIndex(currentFrameB, "B");
            if (frameImgB && frameImgB.complete) {
                // Get slider position from mouse
                const sliderX = this.node.pointerOverPos[0];
                
                // Draw B with clipping
                ctx.save();
                ctx.beginPath();
                ctx.rect(0, y, sliderX, availableHeight - 45);
                ctx.clip();
                this.drawFrame(ctx, frameImgB, y, width, availableHeight - 45);
                ctx.restore();
                
                // Draw slider line
                ctx.beginPath();
                ctx.moveTo(sliderX, y);
                ctx.lineTo(sliderX, y + availableHeight - 45);
                ctx.strokeStyle = "rgba(255,255,255,0.8)";
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Draw B label near slider
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
        const hasVideoA = this.framesA.length > 0;
        const hasVideoB = this.framesB.length > 0;
        
        if (!hasVideoA && !hasVideoB) {
            this.drawNoVideoMessage(ctx, y, width, availableHeight, "both");
            return;
        }
        
        // Calculate current frames
        const currentFrameA = Math.min(this.currentFrameIndex, this.framesA.length - 1);
        const currentFrameB = Math.min(this.currentFrameIndex, this.framesB.length - 1);
        
        // Get opacity from node properties
        const opacity = this.node.properties?.onionSkinOpacity || 0.5;
        
        // Draw video A as the base
        if (hasVideoA) {
            const frameImgA = this.getFrameImageForIndex(currentFrameA, "A");
            if (frameImgA && frameImgA.complete) {
                this.drawFrame(ctx, frameImgA, y, width, availableHeight - 45);
                this.drawFrameCounter(ctx, y + 10, width, currentFrameA + 1, this.framesA.length, "A");
            } else {
                this.drawLoadingMessage(ctx, y, width, availableHeight - 45);
            }
        }
        
        // Draw video B with opacity
        if (hasVideoB) {
            const frameImgB = this.getFrameImageForIndex(currentFrameB, "B");
            if (frameImgB && frameImgB.complete) {
                ctx.save();
                ctx.globalAlpha = opacity;
                this.drawFrame(ctx, frameImgB, y, width, availableHeight - 45);
                ctx.restore();
                
                // Draw B label
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
        // This mode shows a single view that toggles between video A and B on click
        const selectedVideo = this.node.selectedVideo || "A";
        const frames = selectedVideo === "A" ? this.framesA : this.framesB;
        
        if (!frames.length) {
            this.drawNoVideoMessage(ctx, y, width, availableHeight, selectedVideo);
            return;
        }
        
        // Calculate current frame
        const currentFrame = Math.min(this.currentFrameIndex, frames.length - 1);
        const frameImg = this.getFrameImageForIndex(currentFrame, selectedVideo);
        
        if (frameImg && frameImg.complete) {
            // Draw the frame
            this.drawFrame(ctx, frameImg, y, width, availableHeight - 45);
            
            // Draw frame counter and video indicator
            this.drawFrameCounter(ctx, y + 10, width, currentFrame + 1, frames.length, selectedVideo);
            
            // Draw comparison toggle hint - positioned just above the timeline (45px controls + 5px margin)
            ctx.fillStyle = "rgba(0,0,0,0.7)";
            ctx.fillRect(width / 2 - 60, y + availableHeight - 65, 120, 24);
            ctx.fillStyle = "white";
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            ctx.fillText("Click to toggle A/B", width / 2, y + availableHeight - 49);
        } else {
            // Draw loading message
            this.drawLoadingMessage(ctx, y, width, availableHeight - 45);
        }
    }
    
    drawFrame(ctx, img, y, width, availableHeight) {
        if (!img || !img.complete) return;
        
        const imageAspect = img.naturalWidth / img.naturalHeight;
        const canvasAspect = width / availableHeight;
        
        let drawWidth, drawHeight, offsetX = 0, offsetY = 0;
        
        if (imageAspect > canvasAspect) {
            // Image is wider relative to container
            drawWidth = width;
            drawHeight = width / imageAspect;
            offsetY = (availableHeight - drawHeight) / 2;
        } else {
            // Image is taller relative to container
            drawHeight = availableHeight;
            drawWidth = availableHeight * imageAspect;
            offsetX = (width - drawWidth) / 2;
        }
        
        // Draw image centered
        ctx.drawImage(img, offsetX, y + offsetY, drawWidth, drawHeight);
    }
    
    drawFrameInRegion(ctx, img, y, x, regionWidth, regionHeight) {
        if (!img || !img.complete) return;
        
        const imageAspect = img.naturalWidth / img.naturalHeight;
        const regionAspect = regionWidth / regionHeight;
        
        let drawWidth, drawHeight, offsetX = 0, offsetY = 0;
        
        if (imageAspect > regionAspect) {
            // Image is wider relative to region
            drawWidth = regionWidth;
            drawHeight = regionWidth / imageAspect;
            offsetY = (regionHeight - drawHeight) / 2;
        } else {
            // Image is taller relative to region
            drawHeight = regionHeight;
            drawWidth = regionHeight * imageAspect;
            offsetX = (regionWidth - drawWidth) / 2;
        }
        
        // Draw image centered within region
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
        
        if (videoId === "both") {
            ctx.fillText("No videos available", offsetX + width/2, y + height/2);
        } else {
            ctx.fillText(`No video ${videoId} available`, offsetX + width/2, y + height/2);
        }
    }
    
    drawLoadingMessage(ctx, y, width, height, offsetX = 0) {
        ctx.fillStyle = "rgba(0,0,0,0.3)";
        ctx.fillRect(offsetX, y, width, height);
        
        ctx.fillStyle = "white";
        ctx.font = "14px Arial";
        ctx.textAlign = "center";
        ctx.fillText("Loading frames...", offsetX + width/2, y + height/2);
    }
    
    drawPlaybackControls(ctx, y, width) {
        const controlHeight = 45; // Increased from 30 to 45 for larger controls
        
        // Draw control background
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(0, y, width, controlHeight);
        
        // Calculate button dimensions - increased sizes
        const buttonSize = 36; // Increased from 24 to 36
        const buttonPadding = 8; // Increased from 5 to 8
        const playPauseX = 12; // Increased from 10 to 12
        const sliderStart = playPauseX + buttonSize + buttonPadding;
        const sliderWidth = width - sliderStart - buttonPadding - 60; // Increased space for frame counter
        
        // Draw play/pause button with rounded corners for better appearance
        ctx.fillStyle = "rgba(100,100,100,0.8)";
        const buttonY = y + (controlHeight - buttonSize) / 2;
        ctx.beginPath();
        ctx.roundRect(playPauseX, buttonY, buttonSize, buttonSize, 4);
        ctx.fill();
        
        // Draw play/pause icon - larger and better positioned
        ctx.fillStyle = "white";
        ctx.font = "20px Arial"; // Increased from 14px to 20px
        ctx.textAlign = "center";
        ctx.fillText(this.isPlaying ? "⏸" : "▶", playPauseX + buttonSize / 2, y + controlHeight / 2 + 7);
        
        // Draw scrubber track - taller and more visible
        const trackHeight = 8; // Increased from 4 to 8
        ctx.fillStyle = "rgba(60,60,60,0.8)";
        ctx.beginPath();
        ctx.roundRect(sliderStart, y + controlHeight / 2 - trackHeight / 2, sliderWidth, trackHeight, trackHeight / 2);
        ctx.fill();
        
        // Calculate scrubber position
        const totalFrames = Math.max(this.framesA.length, this.framesB.length);
        if (totalFrames > 0) {
            const progress = this.currentFrameIndex / (totalFrames - 1);
            const scrubberPos = sliderStart + progress * sliderWidth;
            
            // Draw scrubber handle - larger and more grabbable
            const scrubberRadius = 10; // Increased from 6 to 10
            
            // Draw scrubber shadow for depth
            ctx.fillStyle = "rgba(0,0,0,0.3)";
            ctx.beginPath();
            ctx.arc(scrubberPos + 1, y + controlHeight / 2 + 1, scrubberRadius, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw main scrubber handle
            ctx.fillStyle = "white";
            ctx.beginPath();
            ctx.arc(scrubberPos, y + controlHeight / 2, scrubberRadius, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw inner highlight for better visual feedback
            ctx.fillStyle = "rgba(255,255,255,0.8)";
            ctx.beginPath();
            ctx.arc(scrubberPos - 2, y + controlHeight / 2 - 2, scrubberRadius * 0.3, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw frame counter - larger font
            ctx.fillStyle = "white";
            ctx.font = "14px Arial"; // Increased from 12px to 14px
            ctx.textAlign = "right";
            ctx.fillText(`${this.currentFrameIndex + 1}/${totalFrames}`, width - 12, y + controlHeight / 2 + 5);
        }
    }
    
    startPlayback() {
        if (this.isPlaying) return;
        
        // Check if we have enough frames loaded for smooth playback
        const minFramesNeeded = Math.min(3, Math.max(this.framesA.length, this.framesB.length));
        const hasMinFramesA = this.framesA.length === 0 || this.getLoadedFrameCount("A") >= minFramesNeeded;
        const hasMinFramesB = this.framesB.length === 0 || this.getLoadedFrameCount("B") >= minFramesNeeded;
        
        if (!hasMinFramesA || !hasMinFramesB) {
            console.log("[VideoComparer] Not enough frames loaded, preloading before playback");
            // Force preload more frames before starting
            this.preloadForPlayback(() => {
                this.startPlaybackImmediate();
            });
            return;
        }
        
        this.startPlaybackImmediate();
    }
    
    getLoadedFrameCount(videoId) {
        const cache = videoId === "A" ? this.loadedFramesA : this.loadedFramesB;
        return Object.values(cache).filter(img => img && img.complete && !img.failed).length;
    }
    
    preloadForPlayback(callback) {
        const framesToLoad = Math.min(5, Math.max(this.framesA.length, this.framesB.length));
        let loadedCount = 0;
        let targetCount = 0;
        
        // Count how many frames we need to load
        for (let i = 0; i < framesToLoad; i++) {
            if (this.framesA.length > i) targetCount++;
            if (this.framesB.length > i) targetCount++;
        }
        
        const onFrameLoaded = () => {
            loadedCount++;
            if (loadedCount >= targetCount) {
                console.log("[VideoComparer] Preload for playback complete");
                callback();
            }
        };
        
        // Load frames with immediate feedback
        for (let i = 0; i < framesToLoad; i++) {
            if (this.framesA.length > i) {
                const img = this.loadFrame(this.framesA[i], "A");
                if (img && img.complete) {
                    onFrameLoaded();
                } else if (img) {
                    const originalOnload = img.onload;
                    img.onload = () => {
                        if (originalOnload) originalOnload();
                        onFrameLoaded();
                    };
                }
            }
            if (this.framesB.length > i) {
                const img = this.loadFrame(this.framesB[i], "B");
                if (img && img.complete) {
                    onFrameLoaded();
                } else if (img) {
                    const originalOnload = img.onload;
                    img.onload = () => {
                        if (originalOnload) originalOnload();
                        onFrameLoaded();
                    };
                }
            }
        }
        
        // Fallback timeout in case some frames fail to load
        setTimeout(() => {
            if (loadedCount < targetCount) {
                console.log("[VideoComparer] Preload timeout, starting playback anyway");
                callback();
            }
        }, 2000);
    }
    
    startPlaybackImmediate() {
        console.log("[VideoComparer] Starting playback at FPS:", this.fps);
        this.isPlaying = true;
        this.lastFrameTime = performance.now();
        
        // Ensure we're not in initial loading state
        this.isInitialLoading = false;
        
        const playbackLoop = () => {
            const now = performance.now();
            const frameDelay = 1000 / this.fps;
            
            if (now - this.lastFrameTime >= frameDelay) {
                this.advanceFrame();
                this.lastFrameTime = now;
            }
            
            if (this.isPlaying) {
                this.animationFrame = requestAnimationFrame(playbackLoop);
            }
        };
        
        this.animationFrame = requestAnimationFrame(playbackLoop);
        this.node.setDirtyCanvas(true, false);
    }
    
    stopPlayback() {
        if (!this.isPlaying) return;
        
        console.log("[VideoComparer] Stopping playback");
        this.isPlaying = false;
        
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        
        this.node.setDirtyCanvas(true, false);
    }
    
    togglePlayback() {
        if (this.isPlaying) {
            this.stopPlayback();
        } else {
            this.startPlayback();
        }
    }
    
    advanceFrame() {
        const totalFrames = Math.max(this.framesA.length, this.framesB.length);
        if (totalFrames <= 1) return;
        
        this.currentFrameIndex++;
        
        // Loop back to beginning if reached the end
        if (this.currentFrameIndex >= totalFrames) {
            this.currentFrameIndex = 0;
        }
        
        // Preload next frames
        this.preloadNextFrames();
        
        // Update display
        this.node.setDirtyCanvas(true, false);
    }

    previousFrame() {
        const totalFrames = Math.max(this.framesA.length, this.framesB.length);
        if (totalFrames === 0) return; // No frames to navigate
        // If only one frame, or currently at the first frame of multiple, and try to go previous, loop to end.
        
        this.currentFrameIndex--;
        if (this.currentFrameIndex < 0) {
            this.currentFrameIndex = totalFrames > 0 ? totalFrames - 1 : 0; // Loop to the last frame or 0 if no frames
        }
        
        this.preloadNextFrames();
        this.node.setDirtyCanvas(true, false);
    }

    nextFrame() {
        const totalFrames = Math.max(this.framesA.length, this.framesB.length);
        if (totalFrames === 0) return; // No frames to navigate
        // If only one frame, or currently at the last frame of multiple, and try to go next, loop to start.

        this.currentFrameIndex++;
        if (this.currentFrameIndex >= totalFrames) {
            this.currentFrameIndex = 0; // Loop to the first frame
        }
        
        this.preloadNextFrames();
        this.node.setDirtyCanvas(true, false);
    }
    
    preloadNextFrames() {
        const totalFrames = Math.max(this.framesA.length, this.framesB.length);
        if (totalFrames <= 1) return;
        
        // Preload more frames ahead for smoother playback
        const framesToPreload = Math.min(3, totalFrames); // Preload 3 frames ahead instead of 1
        
        for (let i = 1; i <= framesToPreload; i++) {
            const nextIndex = (this.currentFrameIndex + i) % totalFrames;
            
            // Load without delays during playback for smoothness
            if (this.framesA.length > nextIndex) {
                this.loadFrame(this.framesA[nextIndex], "A");
            }
            if (this.framesB.length > nextIndex) {
                this.loadFrame(this.framesB[nextIndex], "B");
            }
        }
        
        // Also preload 1 frame behind
        const prevIndex = (this.currentFrameIndex - 1 + totalFrames) % totalFrames;
        if (this.framesA.length > prevIndex) {
            this.loadFrame(this.framesA[prevIndex], "A");
        }
        if (this.framesB.length > prevIndex) {
            this.loadFrame(this.framesB[prevIndex], "B");
        }
    }
    
    seekToFrame(frameIndex) {
        const totalFrames = Math.max(this.framesA.length, this.framesB.length);
        if (totalFrames <= 1) return;
        
        // Clamp index to valid range
        const newIndex = Math.max(0, Math.min(frameIndex, totalFrames - 1));
        
        if (this.currentFrameIndex !== newIndex) {
            this.currentFrameIndex = newIndex;
            this.preloadNextFrames();
            this.node.setDirtyCanvas(true, false);
        }
    }
    
    seekToPosition(x, width) {
        // Calculate button dimensions for scrubber position - updated to match new sizes
        const buttonSize = 36; // Updated from 24 to 36
        const buttonPadding = 8; // Updated from 5 to 8
        const playPauseX = 12; // Updated from 10 to 12
        const sliderStart = playPauseX + buttonSize + buttonPadding;
        const sliderWidth = width - sliderStart - buttonPadding - 60;
        
        // Convert x position to progress (0-1)
        let progress = (x - sliderStart) / sliderWidth;
        progress = Math.max(0, Math.min(1, progress)); // Clamp to 0-1
        
        // Convert progress to frame index
        const totalFrames = Math.max(this.framesA.length, this.framesB.length);
        const frameIndex = Math.floor(progress * (totalFrames - 1));
        
        this.seekToFrame(frameIndex);
    }
    
    mouse(event, pos, node) {
        // Handle mouse events for the widget
        const [x, y] = pos;
        const width = node.size[0];
        const height = node.size[1];
        const availableHeight = height - this.y - 10;
        const controlsY = this.y + availableHeight - 45;
        const mode = node.properties?.comparer_mode || "Playback";
        
        // Handle batch mode controls
        if (["Grid", "Batch"].includes(mode) && this.maxPairs > 1) {
            const batchControlsY = this.y + availableHeight - 30;
            
            if (y >= batchControlsY && event.type === "pointerdown") {
                const buttonHeight = 20;
                const buttonY = batchControlsY + 5;
                
                // Play/pause button
                const playButtonX = 80;
                const buttonWidth = 60;
                if (x >= playButtonX && x <= playButtonX + buttonWidth && 
                    y >= buttonY && y <= buttonY + buttonHeight) {
                    this.togglePlayback();
                    return true;
                }
                
                // Batch navigation buttons (only for batch mode)
                if (mode === "Batch") {
                    // Previous button
                    if (x >= 150 && x <= 210 && y >= buttonY && y <= buttonY + buttonHeight) {
                        this.previousBatchPage();
                    return true;
                    }
                    
                    // Next button  
                    if (x >= 220 && x <= 280 && y >= buttonY && y <= buttonY + buttonHeight) {
                        this.nextBatchPage();
                return true;
                    }
                }
                
            return true;
        }
        
        return false;
    }
    
        // Add robust scrubbing state management
        if (this.isScrubbing && event.type === "pointermove") {
            if (event.buttons === 0 || event.buttons === undefined) {
                console.log("[VideoComparer] Detected mouse move without button press, stopping scrubbing");
                this.isScrubbing = false;
                return false;
            }
        }
        
        // Handle pointerup and other release events globally
        if (event.type === "pointerup" || event.type === "mouseup" || event.type === "pointercancel") {
            if (this.isScrubbing) {
                console.log("[VideoComparer] Stopping scrubbing on:", event.type);
                this.isScrubbing = false;
                return true;
            }
        }
        
        // Check if click is in controls area (for non-batch modes)
        if (y >= controlsY) {
            if (event.type === "pointerdown") {
                // Check if click is on play/pause button
                const buttonSize = 36;
                const playPauseX = 12;
                if (x >= playPauseX && x <= playPauseX + buttonSize && 
                    y >= controlsY + (45 - buttonSize) / 2 && y <= controlsY + (45 + buttonSize) / 2) {
                    this.togglePlayback();
                    return true;
                }
                
                // Check if click is on scrubber
                const buttonPadding = 8;
                const sliderStart = playPauseX + buttonSize + buttonPadding;
                const sliderWidth = width - sliderStart - buttonPadding - 60;
                
                if (x >= sliderStart && x <= sliderStart + sliderWidth) {
                    console.log("[VideoComparer] Starting scrubbing");
                    this.isScrubbing = true;
                    this.seekToPosition(x, width);
                    
                    // Set a safety timeout to auto-stop scrubbing
                    this.scrubTimeout = setTimeout(() => {
                        if (this.isScrubbing) {
                            console.log("[VideoComparer] Auto-stopping scrubbing due to timeout");
                            this.isScrubbing = false;
                        }
                    }, 5000);
                    
                    return true;
                }
            }
        }
        
        // Handle pointermove during scrubbing
        if (event.type === "pointermove" && this.isScrubbing) {
            if (event.buttons > 0) {
                this.seekToPosition(x, width);
                return true;
            } else {
                console.log("[VideoComparer] No button pressed during move, stopping scrubbing");
                this.isScrubbing = false;
                if (this.scrubTimeout) {
                    clearTimeout(this.scrubTimeout);
                    this.scrubTimeout = null;
                }
                return false;
            }
        }
        
        // Handle sync compare mode click
        if (event.type === "pointerdown" && 
            y >= this.y && y < controlsY && 
            node.properties?.comparer_mode === "Sync Compare") {
            
            node.selectedVideo = node.selectedVideo === "A" ? "B" : "A";
            node.setDirtyCanvas(true, false);
            return true;
        }
        
        // Handle slider mode hover
        if (event.type === "pointermove" && 
            !this.isScrubbing &&
            y >= this.y && y < controlsY && 
            node.properties?.comparer_mode === "Slider") {
            
            node.pointerOverPos = [...pos];
            node.isPointerOver = true;
            node.setDirtyCanvas(true, false);
            return true;
        }
        
        return false;
    }
    
    computeSize(width) {
        const mode = this.node?.properties?.comparer_mode || "Playback";
        
        // Base height calculation
        let height = Math.max(300, width * 0.75);
        
        // Adjust height based on layout mode
        switch (mode) {
            case "Stacked":
                height = Math.max(500, width * 1.2);
                break;
            case "Side-by-Side":
                height = Math.max(300, width * 0.6);
                break;
            case "Grid":
                // Grid mode needs more height to show multiple pairs
                const pairs = Math.min(this.maxPairs || 1, 8);
                const cols = Math.ceil(Math.sqrt(pairs * 2));
                const rows = Math.ceil((pairs * 2) / cols);
                height = Math.max(400, (width / cols) * rows + 60);
                break;
            case "Batch":
                // Batch mode shows multiple pairs vertically
                const visiblePairs = Math.min(this.maxPairs || 1, this.pairsPerPage);
                height = Math.max(500, visiblePairs * (width * 0.4) + 60);
                break;
            default:
                height = Math.max(300, width * 0.75);
                break;
        }
        
        // Add space for controls
        height += 55;
        
        return [width, height];
    }
    
    // Cleanup method
    onRemoved() {
        console.log("[VideoComparer] Cleaning up widget resources");
        
        // Clear scrub timeout
        if (this.scrubTimeout) {
            clearTimeout(this.scrubTimeout);
            this.scrubTimeout = null;
        }
        
        // Reset scrubbing state
        this.isScrubbing = false;
        
        // Remove from global instances tracking
        if (window.videoComparerInstances) {
            const index = window.videoComparerInstances.indexOf(this);
            if (index > -1) {
                window.videoComparerInstances.splice(index, 1);
            }
        }
        
        // Clean up all cached frames
        this.cleanupAllFrames();
        
        if (this.playbackInterval) {
            clearInterval(this.playbackInterval);
            this.playbackInterval = null;
        }
    }
    
    cleanupAllFrames() {
        // Clear all frame caches
        Object.keys(this.loadedFramesA).forEach(key => {
            if (this.loadedFramesA[key] && this.loadedFramesA[key].src) {
                this.loadedFramesA[key].src = "";
            }
            delete this.loadedFramesA[key];
        });
        
        Object.keys(this.loadedFramesB).forEach(key => {
            if (this.loadedFramesB[key] && this.loadedFramesB[key].src) {
                this.loadedFramesB[key].src = "";
            }
            delete this.loadedFramesB[key];
        });
        
        this.loadedFramesA = {};
        this.loadedFramesB = {};
    }
    
    // Add memory management methods
    cleanupFrameCache(cache, videoId) {
        const keys = Object.keys(cache);
        if (keys.length <= 10) return;
        
        console.log(`[VideoComparer] Cleaning up frame cache for video ${videoId}, current size: ${keys.length}`);
        
        // Sort keys by frame index and keep only recent frames around current position
        const currentIndex = this.currentFrameIndex;
        const keysToRemove = keys.filter(key => {
            const frameIndex = parseInt(key.split('_')[1]);
            return Math.abs(frameIndex - currentIndex) > 5; // Keep frames within 5 of current
        });
        
        keysToRemove.forEach(key => {
            if (cache[key] && !cache[key].loading) {
                if (cache[key].src) {
                    cache[key].src = "";
                }
                delete cache[key];
            }
        });
        
        console.log(`[VideoComparer] Cleaned up ${keysToRemove.length} frames from ${videoId} cache`);
    }
    
    cleanupOldestFrames() {
        console.log("[VideoComparer] Cleaning up oldest frames due to memory pressure");
        
        const cleanupCache = (cache, videoId) => {
            const keys = Object.keys(cache);
            if (keys.length <= 5) return; // Keep minimum frames
            
            // Sort by distance from current frame, remove furthest ones
            const currentIndex = this.currentFrameIndex;
            const sortedKeys = keys.sort((a, b) => {
                const indexA = parseInt(a.split('_')[1]);
                const indexB = parseInt(b.split('_')[1]);
                const distA = Math.abs(indexA - currentIndex);
                const distB = Math.abs(indexB - currentIndex);
                return distB - distA; // Sort by distance, furthest first
            });
            
            // Remove half of the furthest frames
            const toRemove = sortedKeys.slice(0, Math.floor(keys.length / 2));
            toRemove.forEach(key => {
                if (cache[key] && !cache[key].loading) {
                    if (cache[key].src) {
                        cache[key].src = "";
                    }
                    delete cache[key];
                }
            });
            
            console.log(`[VideoComparer] Cleaned ${toRemove.length} old frames from ${videoId} cache`);
        };
        
        cleanupCache(this.loadedFramesA, "A");
        cleanupCache(this.loadedFramesB, "B");
    }

    // Add debounced canvas update method
    debouncedCanvasUpdate() {
        // Clear any existing debounce timer
        if (this.canvasUpdateTimer) {
            clearTimeout(this.canvasUpdateTimer);
        }
        
        // Set a new timer with a small delay to batch updates
        this.canvasUpdateTimer = setTimeout(() => {
            this.node.setDirtyCanvas(true, false);
            this.canvasUpdateTimer = null;
        }, 50); // 50ms delay to batch multiple frame loads
    }

    // New method for updating selected video pair
    updateSelectedPair() {
        if (this.maxPairs === 0) return;
        
        const pairIndex = Math.min(this.currentPairIndex, this.maxPairs - 1);
        
        // Update current video pair
        this.videoA = this.videosA[pairIndex] || null;
        this.videoB = this.videosB[pairIndex] || null;
        this.framesA = this.videoA ? this.videoA.frames : [];
        this.framesB = this.videoB ? this.videoB.frames : [];
        this.fps = (this.videoA && this.videoA.fps) || (this.videoB && this.videoB.fps) || 8;
        
        // Reset frame index when switching pairs
        this.currentFrameIndex = 0;
        
        // Clear frame caches for new pair
        this.loadedFramesA = {};
        this.loadedFramesB = {};
        
        // Preload frames for new pair
        if (this.framesA.length > 0 || this.framesB.length > 0) {
            this.preloadInitialFrames();
        }
        
        console.log(`[VideoComparer] Updated to pair ${pairIndex + 1}/${this.maxPairs}`);
    }

    // New method: Draw grid mode for multiple video pairs
    drawGridMode(ctx, y, width, availableHeight) {
        const maxPairsToShow = Math.min(this.maxPairs, 8); // Limit to 8 pairs for performance
        const cols = Math.ceil(Math.sqrt(maxPairsToShow * 2)); // 2 videos per pair
        const rows = Math.ceil((maxPairsToShow * 2) / cols);
        
        const cellWidth = width / cols;
        const cellHeight = (availableHeight - 40) / rows; // Reserve space for controls
        
        let cellIndex = 0;
        
        for (let pairIndex = 0; pairIndex < maxPairsToShow; pairIndex++) {
            const videoA = this.videosA[pairIndex];
            const videoB = this.videosB[pairIndex];
            
            // Calculate current frames for this pair
            const currentFrameA = Math.min(this.currentFrameIndex, (videoA?.frames.length || 1) - 1);
            const currentFrameB = Math.min(this.currentFrameIndex, (videoB?.frames.length || 1) - 1);
            
            // Draw video A
            if (videoA && videoA.frames.length > 0) {
                const frameImg = this.getFrameImageForPair(currentFrameA, "A", pairIndex);
                const col = cellIndex % cols;
                const row = Math.floor(cellIndex / cols);
                const cellX = col * cellWidth;
                const cellY = y + row * cellHeight;
                
                this.drawVideoInCell(ctx, frameImg, videoA.name, cellY, cellX, cellWidth, cellHeight);
                cellIndex++;
            }
            
            // Draw video B
            if (videoB && videoB.frames.length > 0) {
                const frameImg = this.getFrameImageForPair(currentFrameB, "B", pairIndex);
                const col = cellIndex % cols;
                const row = Math.floor(cellIndex / cols);
                const cellX = col * cellWidth;
                const cellY = y + row * cellHeight;
                
                this.drawVideoInCell(ctx, frameImg, videoB.name, cellY, cellX, cellWidth, cellHeight);
                cellIndex++;
            }
        }
    }

    // New method: Draw batch mode for multiple video pairs
    drawBatchMode(ctx, y, width, availableHeight) {
        const pairHeight = (availableHeight - 40) / this.pairsPerPage; // Reserve space for controls
        const startPairIndex = this.currentBatchPage * this.pairsPerPage;
        const endPairIndex = Math.min(startPairIndex + this.pairsPerPage, this.maxPairs);
        
        for (let i = 0; i < this.pairsPerPage; i++) {
            const pairIndex = startPairIndex + i;
            if (pairIndex >= this.maxPairs) break;
            
            const videoA = this.videosA[pairIndex];
            const videoB = this.videosB[pairIndex];
            const pairY = y + i * pairHeight;
            
            // Calculate current frames for this pair
            const currentFrameA = Math.min(this.currentFrameIndex, (videoA?.frames.length || 1) - 1);
            const currentFrameB = Math.min(this.currentFrameIndex, (videoB?.frames.length || 1) - 1);
            
            // Draw video A on the left
            if (videoA && videoA.frames.length > 0) {
                const frameImg = this.getFrameImageForPair(currentFrameA, "A", pairIndex);
                this.drawVideoInPair(ctx, frameImg, videoA.name, pairY, 0, width / 2, pairHeight, 0);
            }
            
            // Draw video B on the right
            if (videoB && videoB.frames.length > 0) {
                const frameImg = this.getFrameImageForPair(currentFrameB, "B", pairIndex);
                this.drawVideoInPair(ctx, frameImg, videoB.name, pairY, width / 2, width / 2, pairHeight, 1);
            }
            
            // Draw separator between pairs
            if (i < this.pairsPerPage - 1 && pairIndex < this.maxPairs - 1) {
                ctx.beginPath();
                ctx.moveTo(0, pairY + pairHeight);
                ctx.lineTo(width, pairY + pairHeight);
                ctx.strokeStyle = "rgba(255,255,255,0.3)";
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }
    }

    // New method: Draw video frame in a cell (for grid mode)
    drawVideoInCell(ctx, frameImg, videoName, y, x, cellWidth, cellHeight, label) {
        if (!frameImg || !frameImg.complete || frameImg.failed) {
            // Draw placeholder
            ctx.fillStyle = "rgba(100,100,100,0.5)";
            ctx.fillRect(x + 2, y + 2, cellWidth - 4, cellHeight - 4);
            ctx.fillStyle = "white";
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            ctx.fillText("Loading...", x + cellWidth/2, y + cellHeight/2);
            return;
        }

        const padding = 4;
        const usableWidth = cellWidth - padding * 2;
        const usableHeight = cellHeight - padding * 2;
        
        const imageAspect = frameImg.naturalWidth / frameImg.naturalHeight;
        const cellAspect = usableWidth / usableHeight;
        
        let targetWidth, targetHeight;
        
        if (imageAspect > cellAspect) {
            targetWidth = usableWidth;
            targetHeight = usableWidth / imageAspect;
        } else {
            targetHeight = usableHeight;
            targetWidth = usableHeight * imageAspect;
        }

        const destX = x + padding + (usableWidth - targetWidth) / 2;
        const destY = y + padding + (usableHeight - targetHeight) / 2;

        ctx.save();
        
        // Draw border
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x + padding, y + padding, usableWidth, usableHeight);
        
        // Draw video frame
        ctx.drawImage(
            frameImg,
            0, 0, frameImg.naturalWidth, frameImg.naturalHeight,
            destX, destY, targetWidth, targetHeight
        );
        
        // Draw label
        ctx.fillStyle = "rgba(0,0,0,0.8)";
        ctx.fillRect(destX, destY, 25, 18);
        ctx.fillStyle = "white";
        ctx.font = "11px Arial";
        ctx.textAlign = "center";
        ctx.fillText(videoName, destX + 12, destY + 13);

        ctx.restore();
    }

    // New method: Draw video frame in a pair (for batch mode)
    drawVideoInPair(ctx, frameImg, videoName, y, x, pairWidth, pairHeight, videoIndex) {
        if (!frameImg || !frameImg.complete || frameImg.failed) {
            // Draw placeholder
            ctx.fillStyle = "rgba(100,100,100,0.5)";
            ctx.fillRect(x + 2, y + 2, pairWidth - 4, pairHeight - 4);
            ctx.fillStyle = "white";
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            ctx.fillText("Loading...", x + pairWidth/2, y + pairHeight/2);
            return;
        }

        const padding = 2;
        const usableWidth = pairWidth - padding * 2;
        const usableHeight = pairHeight - padding * 2;
        
        const imageAspect = frameImg.naturalWidth / frameImg.naturalHeight;
        const pairAspect = usableWidth / usableHeight;
        
        let targetWidth, targetHeight;
        
        if (imageAspect > pairAspect) {
            targetWidth = usableWidth;
            targetHeight = usableWidth / imageAspect;
        } else {
            targetHeight = usableHeight;
            targetWidth = usableHeight * imageAspect;
        }

        const destX = x + padding + (usableWidth - targetWidth) / 2;
        const destY = y + padding + (usableHeight - targetHeight) / 2;

        ctx.save();
        
        // Draw video frame
        ctx.drawImage(
            frameImg,
            0, 0, frameImg.naturalWidth, frameImg.naturalHeight,
            destX, destY, targetWidth, targetHeight
        );
        
        // Draw label
        ctx.fillStyle = "rgba(0,0,0,0.8)";
        ctx.fillRect(destX, destY, 25, 18);
        ctx.fillStyle = "white";
        ctx.font = "12px Arial";
        ctx.textAlign = "center";
        ctx.fillText(videoName, destX + 12, destY + 13);
        
        // Draw separator line for side-by-side in batch mode
        if (videoIndex === 0 && pairWidth < this.node.size[0]) {
            ctx.beginPath();
            ctx.moveTo(x + pairWidth, y);
            ctx.lineTo(x + pairWidth, y + pairHeight);
            ctx.strokeStyle = "rgba(255,255,255,0.5)";
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        ctx.restore();
    }

    // New method: Draw batch controls
    drawBatchControls(ctx, y, width, availableHeight) {
        const controlY = y + availableHeight - 30;
        const mode = this.node.properties?.comparer_mode || "Playback";
        
        ctx.save();
        
        // Draw control background
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(0, controlY, width, 30);
        
        ctx.fillStyle = "white";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        
        // Draw current frame info and controls
        ctx.fillText(`Frame: ${this.currentFrameIndex + 1}`, 10, controlY + 18);
        
        // Play/pause button
        const buttonWidth = 60;
        const buttonHeight = 20;
        const buttonY = controlY + 5;
        const playButtonX = 80;
        
        ctx.fillStyle = this.isPlaying ? "rgba(200,100,100,0.8)" : "rgba(100,200,100,0.8)";
        ctx.fillRect(playButtonX, buttonY, buttonWidth, buttonHeight);
        ctx.fillStyle = "white";
        ctx.textAlign = "center";
        ctx.fillText(this.isPlaying ? "⏸ Pause" : "▶ Play", playButtonX + buttonWidth/2, buttonY + 14);
        
        if (mode === "Grid") {
            ctx.textAlign = "right";
            ctx.fillText(`Showing ${Math.min(this.maxPairs, 8)} pairs`, width - 10, controlY + 18);
        } else if (mode === "Batch") {
            // Batch navigation
            ctx.fillStyle = "rgba(100,100,100,0.8)";
            ctx.fillRect(150, buttonY, buttonWidth, buttonHeight);
            ctx.fillStyle = "white";
            ctx.textAlign = "center";
            ctx.fillText("◀ Prev", 150 + buttonWidth/2, buttonY + 14);
            
            ctx.fillStyle = "rgba(100,100,100,0.8)";
            ctx.fillRect(220, buttonY, buttonWidth, buttonHeight);
            ctx.fillStyle = "white";
            ctx.fillText("Next ▶", 220 + buttonWidth/2, buttonY + 14);
            
            const startPair = this.currentBatchPage * this.pairsPerPage + 1;
            const endPair = Math.min((this.currentBatchPage + 1) * this.pairsPerPage, this.maxPairs);
            ctx.textAlign = "right";
            ctx.fillText(`Pairs ${startPair}-${endPair} of ${this.maxPairs}`, width - 10, controlY + 18);
        }
        
        ctx.restore();
    }

    // New method: Get frame image for specific pair
    getFrameImageForPair(frameIndex, videoId, pairIndex) {
        const videos = videoId === "A" ? this.videosA : this.videosB;
        const video = videos[pairIndex];
        
        if (!video || !video.frames || frameIndex >= video.frames.length) {
            return null;
        }
        
        const frameData = video.frames[frameIndex];
        const cacheKey = `${videoId}_${pairIndex}_${frameData.frame_index}`;
        
        // Use same caching mechanism but with pair-specific keys
        return this.loadFrame(frameData, `${videoId}_${pairIndex}`);
    }

    // New method: Navigate to next batch page
    nextBatchPage() {
        if (this.currentBatchPage < this.maxBatchPages - 1) {
            this.currentBatchPage++;
            this.node.setDirtyCanvas(true, false);
        }
    }

    // New method: Navigate to previous batch page
    previousBatchPage() {
        if (this.currentBatchPage > 0) {
            this.currentBatchPage--;
            this.node.setDirtyCanvas(true, false);
        }
    }
}

app.registerExtension({
    name: "VideoComparer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Check for both possible node names
        if (nodeData.name !== "Video Comparer" && nodeData.name !== "VideoComparer") {
            return;
        }
        
        // Add properties
        nodeType.prototype.properties = nodeType.prototype.properties || {};
        nodeType.prototype.properties.comparer_mode = "Playback";
        nodeType.prototype.properties.selected_video = "A";
        nodeType.prototype.properties.onionSkinOpacity = 0.5;
        nodeType.prototype.properties.user_resized = false; // Add flag to track if user has manually resized
        
        nodeType["@comparer_mode"] = {
            type: "combo",
            values: ["Playback", "Side-by-Side", "Stacked", "Slider", "Onion Skin", "Sync Compare"],
        };
        
        nodeType["@selected_video"] = {
            type: "combo",
            values: ["A", "B"],
        };

        // Store the original onDrawForeground function if it exists
        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        
        // Add our own onDrawForeground function for gradient background
        nodeType.prototype.onDrawForeground = function(ctx) {
            // Call the original onDrawForeground if it exists
            if (origOnDrawForeground) {
                origOnDrawForeground.apply(this, arguments);
            }
            
            // Draw our custom gradient title
            drawGradientTitle(this, ctx);
        };

        // Clean up resources when node is removed
        const origOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function() {
            if (origOnRemoved) {
                origOnRemoved.apply(this, arguments);
            }
            
            // Clean up widget resources
            if (this.videoComparerWidget) {
                this.videoComparerWidget.onRemoved();
            }
            
            // Clear cached canvases to prevent memory leaks
            CACHE.titleCanvas = null;
            CACHE.titleCtx = null;
            CACHE.collapsed.canvas = null;
            CACHE.collapsed.ctx = null;
        };

        // Initialize state variables
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            
            // Ensure properties are properly initialized
            this.properties = this.properties || {};
            if (!this.properties.comparer_mode) {
                this.properties.comparer_mode = "Playback";
            }
            if (!this.properties.selected_video) {
                this.properties.selected_video = "A";
            }
            if (this.properties.onionSkinOpacity === undefined) {
                this.properties.onionSkinOpacity = 0.5;
            }
            
            // Initialize state variables
            this.isPointerOver = false;
            this.pointerOverPos = [0, 0];
            this.selectedVideo = "A";
            
            // Add layout control widget
            this.layoutWidget = this.addWidget("combo", "Comparison Mode", this.properties.comparer_mode, (value) => {
                this.properties.comparer_mode = value;
                this.updateControlsVisibility();
                this.setDirtyCanvas(true, false);
            }, {
                values: ["Playback", "Side-by-Side", "Stacked", "Slider", "Onion Skin", "Sync Compare"]
            });

            // Add video selector widget for Playback mode
            this.videoSelectorWidget = this.addWidget("combo", "Video", this.properties.selected_video, (value) => {
                this.properties.selected_video = value;
                this.selectedVideo = value;
                this.setDirtyCanvas(true, false);
            }, {
                values: ["A", "B"]
            });

            // Add Onion Skin opacity slider
            this.onionSkinOpacitySlider = this.addWidget("slider", "Opacity B", this.properties.onionSkinOpacity, (value) => {
                this.properties.onionSkinOpacity = parseFloat(value);
                this.setDirtyCanvas(true, false);
            }, {
                min: 0.0,
                max: 1.0,
                step: 0.01
            });

            // Create the custom widget
            this.videoComparerWidget = this.addCustomWidget(new VideoComparerWidget("video_comparer", this));
            
            // Initialize controls visibility
            this.updateControlsVisibility();
            
            // Force node to be larger to accommodate the widget
            const initialSize = this.computeSize();
            this.setSize([Math.max(400, initialSize[0]), Math.max(300, initialSize[1])]);
            
            // Force immediate redraw
            this.setDirtyCanvas(true, true);
        };

        // Method to show/hide controls based on mode
        nodeType.prototype.updateControlsVisibility = function() {
            const mode = this.properties.comparer_mode;
            
            // Show video selector only in Playback mode
            if (this.videoSelectorWidget) {
                this.videoSelectorWidget.type = mode === "Playback" ? "combo" : "hidden";
            }
            
            // Show opacity slider only in Onion Skin mode
            if (this.onionSkinOpacitySlider) {
                this.onionSkinOpacitySlider.type = mode === "Onion Skin" ? "slider" : "hidden";
            }
        };

        // Override computeSize to account for the widget
        const originalComputeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function(out) {
            const size = originalComputeSize ? originalComputeSize.apply(this, arguments) : [400, 300];
            if (this.videoComparerWidget) {
                const widgetSize = this.videoComparerWidget.computeSize(size[0]);
                
                // Calculate additional space needed for controls
                let extraHeight = 60; // Base padding for layout widget + title bar
                
                const mode = this.properties.comparer_mode;
                
                if (mode === "Playback") {
                    extraHeight += 30; // Additional space for video selector
                } else if (mode === "Onion Skin") {
                    extraHeight += 30; // Additional space for opacity slider
                }
                
                size[1] = Math.max(size[1], widgetSize[1] + extraHeight);
            }
            return size;
        };

        // Override onExecuted to handle video data
        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            console.log("[VideoComparer] onExecuted called!", this.id);
            console.log("[VideoComparer] Message type:", typeof message);
            console.log("[VideoComparer] Message structure:", message ? Object.keys(message).join(", ") : "null");
            
            if (message && message.ui) {
                console.log("[VideoComparer] UI data structure:", message.ui ? Object.keys(message.ui).join(", ") : "no ui object");
            }
            
            // Call the original onExecuted first
            let result;
            if (originalOnExecuted) {
                console.log("[VideoComparer] Calling original onExecuted");
                result = originalOnExecuted.apply(this, arguments);
                console.log("[VideoComparer] Original onExecuted returned:", result ? "result object" : "falsy value");
            }
            
            // Now handle our custom logic
            if (message && typeof message === 'object') {
                console.log("[VideoComparer] Processing message for custom widget");
                console.log("[VideoComparer] Full message:", JSON.stringify(message, null, 2));
                
                // Check for video data in different possible locations
                let videoData = null;
                if (message && message.ui && message.ui.video_data) {
                    videoData = message.ui.video_data;
                    console.log("[VideoComparer] Found video_data in message.ui:", videoData);
                } else if (message && message.video_data) {
                    videoData = message.video_data;
                    console.log("[VideoComparer] Found video_data in message root:", videoData);
                } else {
                    console.log("[VideoComparer] No video_data found. Message structure:");
                    console.log("[VideoComparer] message.ui keys:", message.ui ? Object.keys(message.ui) : "no ui");
                    console.log("[VideoComparer] message root keys:", message ? Object.keys(message) : "no message");
                    if (message && message.ui) {
                        console.log("[VideoComparer] message.ui content:", message.ui);
                    }
                }
                
                if (videoData) {
                    console.log("[VideoComparer] Processing video data with", videoData.length, "videos");
                    
                    // Extract ALL the data types from the message
                    const differenceData = message.ui && message.ui.difference_data ? message.ui.difference_data : {};
                    const histogramData = message.ui && message.ui.histogram_data ? message.ui.histogram_data : [];
                    
                    console.log("[VideoComparer] Extracted difference data:", differenceData);
                    console.log("[VideoComparer] Extracted histogram data:", histogramData);
                    
                    if (this.videoComparerWidget) {
                        console.log("[VideoComparer] Setting widget value, widget exists:", !!this.videoComparerWidget);
                        
                        // Pass ALL the data to the widget
                        this.videoComparerWidget.value = { 
                            video_data: videoData,
                            difference_data: differenceData,
                            histogram_data: histogramData
                        };
                        
                        console.log("[VideoComparer] Widget value set, triggering redraw");
                        
                        // Don't force a resize, just redraw
                        this.setDirtyCanvas(true, true);
                        
                        // Double check that widget is attached and visible
                        console.log("[VideoComparer] Widget after value set:", 
                            this.videoComparerWidget ? 
                            `Widget exists, name: ${this.videoComparerWidget.name}, y: ${this.videoComparerWidget.y}` : 
                            "Widget doesn't exist");
                    } else {
                        console.error("[VideoComparer] No videoComparerWidget found on node!");
                        console.log("[VideoComparer] All widgets:", this.widgets ? this.widgets.map(w => w.name).join(', ') : "No widgets array");
                        
                        // Try to create the widget if it doesn't exist
                        if (!this.widgets || !this.widgets.find(w => w.name === "video_comparer")) {
                            console.log("[VideoComparer] Attempting to create missing widget");
                            this.videoComparerWidget = this.addCustomWidget(new VideoComparerWidget("video_comparer", this));
                            
                            if (this.videoComparerWidget) {
                                console.log("[VideoComparer] Created widget, setting value");
                                this.videoComparerWidget.value = { 
                                    video_data: videoData,
                                    difference_data: differenceData,
                                    histogram_data: histogramData
                                };
                                // Don't force a resize, just redraw
                                this.setDirtyCanvas(true, true);
                            }
                        }
                    }
                } else {
                    console.log("[VideoComparer] No valid video data found, checking if we have standard images");
                    
                    // Check if we have standard animated images and convert them for testing
                    if (message && message.ui && message.ui.images && message.ui.images.length > 0) {
                        console.log("[VideoComparer] Found standard images, creating test video data");
                        
                        // Create fake video data structure for testing
                        videoData = [{
                            name: "video_a",
                            frames: message.ui.images.map((img, index) => ({
                                filename: img.filename,
                                subfolder: img.subfolder || "",
                                type: img.type || "output",
                                frame_index: index
                            })),
                            fps: 8
                        }];
                        
                        console.log("[VideoComparer] Created test video data:", videoData);
                    }
                }
            } else {
                console.warn("[VideoComparer] Message is not a valid object:", message);
            }
            
            return result || message;
        };
    
        // Mouse event handlers
        nodeType.prototype.onMouseDown = function(event, pos, canvas) {
            if (this.videoComparerWidget) {
                return this.videoComparerWidget.mouse(event, pos, this);
            }
            return false;
        };

        // Track when user manually resizes the node
        const originalOnResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function(size) {
            // Call original resize handler if it exists
            if (originalOnResize) {
                originalOnResize.apply(this, arguments);
            }
            
            // Mark that user has manually resized
            this.properties.user_resized = true;
        };

        nodeType.prototype.onMouseUp = function(event, pos, canvas) {
            if (this.videoComparerWidget) {
                return this.videoComparerWidget.mouse(event, pos, this);
            }
            return false;
        };

        nodeType.prototype.onMouseMove = function(event, pos, canvas) {
            if (this.videoComparerWidget) {
                return this.videoComparerWidget.mouse(event, pos, this);
            }
            return false;
        };

        nodeType.prototype.onMouseEnter = function(event) {
            this.isPointerOver = true;
            this.setDirtyCanvas(true, false);
        };
    
        nodeType.prototype.onMouseLeave = function(event) {
            this.isPointerOver = false;
            this.setDirtyCanvas(true, false);
        };

        // Keydown event handler for node
        nodeType.prototype.onKeyDown = function(event) {
            if (!this.videoComparerWidget) return false;

            let handled = false;
            switch (event.key) {
                case "ArrowLeft":
                    this.videoComparerWidget.previousFrame();
                    handled = true;
                    break;
                case "ArrowRight":
                    this.videoComparerWidget.nextFrame();
                    handled = true;
                    break;
                case " ": // Space bar
                    this.videoComparerWidget.togglePlayback();
                    handled = true;
                    break;
                case "p":
                case "P":
                    if (event.altKey) {
                        this.videoComparerWidget.togglePlayback();
                        handled = true;
                    }
                    break;
            }

            if (handled) {
                event.preventDefault(); // Prevent default browser action (e.g., scrolling with space)
                event.stopPropagation(); // Stop event from bubbling up
                return true; // Indicate that the event was handled
            }
            return false; // Event not handled by this node
        };

        // Add context menu options
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            if (getExtraMenuOptions) {
                getExtraMenuOptions.apply(this, arguments);
            }
            
            // Add separator before our options
            options.push(null);
            
            // Add comparison mode submenu
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
                            if (this.layoutWidget) {
                                this.layoutWidget.value = mode;
                            }
                            this.updateControlsVisibility();
                            this.setDirtyCanvas(true, false);
                        }
                    }))
                }
            });
            
            // Add video selection submenu (only show when relevant)
            if (this.properties.comparer_mode === "Playback") {
                const currentVideo = this.properties.selected_video || "A";
                const videos = ["A", "B"];
                
                options.push({
                    content: "Select Video",
                    has_submenu: true,
                    submenu: {
                        options: videos.map(video => ({
                            content: video === currentVideo ? `✓ Video ${video}` : `Video ${video}`,
                            callback: () => {
                                this.properties.selected_video = video;
                                this.selectedVideo = video;
                                if (this.videoSelectorWidget) {
                                    this.videoSelectorWidget.value = video;
                                }
                                this.setDirtyCanvas(true, false);
                            }
                        }))
                    }
                });
            }
            
            // Add opacity controls for Onion Skin mode
            if (this.properties.comparer_mode === "Onion Skin") {
                const opacityPresets = [
                    { label: "25%", value: 0.25 },
                    { label: "50%", value: 0.5 },
                    { label: "75%", value: 0.75 }
                ];
                
                options.push({
                    content: "Onion Skin Opacity",
                    has_submenu: true,
                    submenu: {
                        options: opacityPresets.map(preset => ({
                            content: preset.label,
                            callback: () => {
                                this.properties.onionSkinOpacity = preset.value;
                                if (this.onionSkinOpacitySlider) {
                                    this.onionSkinOpacitySlider.value = preset.value;
                                }
                                this.setDirtyCanvas(true, false);
                            }
                        }))
                    }
                });
            }
            
            // Add separator before existing options
            options.push(null);
            
            options.push(
                {
                    content: "Reset to Default Size",
                    callback: () => {
                        // Reset the user_resized flag
                        this.properties.user_resized = false;
                        this.setSize(this.computeSize());
                        this.setDirtyCanvas(true, false);
                    }
                }
            );
        };
        
        console.log("[VideoComparer] Node setup complete");
    }
});