/**
 * advanced_comparer.ts - Advanced Image Comparer Widget for ComfyUI
 * Supports multiple comparison modes: Slider, Click, Side-by-Side, Stacked, Grid, Carousel, Batch, Onion Skin
 */
// @ts-ignore - Runtime ComfyUI import
import { app } from "../../../scripts/app.js";
// @ts-ignore - Runtime ComfyUI import  
import { api } from "../../../scripts/api.js";
console.log("AdvancedImageComparer module loaded");
const CACHE = {
    lastTime: 0,
    frameCount: 0,
    frameSkip: 2,
};
// ============================
// Helper Functions
// ============================
function imageDataToUrl(data) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const apiObj = api;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const appObj = app;
    return apiObj.apiURL(`/view?filename=${encodeURIComponent(data.filename)}&type=${encodeURIComponent(data.type || "")}&subfolder=${encodeURIComponent(data.subfolder || "")}${appObj.getPreviewFormatParam()}${appObj.getRandParam()}`);
}
function drawGoldenEyeball(ctx, centerX, centerY, size, shimmerPosition) {
    const eyeWidth = size * 1.6;
    const eyeHeight = size * 1.0;
    const irisRadius = size * 0.35;
    const pupilRadius = size * 0.15;
    ctx.save();
    const baseGradient = ctx.createLinearGradient(0, centerY - size * 0.7, 0, centerY + size * 0.7);
    baseGradient.addColorStop(0, "#B8860B");
    baseGradient.addColorStop(0.5, "#FFD700");
    baseGradient.addColorStop(1, "#B8860B");
    const highlightWidth = eyeWidth * 0.4;
    const highlightX = -highlightWidth + (eyeWidth + highlightWidth) * shimmerPosition;
    const shimmerGradient = ctx.createLinearGradient(centerX + highlightX - highlightWidth / 2, 0, centerX + highlightX + highlightWidth / 2, 0);
    shimmerGradient.addColorStop(0, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(0.1, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(0.5, "rgba(255, 255, 200, 0.3)");
    shimmerGradient.addColorStop(0.9, "rgba(255, 255, 200, 0)");
    shimmerGradient.addColorStop(1, "rgba(255, 255, 200, 0)");
    // Draw shadows
    ctx.strokeStyle = "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1.5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.ellipse(centerX + 2, centerY + 2, eyeWidth / 2, eyeHeight / 2, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX + 2, centerY + 2, irisRadius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX + 2, centerY + 2, pupilRadius, 0, Math.PI * 2);
    ctx.stroke();
    // Draw rays
    const rayCount = 8;
    const rayLength = size * 0.7;
    for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        const startX = centerX + Math.cos(angle) * (eyeWidth / 2 + 1);
        const startY = centerY + Math.sin(angle) * (eyeHeight / 2 + 1);
        const endX = centerX + Math.cos(angle) * (eyeWidth / 2 + rayLength);
        const endY = centerY + Math.sin(angle) * (eyeHeight / 2 + rayLength);
        ctx.beginPath();
        ctx.moveTo(startX + 2, startY + 2);
        ctx.lineTo(endX + 2, endY + 2);
        ctx.stroke();
    }
    // Draw golden outlines
    ctx.strokeStyle = baseGradient;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, eyeWidth / 2, eyeHeight / 2, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX, centerY, irisRadius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX, centerY, pupilRadius, 0, Math.PI * 2);
    ctx.stroke();
    for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        const startX = centerX + Math.cos(angle) * (eyeWidth / 2 + 1);
        const startY = centerY + Math.sin(angle) * (eyeHeight / 2 + 1);
        const endX = centerX + Math.cos(angle) * (eyeWidth / 2 + rayLength);
        const endY = centerY + Math.sin(angle) * (eyeHeight / 2 + rayLength);
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
    }
    // Iris texture
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 8; i++) {
        const angle = (i / 8) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(centerX + Math.cos(angle) * pupilRadius * 1.1, centerY + Math.sin(angle) * pupilRadius * 1.1);
        ctx.lineTo(centerX + Math.cos(angle) * irisRadius * 0.9, centerY + Math.sin(angle) * irisRadius * 0.9);
        ctx.stroke();
    }
    // Shimmer effect
    ctx.strokeStyle = shimmerGradient;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, eyeWidth / 2, eyeHeight / 2, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX, centerY, irisRadius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX, centerY, pupilRadius, 0, Math.PI * 2);
    ctx.stroke();
    for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        const startX = centerX + Math.cos(angle) * (eyeWidth / 2 + 1);
        const startY = centerY + Math.sin(angle) * (eyeHeight / 2 + 1);
        const endX = centerX + Math.cos(angle) * (eyeWidth / 2 + rayLength);
        const endY = centerY + Math.sin(angle) * (eyeHeight / 2 + rayLength);
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
    }
    // Glow effect
    const glowIntensity = Math.max(0, 1 - Math.abs(centerX - (centerX + highlightX)) / (eyeWidth / 4));
    ctx.shadowColor = `rgba(255, 255, 200, ${glowIntensity * 0.3})`;
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.strokeStyle = baseGradient;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, eyeWidth / 2, eyeHeight / 2, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
}
function drawGradientTitle(node, ctx) {
    const titleHeight = node.flags.collapsed ? 20 : 30;
    const width = node.flags.collapsed ? 190 : node.size[0];
    const fullHeight = node.size[1];
    const eyeballY = node.flags.collapsed ? titleHeight / 2 : 25;
    const eyeballSize = node.flags.collapsed ? 6 : 10;
    CACHE.frameCount = (CACHE.frameCount + 1) % (CACHE.frameSkip + 1);
    const shouldUpdateAnimation = CACHE.frameCount === 0;
    ctx.save();
    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    const gradient = ctx.createLinearGradient(0, 0, 0, fullHeight);
    gradient.addColorStop(0, "#000000");
    gradient.addColorStop(0.2, "#101010");
    gradient.addColorStop(1, "#101010");
    let shimmerPosition = 0.5;
    if (shouldUpdateAnimation) {
        const time = Date.now() / 3000;
        shimmerPosition = (Math.sin(time) + 1) / 2;
        CACHE.lastTime = time;
    }
    else {
        const time = CACHE.lastTime || Date.now() / 3000;
        shimmerPosition = (Math.sin(time) + 1) / 2;
    }
    if (node.flags.collapsed) {
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, titleHeight);
        drawGoldenEyeball(ctx, width / 2, titleHeight / 2, eyeballSize, shimmerPosition);
        ctx.restore();
        return;
    }
    ctx.fillStyle = gradient;
    if (!node.flags.collapsed) {
        const cornerRadius = 8;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(width, 0);
        ctx.lineTo(width, fullHeight - cornerRadius);
        ctx.arcTo(width, fullHeight, width - cornerRadius, fullHeight, cornerRadius);
        ctx.lineTo(cornerRadius, fullHeight);
        ctx.arcTo(0, fullHeight, 0, fullHeight - cornerRadius, cornerRadius);
        ctx.lineTo(0, 0);
        ctx.closePath();
        ctx.fill();
    }
    else {
        ctx.fillRect(0, 0, width, fullHeight);
    }
    drawGoldenEyeball(ctx, width / 2, eyeballY, eyeballSize, shimmerPosition);
    ctx.restore();
}
// ============================
// Widget Class
// ============================
class AdvancedImageComparerWidget {
    constructor(name, node) {
        this.type = "custom";
        this.options = { serialize: false };
        this.y = 0;
        this.last_y = 0;
        this.imagesA = [];
        this.imagesB = [];
        this.currentPairIndex = 0;
        this.maxPairs = 0;
        this.animationFrame = null;
        this.autoPlayEnabled = false;
        this.autoPlaySpeed = 2000;
        this.currentBatchPage = 0;
        this.pairsPerPage = 3;
        this.maxBatchPages = 0;
        this.name = name;
        this.node = node;
        this._value = { images: [] };
        this.selected = [];
        this.imgs = [];
    }
    set value(v) {
        const images = v.images || [];
        const imagesA = images.filter((img) => 'is_image_a' in img && img.is_image_a === true);
        const imagesB = images.filter((img) => 'is_image_a' in img && img.is_image_a === false);
        this.imagesA = imagesA.map((img, index) => ({
            name: `A${index + 1}`,
            selected: true,
            url: imageDataToUrl(img),
            img: null,
            index
        }));
        this.imagesB = imagesB.map((img, index) => ({
            name: `B${index + 1}`,
            selected: true,
            url: imageDataToUrl(img),
            img: null,
            index
        }));
        this.maxPairs = Math.max(this.imagesA.length, this.imagesB.length);
        this.currentPairIndex = 0;
        this.maxBatchPages = Math.ceil(this.maxPairs / this.pairsPerPage);
        this.currentBatchPage = 0;
        this._value = { images: [...this.imagesA, ...this.imagesB] };
        this.updateSelectedPair();
        this.loadAllImages();
        if (this.node?.updateControlsVisibility) {
            this.node.updateControlsVisibility();
        }
        if (this.node) {
            const minWidth = 700, minHeight = 600;
            const [w, h] = this.node.size;
            this.node.setSize([Math.max(w, minWidth), Math.max(h, minHeight)]);
            this.node.setDirtyCanvas(true, true);
        }
    }
    get value() {
        return this._value || { images: [] };
    }
    loadAllImages() {
        [...this.imagesA, ...this.imagesB].forEach(imageData => {
            if (!imageData.img && imageData.url) {
                imageData.img = new Image();
                imageData.img.onload = () => this.node.setDirtyCanvas(true, false);
                imageData.img.onerror = (e) => console.error("Image load failed:", imageData.name, e);
                imageData.img.src = imageData.url;
            }
        });
    }
    setSelected(selected) {
        this.selected = selected;
        this.imgs = [];
        for (const sel of selected) {
            if (!sel.img && sel.url) {
                sel.img = new Image();
                sel.img.onload = () => this.node.setDirtyCanvas(true, false);
                sel.img.src = sel.url;
            }
            if (sel.img)
                this.imgs.push(sel.img);
        }
    }
    draw(ctx, node, width, y, height) {
        this.y = y;
        this.last_y = y;
        const [, nodeHeight] = node.size;
        const availableHeight = Math.max(200, nodeHeight - y - 10);
        const mode = node.properties?.comparer_mode || "Slider";
        switch (mode) {
            case "Click":
                this.drawClickMode(ctx, y, width, availableHeight);
                break;
            case "Side-by-Side":
                this.drawSideBySideMode(ctx, y, width, availableHeight);
                break;
            case "Stacked":
                this.drawStackedMode(ctx, y, width, availableHeight);
                break;
            case "Grid":
                this.drawGridMode(ctx, y, width, availableHeight);
                break;
            case "Carousel":
                this.drawCarouselMode(ctx, y, width, availableHeight);
                break;
            case "Batch":
                this.drawBatchMode(ctx, y, width, availableHeight);
                break;
            case "Onion Skin":
                this.drawOnionSkinMode(ctx, y, width, availableHeight);
                break;
            default:
                this.drawSliderMode(ctx, y, width, availableHeight);
                break;
        }
        if (["Grid", "Batch"].includes(mode) && this.maxPairs > 1) {
            this.drawBatchControls(ctx, y, width, availableHeight);
        }
        if (mode === "Carousel" && this.maxPairs > 1) {
            this.drawPairIndicator(ctx, y + availableHeight - 25, width);
        }
    }
    drawClickMode(ctx, y, width, h) {
        const idx = this.node.isPointerDown ? 1 : 0;
        this.drawImage(ctx, this.selected[idx], y, width, h);
    }
    drawSideBySideMode(ctx, y, width, h) {
        if (this.selected[0])
            this.drawImageSideBySide(ctx, this.selected[0], y, width, h, 0);
        if (this.selected[1])
            this.drawImageSideBySide(ctx, this.selected[1], y, width, h, 1);
    }
    drawStackedMode(ctx, y, width, h) {
        if (this.selected[0])
            this.drawImageStacked(ctx, this.selected[0], y, width, h, 0);
        if (this.selected[1])
            this.drawImageStacked(ctx, this.selected[1], y, width, h, 1);
    }
    drawSliderMode(ctx, y, width, h) {
        if (this.selected[0])
            this.drawImage(ctx, this.selected[0], y, width, h);
        if (this.selected[1] && this.node.isPointerOver) {
            this.drawImage(ctx, this.selected[1], y, width, h, this.node.pointerOverPos[0]);
        }
    }
    drawGridMode(ctx, y, width, h) {
        const pairs = Math.min(this.maxPairs, 64);
        const cols = Math.ceil(Math.sqrt(pairs * 2));
        const rows = Math.ceil((pairs * 2) / cols);
        const cw = width / cols, ch = (h - 40) / rows;
        let ci = 0;
        for (let i = 0; i < pairs; i++) {
            if (this.imagesA[i]?.img) {
                this.drawImageInCell(ctx, this.imagesA[i], y + Math.floor(ci / cols) * ch, (ci % cols) * cw, cw, ch, `A${i + 1}`);
                ci++;
            }
            if (this.imagesB[i]?.img) {
                this.drawImageInCell(ctx, this.imagesB[i], y + Math.floor(ci / cols) * ch, (ci % cols) * cw, cw, ch, `B${i + 1}`);
                ci++;
            }
        }
    }
    drawCarouselMode(ctx, y, width, h) {
        const a = this.imagesA[this.currentPairIndex], b = this.imagesB[this.currentPairIndex];
        if (a?.img)
            this.drawImageSideBySide(ctx, a, y, width, h, 0);
        if (b?.img)
            this.drawImageSideBySide(ctx, b, y, width, h, 1);
    }
    drawBatchMode(ctx, y, width, h) {
        const ph = (h - 40) / this.pairsPerPage;
        const start = this.currentBatchPage * this.pairsPerPage;
        for (let i = 0; i < this.pairsPerPage && start + i < this.maxPairs; i++) {
            const idx = start + i, py = y + i * ph;
            if (this.imagesA[idx]?.img)
                this.drawImageInPair(ctx, this.imagesA[idx], py, 0, width / 2, ph, 0);
            if (this.imagesB[idx]?.img)
                this.drawImageInPair(ctx, this.imagesB[idx], py, width / 2, width / 2, ph, 1);
            if (i < this.pairsPerPage - 1 && idx < this.maxPairs - 1) {
                ctx.beginPath();
                ctx.moveTo(0, py + ph);
                ctx.lineTo(width, py + ph);
                ctx.strokeStyle = "rgba(255,255,255,0.3)";
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }
    }
    drawOnionSkinMode(ctx, y, width, h) {
        const opacity = this.node.properties?.onionSkinOpacity || 0.5;
        if (this.selected[0])
            this.drawImage(ctx, this.selected[0], y, width, h);
        if (this.selected[1]) {
            ctx.save();
            ctx.globalAlpha = opacity;
            this.drawImage(ctx, this.selected[1], y, width, h);
            ctx.restore();
        }
    }
    drawImage(ctx, imageData, y, nodeWidth, availableHeight, cropX) {
        if (!imageData?.img?.naturalWidth)
            return;
        const img = imageData.img, pad = 3;
        const uw = nodeWidth - pad * 2, uh = availableHeight - pad * 2;
        const ia = img.naturalWidth / img.naturalHeight, ua = uw / uh;
        let tw, th;
        if (ia > ua) {
            tw = uw;
            th = uw / ia;
        }
        else {
            th = uh;
            tw = uh * ia;
        }
        const dx = pad + (uw - tw) / 2, dy = y + pad + (uh - th) / 2;
        ctx.save();
        ctx.beginPath();
        ctx.rect(pad, y + pad, uw, uh);
        ctx.clip();
        if (cropX && cropX > dx) {
            const wm = img.naturalWidth / tw, sw = Math.max(0, (cropX - dx) * wm), dw = Math.max(0, cropX - dx);
            ctx.drawImage(img, 0, 0, sw, img.naturalHeight, dx, dy, dw, th);
        }
        else {
            ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight, dx, dy, tw, th);
        }
        if (cropX != null && cropX > dx && cropX < dx + tw) {
            ctx.beginPath();
            ctx.moveTo(cropX, dy);
            ctx.lineTo(cropX, dy + th);
            ctx.globalCompositeOperation = "difference";
            ctx.strokeStyle = "rgba(255,255,255,1)";
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        ctx.restore();
    }
    drawImageSideBySide(ctx, imageData, y, nodeWidth, availableHeight, imageIndex) {
        if (!imageData?.img?.naturalWidth)
            return;
        const img = imageData.img, hw = nodeWidth / 2, pad = 3, sep = 1;
        const uw = hw - pad - sep / 2, uh = availableHeight - pad * 2;
        const ia = img.naturalWidth / img.naturalHeight, ua = uw / uh;
        let tw, th;
        if (ia > ua) {
            tw = uw;
            th = uw / ia;
        }
        else {
            th = uh;
            tw = uh * ia;
        }
        const dx = imageIndex === 0 ? pad + (uw - tw) / 2 : hw + sep / 2 + pad + (uw - tw) / 2;
        const dy = y + pad + (uh - th) / 2;
        ctx.save();
        const cx = imageIndex === 0 ? 0 : hw + sep / 2, cw = imageIndex === 0 ? hw - sep / 2 : hw - sep / 2;
        ctx.beginPath();
        ctx.rect(cx, y, cw, availableHeight);
        ctx.clip();
        ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight, dx, dy, tw, th);
        ctx.fillStyle = "rgba(0,0,0,0.8)";
        ctx.fillRect(dx, dy, 25, 18);
        ctx.fillStyle = "white";
        ctx.font = "11px Arial";
        ctx.textAlign = "center";
        ctx.fillText(imageData.name, dx + 12, dy + 13);
        ctx.restore();
        if (imageIndex === 0) {
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(hw, y + pad);
            ctx.lineTo(hw, y + availableHeight - pad);
            ctx.strokeStyle = "rgba(255,255,255,0.5)";
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.restore();
        }
    }
    drawImageStacked(ctx, imageData, y, nodeWidth, availableHeight, imageIndex) {
        if (!imageData?.img?.naturalWidth)
            return;
        const img = imageData.img, hh = availableHeight / 2, pad = 3, sep = 1;
        const uw = nodeWidth - pad * 2, uh = hh - pad - sep / 2;
        const ia = img.naturalWidth / img.naturalHeight, ua = uw / uh;
        let tw, th;
        if (ia > ua) {
            tw = uw;
            th = uw / ia;
        }
        else {
            th = uh;
            tw = uh * ia;
        }
        const dx = pad + (uw - tw) / 2;
        const dy = imageIndex === 0 ? y + pad + (uh - th) / 2 : y + hh + sep / 2 + pad + (uh - th) / 2;
        ctx.save();
        const cy = imageIndex === 0 ? y : y + hh + sep / 2, ch = imageIndex === 0 ? hh - sep / 2 : hh - sep / 2;
        ctx.beginPath();
        ctx.rect(0, cy, nodeWidth, ch);
        ctx.clip();
        ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight, dx, dy, tw, th);
        ctx.fillStyle = "rgba(0,0,0,0.8)";
        ctx.fillRect(dx, dy, 25, 18);
        ctx.fillStyle = "white";
        ctx.font = "11px Arial";
        ctx.textAlign = "center";
        ctx.fillText(imageData.name, dx + 12, dy + 13);
        ctx.restore();
        if (imageIndex === 0) {
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(pad, y + hh);
            ctx.lineTo(nodeWidth - pad, y + hh);
            ctx.strokeStyle = "rgba(255,255,255,0.5)";
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.restore();
        }
    }
    drawImageInCell(ctx, imageData, y, x, cw, ch, label) {
        if (!imageData?.img?.naturalWidth)
            return;
        const img = imageData.img, pad = 2, uw = cw - pad * 2, uh = ch - pad * 2;
        const ia = img.naturalWidth / img.naturalHeight, ca = uw / uh;
        let tw, th;
        if (ia > ca) {
            tw = uw;
            th = uw / ia;
        }
        else {
            th = uh;
            tw = uh * ia;
        }
        const dx = x + pad + (uw - tw) / 2, dy = y + pad + (uh - th) / 2;
        ctx.save();
        ctx.beginPath();
        ctx.rect(x + pad, y + pad, uw, uh);
        ctx.clip();
        ctx.strokeStyle = "rgba(255,255,255,0.3)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x + pad, y + pad, uw, uh);
        ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight, dx, dy, tw, th);
        ctx.fillStyle = "rgba(0,0,0,0.8)";
        ctx.fillRect(dx, dy, 28, 16);
        ctx.fillStyle = "white";
        ctx.font = "10px Arial";
        ctx.textAlign = "center";
        ctx.fillText(label, dx + 14, dy + 11);
        ctx.restore();
    }
    drawImageInPair(ctx, imageData, y, x, pw, ph, imageIndex) {
        if (!imageData?.img?.naturalWidth)
            return;
        const img = imageData.img, pad = 3, uw = pw - pad * 2, uh = ph - pad * 2;
        const ia = img.naturalWidth / img.naturalHeight, pa = uw / uh;
        let tw, th;
        if (ia > pa) {
            tw = uw;
            th = uw / ia;
        }
        else {
            th = uh;
            tw = uh * ia;
        }
        const dx = x + pad + (uw - tw) / 2, dy = y + pad + (uh - th) / 2;
        ctx.save();
        ctx.beginPath();
        ctx.rect(x + pad, y + pad, uw, uh);
        ctx.clip();
        ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight, dx, dy, tw, th);
        ctx.fillStyle = "rgba(0,0,0,0.8)";
        ctx.fillRect(dx, dy, 30, 16);
        ctx.fillStyle = "white";
        ctx.font = "11px Arial";
        ctx.textAlign = "center";
        ctx.fillText(imageData.name, dx + 15, dy + 11);
        ctx.restore();
        if (imageIndex === 0 && pw < this.node.size[0]) {
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(x + pw, y + pad);
            ctx.lineTo(x + pw, y + ph - pad);
            ctx.strokeStyle = "rgba(255,255,255,0.5)";
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.restore();
        }
    }
    drawBatchControls(ctx, y, width, h) {
        const cy = y + h - 30;
        ctx.save();
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(0, cy, width, 30);
        ctx.fillStyle = "white";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        ctx.fillText(`Images: A(${this.imagesA.length}) B(${this.imagesB.length})`, 10, cy + 18);
        ctx.textAlign = "right";
        const mode = this.node.properties?.comparer_mode || "Slider";
        if (mode === "Grid")
            ctx.fillText(`Showing ${Math.min(this.maxPairs, 64)} pairs`, width - 10, cy + 18);
        else if (mode === "Batch") {
            const s = this.currentBatchPage * this.pairsPerPage + 1, e = Math.min((this.currentBatchPage + 1) * this.pairsPerPage, this.maxPairs);
            ctx.fillText(`Showing pairs ${s}-${e} of ${this.maxPairs}`, width - 10, cy + 18);
        }
        ctx.restore();
    }
    drawPairIndicator(ctx, y, width) {
        ctx.save();
        const ds = 8, dp = 12, tw = this.maxPairs * dp - (dp - ds), sx = (width - tw) / 2;
        for (let i = 0; i < this.maxPairs; i++) {
            ctx.beginPath();
            ctx.arc(sx + i * dp + ds / 2, y + ds / 2, ds / 2, 0, Math.PI * 2);
            ctx.fillStyle = i === this.currentPairIndex ? "rgba(255,255,255,1)" : "rgba(255,255,255,0.4)";
            ctx.fill();
        }
        ctx.restore();
    }
    nextPair() { if (this.currentPairIndex < this.maxPairs - 1) {
        this.currentPairIndex++;
        this.updateSelectedPair();
        this.node.setDirtyCanvas(true, false);
    } }
    previousPair() { if (this.currentPairIndex > 0) {
        this.currentPairIndex--;
        this.updateSelectedPair();
        this.node.setDirtyCanvas(true, false);
    } }
    toggleAutoPlay() { this.autoPlayEnabled = !this.autoPlayEnabled; this.autoPlayEnabled ? this.startAutoPlay() : this.stopAutoPlay(); this.updateNodeControls(); this.node.setDirtyCanvas(true, false); }
    nextBatchPage() { if (this.currentBatchPage < this.maxBatchPages - 1) {
        this.currentBatchPage++;
        this.updateNodeControls();
        this.node.setDirtyCanvas(true, false);
    } }
    previousBatchPage() { if (this.currentBatchPage > 0) {
        this.currentBatchPage--;
        this.updateNodeControls();
        this.node.setDirtyCanvas(true, false);
    } }
    updateNodeControls() {
        if (this.node.pairInfoWidget)
            this.node.pairInfoWidget.value = `${this.currentPairIndex + 1} / ${this.maxPairs}`;
        if (this.node.autoPlayButton)
            this.node.autoPlayButton.name = this.autoPlayEnabled ? "⏸ Pause" : "▶ Play";
        if (this.node.batchSelectorWidget)
            this.node.batchSelectorWidget.value = (this.currentPairIndex + 1).toString();
        if (this.node.batchPageInfoWidget)
            this.node.batchPageInfoWidget.value = `Page ${this.currentBatchPage + 1} / ${this.maxBatchPages}`;
    }
    updateSelectedPair() {
        const mode = this.node.properties?.comparer_mode || "Slider";
        if (["Slider", "Click", "Side-by-Side", "Stacked", "Onion Skin"].includes(mode)) {
            const processed = [];
            if (this.imagesA[this.currentPairIndex])
                processed.push(this.imagesA[this.currentPairIndex]);
            if (this.imagesB[this.currentPairIndex])
                processed.push(this.imagesB[this.currentPairIndex]);
            this.setSelected(processed);
        }
        this.updateNodeControls();
    }
    startAutoPlay() {
        if (this.animationFrame)
            clearInterval(this.animationFrame);
        this.animationFrame = setInterval(() => {
            this.currentPairIndex = this.currentPairIndex >= this.maxPairs - 1 ? 0 : this.currentPairIndex + 1;
            this.updateSelectedPair();
            this.node.setDirtyCanvas(true, false);
        }, this.autoPlaySpeed);
    }
    stopAutoPlay() { if (this.animationFrame) {
        clearInterval(this.animationFrame);
        this.animationFrame = null;
    } }
    computeSize(width) {
        const mode = this.node?.properties?.comparer_mode || "Slider";
        let height = Math.max(500, width);
        switch (mode) {
            case "Stacked":
                height = Math.max(700, width * 1.5);
                break;
            case "Side-by-Side":
                height = Math.max(500, width);
                break;
            case "Grid":
                const p = Math.min(this.maxPairs || 1, 64), c = Math.ceil(Math.sqrt(p * 2)), r = Math.ceil((p * 2) / c);
                height = Math.max(500, (width / c) * r + 100);
                break;
            case "Carousel":
                height = Math.max(450, width * 0.9 + 100);
                break;
            case "Batch":
                height = Math.max(550, Math.min(this.maxPairs || 1, 3) * (width * 0.6) + 100);
                break;
            default:
                height = Math.max(400, width * 0.9);
                break;
        }
        return [width, height];
    }
    mouse(event, pos, node) {
        const mode = node.properties?.comparer_mode || "Slider";
        if (event.type === "pointermove") {
            node.pointerOverPos = [...pos];
            if (mode === "Slider")
                node.setDirtyCanvas(true, false);
            return true;
        }
        if (event.type === "pointerdown" && mode === "Grid" && this.maxPairs > 1) {
            const p = Math.min(this.maxPairs, 64), c = Math.ceil(Math.sqrt(p * 2)), r = Math.ceil((p * 2) / c);
            const cw = node.size[0] / c, wh = node.size[1] - this.y - 10, ch = (wh - 40) / r;
            const ci = Math.floor(pos[1] / ch) * c + Math.floor(pos[0] / cw), ii = Math.floor(ci / 2);
            if (ii < this.maxPairs) {
                this.currentPairIndex = ii;
                node.properties.comparer_mode = "Carousel";
                if (node.layoutWidget)
                    node.layoutWidget.value = "Carousel";
                node.updateControlsVisibility();
                node.setDirtyCanvas(true, false);
                return true;
            }
        }
        return false;
    }
    onRemoved() { this.stopAutoPlay(); }
}
// ============================
// Extension Registration
// ============================
// eslint-disable-next-line @typescript-eslint/no-explicit-any
app.registerExtension({
    name: "AdvancedImageComparer",
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AdvancedImageComparer")
            return;
        nodeType.prototype.properties = nodeType.prototype.properties || { comparer_mode: "Slider", onionSkinOpacity: 0.5 };
        nodeType["@comparer_mode"] = { type: "combo", values: ["Slider", "Click", "Side-by-Side", "Stacked", "Grid", "Carousel", "Batch", "Onion Skin"] };
        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (origOnDrawForeground)
                origOnDrawForeground.call(this, ctx);
            drawGradientTitle(this, ctx);
        };
        const origOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (origOnRemoved)
                origOnRemoved.call(this);
            // Note: CACHE only holds animation timing state, no cleanup needed
        };
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated)
                onNodeCreated.call(this);
            this.properties = this.properties || { comparer_mode: "Slider", onionSkinOpacity: 0.5 };
            this.isPointerDown = false;
            this.isPointerOver = false;
            this.pointerOverPos = [0, 0];
            this.imageIndex = 0;
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const self = this;
            self.layoutWidget = self.addWidget("combo", "Layout Mode", this.properties.comparer_mode, (value) => {
                this.properties.comparer_mode = value;
                this.updateControlsVisibility();
                self.setDirtyCanvas(true, false);
            }, { values: ["Slider", "Click", "Side-by-Side", "Stacked", "Grid", "Carousel", "Batch", "Onion Skin"] });
            self.batchSelectorWidget = self.addWidget("combo", "View Pair", "1", (value) => {
                const idx = parseInt(value) - 1;
                if (this.comparerWidget && idx >= 0 && idx < this.comparerWidget.maxPairs) {
                    this.comparerWidget.currentPairIndex = idx;
                    this.comparerWidget.updateSelectedPair();
                    self.setDirtyCanvas(true, false);
                }
            }, { values: ["1"] });
            self.prevButton = self.addWidget("button", "◀ Previous", null, () => { this.comparerWidget?.previousPair(); });
            self.nextButton = self.addWidget("button", "Next ▶", null, () => { this.comparerWidget?.nextPair(); });
            self.autoPlayButton = self.addWidget("button", "▶ Play", null, () => {
                this.comparerWidget?.toggleAutoPlay();
                self.autoPlayButton.name = this.comparerWidget?.autoPlayEnabled ? "⏸ Pause" : "▶ Play";
            });
            self.pairInfoWidget = self.addWidget("text", "Pair Info", "1 / 1", () => { }, {});
            self.pairInfoWidget.disabled = true;
            self.batchPrevButton = self.addWidget("button", "◀ Prev Page", null, () => { this.comparerWidget?.previousBatchPage(); });
            self.batchNextButton = self.addWidget("button", "Next Page ▶", null, () => { this.comparerWidget?.nextBatchPage(); });
            self.batchPageInfoWidget = self.addWidget("text", "Page Info", "Page 1 / 1", () => { }, {});
            self.batchPageInfoWidget.disabled = true;
            self.onionSkinOpacitySlider = self.addWidget("slider", "Opacity B", this.properties.onionSkinOpacity, (value) => {
                this.properties.onionSkinOpacity = value;
                self.setDirtyCanvas(true, false);
            }, { min: 0.0, max: 1.0, step: 0.01 });
            self.clearCacheButton = self.addWidget("button", "Clear Cache", null, () => {
                console.log("[AdvancedImageComparer] Clearing image cache");
            });
            self.clearCacheButton.hidden = true;
            this.comparerWidget = self.addCustomWidget(new AdvancedImageComparerWidget("advanced_comparer", this));
            this.updateControlsVisibility();
            self.setSize([700, 600]);
            self.setDirtyCanvas(true, true);
        };
        nodeType.prototype.updateControlsVisibility = function () {
            const mode = this.properties.comparer_mode;
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const self = this;
            const hasMultiplePairs = this.comparerWidget && this.comparerWidget.maxPairs > 1;
            const showBatchSelector = hasMultiplePairs && ["Slider", "Click", "Side-by-Side", "Stacked", "Onion Skin"].includes(mode);
            const showCarouselControls = mode === "Carousel";
            const showBatchPagination = mode === "Batch" && this.comparerWidget && this.comparerWidget.maxBatchPages > 1;
            const showOnionSkinSlider = mode === "Onion Skin";
            if (self.batchSelectorWidget) {
                self.batchSelectorWidget.hidden = !showBatchSelector;
                if (showBatchSelector && this.comparerWidget) {
                    const opts = [];
                    for (let i = 1; i <= this.comparerWidget.maxPairs; i++)
                        opts.push(i.toString());
                    self.batchSelectorWidget.options.values = opts;
                    self.batchSelectorWidget.value = (this.comparerWidget.currentPairIndex + 1).toString();
                }
            }
            if (self.prevButton)
                self.prevButton.hidden = !showCarouselControls;
            if (self.nextButton)
                self.nextButton.hidden = !showCarouselControls;
            if (self.autoPlayButton)
                self.autoPlayButton.hidden = !showCarouselControls;
            if (self.pairInfoWidget) {
                self.pairInfoWidget.hidden = !showCarouselControls;
                if (showCarouselControls && this.comparerWidget)
                    self.pairInfoWidget.value = `${this.comparerWidget.currentPairIndex + 1} / ${this.comparerWidget.maxPairs}`;
            }
            if (self.batchPrevButton)
                self.batchPrevButton.hidden = !showBatchPagination;
            if (self.batchNextButton)
                self.batchNextButton.hidden = !showBatchPagination;
            if (self.batchPageInfoWidget) {
                self.batchPageInfoWidget.hidden = !showBatchPagination;
                if (showBatchPagination && this.comparerWidget)
                    self.batchPageInfoWidget.value = `Page ${this.comparerWidget.currentBatchPage + 1} / ${this.comparerWidget.maxBatchPages}`;
            }
            if (self.onionSkinOpacitySlider) {
                self.onionSkinOpacitySlider.hidden = !showOnionSkinSlider;
                if (showOnionSkinSlider)
                    self.onionSkinOpacitySlider.value = this.properties.onionSkinOpacity;
            }
        };
        const originalComputeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function (out) {
            const size = originalComputeSize ? originalComputeSize.call(this, out) : [700, 600];
            if (this.comparerWidget) {
                const widgetSize = this.comparerWidget.computeSize(size[0]);
                let extra = 60;
                const mode = this.properties.comparer_mode;
                const hasMultiplePairs = this.comparerWidget.maxPairs > 1;
                if (mode === "Carousel")
                    extra += 120;
                else if (["Slider", "Click", "Side-by-Side", "Stacked"].includes(mode) && hasMultiplePairs)
                    extra += 35;
                else if (mode === "Batch" && this.comparerWidget.maxBatchPages > 1)
                    extra += 90;
                else if (mode === "Onion Skin") {
                    extra += 35;
                    if (hasMultiplePairs)
                        extra += 35;
                }
                size[1] = Math.max(size[1], widgetSize[1] + extra);
            }
            return size;
        };
        const originalOnExecuted = nodeType.prototype.onExecuted;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodeType.prototype.onExecuted = function (message) {
            let result;
            if (originalOnExecuted)
                result = originalOnExecuted.call(this, message);
            if (message && typeof message === 'object') {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                const self = this;
                const images = message.ui?.images || message.images;
                if (images && images.length > 0 && this.comparerWidget) {
                    this.comparerWidget.value = { images };
                    const [w, h] = self.size;
                    self.setSize([Math.max(w, 700), Math.max(h, 600)]);
                    self.setDirtyCanvas(true, true);
                }
            }
            return result || message;
        };
        nodeType.prototype.setIsPointerDown = function (down = this.isPointerDown) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const self = this;
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const appObj = app;
            const newIsDown = down && !!appObj.canvas.pointer_is_down;
            if (this.isPointerDown !== newIsDown) {
                this.isPointerDown = newIsDown;
                self.setDirtyCanvas(true, false);
            }
            this.imageIndex = this.isPointerDown ? 1 : 0;
            if (this.isPointerDown)
                requestAnimationFrame(() => { this.setIsPointerDown(); });
        };
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodeType.prototype.onMouseDown = function () { this.setIsPointerDown(true); return false; };
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodeType.prototype.onMouseEnter = function () {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any  
            const appObj = app;
            this.setIsPointerDown(!!appObj.canvas.pointer_is_down);
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.isPointerOver = true;
            this.setDirtyCanvas(true, false);
        };
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodeType.prototype.onMouseLeave = function () {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            this.setIsPointerDown(false);
            this.isPointerOver = false;
            this.setDirtyCanvas(true, false);
        };
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodeType.prototype.onMouseMove = function (_event, pos) {
            this.pointerOverPos = [...pos];
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            if (this.properties.comparer_mode === "Slider")
                this.setDirtyCanvas(true, false);
            else if (this.properties.comparer_mode === "Click")
                this.imageIndex = pos[0] > this.size[0] / 2 ? 1 : 0;
            return true;
        };
        console.log("AdvancedImageComparer node setup complete");
    }
});
//# sourceMappingURL=advanced_comparer.js.map