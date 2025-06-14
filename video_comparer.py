from nodes import PreviewImage
import torch
import base64
import io
from PIL import Image
import numpy as np
import time
import gc
import sys

class VideoComparer:
    # Improved cache with better auto-fill retention
    _video_cache = {
        "last_video_a": None,
        "last_video_b": None,
        "cache_metadata_a": None,
        "cache_metadata_b": None,
        "most_recent_video": None,
        "most_recent_metadata": None,
        "last_update_time": 0,
        "execution_count": 0,
        "memory_usage": 0,
        "max_memory_mb": 200,  # Increased for larger videos
        "auto_fill_history": [],  # New: Keep track of recent videos for auto-fill
        "last_cleanup_time": 0,
    }
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fps": ("FLOAT", {"default": 8.0, "min": 0.01, "max": 60.0, "step": 0.01}),
                "auto_fill": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "video_a": ("IMAGE",),
                "video_b": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "compare_videos"
    OUTPUT_NODE = True
    CATEGORY = "utils"
    DESCRIPTION = "Video comparison with unlimited frames and reliable auto-fill"

    def estimate_frame_size(self, image_tensor):
        """Estimate compressed frame size"""
        h, w, c = image_tensor.shape
        # Very conservative estimate for highly compressed output
        base_size = h * w * c * 0.15  # Aggressive compression estimate
        return base_size / (1024 * 1024)  # Convert to MB

    def auto_determine_settings(self, video_tensor):
        """Automatic settings optimized for no frame limits"""
        if video_tensor is None or len(video_tensor) == 0:
            return {"max_frames": 1000, "frame_skip": 1, "quality": 50, "max_dimension": 512}
        
        total_frames = len(video_tensor)
        h, w, c = video_tensor[0].shape
        
        # Determine max dimension based on original size for compression
        if h > 768 or w > 768:
            max_dimension = 512
            quality = 45  # More aggressive compression for large images
        elif h > 512 or w > 512:
            max_dimension = 512
            quality = 50
        elif h > 256 or w > 256:
            max_dimension = min(h, w)
            quality = 55
        else:
            max_dimension = min(h, w)
            quality = 60
        
        # No frame limits - process all frames
        max_frames = total_frames  # Process ALL frames
        frame_skip = 1  # Never skip frames
        
        print(f"[VideoComparer] No-limit settings for {total_frames} frames ({h}x{w}): "
              f"max_frames={max_frames}, frame_skip={frame_skip}, quality={quality}, max_dim={max_dimension}")
        
        return {
            "max_frames": max_frames,
            "frame_skip": frame_skip, 
            "quality": quality,
            "max_dimension": max_dimension
        }

    def tensor_to_base64(self, image_tensor, quality=60, max_dimension=512):
        """Ultra-compressed frame conversion"""
        try:
            i = 255. * image_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Aggressive resizing to keep files small
            if img.width > max_dimension or img.height > max_dimension:
                ratio = min(max_dimension / img.width, max_dimension / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            
            # Always use JPEG for maximum compression
            if img.mode == 'RGBA':
                # Convert RGBA to RGB for JPEG
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            
            # Use aggressive JPEG compression
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
                
        except Exception as e:
            print(f"[VideoComparer] Error converting frame to base64: {e}")
            return None

    def sample_frames_intelligently(self, video_tensor, max_frames, frame_skip):
        """Sample frames while preserving important ones"""
        if video_tensor is None or len(video_tensor) == 0:
            return []
        
        total_frames = len(video_tensor)
        
        # For short videos, use all frames
        if total_frames <= max_frames:
            return list(range(total_frames))
        
        # Apply frame skipping
        candidates = list(range(0, total_frames, frame_skip))
        
        # If still too many, sample evenly but keep first and last
        if len(candidates) > max_frames:
            step = len(candidates) / max_frames
            indices = [candidates[int(i * step)] for i in range(max_frames)]
            # Ensure first and last frames are included
            if 0 not in indices:
                indices[0] = 0
            if total_frames - 1 not in indices:
                indices[-1] = total_frames - 1
            return sorted(set(indices))  # Remove duplicates and sort
        
        return candidates

    def process_video_to_frames(self, video_tensor, fps):
        """Process video without frame limits"""
        if video_tensor is None or len(video_tensor) == 0:
            return None
        
        # Get settings (now without limits)
        settings = self.auto_determine_settings(video_tensor)
        
        # Sample frames (now processes all frames)
        frame_indices = self.sample_frames_intelligently(
            video_tensor, 
            settings["max_frames"], 
            settings["frame_skip"]
        )
        
        frames = []
        total_size_mb = 0
        
        print(f"[VideoComparer] Processing {len(frame_indices)} frames from {len(video_tensor)} total (no limits)")
        
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx >= len(video_tensor):
                continue
            
            frame = video_tensor[frame_idx]
            estimated_size = self.estimate_frame_size(frame)
            
            data_url = self.tensor_to_base64(
                frame, 
                settings["quality"], 
                settings["max_dimension"]
            )
            
            if data_url:
                frames.append({
                    "data_url": data_url,
                    "frame_index": i,
                    "original_index": frame_idx
                })
                total_size_mb += estimated_size
                
                # Progress logging for large videos
                if i > 0 and i % 20 == 0:
                    print(f"[VideoComparer] Processed {i}/{len(frame_indices)} frames, size: {total_size_mb:.2f}MB")
        
        print(f"[VideoComparer] Generated {len(frames)} frames, total size: {total_size_mb:.2f}MB")
        
        return {
            "frames": frames,
            "fps": fps,
            "frame_count": len(frames),
            "original_frame_count": len(video_tensor),
            "tensor_shape": list(video_tensor.shape),
            "estimated_size_mb": total_size_mb,
            "auto_settings": settings
        }

    def cleanup_memory(self, force=False):
        """More selective memory cleanup that preserves auto-fill capability"""
        if not force:
            # Only clean up if we haven't cleaned recently (avoid cleaning every execution)
            current_time = time.time()
            if current_time - self._video_cache["last_cleanup_time"] < 30:  # Don't clean more than once per 30 seconds
                return
            self._video_cache["last_cleanup_time"] = current_time
        
        print(f"[VideoComparer] Performing {'forced' if force else 'selective'} memory cleanup")
        
        # Keep the most recent video for auto-fill, only clear older cached videos
        if force:
            # Clear all cached tensors
            for key in ["last_video_a", "last_video_b", "most_recent_video"]:
                if self._video_cache[key] is not None:
                    del self._video_cache[key]
                    self._video_cache[key] = None
            
            # Clear metadata
            for key in ["cache_metadata_a", "cache_metadata_b", "most_recent_metadata"]:
                self._video_cache[key] = None
            
            # Clear auto-fill history
            self._video_cache["auto_fill_history"] = []
        else:
            # Selective cleanup - keep most recent for auto-fill
            # Only clear if we have multiple cached videos
            cached_count = sum(1 for key in ["last_video_a", "last_video_b", "most_recent_video"] 
                             if self._video_cache[key] is not None)
            
            if cached_count > 2:  # Only clean if we have more than 2 cached videos
                # Keep the most recent one, clear others
                most_recent_time = 0
                keep_key = None
                
                for video_key, meta_key in [("last_video_a", "cache_metadata_a"), 
                                           ("last_video_b", "cache_metadata_b"),
                                           ("most_recent_video", "most_recent_metadata")]:
                    if (self._video_cache[video_key] is not None and 
                        self._video_cache[meta_key] is not None):
                        timestamp = self._video_cache[meta_key].get("timestamp", 0)
                        if timestamp > most_recent_time:
                            most_recent_time = timestamp
                            keep_key = video_key
                
                # Clear everything except the most recent
                for video_key in ["last_video_a", "last_video_b", "most_recent_video"]:
                    if video_key != keep_key and self._video_cache[video_key] is not None:
                        del self._video_cache[video_key]
                        self._video_cache[video_key] = None
                        
                        # Clear corresponding metadata
                        meta_key = video_key.replace("video", "metadata")
                        if meta_key.replace("most_recent_metadata", "most_recent_metadata") in self._video_cache:
                            self._video_cache[meta_key] = None
        
        # Reset memory counter
        self._video_cache["memory_usage"] = 0
        
        # Force cleanup
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def check_memory_pressure(self):
        """More generous memory monitoring for unlimited frame processing"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Higher thresholds to accommodate larger videos
            if memory_mb > 12000:  # 12GB threshold for selective cleanup
                print(f"[VideoComparer] High memory usage detected: {memory_mb:.2f}MB, performing selective cleanup...")
                self.cleanup_memory(force=False)  # Selective cleanup
                return True
            elif memory_mb > 16000:  # 16GB threshold for force cleanup
                print(f"[VideoComparer] Critical memory usage detected: {memory_mb:.2f}MB, forcing cleanup...")
                self.cleanup_memory(force=True)  # Force cleanup
                return True
        except ImportError:
            # Less frequent periodic cleanup for larger videos
            if self._video_cache["execution_count"] % 30 == 0:  # Every 30 executions instead of 20
                print(f"[VideoComparer] Periodic cleanup after {self._video_cache['execution_count']} executions")
                self.cleanup_memory(force=False)  # Selective cleanup
                return True
        
        return False

    def videos_are_same(self, tensor1, tensor2):
        """Lightweight tensor comparison"""
        if tensor1 is None or tensor2 is None:
            return False
        if tensor1.shape != tensor2.shape:
            return False
        
        # For any large tensor, just compare a few samples
        try:
            if tensor1.numel() > 100000:  # Lower threshold
                # Compare just first, middle, and last pixels
                if not torch.allclose(tensor1.flatten()[:10], tensor2.flatten()[:10], atol=1e-5):
                    return False
                mid = tensor1.numel() // 2
                if not torch.allclose(tensor1.flatten()[mid:mid+10], tensor2.flatten()[mid:mid+10], atol=1e-5):
                    return False
                if not torch.allclose(tensor1.flatten()[-10:], tensor2.flatten()[-10:], atol=1e-5):
                    return False
                return True
            else:
                return torch.allclose(tensor1, tensor2, atol=1e-5)
        except:
            return False

    def get_tensor_hash(self, tensor):
        """Ultra-lightweight hash"""
        if tensor is None:
            return None
        try:
            # Just use shape and a couple values
            return hash((tuple(tensor.shape), float(tensor.sum().item())))
        except:
            return hash(tuple(tensor.shape))

    def add_to_auto_fill_history(self, video_tensor, fps):
        """Add video to auto-fill history for better auto-fill reliability"""
        if video_tensor is None:
            return
            
        # Create metadata for the video
        metadata = {
            "fps": fps,
            "frame_count": len(video_tensor),
            "timestamp": time.time(),
            "hash": self.get_tensor_hash(video_tensor),
            "tensor": video_tensor  # Keep reference for auto-fill
        }
        
        # Add to history
        self._video_cache["auto_fill_history"].append(metadata)
        
        # Keep only the last 5 videos in history for larger video support
        if len(self._video_cache["auto_fill_history"]) > 5:
            old_entry = self._video_cache["auto_fill_history"].pop(0)
            # Clean up the tensor reference
            if "tensor" in old_entry:
                del old_entry["tensor"]
        
        print(f"[VideoComparer] Added video to auto-fill history (history size: {len(self._video_cache['auto_fill_history'])})")

    def update_cache(self, video_a, video_b, fps):
        """Improved cache updates with better auto-fill support"""
        current_time = time.time()
        self._video_cache["execution_count"] += 1
        
        # Less aggressive memory checking - only check every 3 executions
        if self._video_cache["execution_count"] % 3 == 0:
            self.check_memory_pressure()
        
        # Only cache if not excessively large (increased limit for no frame restrictions)
        max_cache_frames = 300  # Increased from 100 to handle larger videos
        
        # Add videos to auto-fill history
        if video_a is not None:
            self.add_to_auto_fill_history(video_a, fps)
        if video_b is not None:
            self.add_to_auto_fill_history(video_b, fps)
        
        # Process video A
        if video_a is not None and len(video_a) <= max_cache_frames:
            if not self.videos_are_same(video_a, self._video_cache["last_video_a"]):
                if self._video_cache["last_video_a"] is not None:
                    del self._video_cache["last_video_a"]
                    gc.collect()
                
                self._video_cache["last_video_a"] = video_a
                self._video_cache["cache_metadata_a"] = {
                    "fps": fps,
                    "frame_count": len(video_a),
                    "timestamp": current_time,
                    "hash": self.get_tensor_hash(video_a)
                }

        # Process video B
        if video_b is not None and len(video_b) <= max_cache_frames:
            if not self.videos_are_same(video_b, self._video_cache["last_video_b"]):
                if self._video_cache["last_video_b"] is not None:
                    del self._video_cache["last_video_b"]
                    gc.collect()
                
                self._video_cache["last_video_b"] = video_b
                self._video_cache["cache_metadata_b"] = {
                    "fps": fps,
                    "frame_count": len(video_b),
                    "timestamp": current_time,
                    "hash": self.get_tensor_hash(video_b)
                }

        # Update most recent (prefer smaller videos for caching)
        most_recent_candidate = None
        if video_a is not None and len(video_a) <= max_cache_frames:
            most_recent_candidate = video_a
        elif video_b is not None and len(video_b) <= max_cache_frames:
            most_recent_candidate = video_b
            
        if most_recent_candidate is not None:
            if not self.videos_are_same(most_recent_candidate, self._video_cache["most_recent_video"]):
                if self._video_cache["most_recent_video"] is not None:
                    del self._video_cache["most_recent_video"]
                    gc.collect()
                
                self._video_cache["most_recent_video"] = most_recent_candidate
                self._video_cache["most_recent_metadata"] = {
                    "fps": fps,
                    "frame_count": len(most_recent_candidate),
                    "timestamp": current_time,
                    "hash": self.get_tensor_hash(most_recent_candidate)
                }
        
        self._video_cache["last_update_time"] = current_time

    def get_most_recent_cached_video(self, exclude_video=None):
        """Improved auto-fill video retrieval with fallback to history"""
        candidates = []
        
        # First, try the traditional cache
        if self._video_cache["cache_metadata_a"] is not None:
            candidates.append({
                'video': self._video_cache["last_video_a"],
                'metadata': self._video_cache["cache_metadata_a"],
                'slot': 'A'
            })
        
        if self._video_cache["cache_metadata_b"] is not None:
            candidates.append({
                'video': self._video_cache["last_video_b"],
                'metadata': self._video_cache["cache_metadata_b"],
                'slot': 'B'
            })
        
        if self._video_cache["most_recent_metadata"] is not None:
            candidates.append({
                'video': self._video_cache["most_recent_video"],
                'metadata': self._video_cache["most_recent_metadata"],
                'slot': 'recent'
            })
        
        # Add candidates from auto-fill history
        for entry in self._video_cache["auto_fill_history"]:
            if "tensor" in entry:
                candidates.append({
                    'video': entry["tensor"],
                    'metadata': entry,
                    'slot': 'history'
                })
        
        # Filter out excluded video
        if exclude_video is not None:
            candidates = [c for c in candidates if not self.videos_are_same(c['video'], exclude_video)]
        
        if not candidates:
            print("[VideoComparer] No candidates found for auto-fill")
            return None
        
        # Return most recent
        candidates.sort(key=lambda x: x['metadata']['timestamp'], reverse=True)
        selected = candidates[0]
        print(f"[VideoComparer] Auto-fill using video from {selected['slot']} slot (timestamp: {selected['metadata']['timestamp']})")
        return selected['video']

    def get_auto_filled_videos(self, video_a, video_b, auto_fill):
        """Improved auto-fill logic with better debugging"""
        if not auto_fill:
            print("[VideoComparer] Auto-fill disabled")
            return video_a, video_b
        
        print(f"[VideoComparer] Auto-fill enabled. Input: video_a={'Present' if video_a is not None else 'None'}, video_b={'Present' if video_b is not None else 'None'}")
        
        if video_a is not None and video_b is not None:
            print("[VideoComparer] Both videos provided, no auto-fill needed")
            return video_a, video_b
        
        if video_a is not None and video_b is None:
            print("[VideoComparer] Auto-filling video_b")
            fill_candidate = self.get_most_recent_cached_video(exclude_video=video_a)
            if fill_candidate is not None:
                print("[VideoComparer] Successfully auto-filled video_b")
            else:
                print("[VideoComparer] Failed to auto-fill video_b - no suitable candidate found")
            return video_a, fill_candidate
        
        if video_a is None and video_b is not None:
            print("[VideoComparer] Auto-filling video_a")
            fill_candidate = self.get_most_recent_cached_video(exclude_video=video_b)
            if fill_candidate is not None:
                print("[VideoComparer] Successfully auto-filled video_a")
            else:
                print("[VideoComparer] Failed to auto-fill video_a - no suitable candidate found")
            return fill_candidate, video_b
        
        if video_a is None and video_b is None:
            print("[VideoComparer] Auto-filling both videos from cache")
            first_video = self.get_most_recent_cached_video()
            second_video = self.get_most_recent_cached_video(exclude_video=first_video)
            if first_video is not None and second_video is not None:
                print("[VideoComparer] Successfully auto-filled both videos")
            elif first_video is not None:
                print("[VideoComparer] Auto-filled only one video")
            else:
                print("[VideoComparer] Failed to auto-fill any videos")
            return first_video, second_video

    def compare_videos(self, fps, auto_fill=True, video_a=None, video_b=None, prompt=None, extra_pnginfo=None):
        print(f"[VideoComparer] Starting comparison (execution #{self._video_cache['execution_count'] + 1}) with auto_fill={auto_fill}, unlimited frames")
        
        # Apply auto-fill logic BEFORE updating cache (important!)
        final_video_a, final_video_b = self.get_auto_filled_videos(video_a, video_b, auto_fill)
        
        # Update cache with the ORIGINAL inputs (not auto-filled ones)
        self.update_cache(video_a, video_b, fps)
        
        video_data = []
        total_estimated_size = 0

        # Process videos with ultra-conservative settings
        if final_video_a is not None and len(final_video_a) > 0:
            print(f"[VideoComparer] Processing video A: {len(final_video_a)} frames")
            video_a_data = self.process_video_to_frames(final_video_a, fps)
            if video_a_data:
                video_data.append({
                    "name": "video_a",
                    "frames": video_a_data["frames"],
                    "fps": fps,
                    "is_auto_filled": auto_fill and video_a is None and final_video_a is not None,
                    "original_frame_count": video_a_data["original_frame_count"],
                    "processed_frame_count": video_a_data["frame_count"],
                    "auto_settings": video_a_data["auto_settings"]
                })
                total_estimated_size += video_a_data.get("estimated_size_mb", 0)

        if final_video_b is not None and len(final_video_b) > 0:
            print(f"[VideoComparer] Processing video B: {len(final_video_b)} frames")
            video_b_data = self.process_video_to_frames(final_video_b, fps)
            if video_b_data:
                video_data.append({
                    "name": "video_b",
                    "frames": video_b_data["frames"],
                    "fps": fps,
                    "is_auto_filled": auto_fill and video_b is None and final_video_b is not None,
                    "original_frame_count": video_b_data["original_frame_count"],
                    "processed_frame_count": video_b_data["frame_count"],
                    "auto_settings": video_b_data["auto_settings"]
                })
                total_estimated_size += video_b_data.get("estimated_size_mb", 0)

        # Prepare response metadata
        auto_fill_info = {
            "auto_fill_enabled": auto_fill,
            "video_a_auto_filled": auto_fill and video_a is None and final_video_a is not None,
            "video_b_auto_filled": auto_fill and video_b is None and final_video_b is not None,
            "execution_count": self._video_cache["execution_count"],
            "total_estimated_size_mb": total_estimated_size,
            "unlimited_frames": True,  # Indicate no frame limits
            "cache_status": {
                "cached_videos": sum(1 for key in ["last_video_a", "last_video_b", "most_recent_video"] 
                                   if self._video_cache[key] is not None),
                "auto_fill_history_size": len(self._video_cache["auto_fill_history"]),
                "last_cleanup": self._video_cache["last_cleanup_time"]
            }
        }
        
        print(f"[VideoComparer] Completed. Response size: {total_estimated_size:.2f}MB")
        print(f"[VideoComparer] Auto-fill results: A={'auto-filled' if auto_fill_info['video_a_auto_filled'] else 'original'}, B={'auto-filled' if auto_fill_info['video_b_auto_filled'] else 'original'}")
        
        # Minimal cleanup - don't clear cache aggressively
        gc.collect()
        
        return {
            "ui": {
                "video_data": video_data,
                "auto_fill_info": auto_fill_info
            }
        }

    @classmethod
    def clear_cache(cls):
        """Clear everything and force cleanup"""
        for key in ["last_video_a", "last_video_b", "most_recent_video"]:
            if cls._video_cache[key] is not None:
                del cls._video_cache[key]
        
        # Clear auto-fill history
        for entry in cls._video_cache["auto_fill_history"]:
            if "tensor" in entry:
                del entry["tensor"]
        
        cls._video_cache = {
            "last_video_a": None,
            "last_video_b": None,
            "cache_metadata_a": None,
            "cache_metadata_b": None,
            "most_recent_video": None,
            "most_recent_metadata": None,
            "last_update_time": 0,
            "execution_count": 0,
            "memory_usage": 0,
            "max_memory_mb": 200,
            "auto_fill_history": [],
            "last_cleanup_time": 0,
        }
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def print_cache_status(cls):
        """Print current cache status"""
        print("\n=== VideoComparer Cache Status ===")
        print(f"Execution count: {cls._video_cache['execution_count']}")
        print(f"Auto-fill history size: {len(cls._video_cache['auto_fill_history'])}")
        
        for video_key, meta_key in [("last_video_a", "cache_metadata_a"), 
                                   ("last_video_b", "cache_metadata_b"),
                                   ("most_recent_video", "most_recent_metadata")]:
            if cls._video_cache[video_key] is not None:
                meta = cls._video_cache[meta_key]
                print(f"{video_key}: {meta['frame_count'] if meta else 'Unknown'} frames")
            else:
                print(f"{video_key}: None")
        
        print("Auto-fill history:")
        for i, entry in enumerate(cls._video_cache["auto_fill_history"]):
            print(f"  [{i}]: {entry['frame_count']} frames, timestamp: {entry['timestamp']}")
        print("===============================================\n")

NODE_CLASS_MAPPINGS = {
    "VideoComparer": VideoComparer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoComparer": "Video Comparer",
}