"""
Video Comparer node for ComfyUI.

This node allows comparing two videos side by side with auto-fill functionality
and unlimited frame support.
"""

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
    """Video comparison node with unlimited frames and reliable auto-fill."""
    
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
        "max_memory_mb": 200,
        "auto_fill_history": [],
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
        """Estimate compressed frame size."""
        h, w, c = image_tensor.shape
        base_size = h * w * c * 0.15
        return base_size / (1024 * 1024)

    def auto_determine_settings(self, video_tensor):
        """Automatic settings optimized for no frame limits."""
        if video_tensor is None or len(video_tensor) == 0:
            return {"max_frames": 1000, "frame_skip": 1, "quality": 50, "max_dimension": 512}
        
        total_frames = len(video_tensor)
        h, w, c = video_tensor[0].shape
        
        if h > 768 or w > 768:
            max_dimension = 512
            quality = 45
        elif h > 512 or w > 512:
            max_dimension = 512
            quality = 50
        elif h > 256 or w > 256:
            max_dimension = min(h, w)
            quality = 55
        else:
            max_dimension = min(h, w)
            quality = 60
        
        max_frames = total_frames
        frame_skip = 1
        
        return {
            "max_frames": max_frames,
            "frame_skip": frame_skip, 
            "quality": quality,
            "max_dimension": max_dimension
        }

    def tensor_to_base64(self, image_tensor, quality=60, max_dimension=512):
        """Ultra-compressed frame conversion."""
        try:
            i = 255. * image_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            if img.width > max_dimension or img.height > max_dimension:
                ratio = min(max_dimension / img.width, max_dimension / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
                
        except Exception as e:
            print(f"[VideoComparer] Error converting frame to base64: {e}")
            return None

    def sample_frames_intelligently(self, video_tensor, max_frames, frame_skip):
        """Sample frames while preserving important ones."""
        if video_tensor is None or len(video_tensor) == 0:
            return []
        
        total_frames = len(video_tensor)
        
        if total_frames <= max_frames:
            return list(range(total_frames))
        
        candidates = list(range(0, total_frames, frame_skip))
        
        if len(candidates) > max_frames:
            step = len(candidates) / max_frames
            indices = [candidates[int(i * step)] for i in range(max_frames)]
            if 0 not in indices:
                indices[0] = 0
            if total_frames - 1 not in indices:
                indices[-1] = total_frames - 1
            return sorted(set(indices))
        
        return candidates

    def process_video_to_frames(self, video_tensor, fps):
        """Process video without frame limits."""
        if video_tensor is None or len(video_tensor) == 0:
            return None
        
        settings = self.auto_determine_settings(video_tensor)
        
        frame_indices = self.sample_frames_intelligently(
            video_tensor, 
            settings["max_frames"], 
            settings["frame_skip"]
        )
        
        frames = []
        total_size_mb = 0
        
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
                
                if i > 0 and i % 20 == 0:
                    print(f"[VideoComparer] Processed {i}/{len(frame_indices)} frames")
        
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
        """Memory cleanup that preserves auto-fill capability."""
        if not force:
            current_time = time.time()
            if current_time - self._video_cache["last_cleanup_time"] < 30:
                return
            self._video_cache["last_cleanup_time"] = current_time
        
        if force:
            for key in ["last_video_a", "last_video_b", "most_recent_video"]:
                if self._video_cache[key] is not None:
                    del self._video_cache[key]
                    self._video_cache[key] = None
            
            for key in ["cache_metadata_a", "cache_metadata_b", "most_recent_metadata"]:
                self._video_cache[key] = None
            
            self._video_cache["auto_fill_history"] = []
        
        self._video_cache["memory_usage"] = 0
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def videos_are_same(self, tensor1, tensor2):
        """Lightweight tensor comparison."""
        if tensor1 is None or tensor2 is None:
            return False
        if tensor1.shape != tensor2.shape:
            return False
        
        try:
            if tensor1.numel() > 100000:
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
        """Ultra-lightweight hash."""
        if tensor is None:
            return None
        try:
            return hash((tuple(tensor.shape), float(tensor.sum().item())))
        except:
            return hash(tuple(tensor.shape))

    def add_to_auto_fill_history(self, video_tensor, fps):
        """Add video to auto-fill history."""
        if video_tensor is None:
            return
            
        metadata = {
            "fps": fps,
            "frame_count": len(video_tensor),
            "timestamp": time.time(),
            "hash": self.get_tensor_hash(video_tensor),
            "tensor": video_tensor
        }
        
        self._video_cache["auto_fill_history"].append(metadata)
        
        if len(self._video_cache["auto_fill_history"]) > 5:
            old_entry = self._video_cache["auto_fill_history"].pop(0)
            if "tensor" in old_entry:
                del old_entry["tensor"]

    def update_cache(self, video_a, video_b, fps):
        """Update cache with better auto-fill support."""
        current_time = time.time()
        self._video_cache["execution_count"] += 1
        
        max_cache_frames = 300
        
        if video_a is not None:
            self.add_to_auto_fill_history(video_a, fps)
        if video_b is not None:
            self.add_to_auto_fill_history(video_b, fps)
        
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
        """Get most recent cached video for auto-fill."""
        candidates = []
        
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
        
        for entry in self._video_cache["auto_fill_history"]:
            if "tensor" in entry:
                candidates.append({
                    'video': entry["tensor"],
                    'metadata': entry,
                    'slot': 'history'
                })
        
        if exclude_video is not None:
            candidates = [c for c in candidates if not self.videos_are_same(c['video'], exclude_video)]
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x['metadata']['timestamp'], reverse=True)
        return candidates[0]['video']

    def get_auto_filled_videos(self, video_a, video_b, auto_fill):
        """Auto-fill logic with debugging."""
        if not auto_fill:
            return video_a, video_b
        
        if video_a is not None and video_b is not None:
            return video_a, video_b
        
        if video_a is not None and video_b is None:
            fill_candidate = self.get_most_recent_cached_video(exclude_video=video_a)
            return video_a, fill_candidate
        
        if video_a is None and video_b is not None:
            fill_candidate = self.get_most_recent_cached_video(exclude_video=video_b)
            return fill_candidate, video_b
        
        if video_a is None and video_b is None:
            first_video = self.get_most_recent_cached_video()
            second_video = self.get_most_recent_cached_video(exclude_video=first_video)
            return first_video, second_video

    def compare_videos(self, fps, auto_fill=True, video_a=None, video_b=None, prompt=None, extra_pnginfo=None):
        """Main comparison function."""
        final_video_a, final_video_b = self.get_auto_filled_videos(video_a, video_b, auto_fill)
        
        self.update_cache(video_a, video_b, fps)
        
        video_data = []
        total_estimated_size = 0

        if final_video_a is not None and len(final_video_a) > 0:
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

        auto_fill_info = {
            "auto_fill_enabled": auto_fill,
            "video_a_auto_filled": auto_fill and video_a is None and final_video_a is not None,
            "video_b_auto_filled": auto_fill and video_b is None and final_video_b is not None,
            "execution_count": self._video_cache["execution_count"],
            "total_estimated_size_mb": total_estimated_size,
            "unlimited_frames": True,
        }
        
        gc.collect()
        
        return {
            "ui": {
                "video_data": video_data,
                "auto_fill_info": auto_fill_info
            }
        }

    @classmethod
    def clear_cache(cls):
        """Clear everything and force cleanup."""
        for key in ["last_video_a", "last_video_b", "most_recent_video"]:
            if cls._video_cache[key] is not None:
                del cls._video_cache[key]
        
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
