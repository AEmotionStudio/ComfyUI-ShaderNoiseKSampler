from nodes import PreviewImage
import torch
import base64
import io
from PIL import Image
import numpy as np
import time

class VideoComparer:
    # Enhanced cache to store video data with timestamps and better tracking
    _video_cache = {
        "last_video_a": None,
        "last_video_b": None,
        "cache_metadata_a": None,
        "cache_metadata_b": None,
        "most_recent_video": None,  # Track the most recently processed video
        "most_recent_metadata": None,
        "last_update_time": 0,
        "execution_count": 0
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
    DESCRIPTION = "Auto-filling video comparison widget with intelligent slot detection"

    def tensor_to_base64(self, image_tensor):
        """Convert a single image tensor to base64 data URL"""
        i = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def process_video_to_frames(self, video_tensor, fps):
        """Convert video tensor to frame data"""
        if video_tensor is None or len(video_tensor) == 0:
            return None
            
        frames = []
        for i, frame in enumerate(video_tensor):
            data_url = self.tensor_to_base64(frame)
            frames.append({
                "data_url": data_url,
                "frame_index": i
            })
        
        return {
            "frames": frames,
            "fps": fps,
            "frame_count": len(frames),
            "tensor_shape": list(video_tensor.shape)
        }

    def videos_are_same(self, tensor1, tensor2):
        """Check if two video tensors are the same"""
        if tensor1 is None or tensor2 is None:
            return False
        if tensor1.shape != tensor2.shape:
            return False
        return torch.equal(tensor1, tensor2)

    def get_tensor_hash(self, tensor):
        """Get a simple hash for a tensor to help with tracking"""
        if tensor is None:
            return None
        return hash(tuple(tensor.shape) + (float(tensor.sum()),))

    def update_cache(self, video_a, video_b, fps):
        """Enhanced cache update with better tracking of most recent videos"""
        current_time = time.time()
        self._video_cache["execution_count"] += 1
        
        # Track what's being processed in this execution
        current_videos = {}
        
        # Process video A
        if video_a is not None:
            video_a_hash = self.get_tensor_hash(video_a)
            current_videos['A'] = {'tensor': video_a, 'hash': video_a_hash}
            
            # Update cache A if it's different
            if not self.videos_are_same(video_a, self._video_cache["last_video_a"]):
                self._video_cache["last_video_a"] = video_a.clone() if hasattr(video_a, 'clone') else video_a
                self._video_cache["cache_metadata_a"] = {
                    "fps": fps,
                    "frame_count": len(video_a),
                    "shape": list(video_a.shape),
                    "timestamp": current_time,
                    "execution": self._video_cache["execution_count"],
                    "hash": video_a_hash
                }

        # Process video B
        if video_b is not None:
            video_b_hash = self.get_tensor_hash(video_b)
            current_videos['B'] = {'tensor': video_b, 'hash': video_b_hash}
            
            # Update cache B if it's different
            if not self.videos_are_same(video_b, self._video_cache["last_video_b"]):
                self._video_cache["last_video_b"] = video_b.clone() if hasattr(video_b, 'clone') else video_b
                self._video_cache["cache_metadata_b"] = {
                    "fps": fps,
                    "frame_count": len(video_b),
                    "shape": list(video_b.shape),
                    "timestamp": current_time,
                    "execution": self._video_cache["execution_count"],
                    "hash": video_b_hash
                }

        # Update most recent video tracker
        # Priority: 1) Any newly provided video, 2) Most recently cached video
        most_recent_candidate = None
        most_recent_slot = None
        
        if len(current_videos) > 0:
            # If we have current videos, pick one as most recent (prefer A if both present)
            if 'A' in current_videos:
                most_recent_candidate = current_videos['A']['tensor']
                most_recent_slot = 'A'
            elif 'B' in current_videos:
                most_recent_candidate = current_videos['B']['tensor']
                most_recent_slot = 'B'
        
        # Update most recent video if we have a candidate
        if most_recent_candidate is not None:
            if not self.videos_are_same(most_recent_candidate, self._video_cache["most_recent_video"]):
                self._video_cache["most_recent_video"] = most_recent_candidate.clone() if hasattr(most_recent_candidate, 'clone') else most_recent_candidate
                self._video_cache["most_recent_metadata"] = {
                    "fps": fps,
                    "frame_count": len(most_recent_candidate),
                    "shape": list(most_recent_candidate.shape),
                    "timestamp": current_time,
                    "execution": self._video_cache["execution_count"],
                    "source_slot": most_recent_slot,
                    "hash": self.get_tensor_hash(most_recent_candidate)
                }
        
        self._video_cache["last_update_time"] = current_time

    def get_most_recent_cached_video(self, exclude_video=None):
        """Get the most recently cached video, optionally excluding one that matches exclude_video"""
        candidates = []
        
        # Add cached videos with their metadata
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
        
        # Add most recent video if it's different from slot-specific caches
        if (self._video_cache["most_recent_metadata"] is not None and 
            self._video_cache["most_recent_video"] is not None):
            
            # Check if most_recent is different from slot caches
            is_different_from_a = not self.videos_are_same(
                self._video_cache["most_recent_video"], 
                self._video_cache["last_video_a"]
            )
            is_different_from_b = not self.videos_are_same(
                self._video_cache["most_recent_video"], 
                self._video_cache["last_video_b"]
            )
            
            if is_different_from_a or is_different_from_b:
                candidates.append({
                    'video': self._video_cache["most_recent_video"],
                    'metadata': self._video_cache["most_recent_metadata"],
                    'slot': 'most_recent'
                })
        
        # Filter out videos that match the exclude_video
        if exclude_video is not None:
            candidates = [c for c in candidates if not self.videos_are_same(c['video'], exclude_video)]
        
        if not candidates:
            return None
        
        # Sort by timestamp (most recent first)
        candidates.sort(key=lambda x: x['metadata']['timestamp'], reverse=True)
        
        return candidates[0]['video']

    def get_auto_filled_videos(self, video_a, video_b, auto_fill):
        """Enhanced auto-fill logic with better most-recent video tracking"""
        if not auto_fill:
            return video_a, video_b
        
        # Case 1: Both videos provided - no auto-fill needed
        if video_a is not None and video_b is not None:
            return video_a, video_b
        
        # Case 2: Only video_a provided - auto-fill video_b
        if video_a is not None and video_b is None:
            fill_candidate = self.get_most_recent_cached_video(exclude_video=video_a)
            return video_a, fill_candidate
        
        # Case 3: Only video_b provided - auto-fill video_a
        if video_a is None and video_b is not None:
            fill_candidate = self.get_most_recent_cached_video(exclude_video=video_b)
            return fill_candidate, video_b
        
        # Case 4: No videos provided - use most recent cached videos
        if video_a is None and video_b is None:
            first_video = self.get_most_recent_cached_video()
            second_video = self.get_most_recent_cached_video(exclude_video=first_video)
            return first_video, second_video

    def compare_videos(self, fps, auto_fill=True, video_a=None, video_b=None, prompt=None, extra_pnginfo=None):
        # Apply auto-fill logic BEFORE updating cache
        final_video_a, final_video_b = self.get_auto_filled_videos(video_a, video_b, auto_fill)
        
        # Now update cache with new inputs
        self.update_cache(video_a, video_b, fps)
        
        video_data = []

        # Process final video A
        if final_video_a is not None and len(final_video_a) > 0:
            video_a_data = self.process_video_to_frames(final_video_a, fps)
            if video_a_data:
                video_data.append({
                    "name": "video_a",
                    "frames": video_a_data["frames"],
                    "fps": fps,
                    "is_auto_filled": auto_fill and video_a is None and final_video_a is not None
                })

        # Process final video B
        if final_video_b is not None and len(final_video_b) > 0:
            video_b_data = self.process_video_to_frames(final_video_b, fps)
            if video_b_data:
                video_data.append({
                    "name": "video_b",
                    "frames": video_b_data["frames"],
                    "fps": fps,
                    "is_auto_filled": auto_fill and video_b is None and final_video_b is not None
                })

        # Add metadata about auto-fill status
        auto_fill_info = {
            "auto_fill_enabled": auto_fill,
            "video_a_auto_filled": auto_fill and video_a is None and final_video_a is not None,
            "video_b_auto_filled": auto_fill and video_b is None and final_video_b is not None,
            "execution_count": self._video_cache["execution_count"],
            "cache_status": {
                "has_cached_a": self._video_cache["last_video_a"] is not None,
                "has_cached_b": self._video_cache["last_video_b"] is not None,
                "has_most_recent": self._video_cache["most_recent_video"] is not None,
                "cached_a_frames": self._video_cache["cache_metadata_a"]["frame_count"] if self._video_cache["cache_metadata_a"] else 0,
                "cached_b_frames": self._video_cache["cache_metadata_b"]["frame_count"] if self._video_cache["cache_metadata_b"] else 0,
                "most_recent_frames": self._video_cache["most_recent_metadata"]["frame_count"] if self._video_cache["most_recent_metadata"] else 0
            }
        }
        
        return {
            "ui": {
                "video_data": video_data,
                "auto_fill_info": auto_fill_info
            }
        }

    @classmethod
    def clear_cache(cls):
        """Clear the video cache - useful for debugging"""
        cls._video_cache = {
            "last_video_a": None,
            "last_video_b": None,
            "cache_metadata_a": None,
            "cache_metadata_b": None,
            "most_recent_video": None,
            "most_recent_metadata": None,
            "last_update_time": 0,
            "execution_count": 0
        }

    @classmethod
    def print_cache_status(cls):
        """Print concise cache status for debugging"""
        print(f"[VideoComparer] Cache: A({cls._video_cache['cache_metadata_a']['frame_count'] if cls._video_cache['cache_metadata_a'] else 0}), B({cls._video_cache['cache_metadata_b']['frame_count'] if cls._video_cache['cache_metadata_b'] else 0}), Recent({cls._video_cache['most_recent_metadata']['frame_count'] if cls._video_cache['most_recent_metadata'] else 0})")

NODE_CLASS_MAPPINGS = {
    "VideoComparer": VideoComparer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoComparer": "Video Comparer",
}