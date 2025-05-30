from nodes import PreviewImage
import torch
import base64
import io
from PIL import Image
import numpy as np

class VideoComparer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fps": ("FLOAT", {"default": 8.0, "min": 0.01, "max": 60.0, "step": 0.01}),
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
    DESCRIPTION = "Real-time video comparison widget with 6 viewing modes"

    def compare_videos(self, fps, video_a=None, video_b=None, prompt=None, extra_pnginfo=None):
        print(f"[VideoComparer] Pure comparison mode - FPS: {fps}")
        print(f"[VideoComparer] Video A: {video_a.shape if video_a is not None else 'None'}")
        print(f"[VideoComparer] Video B: {video_b.shape if video_b is not None else 'None'}")
        
        video_data = []

        def tensor_to_base64(image_tensor):
            """Convert a single image tensor to base64 data URL"""
            i = 255. * image_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"

        # Process video A
        if video_a is not None and len(video_a) > 0:
            print(f"[VideoComparer] Processing Video A: {len(video_a)} frames")
            video_a_frames = []
            
            for i, frame in enumerate(video_a):
                data_url = tensor_to_base64(frame)
                video_a_frames.append({
                    "data_url": data_url,
                    "frame_index": i
                })
            
            video_data.append({
                "name": "video_a",
                "frames": video_a_frames,
                "fps": fps
            })

        # Process video B
        if video_b is not None and len(video_b) > 0:
            print(f"[VideoComparer] Processing Video B: {len(video_b)} frames")
            video_b_frames = []
            
            for i, frame in enumerate(video_b):
                data_url = tensor_to_base64(frame)
                video_b_frames.append({
                    "data_url": data_url,
                    "frame_index": i
                })
            
            video_data.append({
                "name": "video_b",
                "frames": video_b_frames,
                "fps": fps
            })

        print(f"[VideoComparer] Sending {len(video_data)} videos to widget")
        
        return {
            "ui": {
                "video_data": video_data
            }
        }

NODE_CLASS_MAPPINGS = {
    "VideoComparer": VideoComparer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoComparer": "Video Comparer",
}