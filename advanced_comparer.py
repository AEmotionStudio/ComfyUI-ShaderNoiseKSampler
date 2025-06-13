from nodes import PreviewImage
import torch

class AdvancedImageComparer(PreviewImage):
    """Advanced custom node that compares two images in the UI with auto-fill functionality."""

    NAME = 'Advanced Image Comparer'
    CATEGORY = 'utils'
    FUNCTION = "compare_images"
    OUTPUT_NODE = True
    
    # Class-level storage for previous images
    _last_images_a = None
    _last_images_b = None
    _last_prompt = None
    _last_extra_pnginfo = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auto_fill": ("BOOLEAN", {"default": True, "tooltip": "Auto-fill empty slot with last generated images"}),
            },
            "optional": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    def compare_images(self,
                       auto_fill=True,
                       image_a=None,
                       image_b=None,
                       filename_prefix="advanced.compare.",
                       prompt=None,
                       extra_pnginfo=None):

        print(f"[AdvancedImageComparer] Auto-fill enabled: {auto_fill}")
        print(f"[AdvancedImageComparer] Input A: {'provided' if image_a is not None else 'None'}")
        print(f"[AdvancedImageComparer] Input B: {'provided' if image_b is not None else 'None'}")

        result = {"ui": {"images": []}}
        
        # Track which inputs are originally provided (not auto-filled)
        original_image_a = image_a
        original_image_b = image_b
        
        # Auto-fill logic when enabled
        if auto_fill:
            # Count provided inputs
            inputs_provided = sum([image_a is not None, image_b is not None])
            
            if inputs_provided == 1:
                print("[AdvancedImageComparer] Only one input provided, attempting auto-fill")
                
                # If only image_a is provided, try to fill image_b with most recent images
                if image_a is not None and image_b is None:
                    # Use the most recently cached images (prefer opposite slot, fallback to same slot)
                    if self._last_images_b is not None:
                        print("[AdvancedImageComparer] Auto-filling slot B with last images B")
                        image_b = self._last_images_b
                    elif self._last_images_a is not None:
                        print("[AdvancedImageComparer] Auto-filling slot B with last images A")
                        image_b = self._last_images_a
                
                # If only image_b is provided, try to fill image_a with most recent images
                elif image_b is not None and image_a is None:
                    # Use the most recently cached images (prefer opposite slot, fallback to same slot)
                    if self._last_images_a is not None:
                        print("[AdvancedImageComparer] Auto-filling slot A with last images A")
                        image_a = self._last_images_a
                    elif self._last_images_b is not None:
                        print("[AdvancedImageComparer] Auto-filling slot A with last images B")
                        image_a = self._last_images_b
            
            elif inputs_provided == 0:
                print("[AdvancedImageComparer] No inputs provided, using last available images")
                # If no inputs provided, use the most recent images if available
                if self._last_images_a is not None:
                    image_a = self._last_images_a
                if self._last_images_b is not None:
                    image_b = self._last_images_b
        
        # Process and save image A if provided
        if image_a is not None and len(image_a) > 0:
            print(f"[AdvancedImageComparer] Processing {len(image_a)} images for slot A")
            images_a = self.save_images(image_a, filename_prefix + "a.", prompt, extra_pnginfo)
            if "ui" in images_a and "images" in images_a["ui"]:
                for img in images_a["ui"]["images"]:
                    if isinstance(img, dict) and "filename" in img:
                        # Add flag to identify this as image A
                        img["is_image_a"] = True
                        result["ui"]["images"].append(img)
            
            # Only cache if this was an original input (not auto-filled)
            # This ensures we always cache the latest NEWLY GENERATED images
            if original_image_a is not None:
                self._last_images_a = image_a.clone() if hasattr(image_a, 'clone') else image_a
                print("[AdvancedImageComparer] Stored NEW images A for future auto-fill")
                
                # IMPORTANT: When we get fresh images in slot A, also update the cache for potential 
                # auto-fill of slot B in future runs (this ensures latest images are always used)
                if original_image_b is None and auto_fill:
                    self._last_images_b = image_a.clone() if hasattr(image_a, 'clone') else image_a
                    print("[AdvancedImageComparer] Updated slot B cache with latest images from A")
        
        # Process and save image B if provided
        if image_b is not None and len(image_b) > 0:
            print(f"[AdvancedImageComparer] Processing {len(image_b)} images for slot B")
            images_b = self.save_images(image_b, filename_prefix + "b.", prompt, extra_pnginfo)
            if "ui" in images_b and "images" in images_b["ui"]:
                for img in images_b["ui"]["images"]:
                    if isinstance(img, dict) and "filename" in img:
                        # Add flag to identify this as image B
                        img["is_image_b"] = True
                        result["ui"]["images"].append(img)
            
            # Only cache if this was an original input (not auto-filled)
            # This ensures we always cache the latest NEWLY GENERATED images
            if original_image_b is not None:
                self._last_images_b = image_b.clone() if hasattr(image_b, 'clone') else image_b
                print("[AdvancedImageComparer] Stored NEW images B for future auto-fill")
                
                # IMPORTANT: When we get fresh images in slot B, also update the cache for potential 
                # auto-fill of slot A in future runs (this ensures latest images are always used)
                if original_image_a is None and auto_fill:
                    self._last_images_a = image_b.clone() if hasattr(image_b, 'clone') else image_b
                    print("[AdvancedImageComparer] Updated slot A cache with latest images from B")
        
        # Store prompt and extra_pnginfo for potential future use
        if prompt is not None:
            self._last_prompt = prompt
        if extra_pnginfo is not None:
            self._last_extra_pnginfo = extra_pnginfo
        
        print(f"[AdvancedImageComparer] Returning {len(result['ui']['images'])} total images")
        return result

    @classmethod
    def clear_cache(cls):
        """Clear the cached images. Useful for debugging or memory management."""
        cls._last_images_a = None
        cls._last_images_b = None
        cls._last_prompt = None
        cls._last_extra_pnginfo = None
        print("[AdvancedImageComparer] Cache cleared")

    @classmethod
    def get_cache_info(cls):
        """Get information about cached images for debugging."""
        return {
            "has_images_a": cls._last_images_a is not None,
            "has_images_b": cls._last_images_b is not None,
            "images_a_shape": cls._last_images_a.shape if cls._last_images_a is not None else None,
            "images_b_shape": cls._last_images_b.shape if cls._last_images_b is not None else None,
        }

NODE_CLASS_MAPPINGS = {
    "AdvancedImageComparer": AdvancedImageComparer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedImageComparer": "Advanced Image Comparer",
}