from nodes import PreviewImage

class AdvancedImageComparer(PreviewImage):
  """Advanced custom node that compares two images in the UI."""

  NAME = 'Advanced Image Comparer'
  CATEGORY = 'utils'
  FUNCTION = "compare_images"
  OUTPUT_NODE = True

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {},
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
                     image_a=None,
                     image_b=None,
                     filename_prefix="advanced.compare.",
                     prompt=None,
                     extra_pnginfo=None):

    result = {"ui": {"images": []}}
    
    # Process and save image A if provided
    if image_a is not None and len(image_a) > 0:
      images_a = self.save_images(image_a, filename_prefix + "a.", prompt, extra_pnginfo)
      if "ui" in images_a and "images" in images_a["ui"]:
        for img in images_a["ui"]["images"]:
          if isinstance(img, dict) and "filename" in img:
            # Add flag to identify this as image A
            img["is_image_a"] = True
            result["ui"]["images"].append(img)
            print(f"Added image A: {img['filename']}")
    
    # Process and save image B if provided
    if image_b is not None and len(image_b) > 0:
      images_b = self.save_images(image_b, filename_prefix + "b.", prompt, extra_pnginfo)
      if "ui" in images_b and "images" in images_b["ui"]:
        for img in images_b["ui"]["images"]:
          if isinstance(img, dict) and "filename" in img:
            # Add flag to identify this as image B
            img["is_image_b"] = True
            result["ui"]["images"].append(img)
            print(f"Added image B: {img['filename']}")
    
    print(f"Final result: {result}")
    return result 