import torch
import numpy as np
import cv2

import numpy as np
import cv2

class CV2InpaintEdgeFixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 支持 ComfyUI IMAGE 类型
                "mask": ("IMAGE",),   # 通常 mask 也是 IMAGE 或 NPARRAY
                "radius": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 32.0}),
            },
            "optional": {
                "flags": ("INT", {"default": cv2.INPAINT_TELEA}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/OpenCV"

    def to_uint8(self, img):
        """
        Convert ComfyUI image (float32 0-1) or already uint8 image to proper uint8 BGR.
        """
        if img.dtype == np.float32 and img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return img

    def execute(self, image, mask, radius, flags=cv2.INPAINT_TELEA):
        img_bgr = self.to_uint8(image)
        mask_uint8 = self.to_uint8(mask)

        # ensure single channel mask
        if len(mask_uint8.shape) == 3:
            mask_gray = cv2.cvtColor(mask_uint8, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask_uint8

        result = cv2.inpaint(img_bgr, mask_gray, radius, flags)

        # convert back to float32 0-1 for ComfyUI output
        result = result.astype(np.float32) / 255.0
        return (result,)
