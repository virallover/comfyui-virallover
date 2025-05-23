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
        支持 ComfyUI tensor 或 numpy array，统一转为 uint8 numpy。
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            # 如果是 [1, 3, H, W] 或 [1, H, W]，去掉 batch 维
            if img.shape[0] == 1:
                img = img[0]
            # 如果是 [3, H, W]，转为 [H, W, 3]
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
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
