import torch
import numpy as np
import cv2

class CV2InpaintEdgeFixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),          # 输入图像 [1, 3, H, W]
                "edge_mask": ("MASK",),       # 修复区域 mask [1, 1, H, W]
                "inpaintRadius": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 50.0}),
                "method": (["telea", "ns"],),  # 选择算法
            }
        }

    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("fixed_nparray",)
    FUNCTION = "run"
    CATEGORY = "inpainting"

    def run(self, image, edge_mask, inpaintRadius, method):
        img = image[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        mask = (edge_mask[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255  # [H, W]

        # Convert to uint8 for OpenCV
        img8 = (img * 255).astype(np.uint8)

        # Select method
        if method == "telea":
            flags = cv2.INPAINT_TELEA
        else:
            flags = cv2.INPAINT_NS

        # Inpaint
        inpainted = cv2.inpaint(img8, mask, inpaintRadius, flags=flags)

        # 直接返回 nparray 格式 [H, W, 3]，uint8
        return (inpainted,)
