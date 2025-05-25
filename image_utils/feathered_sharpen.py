import torch
import numpy as np
import cv2
from PIL import Image

class FeatheredSharpen:
    @staticmethod
    def _ensure_chw(tensor):
        arr = tensor.cpu().numpy()
        # [1, H, W, 3] -> [1, 3, H, W]
        if arr.ndim == 4 and arr.shape[-1] == 3:
            arr = arr.transpose(0, 3, 1, 2)
        # [H, W, 3] -> [1, 3, H, W]
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr.transpose(2, 0, 1)[None, ...]
        # [1, H, W] -> [1, 3, H, W]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)[None, ...]
        # [H, W] -> [1, 3, H, W]
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=0)[None, ...]
        return torch.from_numpy(arr).to(tensor.device).float()

    @staticmethod
    def _ensure_hw(mask):
        arr = mask.cpu().numpy()
        # [1, H, W] -> [H, W]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        # [H, W] 保持
        elif arr.ndim == 2:
            pass
        else:
            raise ValueError(f"mask shape must be [1,H,W] or [H,W], got {arr.shape}")
        return arr

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "feather_radius": ("INT", {"default": 20, "min": 1, "max": 200}),
                "sharpen_strength": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sharpened_image",)
    FUNCTION = "feathered_sharpen"
    CATEGORY = "Image Processing"

    def feathered_sharpen(self, image, mask, feather_radius, sharpen_strength):
        # 记录输入格式
        input_is_nhwc = (image.ndim == 4 and image.shape[-1] == 3)
        device = image.device

        # 全流程转为 NCHW
        image_chw = self._ensure_chw(image)
        mask_hw = self._ensure_hw(mask)

        img = image_chw[0].detach().cpu().numpy()  # [3, H, W], float32
        msk = mask_hw.astype(np.float32)           # [H, W], float32

        img = np.transpose(img, (1, 2, 0))  # -> [H, W, 3]
        img = np.clip(img, 0, 1)

        # === Step 1: Generate edge feather weight (只锐化人物边缘带) ===
        # msk: 人物区域为1，背景为0
        inv_mask = 1.0 - msk
        dist = cv2.distanceTransform((inv_mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
        weight = np.clip((feather_radius - dist) / feather_radius, 0, 1)[..., None]  # [H, W, 1]

        # === Step 2: Unsharp Mask ===
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(img, 1 + sharpen_strength, blurred, -sharpen_strength, 0)

        # === Step 3: Blend sharpened edge ===
        out = img * (1 - weight) + sharpened * weight
        out = np.clip(out, 0, 1)
        out_tensor = torch.from_numpy(out.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)  # [1, 3, H, W]

        # 输出格式还原
        if input_is_nhwc:
            out_tensor = out_tensor.permute(0, 2, 3, 1)  # [1, 3, H, W] -> [1, H, W, 3]
        return (out_tensor,)

