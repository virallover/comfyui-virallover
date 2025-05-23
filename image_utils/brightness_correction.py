import numpy as np
import cv2
from PIL import Image
from comfy.utils import pil2tensor, tensor2pil

class BrightnessCorrectionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "original_mask": ("IMAGE",),
                "target_image": ("IMAGE",),
                "target_mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("corrected_target_image",)
    FUNCTION = "adjust_brightness"
    CATEGORY = "Image Processing"

    @staticmethod
    def _to_single_mask(mask_tensor):
        mask_np = mask_tensor.cpu().numpy()
        if mask_np.ndim == 3:
            if mask_np.shape[0] == 1:
                return (mask_np[0] > 0.5).astype(np.uint8)
            else:
                return (np.any(mask_np > 0.5, axis=0)).astype(np.uint8)
        elif mask_np.ndim == 2:
            return (mask_np > 0.5).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported mask shape: {mask_np.shape}")

    def adjust_brightness(self, original_image, original_mask, target_image, target_mask):
        # PIL
        ori = tensor2pil(original_image)
        tgt = tensor2pil(target_image)
        ori_np = np.array(ori)
        tgt_np = np.array(tgt)

        # 单通道二值 mask
        ori_mask_np = self._to_single_mask(original_mask)
        tgt_mask_np = self._to_single_mask(target_mask)

        # 原图亮度
        ori_gray = cv2.cvtColor(ori_np, cv2.COLOR_RGB2GRAY)
        ori_pixels = ori_gray[ori_mask_np == 1]
        mean_ori_brightness = np.mean(ori_pixels) if ori_pixels.size > 0 else 128

        # 目标图亮度
        tgt_gray = cv2.cvtColor(tgt_np, cv2.COLOR_RGB2GRAY)
        tgt_pixels = tgt_gray[tgt_mask_np == 1]
        mean_tgt_brightness = np.mean(tgt_pixels) if tgt_pixels.size > 0 else 1

        # 补偿因子
        brightness_factor = mean_ori_brightness / mean_tgt_brightness

        # 亮度调整
        tgt_float = tgt_np.astype(np.float32)
        corrected = tgt_float.copy()
        for c in range(3):
            channel = tgt_float[:, :, c]
            channel[tgt_mask_np == 1] = np.clip(channel[tgt_mask_np == 1] * brightness_factor, 0, 255)
            corrected[:, :, c] = channel

        corrected_img = Image.fromarray(corrected.astype(np.uint8))
        return (pil2tensor(corrected_img),)
