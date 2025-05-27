import numpy as np
from PIL import Image
from typing import Any
import torch

class ApplyBrightnessFromGrayWithMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "gray_image": ("IMAGE",),
                "mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Custom/Image"

    @staticmethod
    def _ensure_chw(img):
        arr = np.array(img)
        if arr.ndim == 4 and arr.shape[-1] == 3:
            arr = arr.transpose(0, 3, 1, 2)
        elif arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr.transpose(2, 0, 1)[None, ...]
        elif arr.ndim == 2:
            arr = arr[None, None, ...]
        return arr.astype(np.float32)

    @staticmethod
    def _ensure_mask_chw(mask, target_shape):
        arr = np.array(mask)
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr.transpose(0, 3, 1, 2)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.transpose(2, 0, 1)[None, ...]
        elif arr.ndim == 2:
            arr = arr[None, None, ...]
        arr = arr.astype(np.float32)
        if arr.max() > 1.1:
            arr = arr / 255.0
        # 扩展到[1,3,H,W]以便广播
        if arr.shape[1] == 1 and target_shape[1] == 3:
            arr = np.repeat(arr, 3, axis=1)
        return arr

    def apply(self, target_image: Any, gray_image: Any, mask: Any):
        # 记录输入格式
        input_is_nhwc = False
        arr = np.array(target_image)
        if arr.ndim == 4 and arr.shape[-1] == 3:
            input_is_nhwc = True
        # 全流程转为NCHW
        tgt = self._ensure_chw(target_image)
        gray = self._ensure_chw(gray_image)
        mask = self._ensure_mask_chw(mask, tgt.shape)
        # 灰度图通道对齐
        g = gray[0]
        if g.ndim == 3 and g.shape[0] == 3:
            g = g.transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        elif g.ndim == 3 and g.shape[0] == 1:
            g = np.repeat(g.squeeze(0), 3, axis=0).transpose(1, 2, 0)  # [1, H, W] -> [H, W, 3]
        elif g.ndim == 2:
            g = np.stack([g, g, g], axis=-1)  # [H, W] -> [H, W, 3]
        else:
            raise ValueError(f"Unsupported gray shape: {g.shape}")

        g = g.astype(np.uint8)
        _, _, h, w = tgt.shape
        if g.shape[1:] != (h, w):
            g = np.array(Image.fromarray(g).resize((w, h), Image.BILINEAR)).astype(np.float32)
            g = g if g.ndim == 3 else np.stack([g]*3, axis=-1)
            gray = g[None, ...]  # [1, H, W, 3]
            gray = gray.transpose(0, 3, 1, 2)  # [1, 3, H, W]
        if mask.shape[2:] != (h, w):
            mask_img = (mask[0,0]*255).astype(np.uint8)
            mask_img = np.array(Image.fromarray(mask_img).resize((w, h), Image.BILINEAR)) / 255.0
            mask = mask_img[None, None, ...]
            if tgt.shape[1] == 3:
                mask = np.repeat(mask, 3, axis=1)
        # 混合
        output = tgt * (1 - mask) + gray * mask
        output = np.clip(output, 0, 255).astype(np.uint8)
        # 输出还原为输入格式
        if input_is_nhwc:
            output = output.transpose(0,2,3,1)  # [1,3,H,W]->[1,H,W,3]
        output_img = Image.fromarray(output[0])
        return (output_img,)
