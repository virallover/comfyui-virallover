import torch
import torch.nn.functional as F
import numpy as np

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

    def adjust_brightness(self, original_image, original_mask, target_image, target_mask):
        # 输入形状标准：[1, 3, H, W] 和 [1, 1, H, W]
        ori_img = original_image.clone()
        tgt_img = target_image.clone()
        ori_mask = (original_mask > 0.5).float()
        tgt_mask = (target_mask > 0.5).float()

        # 转灰度图
        def rgb_to_grayscale_torch(img):
            return 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]

        ori_gray = rgb_to_grayscale_torch(ori_img)
        tgt_gray = rgb_to_grayscale_torch(tgt_img)

        # 提取被 mask 遮盖的区域像素亮度
        ori_pixels = ori_gray[ori_mask.bool()]
        tgt_pixels = tgt_gray[tgt_mask.bool()]

        if ori_pixels.numel() == 0 or tgt_pixels.numel() == 0:
            print("Mask 区域为空，跳过校正")
            return (tgt_img,)

        # 计算亮度平均值比例
        factor = ori_pixels.mean() / tgt_pixels.mean()

        # 只对 target 的 mask 区域做亮度调整
        corrected = tgt_img.clone()
        corrected = corrected * (1 - tgt_mask) + torch.clamp(corrected * factor, 0, 1) * tgt_mask

        return (corrected,)
