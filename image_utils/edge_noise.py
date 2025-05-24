import torch
import numpy as np

class EdgeNoise:
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
    def _ensure_mask_chw(mask, target_shape, device):
        arr = mask.cpu().numpy()
        # [1, H, W, 1] -> [1, 1, H, W]
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr.transpose(0, 3, 1, 2)
        # [1, H, W] -> [1, 1, H, W]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[None, ...]
        # [H, W] -> [1, 1, H, W]
        elif arr.ndim == 2:
            arr = arr[None, None, ...]
        # [1, 1, H, W] 保持
        mask_tensor = torch.from_numpy(arr).to(device).float()
        # 扩展到 [1, 3, H, W]
        if mask_tensor.shape[1] == 1 and target_shape[1] == 3:
            mask_tensor = mask_tensor.repeat(1, 3, 1, 1)
        return mask_tensor

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("noised_image",)
    FUNCTION = "add_noise"
    CATEGORY = "Image Processing"

    def add_noise(self, image, mask, strength):
        input_is_nhwc = (image.ndim == 4 and image.shape[-1] == 3)
        device = image.device

        # 全流程转为 NCHW
        image_chw = self._ensure_chw(image)
        mask_chw = self._ensure_mask_chw(mask, image_chw.shape, device)

        # 计算亮度（Y通道，仿照灰度公式）
        # Y = 0.299*R + 0.587*G + 0.114*B
        Y = 0.299 * image_chw[:, 0:1, :, :] + 0.587 * image_chw[:, 1:2, :, :] + 0.114 * image_chw[:, 2:3, :, :]

        # 只对亮度加噪声
        noise = torch.randn_like(Y) * strength
        Y_noised = (Y * (1 - mask_chw[:, 0:1, :, :]) + (Y + noise).clamp(0, 1) * mask_chw[:, 0:1, :, :]).clamp(0, 1)

        # 用原始色彩和新亮度重建RGB
        # 线性重建法（近似）：保持色彩比例
        eps = 1e-6
        Y_orig = Y + eps
        ratio = Y_noised / Y_orig
        ratio = ratio.expand_as(image_chw)
        noised_image = (image_chw * ratio).clamp(0, 1)

        # 输出格式还原
        if input_is_nhwc:
            noised_image = noised_image.permute(0, 2, 3, 1)
        return (noised_image,)
