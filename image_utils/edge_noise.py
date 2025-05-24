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
        # 记录输入格式
        input_is_nhwc = (image.ndim == 4 and image.shape[-1] == 3)
        device = image.device

        # 全流程转为 NCHW
        image_chw = self._ensure_chw(image)
        mask_chw = self._ensure_mask_chw(mask, image_chw.shape, device)

        # 添加高斯噪声
        noise = torch.randn_like(image_chw) * strength
        noisy_image = image_chw * (1 - mask_chw) + (image_chw + noise).clamp(0, 1) * mask_chw
        noisy_image = noisy_image.clamp(0, 1)

        # 输出格式还原
        if input_is_nhwc:
            noisy_image = noisy_image.permute(0, 2, 3, 1)  # [1, 3, H, W] -> [1, H, W, 3]
        return (noisy_image,)
