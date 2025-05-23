import numpy as np
import torch
from PIL import Image

def pil2tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(arr).unsqueeze(0)

def tensor2pil(tensor):
    arr = tensor[0].cpu().numpy()
    if arr.ndim == 3:
        if arr.shape[0] == 3:  # [3, H, W]
            arr = arr.transpose(1, 2, 0)
        elif arr.shape[2] == 3:  # [H, W, 3]
            pass
        else:
            raise ValueError(f"tensor2pil: 不支持的shape: {arr.shape}")
    elif arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    else:
        raise ValueError(f"tensor2pil: 不支持的shape: {arr.shape}")
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)

class BrightnessCorrectionNode:
    @staticmethod
    def _ensure_chw(tensor):
        arr = tensor.cpu().numpy()
        if arr.ndim == 4 and arr.shape[-1] == 3:  # [1, H, W, 3]
            arr = arr.transpose(0, 3, 1, 2)  # -> [1, 3, H, W]
            return torch.from_numpy(arr).to(tensor.device)
        return tensor

    @staticmethod
    def _ensure_mask_single_channel(tensor):
        arr = tensor.cpu().numpy()
        if arr.ndim == 4 and arr.shape[-1] == 3:  # [1, H, W, 3]
            arr = arr[..., 0]  # 只取第一个通道
            arr = arr[:, None, :, :]  # [1, 1, H, W]
            return torch.from_numpy(arr).to(tensor.device)
        elif arr.ndim == 4 and arr.shape[1] == 3:  # [1, 3, H, W]
            arr = arr[:, 0:1, :, :]  # 只取第一个通道
            return torch.from_numpy(arr).to(tensor.device)
        return tensor

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
        # 处理 [1, 1, H, W]
        if mask_np.ndim == 4 and mask_np.shape[0] == 1 and mask_np.shape[1] == 1:
            return (mask_np[0, 0] > 0.5).astype(np.float32)
        # 处理 [1, H, W]
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            return (mask_np[0] > 0.5).astype(np.float32)
        # 处理 [H, W]
        if mask_np.ndim == 2:
            return (mask_np > 0.5).astype(np.float32)
        # 处理 [C, H, W]
        if mask_np.ndim == 3:
            return (np.any(mask_np > 0.5, axis=0)).astype(np.float32)
        raise ValueError(f"Unsupported mask shape: {mask_np.shape}")

    def adjust_brightness(self, original_image, original_mask, target_image, target_mask):
        # 保证输入都是 [1, 3, H, W] float32
        original_image = self._ensure_chw(original_image)
        target_image = self._ensure_chw(target_image)
        original_mask = self._ensure_mask_single_channel(original_mask)
        target_mask = self._ensure_mask_single_channel(target_mask)

        ori = original_image[0].cpu().numpy()  # [3, H, W]
        tgt = target_image[0].cpu().numpy()    # [3, H, W]
        ori_mask = self._to_single_mask(original_mask)  # [H, W]
        tgt_mask = self._to_single_mask(target_mask)    # [H, W]

        # 灰度
        ori_gray = 0.299 * ori[0] + 0.587 * ori[1] + 0.114 * ori[2]
        tgt_gray = 0.299 * tgt[0] + 0.587 * tgt[1] + 0.114 * tgt[2]

        ori_pixels = ori_gray[ori_mask == 1]
        tgt_pixels = tgt_gray[tgt_mask == 1]
        mean_ori = np.mean(ori_pixels) if ori_pixels.size > 0 else 0.5
        mean_tgt = np.mean(tgt_pixels) if tgt_pixels.size > 0 else 1.0

        factor = mean_ori / mean_tgt if mean_tgt > 0 else 1.0

        # 只在 target_mask 区域调整亮度
        corrected = tgt.copy()
        for c in range(3):
            channel = corrected[c]
            channel[tgt_mask == 1] = np.clip(channel[tgt_mask == 1] * factor, 0, 1)
            corrected[c] = channel

        corrected = torch.from_numpy(corrected).unsqueeze(0).float()
        return (corrected,)
