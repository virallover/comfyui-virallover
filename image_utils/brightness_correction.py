import torch
import numpy as np

class BrightnessCorrectionNode:
    @staticmethod
    def _to_single_mask(mask_tensor):
        mask_np = mask_tensor.cpu().numpy()
        # 支持 [1, 1, H, W]、[1, 3, H, W]、[1, H, W, 1]、[1, H, W, 3]、[1, H, W]、[H, W]
        if mask_np.ndim == 4:
            mask_np = mask_np[0]
        if mask_np.ndim == 3:
            if mask_np.shape[0] in [1, 3]:
                mask_np = mask_np.max(axis=0)
            elif mask_np.shape[2] in [1, 3]:
                mask_np = mask_np.max(axis=2)
        return (mask_np > 0.5)

    @staticmethod
    def rgb_to_grayscale_torch(img):
        # img: [1, 3, H, W] or [3, H, W]
        if img.ndim == 4:
            return 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        elif img.ndim == 3:
            return 0.299 * img[0:1, :, :] + 0.587 * img[1:2, :, :] + 0.114 * img[2:3, :, :]
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

    @staticmethod
    def _ensure_chw(tensor):
        arr = tensor.cpu().numpy()
        if arr.ndim == 4 and arr.shape[-1] == 3:  # [1, H, W, 3]
            arr = arr.transpose(0, 3, 1, 2)  # -> [1, 3, H, W]
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

    def adjust_brightness(self, original_image, original_mask, target_image, target_mask):
        print(f"[调试] original_image shape: {getattr(original_image, 'shape', None)}, dtype: {getattr(original_image, 'dtype', None)}")
        print(f"[调试] original_mask shape: {getattr(original_mask, 'shape', None)}, dtype: {getattr(original_mask, 'dtype', None)}")
        print(f"[调试] target_image shape: {getattr(target_image, 'shape', None)}, dtype: {getattr(target_image, 'dtype', None)}")
        print(f"[调试] target_mask shape: {getattr(target_mask, 'shape', None)}, dtype: {getattr(target_mask, 'dtype', None)}")

        ori_img = self._ensure_chw(original_image.clone())
        tgt_img = self._ensure_chw(target_image.clone())
        ori_mask_img = self._ensure_chw(original_mask.clone())
        tgt_mask_img = self._ensure_chw(target_mask.clone())

        ori_mask = self._to_single_mask(ori_mask_img)
        tgt_mask = self._to_single_mask(tgt_mask_img)

        ori_gray = self.rgb_to_grayscale_torch(ori_img).cpu().numpy().squeeze()
        tgt_gray = self.rgb_to_grayscale_torch(tgt_img).cpu().numpy().squeeze()

        if ori_gray.shape != ori_mask.shape:
            raise ValueError(f"original_image灰度图与original_mask尺寸不一致: ori_gray shape={ori_gray.shape}, ori_mask shape={ori_mask.shape}")
        if tgt_gray.shape != tgt_mask.shape:
            raise ValueError(f"target_image灰度图与target_mask尺寸不一致: tgt_gray shape={tgt_gray.shape}, tgt_mask shape={tgt_mask.shape}")

        ori_pixels = ori_gray[ori_mask]
        tgt_pixels = tgt_gray[tgt_mask]

        if ori_pixels.size == 0 or tgt_pixels.size == 0:
            print("Mask 区域为空，跳过校正")
            corrected = tgt_img
        else:
            factor = float(ori_pixels.mean() / tgt_pixels.mean())
            corrected = tgt_img.clone()
            mask_tensor = torch.from_numpy(tgt_mask.astype(np.float32)).to(tgt_img.device)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            corrected = corrected * (1 - mask_tensor) + torch.clamp(corrected * factor, 0, 1) * mask_tensor

        # 输出标准化处理：RGB、float32、[0, 1]、[1, 3, H, W]
        if corrected.shape[1] == 1:
            corrected = corrected.repeat(1, 3, 1, 1)
        corrected = corrected.to
