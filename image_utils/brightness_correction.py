import torch
import numpy as np

class BrightnessCorrectionNode:
    @staticmethod
    def _to_single_mask(mask_tensor):
        mask_np = mask_tensor.cpu().numpy()
        if mask_np.ndim == 4 and mask_np.shape[0] == 1 and mask_np.shape[1] == 1:
            return (mask_np[0, 0] > 0.5)
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            return (mask_np[0] > 0.5)
        if mask_np.ndim == 2:
            return (mask_np > 0.5)
        if mask_np.ndim == 3:
            return (np.any(mask_np > 0.5, axis=0))
        if mask_np.ndim == 4 and mask_np.shape[0] == 1 and mask_np.shape[-1] == 3:
            return (mask_np[0, ..., 0] > 0.5)
        raise ValueError(f"Unsupported mask shape: {mask_np.shape}")

    @staticmethod
    def rgb_to_grayscale_torch(img):
        # img: [1, 3, H, W] or [3, H, W]
        if img.ndim == 4:
            return 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        elif img.ndim == 3:
            return 0.299 * img[0:1, :, :] + 0.587 * img[1:2, :, :] + 0.114 * img[2:3, :, :]
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

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
        ori_img = original_image.clone()
        tgt_img = target_image.clone()
        ori_mask = self._to_single_mask(original_mask)
        tgt_mask = self._to_single_mask(target_mask)

        ori_gray = self.rgb_to_grayscale_torch(ori_img).cpu().numpy().squeeze()
        tgt_gray = self.rgb_to_grayscale_torch(tgt_img).cpu().numpy().squeeze()

        ori_pixels = ori_gray[ori_mask]
        tgt_pixels = tgt_gray[tgt_mask]

        if ori_pixels.size == 0 or tgt_pixels.size == 0:
            print("Mask 区域为空，跳过校正")
            return (tgt_img,)

        factor = float(ori_pixels.mean() / tgt_pixels.mean())

        corrected = tgt_img.clone()
        # 只对 mask 区域做亮度调整
        mask_tensor = torch.from_numpy(tgt_mask.astype(np.float32)).to(tgt_img.device)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        corrected = corrected * (1 - mask_tensor) + torch.clamp(corrected * factor, 0, 1) * mask_tensor

        return (corrected,)
