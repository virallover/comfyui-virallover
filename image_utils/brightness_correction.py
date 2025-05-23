import torch
import torchvision.transforms.functional as TF
import numpy as np

class BrightnessCorrectionNodeTorchVision:
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

    def _ensure_tensor_format(self, image):
        # [B, C, H, W] and float32
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.dtype != torch.float32:
            image = image.float()
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.unsqueeze(0)
        return image

    def _extract_masked_mean_luminance(self, image, mask):
        # image: [1, 3, H, W], mask: [1, 1, H, W]
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        masked_pixels = gray[mask[:, 0] > 0.5]
        if masked_pixels.numel() == 0:
            return torch.tensor(1.0, device=image.device)
        return masked_pixels.mean()

    def adjust_brightness(self, original_image, original_mask, target_image, target_mask):
        original_image = self._ensure_tensor_format(original_image)
        original_mask = self._ensure_tensor_format(original_mask)
        target_image = self._ensure_tensor_format(target_image)
        target_mask = self._ensure_tensor_format(target_mask)

        mean_ori = self._extract_masked_mean_luminance(original_image, original_mask)
        mean_tgt = self._extract_masked_mean_luminance(target_image, target_mask)
        factor = (mean_ori / mean_tgt).clamp(0.5, 2.0) if mean_tgt > 0 else 1.0

        adjusted = TF.adjust_brightness(target_image.squeeze(0), factor)
        adjusted = adjusted.unsqueeze(0).clamp(0, 1)
        return (adjusted,)
