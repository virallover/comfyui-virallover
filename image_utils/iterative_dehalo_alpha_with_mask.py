import torch
import torch.nn.functional as F
import numpy as np

class IterativeDeHaloAlphaWithMaskTorch:
    @staticmethod
    def _ensure_chw(tensor):
        arr = tensor.cpu().numpy()
        if arr.ndim == 4 and arr.shape[-1] == 3:
            arr = arr.transpose(0, 3, 1, 2)
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr.transpose(2, 0, 1)[None, ...]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)[None, ...]
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=0)[None, ...]
        return torch.from_numpy(arr).to(tensor.device).float()

    @staticmethod
    def _ensure_mask_chw(mask, device=None):
        arr = mask.detach().cpu().numpy() if hasattr(mask, 'detach') else np.array(mask)
        if arr.max() > 1.1:
            arr = arr / 255.0
        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
            arr = arr[0, 0]
        elif arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[-1] == 1:
            arr = arr[0, ..., 0]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        arr = arr[None, None, ...]
        tensor = torch.from_numpy(arr).float()
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "passes": ("INT", {"default": 3, "min": 1, "max": 10}),
                "low_sens": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 255.0}),
                "high_sens": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 255.0}),
                "darkstr": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "brightstr": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "dehalo_with_dynamic_mask"
    CATEGORY = "image/fix"

    def dehalo_with_dynamic_mask(self, image, mask, passes, low_sens, high_sens, darkstr, brightstr):
        input_is_nhwc = (image.ndim == 4 and image.shape[-1] == 3)
        device = image.device
        img_chw = self._ensure_chw(image)
        mask_chw = self._ensure_mask_chw(mask, device=device)

        img = img_chw[0]  # [3, H, W]
        base_mask = mask_chw[0, 0]  # [H, W]

        for _ in range(passes):
            img_ = img.unsqueeze(0)
            img_down = F.interpolate(img_, scale_factor=0.5, mode='bicubic', align_corners=False)
            img_blur = F.interpolate(img_down, size=img.shape[1:], mode='bilinear', align_corners=False).squeeze(0)

            diff = torch.abs(img - img_blur)
            diff_gray = 0.2989 * diff[0] + 0.5870 * diff[1] + 0.1140 * diff[2]

            temp = (diff_gray + 256.0) / (512.0 + high_sens)
            halo_mask = diff_gray * (255.0 - low_sens * temp) / 255.0
            halo_mask = torch.clamp(halo_mask, 0.0, 1.0)

            dynamic_mask = halo_mask * base_mask
            merged = img * (1 - dynamic_mask) + img_blur * dynamic_mask
            correction_factor = torch.where(img < merged, darkstr, brightstr)
            img = img - (img - merged) * correction_factor

        out = img.unsqueeze(0)
        if input_is_nhwc:
            out = out.permute(0, 2, 3, 1).contiguous()
        out = out.clamp(0, 1).to(torch.float32)
        return (out,)
