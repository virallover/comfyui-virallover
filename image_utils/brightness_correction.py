import torch
import numpy as np

class BrightnessCorrectionNode:
    @staticmethod
    def _to_single_mask(mask_tensor, threshold=0.5):
        """
        自动将任意图片/张量转为二值 mask，支持 [1,3,H,W]、[1,H,W,3]、[1,H,W]、[H,W]、[1,1,H,W]、[1,H,W,1] 等格式
        """
        arr = mask_tensor.detach().cpu().float().numpy()
        # [1, H, W, 3] -> [1, 3, H, W]
        if arr.ndim == 4 and arr.shape[-1] == 3:
            arr = arr.transpose(0, 3, 1, 2)
        # [1, 3, H, W] 或 [3, H, W]
        if arr.ndim == 4 and arr.shape[1] == 3:
            arr = arr.mean(axis=1, keepdims=True)  # [1, 1, H, W]
        elif arr.ndim == 4 and arr.shape[1] == 1:
            pass  # [1, 1, H, W]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr  # [1, H, W]
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.transpose(2, 0, 1)  # [1, H, W]
        elif arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr.transpose(2, 0, 1).mean(axis=0, keepdims=True)  # [1, H, W]
        elif arr.ndim == 2:
            arr = arr[None, ...]  # [1, H, W]
        # 归一化到0~1
        if arr.max() > 1.1:
            arr = arr / 255.0
        mask = (arr > threshold)
        # squeeze到[H,W]
        mask = mask.squeeze()
        return mask

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
                "original_mask": ("MASK",),
                "target_image": ("IMAGE",),
                "target_mask": ("MASK",),
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
            if mask_tensor.shape[1] == 1 and corrected.shape[1] == 3:
                mask_tensor = mask_tensor.repeat(1, 3, 1, 1)  # [1, 3, H, W]
            elif mask_tensor.shape[1] != corrected.shape[1]:
                mask_tensor = mask_tensor.expand(-1, corrected.shape[1], -1, -1)
            corrected = corrected * (1 - mask_tensor) + torch.clamp(corrected * factor, 0, 1) * mask_tensor

        # 输出标准化处理：RGB、float32、[0, 1]、[1, 3, H, W]
        if corrected.shape[1] == 1:
            corrected = corrected.repeat(1, 3, 1, 1)
        corrected = corrected.clamp(0, 1).to(torch.float32)
        assert corrected.shape[1] == 3, f"corrected shape error: {corrected.shape}"
        return (corrected,)
