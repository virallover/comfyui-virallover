from skimage.transform import resize
import numpy as np
import torch

class DepthFitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "new_depth": ("IMAGE",),
                "old_depth": ("IMAGE",),
                "old_mask": ("MASK",),
                "new_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("fitted_depth",)
    FUNCTION = "fit_depth"
    CATEGORY = "custom"

    @staticmethod
    def _to_single_mask(mask):
        mask_np = mask.cpu().numpy()
        # 处理 [C, H, W]，C=1 或 C=3
        if mask_np.ndim == 3:
            if mask_np.shape[0] == 1:
                mask_np = mask_np[0]
            else:
                mask_np = np.any(mask_np > 0, axis=0).astype(np.uint8)
        elif mask_np.ndim == 2:
            pass
        else:
            raise ValueError(f"Unsupported mask shape: {mask_np.shape}")
        return mask_np

    @staticmethod
    def _extract_gray(depth):
        arr = depth.cpu().numpy()
        print(f'raw depth shape: {arr.shape}')
        if arr.ndim == 4 and arr.shape[-1] == 3:
            arr = arr[0, :, :, 0]
        elif arr.ndim == 4 and arr.shape[1] == 3:
            arr = arr[0, 0]
        elif arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr[0]
        elif arr.ndim == 3:
            arr = arr[0]
        else:
            arr = arr.squeeze()
        return arr

    def fit_depth(self, new_depth, old_depth, old_mask, new_mask):
        # Convert to numpy arrays
        old_depth_np = self._extract_gray(old_depth)
        new_depth_np = self._extract_gray(new_depth)
        old_mask_np = self._to_single_mask(old_mask[0])
        new_mask_np = self._to_single_mask(new_mask[0])
        # 调试用，输出shape
        print(f"old_depth_np shape: {old_depth_np.shape}, old_mask_np shape: {old_mask_np.shape}")
        print(f"new_depth_np shape: {new_depth_np.shape}, new_mask_np shape: {new_mask_np.shape}")
        print(f"raw old_depth shape: {old_depth.shape}, raw new_depth shape: {new_depth.shape}")

        # resize depth 到 mask 的 shape
        if old_depth_np.shape != old_mask_np.shape:
            print(f"Resizing old_depth from {old_depth_np.shape} to {old_mask_np.shape}")
            old_depth_np = resize(
                old_depth_np, old_mask_np.shape, order=1, preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
        if new_depth_np.shape != new_mask_np.shape:
            print(f"Resizing new_depth from {new_depth_np.shape} to {new_mask_np.shape}")
            new_depth_np = resize(
                new_depth_np, new_mask_np.shape, order=1, preserve_range=True, anti_aliasing=True
            ).astype(np.float32)

        # Extract old depth values under old_mask
        old_depth_masked = old_depth_np[old_mask_np > 0]

        if old_depth_masked.size == 0:
            old_min, old_max = 0.0, 1.0
        else:
            old_min = np.min(old_depth_masked)
            old_max = np.max(old_depth_masked)

        # Extract new depth values under new_mask
        new_depth_masked = new_depth_np[new_mask_np > 0]

        new_min = np.min(new_depth_masked) if new_depth_masked.size > 0 else 0.0
        new_max = np.max(new_depth_masked) if new_depth_masked.size > 0 else 1.0

        # Prepare output map (default: 0)
        aligned = np.zeros_like(new_depth_np)

        if abs(new_max - new_min) < 1e-6:
            aligned[new_mask_np > 0] = (old_min + old_max) / 2
        else:
            aligned[new_mask_np > 0] = (
                (new_depth_np[new_mask_np > 0] - new_min)
                * (old_max - old_min)
                / (new_max - new_min)
                + old_min
            )

        # Convert back to tensor: [1, 1, H, W]
        result = torch.from_numpy(aligned).unsqueeze(0).unsqueeze(0).to(dtype=new_depth.dtype)

        return (result,)
