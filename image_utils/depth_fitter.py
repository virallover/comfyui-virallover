from skimage.transform import resize
import numpy as np
import torch
from scipy.interpolate import interp1d

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

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("fitted_depth", "final_mask")
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

    @staticmethod
    def warp_depth_by_masked_distribution(old_depth_np, new_depth_np, new_mask_np, bins=1000):
        """
        将 old_depth 在 new_mask 区域的像素值分布压缩成 new_depth 中对应区域的分布，其它区域设为0。

        参数:
            old_depth_np: 原深度图，shape [H, W]
            new_depth_np: 新深度图，shape [H, W]
            new_mask_np: 二值掩码图，shape [H, W]，值为 0 或 1
            bins: 拟合精度（histogram bin 数）

        返回:
            warped_depth: np.ndarray，mask 区域值被压缩，其他区域为0
        """
        # 1. 提取掩码区域的深度值
        old_vals = old_depth_np[new_mask_np > 0]
        new_vals = new_depth_np[new_mask_np > 0]

        # 2. 构建 CDF 和反 CDF
        old_hist, old_edges = np.histogram(old_vals, bins=bins, density=True)
        old_cdf = np.cumsum(old_hist) / np.sum(old_hist)
        old_centers = (old_edges[:-1] + old_edges[1:]) / 2
        value_to_cdf = interp1d(old_centers, old_cdf, bounds_error=False, fill_value=(0.0, 1.0))

        new_hist, new_edges = np.histogram(new_vals, bins=bins, density=True)
        new_cdf = np.cumsum(new_hist) / np.sum(new_hist)
        new_centers = (new_edges[:-1] + new_edges[1:]) / 2
        cdf_to_value = interp1d(new_cdf, new_centers, bounds_error=False, fill_value=(new_centers[0], new_centers[-1]))

        # 3. 执行变换
        warped = np.zeros_like(old_depth_np)  # 其他区域设为0
        cdf_vals = value_to_cdf(old_vals)
        warped_vals = cdf_to_value(cdf_vals)
        warped[new_mask_np > 0] = warped_vals

        return warped

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

        print("old_depth min/max:", np.min(old_depth_np), np.max(old_depth_np))
        print("new_depth min/max:", np.min(new_depth_np), np.max(new_depth_np))

        # 用 warp_depth_by_masked_distribution 拟合 old_depth 到 new_depth 的分布，仅在 new_mask 区域
        aligned = self.warp_depth_by_masked_distribution(old_depth_np, new_depth_np, new_mask_np)
        print("aligned min/max:", np.min(aligned), np.max(aligned))

        # Convert back to tensor: [1, H, W]
        result = torch.from_numpy(aligned).unsqueeze(0).to(dtype=new_depth.dtype)

        # 计算 final_mask
        # 1. 重叠区域（new_mask=1 且 old_mask=1）强制为1
        overlap = (old_mask_np > 0) & (new_mask_np > 0)
        # 2. 非重叠区域（new_mask=1 且 old_mask=0）根据深度比较
        non_overlap = (old_mask_np == 0) & (new_mask_np > 0)
        in_front = (new_depth_np < aligned)

        final_mask_np = np.zeros_like(aligned, dtype=np.float32)
        final_mask_np[overlap] = 1.0
        final_mask_np[non_overlap] = in_front[non_overlap].astype(np.float32)
        final_mask = torch.from_numpy(final_mask_np).unsqueeze(0)

        return (result, final_mask)
