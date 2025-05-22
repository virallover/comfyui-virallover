from skimage.transform import resize
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf

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
    def tps_depth_warp(old_depth, old_mask, new_mask):
        """
        使用TPS（Thin Plate Spline）将 old_mask 区域的深度结构挤压映射到 new_mask 区域。
        返回一张深度图，仅在 new_mask 区域有效，其余为0。
        """
        # 1. 提取 old_mask 区域坐标和值
        old_coords = np.argwhere(old_mask > 0)
        if old_coords.shape[0] < 3:
            return np.zeros_like(old_depth)  # 不足以做拟合
        old_z = old_depth[old_mask > 0]

        # 2. 提取 new_mask 区域坐标
        new_coords = np.argwhere(new_mask > 0)
        if new_coords.shape[0] < 3:
            return np.zeros_like(old_depth)

        # 3. 创建 TPS 插值函数（使用RBF模拟TPS）
        old_x, old_y = old_coords[:, 1], old_coords[:, 0]  # (x, y)
        rbf_func = Rbf(old_x, old_y, old_z, function='thin_plate', smooth=1e-5)

        # 4. 应用于 new_mask 区域坐标
        new_x, new_y = new_coords[:, 1], new_coords[:, 0]
        new_z = rbf_func(new_x, new_y)

        # 5. 构建输出深度图
        result = np.zeros_like(old_depth)
        result[new_coords[:, 0], new_coords[:, 1]] = new_z

        return result

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

        # 用 tps_depth_warp 拟合 old_mask 区域深度到 new_mask 区域
        aligned = self.tps_depth_warp(old_depth_np, old_mask_np, new_mask_np)
        print("aligned min/max:", np.min(aligned), np.max(aligned))

        # Convert back to tensor: [1, H, W]
        result = torch.from_numpy(aligned).unsqueeze(0).to(dtype=new_depth.dtype)

        # 计算 final_mask
        # 1. 重叠区域（new_mask=1 且 old_mask=1）强制为1
        overlap = (old_mask_np > 0) & (new_mask_np > 0)
        # 2. 非重叠区域（new_mask=1 且 old_mask=0）根据深度比较
        non_overlap = (old_mask_np == 0) & (new_mask_np > 0)
        in_front = (old_depth_np > aligned)

        final_mask_np = np.zeros_like(aligned, dtype=np.float32)
        final_mask_np[overlap] = 1.0
        final_mask_np[non_overlap] = in_front[non_overlap].astype(np.float32)
        final_mask = torch.from_numpy(final_mask_np).unsqueeze(0)

        return (result, final_mask)
