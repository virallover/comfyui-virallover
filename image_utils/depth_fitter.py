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
    def _to_single_mask(mask_tensor):
        mask_np = mask_tensor.cpu().numpy()
        # 处理 [1, 1, H, W]
        if mask_np.ndim == 4 and mask_np.shape[0] == 1 and mask_np.shape[1] == 1:
            return (mask_np[0, 0] > 0.5).astype(np.uint8)
        # 处理 [1, H, W]
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            return (mask_np[0] > 0.5).astype(np.uint8)
        # 处理 [H, W]
        if mask_np.ndim == 2:
            return (mask_np > 0.5).astype(np.uint8)
        # 处理 [C, H, W]
        if mask_np.ndim == 3:
            return (np.any(mask_np > 0.5, axis=0)).astype(np.uint8)
        raise ValueError(f"Unsupported mask shape: {mask_np.shape}")

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
    def uniform_sample_coords(coords, values, max_points, grid_size=32):
        """
        在图像空间内均匀采样点。
        coords: (N, 2) array of [y, x] positions
        values: (N,) array of depth values
        """
        h, w = coords[:, 0].max() + 1, coords[:, 1].max() + 1
        sampled_idx = []

        grid_h = h // grid_size + 1
        grid_w = w // grid_size + 1

        for i in range(grid_h):
            for j in range(grid_w):
                mask = (
                    (coords[:, 0] >= i * grid_size) & (coords[:, 0] < (i + 1) * grid_size) &
                    (coords[:, 1] >= j * grid_size) & (coords[:, 1] < (j + 1) * grid_size)
                )
                cell_indices = np.where(mask)[0]
                if len(cell_indices) > 0:
                    chosen = np.random.choice(cell_indices, size=1)
                    sampled_idx.append(chosen)

        sampled_idx = np.concatenate(sampled_idx)
        if sampled_idx.size > max_points:
            sampled_idx = np.random.choice(sampled_idx, size=max_points, replace=False)

        return coords[sampled_idx], values[sampled_idx]

    @staticmethod
    def tps_depth_warp(old_depth, old_mask, new_mask, max_points=1000):
        """
        使用TPS（Thin Plate Spline）将 old_mask 区域的深度结构挤压映射到 new_mask 区域。
        返回一张深度图，仅在 new_mask 区域有效，其余为0。
        max_points: 用于拟合的最大采样点数，防止内存爆炸
        """
        # 1. 提取 old_mask 区域坐标和值
        old_coords = np.argwhere(old_mask > 0)
        if old_coords.shape[0] < 3:
            return np.zeros_like(old_depth)
        old_z = old_depth[old_mask > 0]

        # 均匀降采样
        if old_coords.shape[0] > max_points:
            old_coords, old_z = DepthFitter.uniform_sample_coords(old_coords, old_z, max_points)

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
        in_front = (old_depth_np < aligned)

        final_mask_np = np.zeros_like(aligned, dtype=np.float32)
        final_mask_np[overlap] = 1.0
        final_mask_np[non_overlap] = in_front[non_overlap].astype(np.float32)
        final_mask = torch.from_numpy(final_mask_np).unsqueeze(0)

        return (result, final_mask)
