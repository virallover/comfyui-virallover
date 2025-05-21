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

    def fit_depth(self, new_depth, old_depth, old_mask, new_mask):
        # Convert to numpy arrays: [H, W]
        new_depth_np = new_depth[0, 0].cpu().numpy()
        old_depth_np = old_depth[0, 0].cpu().numpy()
        old_mask_np = old_mask[0].cpu().numpy()
        new_mask_np = new_mask[0].cpu().numpy()

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
