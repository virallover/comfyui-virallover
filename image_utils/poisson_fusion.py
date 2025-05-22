import numpy as np
import torch
import cv2

class PoissonImageFusion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "new_image": ("IMAGE",),
                "blurred_mask": ("MASK",),
                "background_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    FUNCTION = "fuse"
    CATEGORY = "custom"

    def fuse(self, new_image, blurred_mask, background_image):
        # [1, C, H, W] → [H, W, C]
        fg_np = new_image[0].cpu().numpy().transpose(1, 2, 0)
        bg_np = background_image[0].cpu().numpy().transpose(1, 2, 0)
        mask_np = blurred_mask[0].cpu().numpy()

        # 类型转换到 uint8，归一化到 0~255
        fg_np = (np.clip(fg_np, 0, 1) * 255).astype(np.uint8)
        bg_np = (np.clip(bg_np, 0, 1) * 255).astype(np.uint8)
        mask_np = (np.clip(mask_np, 0, 1) * 255).astype(np.uint8)

        # 保证 mask 是单通道 [H, W]
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        if mask_np.ndim == 3 and mask_np.shape[2] == 1:
            mask_np = mask_np[:, :, 0]
        if mask_np.ndim == 2:
            mask_np = mask_np  # [H, W]
        # mask 必须和 fg/bg shape 一致，扩展为三通道
        if mask_np.ndim == 2:
            mask_np = cv2.merge([mask_np]*3)
        # 再次确保 shape 完全一致
        mask_np = mask_np[:fg_np.shape[0], :fg_np.shape[1], :3]
        fg_np = fg_np[:bg_np.shape[0], :bg_np.shape[1], :3]
        bg_np = bg_np[:fg_np.shape[0], :fg_np.shape[1], :3]

        # shape 检查
        if fg_np.shape != bg_np.shape or fg_np.shape != mask_np.shape:
            raise ValueError(f"[PoissonImageFusion] shape mismatch: fg_np {fg_np.shape}, bg_np {bg_np.shape}, mask_np {mask_np.shape}")

        # 定位融合中心
        ys, xs = np.where(mask_np[:, :, 0] > 10)
        if len(xs) == 0 or len(ys) == 0:
            center = (bg_np.shape[1] // 2, bg_np.shape[0] // 2)
        else:
            center = (int(xs.mean()), int(ys.mean()))

        # 执行泊松融合
        blended = cv2.seamlessClone(fg_np, bg_np, mask_np, center, cv2.NORMAL_CLONE)

        # [H, W, C] → [1, C, H, W]
        blended_tensor = torch.from_numpy(blended.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return (blended_tensor,) 