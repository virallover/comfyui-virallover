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
        try:
            # 解包 tensor → numpy
            fg = new_image[0].cpu().numpy()
            bg = background_image[0].cpu().numpy()
            mask = blurred_mask[0].cpu().numpy()

            print(f"[DEBUG] Input fg shape: {fg.shape}")
            print(f"[DEBUG] Input bg shape: {bg.shape}")
            print(f"[DEBUG] Input mask shape: {mask.shape}")

            # 检查通道数
            if fg.shape[0] != 3 or bg.shape[0] != 3:
                raise ValueError(f"[PoissonImageFusion] 期望 RGB 图像，实际 fg/bg 通道为 {fg.shape[0]}, {bg.shape[0]}")

            # 转换为 [H, W, C]
            fg_np = np.transpose(fg, (1, 2, 0))
            bg_np = np.transpose(bg, (1, 2, 0))
            mask_np = mask.squeeze()  # (1, H, W) → (H, W)

            # 归一化
            fg_np = (np.clip(fg_np, 0, 1) * 255).astype(np.uint8)
            bg_np = (np.clip(bg_np, 0, 1) * 255).astype(np.uint8)
            mask_np = (np.clip(mask_np, 0, 1) * 255).astype(np.uint8)

            # 检查尺寸一致
            h = min(fg_np.shape[0], bg_np.shape[0], mask_np.shape[0])
            w = min(fg_np.shape[1], bg_np.shape[1], mask_np.shape[1])

            fg_np = fg_np[:h, :w, :]
            bg_np = bg_np[:h, :w, :]
            if mask_np.ndim == 2:
                mask_np = cv2.merge([mask_np[:h, :w]] * 3)
            else:
                mask_np = mask_np[:h, :w, :3]

            print(f"[DEBUG] Resized fg/bg/mask to: {fg_np.shape}")

            # 中心点定位
            ys, xs = np.where(mask_np[:, :, 0] > 10)
            if len(xs) == 0 or len(ys) == 0:
                print("[WARNING] Mask 非法，默认使用图像中心作为融合中心")
                center = (w // 2, h // 2)
            else:
                center = (int(xs.mean()), int(ys.mean()))

            print(f"[DEBUG] Blending center: {center}")

            # 泊松融合
            blended = cv2.seamlessClone(fg_np, bg_np, mask_np, center, cv2.NORMAL_CLONE)

            # 回转为 tensor
            blended_tensor = torch.from_numpy(blended.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            return (blended_tensor,)

        except Exception as e:
            print(f"[ERROR] PoissonImageFusion 失败: {str(e)}")
            raise RuntimeError(f"泊松融合失败: {str(e)}")
