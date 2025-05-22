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

            # 兼容 [C, H, W] 或 [H, W, C]
            if fg.ndim == 3 and fg.shape[0] == 3:
                fg_np = np.transpose(fg, (1, 2, 0))
            elif fg.ndim == 3 and fg.shape[2] == 3:
                fg_np = fg
            else:
                raise ValueError(f"[PoissonImageFusion] fg shape不支持: {fg.shape}")

            if bg.ndim == 3 and bg.shape[0] == 3:
                bg_np = np.transpose(bg, (1, 2, 0))
            elif bg.ndim == 3 and bg.shape[2] == 3:
                bg_np = bg
            else:
                raise ValueError(f"[PoissonImageFusion] bg shape不支持: {bg.shape}")

            # mask 兼容 [1, H, W]、[H, W, 1]、[H, W]
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask_np = mask[0]
            elif mask.ndim == 3 and mask.shape[2] == 1:
                mask_np = mask[:, :, 0]
            elif mask.ndim == 2:
                mask_np = mask
            else:
                raise ValueError(f"[PoissonImageFusion] mask shape不支持: {mask.shape}")

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

            # shape 检查
            if fg_np.shape != bg_np.shape or fg_np.shape != mask_np.shape:
                raise ValueError(f"[PoissonImageFusion] shape mismatch: fg_np {fg_np.shape}, bg_np {bg_np.shape}, mask_np {mask_np.shape}")

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
            print(f"[DEBUG] Output blended_tensor shape: {blended_tensor.shape}")

            # 强制修正为 [1, 3, H, W]
            if blended_tensor.ndim == 4 and blended_tensor.shape[1] == 3:
                return (blended_tensor,)
            elif blended_tensor.ndim == 3 and blended_tensor.shape[0] == 3:
                blended_tensor = blended_tensor.unsqueeze(0)
                return (blended_tensor,)
            else:
                raise RuntimeError(f"泊松融合输出 shape 异常: {blended_tensor.shape}")

        except Exception as e:
            print(f"[ERROR] PoissonImageFusion 失败: {str(e)}")
            raise RuntimeError(f"泊松融合失败: {str(e)}")
