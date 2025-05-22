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

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("blended_tensor",)
    FUNCTION = "fuse"
    CATEGORY = "custom"

    @staticmethod
    def _to_single_mask(mask):
        mask_np = mask.cpu().numpy()
        if mask_np.ndim == 3:
            mask_np = mask_np[0] if mask_np.shape[0] == 1 else np.any(mask_np > 0, axis=0).astype(np.uint8)
        elif mask_np.ndim != 2:
            raise ValueError(f"Unsupported mask shape: {mask_np.shape}")
        return mask_np

    def fuse(self, new_image, blurred_mask, background_image):
        fg = new_image[0].cpu().numpy()
        bg = background_image[0].cpu().numpy()
        mask_np = self._to_single_mask(blurred_mask[0])

        # 兼容 [3, H, W] 和 [H, W, 3]
        if fg.ndim == 3 and fg.shape[0] == 3:
            fg_np = np.transpose(fg, (1, 2, 0))
        elif fg.ndim == 3 and fg.shape[2] == 3:
            fg_np = fg
        else:
            raise ValueError(f"Invalid fg shape: {fg.shape}")

        if bg.ndim == 3 and bg.shape[0] == 3:
            bg_np = np.transpose(bg, (1, 2, 0))
        elif bg.ndim == 3 and bg.shape[2] == 3:
            bg_np = bg
        else:
            raise ValueError(f"Invalid bg shape: {bg.shape}")

        fg_np = (np.clip(fg_np, 0, 1) * 255).astype(np.uint8)
        bg_np = (np.clip(bg_np, 0, 1) * 255).astype(np.uint8)
        mask_np = (np.clip(mask_np, 0, 1) * 255).astype(np.uint8)

        h, w = min(fg_np.shape[0], bg_np.shape[0], mask_np.shape[0]), min(fg_np.shape[1], bg_np.shape[1], mask_np.shape[1])
        fg_np, bg_np, mask_np = fg_np[:h, :w], bg_np[:h, :w], mask_np[:h, :w]

        ys, xs = np.where(mask_np > 10)
        center = (w // 2, h // 2) if len(xs) == 0 else (int(xs.mean()), int(ys.mean()))
        center = (min(center[0], w - 1), min(center[1], h - 1))

        blended = cv2.seamlessClone(fg_np, bg_np, mask_np, center, cv2.NORMAL_CLONE)
        if blended.ndim != 3 or blended.shape[2] != 3:
            blended = cv2.cvtColor(blended, cv2.COLOR_GRAY2RGB)

        out_tensor = torch.from_numpy(blended).float() / 255.0
        out_tensor = out_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        return (out_tensor,)


class DebugShape:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "debug"
    CATEGORY = "debug"

    def debug(self, image):
        print(f"[DEBUG] DebugShape input shape: {getattr(image, 'shape', type(image))}, dtype: {getattr(image, 'dtype', None)}")
        if hasattr(image, 'min') and hasattr(image, 'max'):
            print(f"[DEBUG] DebugShape min: {image.min()}, max: {image.max()}")
        if hasattr(image, 'shape') and (image.shape[1] != 3 or image.ndim != 4):
            print(f"[ERROR] DebugShape: 非 RGB 4D 图像，实际 shape: {image.shape}")
        return (image,)

class TensorBatchToImage:
    """
    将一个 Tensor 批次 [B, C, H, W] 中的某一张图像提取出来，返回单张 ComfyUI IMAGE。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_batch": ("TENSOR",),
                "batch_image_number": ("INT", {"default": 0, "min": 0, "max": 9999})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "extract"

    CATEGORY = "Custom/Tensor"

    def extract(self, images_batch: torch.Tensor, batch_image_number: int):
        if not isinstance(images_batch, torch.Tensor):
            raise TypeError("images_batch must be a torch.Tensor")

        batch_size = images_batch.shape[0]
        index = min(max(batch_image_number, 0), batch_size - 1)  # Clamp index

        single_tensor = images_batch[index].unsqueeze(0)  # 保持 [1, C, H, W] 维度
        # 自动归一化到 0~1
        if single_tensor.dtype == torch.uint8:
            single_tensor = single_tensor.float() / 255.0
        else:
            single_tensor = single_tensor.clamp(0, 1)
        return (single_tensor,)