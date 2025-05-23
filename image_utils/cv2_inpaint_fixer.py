import torch
import numpy as np
import cv2

class CV2InpaintEdgeFixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),          # 输入图像 [1, 3, H, W]
                "edge_mask": ("MASK",),       # 修复区域 mask [1, 1, H, W]
                "inpaintRadius": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 50.0}),
                "method": (["telea", "ns"],),  # 选择算法
            }
        }

    RETURN_TYPES = ("NPARRAY",)
    RETURN_NAMES = ("fixed_nparray",)
    FUNCTION = "run"
    CATEGORY = "inpainting"

    def run(self, image, edge_mask, inpaintRadius, method):
        # 兼容 tensor 和 nparray
        if isinstance(image, torch.Tensor):
            img = image[0].cpu().numpy().transpose(1, 2, 0)
        elif isinstance(image, np.ndarray):
            if image.ndim == 4 and image.shape[1] == 3:
                img = np.transpose(image[0], (1, 2, 0))
            elif image.ndim == 3:
                img = image
            else:
                raise ValueError(f"image shape不支持: {image.shape}")
        else:
            raise TypeError(f"image类型不支持: {type(image)}")

        if isinstance(edge_mask, torch.Tensor):
            mask = (edge_mask[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
        elif isinstance(edge_mask, np.ndarray):
            if edge_mask.ndim == 4:
                mask = (edge_mask[0, 0] > 0.5).astype(np.uint8) * 255
            elif edge_mask.ndim == 3:
                mask = (edge_mask[0] > 0.5).astype(np.uint8) * 255
            elif edge_mask.ndim == 2:
                mask = (edge_mask > 0.5).astype(np.uint8) * 255
            else:
                raise ValueError(f"edge_mask shape不支持: {edge_mask.shape}")
        else:
            raise TypeError(f"edge_mask类型不支持: {type(edge_mask)}")

        img8 = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img

        # Select method
        if method == "telea":
            flags = cv2.INPAINT_TELEA
        else:
            flags = cv2.INPAINT_NS

        # Inpaint
        inpainted = cv2.inpaint(img8, mask, inpaintRadius, flags=flags)

        # 直接返回 nparray 格式 [H, W, 3]，uint8
        return (inpainted,)
