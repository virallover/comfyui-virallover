import torch
import numpy as np
# from PIL import Image  # 不再需要PIL
from typing import Optional, Tuple

# 复用edge_noise.py的格式转换函数

class ConcatHorizontalWithMask:
    @staticmethod
    def _ensure_chw(tensor):
        arr = tensor.cpu().numpy()
        if arr.ndim == 4 and arr.shape[-1] == 3:
            arr = arr.transpose(0, 3, 1, 2)
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr.transpose(2, 0, 1)[None, ...]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)[None, ...]
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=0)[None, ...]
        return torch.from_numpy(arr).to(tensor.device).float()

    @staticmethod
    def _ensure_mask_chw(mask, target_shape=None, device=None):
        # 支持输入[1,1,H,W]、[1,H,W]、[H,W]、[1,H,W,1]、[H,W,1]
        arr = mask.detach().cpu().numpy() if hasattr(mask, 'detach') else np.array(mask)
        # 归一化到0~1
        if arr.max() > 1.1:
            arr = arr / 255.0
        # squeeze到[H, W]
        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
            arr = arr[0, 0]
        elif arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[-1] == 1:
            arr = arr[0, ..., 0]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.ndim == 2:
            pass
        else:
            raise ValueError(f"Unsupported mask shape: {arr.shape}")
        # 再加回batch和channel维，变[1,1,H,W]
        arr = arr[None, None, ...]
        tensor = torch.from_numpy(arr).float()
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
            },
            "optional": {
                "left_mask": ("MASK",),
                "right_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "output_width", "output_height", "slice_width")
    FUNCTION = "concat"
    CATEGORY = "custom/image"

    def concat(
        self,
        left_image,
        right_image,
        left_mask: Optional[torch.Tensor] = None,
        right_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        # 记录输入格式
        input_is_nhwc = (left_image.ndim == 4 and left_image.shape[-1] == 3)
        device = left_image.device

        # 转为NCHW
        left_image_chw = self._ensure_chw(left_image)
        right_image_chw = self._ensure_chw(right_image)
        H = left_image_chw.shape[2]
        assert H == right_image_chw.shape[2], "左右图片高度不一致"
        left_width = left_image_chw.shape[3]
        right_width = right_image_chw.shape[3]
        output_width = left_width + right_width
        output_height = H
        slice_width = left_width

        # 横向拼接图片
        out_image = torch.cat([left_image_chw, right_image_chw], dim=3)

        # mask处理
        # 归一化和格式统一
        def norm_and_to_hw(mask, width):
            if mask is None:
                return None
            arr = mask.detach().cpu().numpy() if hasattr(mask, 'detach') else np.array(mask)
            if arr.max() > 1.1:
                arr = arr / 255.0
            # squeeze到[H, W]
            if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
                arr = arr[0, 0]
            elif arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[-1] == 1:
                arr = arr[0, ..., 0]
            elif arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            elif arr.ndim == 2:
                pass
            else:
                raise ValueError(f"Unsupported mask shape: {arr.shape}")
            # 检查宽度
            if arr.shape[1] != width:
                raise ValueError(f"mask宽度({arr.shape[1]})与图片宽度({width})不一致")
            return arr

        left_mask_hw = norm_and_to_hw(left_mask, left_width) if left_mask is not None else None
        right_mask_hw = norm_and_to_hw(right_mask, right_width) if right_mask is not None else None

        # 新建output_mask
        output_mask = np.zeros((H, output_width), dtype=np.float32)
        if left_mask_hw is not None:
            output_mask[:, :left_width] = left_mask_hw
        else:
            output_mask[:, :left_width] = 0.0
        if right_mask_hw is not None:
            output_mask[:, left_width:] = right_mask_hw
        else:
            output_mask[:, left_width:] = 1.0
        # 保证输出为[1, H, W]，float32
        out_mask = torch.from_numpy(output_mask).float().to(device)
        if out_mask.ndim == 2:
            out_mask = out_mask.unsqueeze(0)
        elif out_mask.ndim == 3 and out_mask.shape[0] != 1:
            out_mask = out_mask[:1]
        # 调试用shape打印
        print("out_mask shape:", out_mask.shape)

        # 输出图片格式转换 [1, 3, H, W] -> [1, H, W, 3]
        if out_image.ndim == 4 and out_image.shape[1] == 3:
            out_image = out_image.permute(0, 2, 3, 1).contiguous()

        # 检查最终mask的宽高和输出图片一致
        if out_mask.shape[1:] != out_image.shape[1:3]:
            raise ValueError(f"输出mask的宽高{out_mask.shape[1:]}与输出图片的宽高{out_image.shape[1:3]}不一致")

        return (out_image, out_mask, output_width, output_height, slice_width)
