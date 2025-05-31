import torch
import numpy as np
# from PIL import Image  # 不再需要PIL
from typing import Optional, Tuple

# 复用edge_noise.py的格式转换函数

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

def _ensure_mask_chw(mask, target_shape, device):
    arr = mask.cpu().numpy()
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr.transpose(0, 3, 1, 2)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[None, ...]
    elif arr.ndim == 2:
        arr = arr[None, None, ...]
    mask_tensor = torch.from_numpy(arr).to(device).float()
    if mask_tensor.shape[1] == 1 and target_shape[1] == 3:
        mask_tensor = mask_tensor.repeat(1, 3, 1, 1)
    return mask_tensor

class ConcatHorizontalWithMask:
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
        left_image_chw = _ensure_chw(left_image)
        right_image_chw = _ensure_chw(right_image)
        H = left_image_chw.shape[2]
        assert H == right_image_chw.shape[2], "左右图片高度不一致"
        left_width = left_image_chw.shape[3]
        right_width = right_image_chw.shape[3]
        output_width = left_width + right_width
        output_height = H
        slice_width = left_width

        # mask处理
        if left_mask is None:
            left_mask_chw = torch.zeros((1, 1, H, left_width), device=device)
        else:
            left_mask_chw = _ensure_mask_chw(left_mask, left_image_chw.shape, device)
        if right_mask is None:
            right_mask_chw = torch.ones((1, 1, H, right_width), device=device)
        else:
            right_mask_chw = _ensure_mask_chw(right_mask, right_image_chw.shape, device)

        # 横向拼接
        out_image = torch.cat([left_image_chw, right_image_chw], dim=3)
        out_mask = torch.cat([left_mask_chw, right_mask_chw], dim=3)

        # 输出格式还原
        if input_is_nhwc:
            out_image = out_image.permute(0, 2, 3, 1)  # [1, 3, H, W] -> [1, H, W, 3]
            out_mask = out_mask.permute(0, 2, 3, 1)    # [1, 1, H, W] -> [1, H, W, 1]
        return (out_image, out_mask, output_width, output_height, slice_width)
