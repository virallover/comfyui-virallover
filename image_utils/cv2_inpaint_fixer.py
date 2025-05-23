import torch
import numpy as np
import cv2

class CV2InpaintEdgeFixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),           # ComfyUI 图像 [1, 3, H, W]
                "edge_mask": ("MASK",),        # 修复区域 [1, 1, H, W]
                "inpaintRadius": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 50.0}),
                "method": (["telea", "ns"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "NPARRAY")
    RETURN_NAMES = ("fixed_image", "fixed_np_array")
    FUNCTION = "run"
    CATEGORY = "inpainting"

    def run(self, image, edge_mask, inpaintRadius=3, method="telea"):
        # 保证 image 是 [1, 3, H, W]
        if isinstance(image, torch.Tensor) and image.dim() == 4:
            image = image[0]
        elif isinstance(image, list) and isinstance(image[0], torch.Tensor):
            image = image[0]

        # edge_mask 是 [1, 1, H, W]
        if isinstance(edge_mask, torch.Tensor) and edge_mask.dim() == 4:
            edge_mask = edge_mask[0, 0]
        elif isinstance(edge_mask, list) and isinstance(edge_mask[0], torch.Tensor):
            edge_mask = edge_mask[0][0]

        # 转成 numpy 格式
        img_np = image.detach().cpu().numpy()  # shape: [3, H, W]
        mask_np = edge_mask.detach().cpu().numpy()  # shape: [H, W]

        # 转换成 BGR 格式图像
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        img_bgr = np.transpose(img_np, (1, 2, 0))[..., ::-1]  # HWC + RGB→BGR

        # 转换 mask
        mask_np = (mask_np > 0.5).astype(np.uint8) * 255  # 单通道

        # 检查形状是否正确
        assert img_bgr.dtype == np.uint8 and img_bgr.shape[2] == 3, f"img_bgr invalid: {img_bgr.shape}, {img_bgr.dtype}"
        assert mask_np.dtype == np.uint8 and len(mask_np.shape) == 2, f"mask_np invalid: {mask_np.shape}, {mask_np.dtype}"

        flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        inpainted = cv2.inpaint(img_bgr, mask_np, inpaintRadius, flags)

        # 返回修复后的 RGB 图像（torch 格式）
        inpainted_rgb = inpainted[..., ::-1].astype(np.float32) / 255.0  # BGR → RGB
        out_tensor = torch.from_numpy(inpainted_rgb.transpose(2, 0, 1)).unsqueeze(0)  # [1, 3, H, W]

        return (out_tensor,)


