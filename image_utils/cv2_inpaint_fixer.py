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

    def run(self, image, edge_mask, inpaintRadius, method):
        img_tensor = image[0]  # shape: [3, H, W]
        mask_tensor = edge_mask[0, 0]  # shape: [H, W]

        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        mask_np = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8) * 255  # [H, W]

        # Convert to uint8 and BGR
        img_bgr = (img_np * 255).astype(np.uint8)[..., ::-1]  # RGB → BGR

        flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        inpainted_bgr = cv2.inpaint(img_bgr, mask_np, inpaintRadius, flags=flags)  # BGR uint8

        # Convert back to RGB float32 for IMAGE
        inpainted_rgb = inpainted_bgr[..., ::-1].astype(np.float32) / 255.0  # BGR → RGB
        img_tensor_out = torch.from_numpy(inpainted_rgb.transpose(2, 0, 1)).unsqueeze(0).to(dtype=img_tensor.dtype)

        return (img_tensor_out, inpainted_bgr)  # 第二个输出为 BGR np.uint8
