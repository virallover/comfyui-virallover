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
        img_tensor = image[0]            # [3, H, W]
        mask_tensor = edge_mask[0, 0]    # [H, W]

        # Convert to numpy image
        img_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        img_bgr = (img_np * 255).clip(0, 255).astype(np.uint8)[..., ::-1]  # BGR

        mask_np = (mask_tensor.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255  # [H, W]

        # Confirm OpenCV input types
        assert img_bgr.dtype == np.uint8 and img_bgr.shape[2] == 3
        assert mask_np.dtype == np.uint8 and len(mask_np.shape) == 2

        # Choose method
        flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

        # Do inpaint
        inpainted_bgr = cv2.inpaint(img_bgr, mask_np, inpaintRadius, flags)

        # Convert back to tensor (RGB, float32)
        inpainted_rgb = inpainted_bgr[..., ::-1].astype(np.float32) / 255.0
        img_tensor_out = torch.from_numpy(inpainted_rgb.transpose(2, 0, 1)).unsqueeze(0).to(dtype=img_tensor.dtype)

        return (img_tensor_out, inpainted_bgr)

