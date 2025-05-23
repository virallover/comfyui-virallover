import torch

class MaskSubtract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("diff_mask",)
    FUNCTION = "subtract"
    CATEGORY = "mask"

    def subtract(self, mask_a, mask_b):
        a = mask_a[0, 0]
        b = mask_b[0, 0]

        diff = ((a > 0.5) & (b < 0.5)).float()  # A中有而B中没有
        result = diff.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return (result,)
