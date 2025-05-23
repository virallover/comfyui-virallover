from .lora_downloader import DownloadAndLoadLoraModelOnly
from .image_utils.depth_fitter import DepthFitter
from .image_utils.mask_utils import MaskSubtract
from .image_utils.cv2_inpaint_fixer import CV2InpaintEdgeFixer


# 注册到NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    # ... 其他节点 ...
    "DownloadAndLoadLoraModelOnly": DownloadAndLoadLoraModelOnly,
    "DepthFitter": DepthFitter,
    "MaskSubtract": MaskSubtract,
    "CV2InpaintEdgeFixer": CV2InpaintEdgeFixer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadLoraModelOnly": "Download and Load Lora Model Only",
    "DepthFitter": "Depth Fitter",
    "MaskSubtract": "Mask Subtract",
    "CV2InpaintEdgeFixer": "CV2 Inpaint Edge Fixer",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']