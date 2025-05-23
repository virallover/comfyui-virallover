from .lora_downloader import DownloadAndLoadLoraModelOnly
from .image_utils.depth_fitter import DepthFitter
from .image_utils.brightness_correction import BrightnessCorrectionNode


# 注册到NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    # ... 其他节点 ...
    "DownloadAndLoadLoraModelOnly": DownloadAndLoadLoraModelOnly,
    "DepthFitter": DepthFitter,
    "BrightnessCorrectionNode": BrightnessCorrectionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadLoraModelOnly": "Download and Load Lora Model Only",
    "DepthFitter": "Depth Fitter",
    "BrightnessCorrectionNode": "Brightness Correction",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']