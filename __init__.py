from .lora_downloader import DownloadAndLoadLoraModelOnly
from .image_utils.depth_fitter import DepthFitter


# 注册到NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    # ... 其他节点 ...
    "DownloadAndLoadLoraModelOnly": DownloadAndLoadLoraModelOnly,
    "DepthFitter": DepthFitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadLoraModelOnly": "Download and Load Lora Model Only",
    "DepthFitter": "Depth Fitter",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']