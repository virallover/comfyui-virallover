from .lora_downloader import DownloadAndLoadLoraModelOnly
from .image_utils.depth_fitter import DepthFitter
from .image_utils.brightness_correction import BrightnessCorrectionNode
from .image_utils.edge_noise import EdgeNoise
from .image_utils.feathered_sharpen import FeatheredSharpen


# 注册到NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    # ... 其他节点 ...
    "DownloadAndLoadLoraModelOnly": DownloadAndLoadLoraModelOnly,
    "DepthFitter": DepthFitter,
    "BrightnessCorrectionNode": BrightnessCorrectionNode,
    "EdgeNoise": EdgeNoise,
    "FeatheredSharpen": FeatheredSharpen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadLoraModelOnly": "Download and Load Lora Model Only",
    "DepthFitter": "Depth Fitter",
    "BrightnessCorrectionNode": "Brightness Correction",
    "EdgeNoise": "Edge Noise",
    "FeatheredSharpen": "Feathered Sharpen",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']