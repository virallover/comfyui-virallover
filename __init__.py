from .lora_downloader import DownloadAndLoadLoraModelOnly
from .image_utils.depth_fitter import DepthFitter
from .image_utils.brightness_correction import BrightnessCorrectionNode
from .image_utils.edge_noise import EdgeNoise
from .image_utils.feathered_sharpen import FeatheredSharpen
from .image_utils.apply_brightness_with_mask import ApplyBrightnessFromGrayWithMask


# 注册到NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    # ... 其他节点 ...
    "DownloadAndLoadLoraModelOnly": DownloadAndLoadLoraModelOnly,
    "DepthFitter": DepthFitter,
    "BrightnessCorrectionNode": BrightnessCorrectionNode,
    "EdgeNoise": EdgeNoise,
    "FeatheredSharpen": FeatheredSharpen,
    "ApplyBrightnessFromGrayWithMask": ApplyBrightnessFromGrayWithMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadLoraModelOnly": "Download and Load Lora Model Only",
    "DepthFitter": "Depth Fitter",
    "BrightnessCorrectionNode": "Brightness Correction",
    "EdgeNoise": "Edge Noise",
    "FeatheredSharpen": "Feathered Sharpen",
    "ApplyBrightnessFromGrayWithMask": "Apply Brightness From Gray With Mask",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']