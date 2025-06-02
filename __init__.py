from .lora_downloader import DownloadAndLoadLoraModelOnly
from .image_utils.depth_fitter import DepthFitter
from .image_utils.brightness_correction import BrightnessCorrectionNode
from .image_utils.edge_noise import EdgeNoise
from .image_utils.feathered_sharpen import FeatheredSharpen
from .image_utils.concat_image_horizontal import ConcatHorizontalWithMask
from .image_utils.dehalo_alpha_with_mask import DeHaloAlphaWithMaskTorch


# 注册到NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    # ... 其他节点 ...
    "DownloadAndLoadLoraModelOnly": DownloadAndLoadLoraModelOnly,
    "DepthFitter": DepthFitter,
    "BrightnessCorrectionNode": BrightnessCorrectionNode,
    "EdgeNoise": EdgeNoise,
    "FeatheredSharpen": FeatheredSharpen,
    "ConcatHorizontalWithMask": ConcatHorizontalWithMask,
    "DeHaloAlphaWithMaskTorch": DeHaloAlphaWithMaskTorch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadLoraModelOnly": "Download and Load Lora Model Only",
    "DepthFitter": "Depth Fitter",
    "BrightnessCorrectionNode": "Brightness Correction",
    "EdgeNoise": "Edge Noise",
    "FeatheredSharpen": "Feathered Sharpen",
    "ConcatHorizontalWithMask": "Concat Horizontal With Mask",
    "DeHaloAlphaWithMaskTorch": "DeHalo Alpha With Mask",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']