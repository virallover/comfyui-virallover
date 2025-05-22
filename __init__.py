from .lora_downloader import DownloadAndLoadLoraModelOnly
from .image_utils.depth_fitter import DepthFitter
from .image_utils.poisson_fusion import PoissonImageFusion, DebugShape, TensorBatchToImage


# 注册到NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    # ... 其他节点 ...
    "DownloadAndLoadLoraModelOnly": DownloadAndLoadLoraModelOnly,
    "DepthFitter": DepthFitter,
    "PoissonImageFusion": PoissonImageFusion,
    "DebugShape": DebugShape,
    "TensorBatchToImage": TensorBatchToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadLoraModelOnly": "Download and Load Lora Model Only",
    "DepthFitter": "Depth Fitter",
    "PoissonImageFusion": "Poisson Image Fusion",
    "DebugShape": "Debug Shape",
    "TensorBatchToImage": "Tensor Batch to Image",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']