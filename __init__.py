from .lora_downloader import DownloadAndLoadLoraModelOnly


# 注册到NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    # ... 其他节点 ...
    "DownloadAndLoadLoraModelOnly": DownloadAndLoadLoraModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadLoraModelOnly": "Download and Load Lora Model Only",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']