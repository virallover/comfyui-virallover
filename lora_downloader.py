import os
import requests
from urllib.parse import unquote
import comfy.sd
import comfy.utils
from folder_paths import get_folder_paths

class DownloadAndLoadLoraModelOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_link": ("STRING", {}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "file_name": ("STRING", {}),  # 现在file_name只作为文件名
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "download_and_load_lora_model_only"
    CATEGORY = "loaders"

    def download_and_load_lora_model_only(self, model, lora_link, strength_model, file_name):
        lora_path = self.download_lora(lora_link, file_name)
        if lora_path:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
            return (model_lora,)
        else:
            print("Error loading Lora. Downloaded file not found.")
            return (model,)

    def download_lora(self, link, file_name):
        try:
            # 自动检测loras目录
            loras_dirs = get_folder_paths("loras")
            if not loras_dirs:
                raise Exception("No loras directory found!")
            loras_dir = loras_dirs[0]
            os.makedirs(loras_dir, exist_ok=True)
            # file_name 现在是文件名
            downloaded_file = os.path.join(loras_dir, file_name)
            temp_file = downloaded_file + ".part"

            # 获取已下载部分大小
            pos = 0
            if os.path.exists(temp_file):
                pos = os.path.getsize(temp_file)

            headers = {}
            if pos > 0:
                headers['Range'] = f'bytes={pos}-'

            with requests.get(link, headers=headers, stream=True) as r:
                if r.status_code in (200, 206):
                    mode = 'ab' if pos else 'wb'
                    with open(temp_file, mode) as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    # 下载完成后重命名
                    total_size = int(r.headers.get('Content-Range', '').split('/')[-1] or r.headers.get('Content-Length', 0))
                    if os.path.getsize(temp_file) == total_size or not total_size:
                        os.rename(temp_file, downloaded_file)
                        return downloaded_file
                    else:
                        print("Download incomplete, please retry.")
                        return None
                else:
                    print(f"Error downloading Lora file: HTTP status code {r.status_code}")
                    return None
        except Exception as e:
            print(f"Error downloading Lora file: {e}")
            return None

