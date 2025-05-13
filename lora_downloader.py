import os
import requests
from urllib.parse import unquote
import comfy.sd
import comfy.utils

class DownloadAndLoadLoraModelOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_link": ("STRING", {}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "output": ("STRING", {}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "download_and_load_lora_model_only"
    CATEGORY = "loaders"

    def download_and_load_lora_model_only(self, model, lora_link, strength_model, output):
        lora_path = self.download_lora(lora_link, output)
        if lora_path:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
            return (model_lora,)
        else:
            print("Error loading Lora. Downloaded file not found.")
            return (model,)

    def download_lora(self, link, output):
        try:
            os.makedirs(output, exist_ok=True)
            # 获取文件名
            filename = "example.safetensors"
            response_head = requests.head(link, allow_redirects=True)
            content_disposition = response_head.headers.get('Content-Disposition')
            if content_disposition:
                filename = content_disposition.split('filename=')[1]
                filename = unquote(filename).strip('"')
            downloaded_file = os.path.join(output, filename)
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

