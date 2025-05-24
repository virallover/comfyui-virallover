import argparse
import numpy as np
import torch
from PIL import Image
import os

def load_image(path, to_tensor=True):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # [3, H, W]
    arr = arr[None, ...]  # [1, 3, H, W]
    if to_tensor:
        return torch.from_numpy(arr)
    return arr

def load_mask(path, to_tensor=True):
    img = Image.open(path)
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]  # 取第一个通道
    arr = arr[None, ...]  # [1, H, W]
    if arr.max() > 1.1:
        arr = arr / 255.0
    if to_tensor:
        return torch.from_numpy(arr)
    return arr

def save_image(tensor, path):
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    arr = arr.squeeze()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)  # [H, W, 3]
    img = Image.fromarray(arr)
    img.save(path)

def to_single_mask(mask_tensor, threshold=0.5):
    arr = mask_tensor.detach().cpu().float().numpy()
    if arr.ndim == 4 and arr.shape[-1] == 3:
        arr = arr.transpose(0, 3, 1, 2)
    if arr.ndim == 4 and arr.shape[1] == 3:
        arr = arr.mean(axis=1, keepdims=True)
    elif arr.ndim == 4 and arr.shape[1] == 1:
        pass
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.transpose(2, 0, 1)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr.transpose(2, 0, 1).mean(axis=0, keepdims=True)
    elif arr.ndim == 2:
        arr = arr[None, ...]
    if arr.max() > 1.1:
        arr = arr / 255.0
    mask = (arr > threshold)
    mask = mask.squeeze()
    return mask

def rgb_to_grayscale_torch(img):
    if img.ndim == 4:
        return 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
    elif img.ndim == 3:
        return 0.299 * img[0:1, :, :] + 0.587 * img[1:2, :, :] + 0.114 * img[2:3, :, :]
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

def ensure_chw(tensor):
    arr = tensor.cpu().numpy()
    if arr.ndim == 4 and arr.shape[-1] == 3:
        arr = arr.transpose(0, 3, 1, 2)
        return torch.from_numpy(arr).to(tensor.device)
    return tensor

def adjust_brightness(original_image, original_mask, target_image, target_mask):
    ori_img = ensure_chw(original_image.clone())
    tgt_img = ensure_chw(target_image.clone())
    ori_mask_img = ensure_chw(original_mask.clone())
    tgt_mask_img = ensure_chw(target_mask.clone())

    ori_mask = to_single_mask(ori_mask_img)
    tgt_mask = to_single_mask(tgt_mask_img)

    ori_gray = rgb_to_grayscale_torch(ori_img).cpu().numpy().squeeze()
    tgt_gray = rgb_to_grayscale_torch(tgt_img).cpu().numpy().squeeze()

    if ori_gray.shape != ori_mask.shape:
        raise ValueError(f"original_image灰度图与original_mask尺寸不一致: ori_gray shape={ori_gray.shape}, ori_mask shape={ori_mask.shape}")
    if tgt_gray.shape != tgt_mask.shape:
        raise ValueError(f"target_image灰度图与target_mask尺寸不一致: tgt_gray shape={tgt_gray.shape}, tgt_mask shape={tgt_mask.shape}")

    ori_pixels = ori_gray[ori_mask]
    tgt_pixels = tgt_gray[tgt_mask]

    if ori_pixels.size == 0 or tgt_pixels.size == 0:
        print("Mask 区域为空，跳过校正")
        corrected = tgt_img
    else:
        factor = float(ori_pixels.mean() / tgt_pixels.mean())
        corrected = tgt_img.clone()
        mask_tensor = torch.from_numpy(tgt_mask.astype(np.float32)).to(tgt_img.device)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        if mask_tensor.shape[1] == 1 and corrected.shape[1] == 3:
            mask_tensor = mask_tensor.repeat(1, 3, 1, 1)
        elif mask_tensor.shape[1] != corrected.shape[1]:
            mask_tensor = mask_tensor.expand(-1, corrected.shape[1], -1, -1)
        corrected = corrected * (1 - mask_tensor) + torch.clamp(corrected * factor, 0, 1) * mask_tensor
    if corrected.shape[1] == 1:
        corrected = corrected.repeat(1, 3, 1, 1)
    corrected = corrected.clamp(0, 1).to(torch.float32)
    assert corrected.shape[1] == 3, f"corrected shape error: {corrected.shape}"
    return corrected

def main():
    parser = argparse.ArgumentParser(description='亮度校正脚本')
    parser.add_argument('--original_image', required=True, help='原图路径')
    parser.add_argument('--original_mask', required=True, help='原mask路径')
    parser.add_argument('--target_image', required=True, help='目标图路径')
    parser.add_argument('--target_mask', required=True, help='目标mask路径')
    parser.add_argument('--output', required=True, help='输出校正后图片路径')
    args = parser.parse_args()

    ori_img = load_image(args.original_image)
    ori_mask = load_mask(args.original_mask)
    tgt_img = load_image(args.target_image)
    tgt_mask = load_mask(args.target_mask)

    corrected = adjust_brightness(ori_img, ori_mask, tgt_img, tgt_mask)
    save_image(corrected, args.output)
    print(f'已保存校正后图片到: {args.output}')

if __name__ == '__main__':
    main()
