import numpy as np
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def tensor_to_image(t):
    """
    Convert a torch tensor to a NumPy uint8 image.

    Args:
        t: Input tensor, expected shapes: [1, C, H, W], [C, H, W], [1, H, W], or [H, W].

    Returns:
        np.ndarray: Image in [H, W, C] format with uint8 dtype.
    """
    if not isinstance(t, torch.Tensor):
        return t
    if t.ndim == 4 and t.shape[0] == 1:  # [1, C, H, W]
        t = t.squeeze(0)  # -> [C, H, W]
    elif t.ndim == 4 and t.shape[1] == 1:  # [1, 1, H, W]
        t = t.squeeze(0).repeat(3, 1, 1)  # -> [3, H, W]
    elif t.ndim == 3 and t.shape[0] == 1:  # [1, H, W]
        t = t.repeat(3, 1, 1)  # -> [3, H, W]
    elif t.ndim == 2:  # [H, W]
        t = t.unsqueeze(0).repeat(3, 1, 1)  # -> [3, H, W]
    elif t.ndim == 3 and t.shape[0] == 3:  # [3, H, W]
        pass  # Already in correct format
    else:
        raise ValueError(f"Unsupported tensor shape: {t.shape}")
    img = t.permute(1, 2, 0).detach().cpu().numpy()  # -> [H, W, C]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

class BrightnessCorrectionNode:
    """
    A node for adjusting the brightness of a target image to match an original image
    based on masked regions.

    Inputs:
        original_image: Tensor of shape [1, 3, H, W], [H, W, 3], or [3, H, W], float32 in [0, 1].
        original_mask: Tensor of shape [1, 1, H, W], [1, H, W], [H, W], or [1, 3, H, W].
        target_image: Tensor of shape [1, 3, H, W], [H, W, 3], or [3, H, W], float32 in [0, 1].
        target_mask: Tensor of shape [1, 1, H, W], [1, H, W], [H, W], or [1, 3, H, W].

    Output:
        corrected_target_image: Tensor of shape [1, 3, H, W], float32 in [0, 1].
    """
    
    @staticmethod
    def _ensure_chw(tensor):
        """
        Ensure tensor is in [1, 3, H, W] format.

        Args:
            tensor: Input tensor, possibly [1, H, W, 3], [H, W, 3], or [3, H, W].

        Returns:
            torch.Tensor: Tensor in [1, 3, H, W] format.
        """
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        if tensor.ndim == 4 and tensor.shape[-1] == 3:  # [1, H, W, 3]
            return tensor.permute(0, 3, 1, 2)  # -> [1, 3, H, W]
        elif tensor.ndim == 3 and tensor.shape[-1] == 3:  # [H, W, 3]
            return tensor.permute(2, 0, 1).unsqueeze(0)  # -> [1, 3, H, W]
        elif tensor.ndim == 3 and tensor.shape[0] == 3:  # [3, H, W]
            return tensor.unsqueeze(0)  # -> [1, 3, H, W]
        elif tensor.ndim == 4 and tensor.shape[1] == 3:  # [1, 3, H, W]
            return tensor
        raise ValueError(f"Unsupported image tensor shape: {tensor.shape}")

    @staticmethod
    def _ensure_mask_single_channel(tensor):
        """
        Ensure mask tensor is in [1, 1, H, W] format.

        Args:
            tensor: Input tensor, possibly [1, H, W, 3], [1, 3, H, W], [1, 1, H, W], [1, H, W], or [H, W].

        Returns:
            torch.Tensor: Tensor in [1, 1, H, W] format.
        """
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        if tensor.ndim == 4 and tensor.shape[1] == 1:  # [1, 1, H, W]
            return tensor
        if tensor.ndim == 3 and tensor.shape[0] == 1:  # [1, H, W]
            return tensor.unsqueeze(1)  # -> [1, 1, H, W]
        if tensor.ndim == 2:  # [H, W]
            return tensor.unsqueeze(0).unsqueeze(0)  # -> [1, 1, H, W]
        if tensor.ndim == 4 and tensor.shape[-1] == 3:  # [1, H, W, 3]
            return tensor[..., 0].unsqueeze(1)  # -> [1, 1, H, W]
        if tensor.ndim == 4 and tensor.shape[1] == 3:  # [1, 3, H, W]
            return tensor[:, 0:1, :, :]  # -> [1, 1, H, W]
        raise ValueError(f"Unsupported mask tensor shape: {tensor.shape}")

    @staticmethod
    def _to_single_mask(mask_tensor):
        """
        Convert mask tensor to a binary [H, W] NumPy array.

        Args:
            mask_tensor: Tensor in [1, 1, H, W], [1, H, W], [H, W], or [1, 3, H, W].

        Returns:
            np.ndarray: Binary mask in [H, W] format, float32.
        """
        mask_np = mask_tensor.cpu().numpy()
        if mask_np.max() > 1.0:
            mask_np = mask_np / mask_np.max()  # Normalize to [0, 1]
        if mask_np.ndim == 4 and mask_np.shape[0] == 1 and mask_np.shape[1] == 1:  # [1, 1, H, W]
            return (mask_np[0, 0] > 0.5).astype(np.float32)
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:  # [1, H, W]
            return (mask_np[0] > 0.5).astype(np.float32)
        if mask_np.ndim == 2:  # [H, W]
            return (mask_np > 0.5).astype(np.float32)
        if mask_np.ndim == 3:  # [C, H, W]
            return (np.any(mask_np > 0.5, axis=0)).astype(np.float32)
        raise ValueError(f"Unsupported mask shape: {mask_np.shape}")

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define input types for the node.

        Returns:
            dict: Input specifications.
        """
        return {
            "required": {
                "original_image": ("IMAGE",),
                "original_mask": ("IMAGE",),
                "target_image": ("IMAGE",),
                "target_mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("corrected_target_image",)
    FUNCTION = "adjust_brightness"
    CATEGORY = "Image Processing"

    def adjust_brightness(self, original_image, original_mask, target_image, target_mask):
        """
        Adjust the brightness of the target image to match the original image in masked regions.

        Args:
            original_image: Tensor of shape [1, 3, H, W], [H, W, 3], or [3, H, W], float32 in [0, 1].
            original_mask: Mask tensor specifying region of interest in original_image.
            target_image: Tensor of shape [1, 3, H, W], [H, W, 3], or [3, H, W], float32 in [0, 1].
            target_mask: Mask tensor specifying region to adjust in target_image.

        Returns:
            tuple: (corrected_target_image,) where corrected_target_image is a tensor in [1, 3, H, W].
        """
        # Ensure correct shapes and dtype
        original_image = self._ensure_chw(original_image)
        target_image = self._ensure_chw(target_image)
        original_mask = self._ensure_mask_single_channel(original_mask)
        target_mask = self._ensure_mask_single_channel(target_mask)

        # Validate shapes
        for name, img in [("original_image", original_image), ("target_image", target_image)]:
            if img.ndim != 4 or img.shape[1] != 3:
                raise ValueError(f"{name} must be [1, 3, H, W], got {img.shape}")
        for name, mask in [("original_mask", original_mask), ("target_mask", target_mask)]:
            if mask.ndim != 4 or mask.shape[1] != 1:
                raise ValueError(f"{name} must be [1, 1, H, W], got {mask.shape}")

        # Convert to NumPy for brightness adjustment
        ori = original_image[0].cpu().numpy()  # [3, H, W]
        tgt = target_image[0].cpu().numpy()    # [3, H, W]
        ori_mask = self._to_single_mask(original_mask)  # [H, W]
        tgt_mask = self._to_single_mask(target_mask)    # [H, W]

        # Compute grayscale images
        ori_gray = 0.299 * ori[0] + 0.587 * ori[1] + 0.114 * ori[2]
        tgt_gray = 0.299 * tgt[0] + 0.587 * tgt[1] + 0.114 * tgt[2]

        # Compute mean brightness in masked regions
        ori_pixels = ori_gray[ori_mask == 1]
        tgt_pixels = tgt_gray[tgt_mask == 1]
        if ori_pixels.size == 0 or tgt_pixels.size == 0:
            logging.warning("One or both masks are empty. Returning target image unchanged.")
            return (target_image,)

        mean_ori = np.mean(ori_pixels)
        mean_tgt = np.mean(tgt_pixels)
        factor = mean_ori / mean_tgt if mean_tgt > 0 else 1.0

        # Adjust brightness in target mask region
        corrected = tgt.copy()
        for c in range(3):
            channel = corrected[c]
            channel[tgt_mask == 1] = np.clip(channel[tgt_mask == 1] * factor, 0, 1)
            corrected[c] = channel

        # Convert back to tensor
        corrected = torch.from_numpy(corrected).unsqueeze(0).float()  # -> [1, 3, H, W]
        corrected = corrected.clamp(0, 1)

        # Log shape and dtype for debugging
        logging.debug(f"corrected shape: {corrected.shape}, dtype: {corrected.dtype}")

        return (corrected,)