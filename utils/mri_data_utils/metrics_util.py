from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np


def compute_mse(gt, pred):
    """
    Compute MSE
    """
    return np.mean((gt - pred) ** 2)


def compute_psnr(gt, pred):
    """
    Compute PSNR.
    """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def compute_nmse(gt, pred):
    """
    Compute Normalized Mean Squared Error (NMSE)
    """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def compute_ssim(gt, pred):
    """
    Compute Structural Similarity (SSIM), gt and pred are with shape of (C, H, W).
    """
    return structural_similarity(
        gt, pred,
        data_range=gt.max(),
        channel_axis=0
    )


METRICS = ["PSNR", "SSIM", "NMSE"]
METRICS_FUNC = {
    "PSNR": compute_psnr,
    "SSIM": compute_ssim,
    "NMSE": compute_nmse,
}
