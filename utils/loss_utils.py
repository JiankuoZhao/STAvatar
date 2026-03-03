#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from math import exp
import torch.fft

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def compute_ssim_map(
    pred: torch.Tensor,  # shape: (C, H, W)
    gt: torch.Tensor,  # shape: (C, H, W)
    window_size=11,
    C1=0.01**2,
    C2=0.03**2,
) -> torch.Tensor:
    """
    Efficient SSIM map (1 - SSIM) computation for a single image (C x H x W).
    Returns per-pixel SSIM loss map of shape (H x W).
    """
    assert pred.shape == gt.shape and pred.ndim == 3, "Input must be (C, H, W)"
    C, H, W = pred.shape
    pad = window_size // 2

    # Create depthwise kernel
    kernel = torch.ones(
        (C, 1, window_size, window_size), dtype=pred.dtype, device=pred.device
    ) / (window_size**2)

    # Add batch dim: (1, C, H, W)
    pred = pred.unsqueeze(0)
    gt = gt.unsqueeze(0)

    def filter2d(x):
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu_x = filter2d(pred)
    mu_y = filter2d(gt)

    sigma_x = filter2d(pred * pred) - mu_x * mu_x
    sigma_y = filter2d(gt * gt) - mu_y * mu_y
    sigma_xy = filter2d(pred * gt) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = ssim_n / (ssim_d + 1e-8)  # numerical stability

    # Per-pixel SSIM loss: shape (1, C, H, W)
    ssim_loss_map = (1 - ssim_map) / 2

    # Average over channels → shape (H, W)
    return ssim_loss_map.mean(dim=1).squeeze(0)


def compute_per_gs_error(
    error_map: torch.Tensor,  # (H, W), float32
    gs_centers: torch.Tensor,  # (N, 2), long, screen-space pixel coords (x, y)
    accum_alpha: torch.Tensor,  # (N,), float32
    pixels_per_gs: torch.Tensor,  # (N,), int
):
    """
    Compute per-GS aggregated image error and edge error in one pass.
    Returns two tensors of shape (N,):
        img_errors: aggregated from error_map
    """

    device = error_map.device
    H, W = error_map.shape
    N = gs_centers.shape[0]
    gs_centers = gs_centers.int()

    # 1. Build integral images for both maps (1-pixel padded)
    integral_err = torch.zeros((H + 1, W + 1), dtype=error_map.dtype, device=device)
    integral_err[1:, 1:] = error_map.cumsum(dim=0).cumsum(dim=1)

    # 2. Prepare center coords and radii
    xs = gs_centers[:, 0].clamp(0, W - 1)
    ys = gs_centers[:, 1].clamp(0, H - 1)
    radii = torch.floor(pixels_per_gs.sqrt()).to(torch.int64) / 2

    x1 = (xs - radii).clamp(0, W - 1)
    x2 = (xs + radii).clamp(0, W - 1)
    y1 = (ys - radii).clamp(0, H - 1)
    y2 = (ys + radii).clamp(0, H - 1)

    # shift for integral indexing
    xa, xb = x1 + 1, x2 + 1
    ya, yb = y1 + 1, y2 + 1

    # 3. Allocate outputs
    img_errors = torch.empty(N, dtype=error_map.dtype, device=device)

    # 4. Compute all GS errors via integral image
    idx = torch.arange(N, device=device)

    # image error sum
    A1 = integral_err[yb[idx].long(), xb[idx].long()]
    B1 = integral_err[ya[idx].long() - 1, xb[idx].long()]
    C1 = integral_err[yb[idx].long(), xa[idx].long() - 1]
    D1 = integral_err[ya[idx].long() - 1, xa[idx].long() - 1]
    sum_err = A1 - B1 - C1 + D1

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    img_errors[idx] = sum_err / area.clamp(min=1)

    # 5. Modulate by alpha and pixel count
    factor = accum_alpha / (pixels_per_gs.to(error_map.dtype) + 1)
    img_errors *= factor

    return img_errors
