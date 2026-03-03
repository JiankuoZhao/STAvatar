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

import torch
import sys
from datetime import datetime
import numpy as np
import random
import torch.nn.functional as F

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)
    seed = 58
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slerp(q1, q2, t):
    """Spherical Linear Interpolation between q1 and q2

    Args:
        q1, q2: [B, 4]
        t: float or [B, 1] interpolation factor
    """
    cos_theta = torch.sum(q1 * q2, dim=-1, keepdim=True)  # [B, 1]
    
    do_linear = cos_theta.abs() > 0.9995

    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=q1.dtype, device=q1.device)
    if t.dim() == 0:
        t = t.expand_as(cos_theta)  # [B, 1]
    elif t.dim() == 1:
        t = t.unsqueeze(-1)  # [B, 1]

    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
    sin_theta = torch.sin(theta)

    w1 = torch.sin((1.0 - t) * theta) / sin_theta
    w2 = torch.sin(t * theta) / sin_theta

    q_interp = w1 * q1 + w2 * q2

    # Fall back to linear if angle is very small
    do_linear = do_linear.squeeze(-1)
    q_interp[do_linear] = (
        (1.0 - t[do_linear]) * q1[do_linear] + t[do_linear] * q2[do_linear]
    )

    return F.normalize(q_interp, dim=-1)


def quatProduct_batch(q1, q2):
    """
    Compute Hamilton product of two quaternions batch-wise.
    Apply q2 first, then apply q1 !!!
    Input:
        q1: [B, 4], format [r, x, y, z]
        q2: [B, 4], format [r, x, y, z]
    Output:
        q:  [B, 4], normalized quaternion result
    """
    assert q1.shape[1] == 4 and q2.shape[1] == 4, "Quaternions must be shape [B, 4]"

    r1, r2 = q1[:, 0], q2[:, 0]
    v1, v2 = q1[:, 1:4], q2[:, 1:4]

    r = r1 * r2 - torch.sum(v1 * v2, dim=1)
    v = r1.unsqueeze(1) * v2 + r2.unsqueeze(1) * v1 + torch.cross(v1, v2, dim=1)

    q = torch.stack((r, v[:, 0], v[:, 1], v[:, 2]), dim=1)
    
    return F.normalize(q, dim=1)  # Ensure output is a unit quaternion
