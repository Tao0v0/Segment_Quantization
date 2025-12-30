import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


def bilinear_sampler(img, coords, mode='bilinear', mask=False):     # 输入corr(B*H*W, 1, H_I, W_I) 和 coords (B*H*W, K, K, 2) 
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    # Use slicing instead of `split` for better ONNX export stability.
    xgrid = coords[..., 0:1]    #shape: (B*H*W, K, K, 1)    
    ygrid = coords[..., 1:2]
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1   # 归一化

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img  #shape: (B*H*W, 1, K, K)


def coords_grid(batch, ht, wd):
    # ONNX-friendly grid generation without `meshgrid`, keeping tensors 4D.
    x = torch.arange(wd).view(1, 1, 1, wd).expand(batch, 1, ht, wd)
    y = torch.arange(ht).view(1, 1, ht, 1).expand(batch, 1, ht, wd)
    return torch.cat([x, y], dim=1).float()


def upflowX(flow, mode='bilinear', X=8):
    new_size = (X * flow.shape[2], X * flow.shape[3])
    return  X * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
