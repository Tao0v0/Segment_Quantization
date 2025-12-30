import math

import torch
import torch.nn.functional as F

from .utils import bilinear_sampler

try:
    import alt_cuda_corr
except Exception:
    # alt_cuda_corr is not compiled
    alt_cuda_corr = None


# 2x2 average pooling implemented via conv2d (no pooling op). Weight is 4D and constant.
_AVG_POOL2X2_W = torch.tensor([[[[0.25, 0.25], [0.25, 0.25]]]], dtype=torch.float32)


def _avg_pool2x2_s2(x: torch.Tensor) -> torch.Tensor:
    w = _AVG_POOL2X2_W.to(device=x.device, dtype=x.dtype)
    return F.conv2d(x, w, stride=2, padding=0)


class CorrBlock:
    """RAFT correlation volume, implemented to keep activation tensors strictly 4D."""

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, num_levels: int = 4, radius: int = 4):
        self.num_levels = int(num_levels)
        self.radius = int(radius)
        self.corr_pyramid = []

        # all pairs correlation (B*H*W, 1, H, W)
        corr = CorrBlock.corr(fmap1, fmap2)     # 计算fmap1和fmap2的相关性，输出shape：(B*H*W, 1, H, W)

        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels - 1):    # 金字塔构建，每次下采样2倍，4级
            corr = _avg_pool2x2_s2(corr)
            self.corr_pyramid.append(corr)

        # Precompute local offset grid as a 4D tensor: (1, K, K, 2).
        # NOTE: This keeps the legacy order from `torch.stack(torch.meshgrid(dy, dx), ...)`,
        # i.e. (dy, dx) == (y, x), to stay equivalent to older checkpoints.
        r = self.radius
        coords = []
        for y in range(-r, r + 1):
            row = []
            for x in range(-r, r + 1):
                row.append([y, x])  # (y, x) legacy order       shape: (K, 2) K=2r+1
            coords.append(row)
        self.delta = torch.tensor(coords, dtype=torch.float32).unsqueeze(0) # (1, K, K, 2) K=2r+1，常数,delta表示局部偏移量

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.permute(0, 2, 3, 1)  # (B,2,H,W) -> (B,H,W,2)
        batch, h1, w1, _ = coords.shape

        delta = self.delta.to(device=coords.device, dtype=coords.dtype)

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]     # 每次的corr shape: (B*H*W, 1, H_i, W_i)

            centroid_lvl = coords.reshape(-1, 1, 1, 2) * (1.0 / float(2**i))    #shape: (B*H*W, 1, 1, 2)
            coords_lvl = centroid_lvl + delta  # (B*H*W, 1,1, 2)+(1,k,k,2) = (B*H*W, K,K, 2) 每个像素有k*k个范围，每个点为xy坐标

            corr = bilinear_sampler(corr, coords_lvl) #输入的corr shape: (B*H*W, 1, H_i, W_i)  输出shape: (B*H*W, K, K, 1)，这个值就是告诉第H,W的像素，在k*k范围内的每个偏移位置的相关性是多少
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor: #对fmap1和fmap2计算相关性
        batch, dim, ht, wd = fmap1.shape

        # Keep activations 4D without broadcasting HW into batch dims (which can OOM).
        # q=(B,1,HW,C), k=(B,1,C,HW) -> corr=(B,1,HW,HW)
        q = fmap1.permute(0, 2, 3, 1).contiguous().reshape(batch, 1, -1, dim)
        k = fmap2.contiguous().reshape(batch, 1, dim, -1)
        corr = torch.matmul(q, k)

        # (B, 1, HW, HW) -> (B, HW, 1, HW) -> (B*HW, 1, H, W)
        corr = corr.permute(0, 2, 1, 3).contiguous().reshape(-1, 1, ht, wd)
        return corr * (1.0 / math.sqrt(float(dim)))
