import torch
from torch import nn, Tensor
from typing import Tuple
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW tensors, keeping activations 4D."""

    def __init__(self, num_channels: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: Tensor) -> Tensor:
        # (B,C,H,W) -> (B,H,W,C) for LayerNorm over channels, then back.
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        # Use a 1x1 Conv to keep activations strictly 4D (B,C,H,W).
        # This is functionally equivalent to applying a Linear layer per spatial location,
        # but avoids flatten/transpose token representations (B,HW,C).
        self.proj = nn.Conv2d(dim, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Backwards-compatible load for older checkpoints using nn.Linear.

        Old: proj.weight (out,in), proj.bias (out,)
        New: proj.weight (out,in,1,1), proj.bias (out,)
        """
        w_key = prefix + "proj.weight"
        if w_key in state_dict:
            w = state_dict[w_key]
            if isinstance(w, torch.Tensor) and w.ndim == 2 and self.proj.weight.ndim == 4:
                if self.proj.weight.shape[:2] == w.shape and self.proj.weight.shape[2:] == (1, 1):
                    state_dict[w_key] = w[:, :, None, None]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        # Replace GroupNorm with LayerNorm2d to avoid any token/3D-style normalization paths.
        self.bn = LayerNorm2d(c2)
        self.activate = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i+1}", MLP(dim, embed_dim))

        self.linear_fuse = ConvModule(embed_dim*4, embed_dim) # 卷积->batchnorm->relu
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        _, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0])]
        for i, feature in enumerate(features[1:]):
            proj = getattr(self, f"linear_c{i+2}")
            cf = proj(feature)
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))  # 先用nn.Upsample(scale_factor=2, mode = 'bilinear')
        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        seg = self.linear_pred(self.dropout(seg))
        return seg
