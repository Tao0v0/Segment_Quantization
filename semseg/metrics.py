import torch
from torch import Tensor
from typing import Tuple, Optional


class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(
            target[keep] * self.num_classes + pred[keep],
            minlength=self.num_classes**2,
        ).view(self.num_classes, self.num_classes)

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        ious[ious.isnan()] = 0.0
        miou = ious.mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        f1[f1.isnan()] = 0.0
        mf1 = f1.mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        acc[acc.isnan()] = 0.0
        macc = acc.mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)


def _as_valid_mask(valid: Optional[Tensor], ref_flow: Tensor) -> Tensor:
    """Return a boolean mask of shape (B, H, W) on ref_flow.device."""
    if valid is None:
        return torch.ones_like(ref_flow[:, 0], dtype=torch.bool)
    if valid.ndim == 4 and valid.shape[1] == 1:
        valid = valid[:, 0]
    if valid.ndim != 3:
        raise ValueError(f"valid must have shape (B, H, W) or (B, 1, H, W), got {tuple(valid.shape)}")
    return valid >= 0.5


def compute_epe(pred_flow: Tensor, gt_flow: Tensor, valid: Optional[Tensor] = None) -> Tensor:
    """Compute mean endpoint error (EPE), optionally masked by `valid`."""
    if pred_flow.ndim != 4 or pred_flow.shape[1] != 2:
        raise ValueError(f"pred_flow must have shape (B, 2, H, W), got {tuple(pred_flow.shape)}")
    if gt_flow.ndim != 4 or gt_flow.shape[1] != 2:
        raise ValueError(f"gt_flow must have shape (B, 2, H, W), got {tuple(gt_flow.shape)}")

    epe = torch.sqrt(torch.sum((pred_flow - gt_flow) ** 2, dim=1))  # (B, H, W)
    valid_mask = _as_valid_mask(valid, gt_flow)
    finite_mask = torch.isfinite(gt_flow).all(dim=1) & torch.isfinite(pred_flow).all(dim=1)
    valid_mask = valid_mask & finite_mask

    if not torch.any(valid_mask):
        return epe.new_zeros(())
    return epe[valid_mask].mean()


def compute_npe(
    pred_flow: Tensor,
    gt_flow: Tensor,
    n_values=(1, 2, 3),
    valid: Optional[Tensor] = None,
):
    """Compute NPE: percentage of valid pixels with EPE > N for each N in `n_values`."""
    if pred_flow.ndim != 4 or pred_flow.shape[1] != 2:
        raise ValueError(f"pred_flow must have shape (B, 2, H, W), got {tuple(pred_flow.shape)}")
    if gt_flow.ndim != 4 or gt_flow.shape[1] != 2:
        raise ValueError(f"gt_flow must have shape (B, 2, H, W), got {tuple(gt_flow.shape)}")

    epe = torch.sqrt(torch.sum((pred_flow - gt_flow) ** 2, dim=1))  # (B, H, W)
    valid_mask = _as_valid_mask(valid, gt_flow)
    finite_mask = torch.isfinite(gt_flow).all(dim=1) & torch.isfinite(pred_flow).all(dim=1)
    valid_mask = valid_mask & finite_mask

    denom = valid_mask.sum().clamp_min(1).float()
    out = []
    for n in n_values:
        pct = ((epe > float(n)) & valid_mask).sum().float() / denom * 100.0
        out.append(float(pct.item()))
    return tuple(out)
