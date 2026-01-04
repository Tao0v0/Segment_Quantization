import torch
from torch import nn, Tensor
from torch.nn import functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def outlier_penalty_loss(X, r=3):
    """
    Compute the outlier penalty loss (OPL) for a given latent variable X.

    Args:
        X (torch.Tensor): The input tensor of latent values with shape (batch_size, channels, height, width).
        r (float): Scaling factor that determines how far outside of the standard deviation a latent value needs to be to be penalized.

    Returns:
        torch.Tensor: The outlier penalty loss value.
    """
    # Compute mean and standard deviation of X over height and width dimensions (H, W)
    mean_X = X.mean(dim=(2, 3), keepdim=True)
    std_X = X.std(dim=(2, 3), keepdim=True)
    
    # Compute the penalty for each latent value
    penalty = torch.norm(X - mean_X, dim=1, keepdim=True) - r * std_X
    
    # Apply the max(., 0) operation
    penalty = torch.maximum(penalty, torch.zeros_like(penalty))
    
    # Average over height and width dimensions
    loss = penalty.mean(dim=(2, 3))
    
    return loss.mean()

def get_vgg16_feature_extractor(layers):
    from torchvision.models import vgg16, VGG16_Weights
    from torchvision.models.feature_extraction import create_feature_extractor
    m = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    return_nodes = [f"features.{l}" for l in layers]
    return create_feature_extractor(m, return_nodes)

class VGGLoss(nn.Module):
    """The feature reconstruction loss in Perceptual Losses for Real-Time Style Transfer and Super-Resolution (https://arxiv.org/abs/1603.08155)."""
    
    def __init__(self, layers=[3, 8, 15, 22]):
        super().__init__()
        self.feature_extractor = get_vgg16_feature_extractor(layers).eval()
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, x, y):
        x = self.feature_extractor(x)
        y = self.feature_extractor(y)
        loss = 0
        for k in x.keys():
            loss += torch.nn.functional.l1_loss(x[k], y[k], reduction="mean")
        return loss / len(x.keys())
    
def gaussian(x, sigma=1.0):
    return torch.exp(-(x**2) / (2*(sigma**2)))

def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, device=None):
    """Construct the convolution kernel for a gaussian blur

    See https://en.wikipedia.org/wiki/Gaussian_blur for a definition.
    Overall I first generate a NxNx2 matrix of indices, and then use those to
    calculate the gaussian function on each element. The two dimensional
    Gaussian function is then the product along axis=2.
    Also, in_channels == out_channels == n_channels
    """
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    
    # 使用 PyTorch 创建网格
    grid = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size)), dim=-1).float().to(device) - size // 2
    kernel = torch.prod(gaussian(grid, sigma), dim=-1)
    kernel /= torch.sum(kernel)

    # repeat same kernel for all pictures and all channels
    # Also, conv weight should be (out_channels, in_channels/groups, h, w)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
    kernel = kernel.repeat(n_channels, 1, 1, 1)  # 重复核以匹配通道数
    return kernel


def blur_images(images, kernel):
    """Convolve the gaussian kernel with the given stack of images"""
    _, n_channels, _, _ = images.shape
    _, _, kw, kh = kernel.shape
    imgs_padded = F.pad(images, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return F.conv2d(imgs_padded, kernel, groups=n_channels)


def laplacian_pyramid(images, kernel, max_levels=5):
    """Laplacian pyramid of each image

    https://en.wikipedia.org/wiki/Pyramid_(image_processing)#Laplacian_pyramid
    """
    current = images
    pyramid = []

    for level in range(max_levels):
        filtered = blur_images(current, kernel)
        diff = current - filtered
        pyramid.append(diff)
        current = F.avg_pool2d(filtered, 2)
    pyramid.append(current)
    return pyramid


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, kernel_size=5, sigma=1.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, output, target):
        if (self._gauss_kernel is None
                or self._gauss_kernel.shape[1] != output.shape[1]):
            self._gauss_kernel = build_gauss_kernel(
                n_channels=output.shape[1],
                device=output.device)
        output_pyramid = laplacian_pyramid(
            output, self._gauss_kernel, max_levels=self.max_levels)
        target_pyramid = laplacian_pyramid(
            target, self._gauss_kernel, max_levels=self.max_levels)
        diff_levels = [F.l1_loss(o, t)
                        for o, t in zip(output_pyramid, target_pyramid)]
        loss = sum([2**(-2*j) * diff_levels[j]
                    for j in range(self.max_levels)])
        return loss
    
def SSIM_(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMilarity (SSIM) distance between two images.

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim

def SSIM(x, y, kernel_size=3):
    """
    Calculates the SSIM (Structural SIMilarity) loss

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    kernel_size : int
        Convolutional parameter

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM loss
    """
    ssim_value = SSIM_(x, y, kernel_size=kernel_size)
    return torch.clamp((1. - ssim_value) / 2., 0., 1.)

def calc_photometric_loss(t_est, images, ssim_loss_weight=0.85, clip_loss=0):
    """
    Calculates the photometric loss (L1 + SSIM)
    Parameters
    ----------
    t_est : list of torch.Tensor [B,3,H,W]
        List of warped reference images in multiple scales
    images : list of torch.Tensor [B,3,H,W]
        List of original images in multiple scales

    Returns
    -------
    photometric_loss : torch.Tensor [1]
        Photometric loss
    """
    # L1 loss
    l1_loss = [torch.abs(t_est[i] - images[i])
                for i in range(len(t_est))]
    # SSIM loss
    if ssim_loss_weight > 0.0:
        ssim_loss = [SSIM(t_est[i], images[i], kernel_size=3)
                        for i in range(len(t_est))]
        # Weighted Sum: alpha * ssim + (1 - alpha) * l1
        photometric_loss = [ssim_loss_weight * ssim_loss[i].mean(1, True) +
                            (1 - ssim_loss_weight) * l1_loss[i].mean(1, True)
                            for i in range(len(t_est))]
    else:
        photometric_loss = l1_loss
    # Clip loss
    if clip_loss > 0.0:
        for i in range(len(photometric_loss)):
            mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
            photometric_loss[i] = torch.clamp(
                photometric_loss[i], max=float(mean + clip_loss * std))
    # Return total photometric loss
    return photometric_loss

def reduce_photometric_loss(photometric_losses, photometric_reduce_op='min'):
    """
    Combine the photometric loss from all context images

    Parameters
    ----------
    photometric_losses : list of torch.Tensor [B,3,H,W]
        Pixel-wise photometric losses from the entire context

    Returns
    -------
    photometric_loss : torch.Tensor [1]
        Reduced photometric loss
    """
    # Reduce function
    def reduce_function(losses):
        if photometric_reduce_op == 'mean':
            return sum([l.mean() for l in losses]) / len(losses)
        elif photometric_reduce_op == 'min':
            return torch.cat(losses, 1).min(1, True)[0].mean()
        else:
            raise NotImplementedError(
                'Unknown photometric_reduce_op: {}'.format(photometric_reduce_op))
    # Reduce photometric loss
    photometric_loss = sum([reduce_function(photometric_losses[i])
                            for i in range(len(photometric_losses))]) / len(photometric_losses)
    return photometric_loss

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        # Store as python float to avoid device-mismatch issues (CPU vs CUDA tensors).
        # thresh is a probability; convert to loss threshold via -log(thresh).
        self.thresh = float(-math.log(float(thresh)))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        valid = labels != self.ignore_label
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            # No supervised pixels in this crop; return a zero loss instead of NaN.
            return preds.new_zeros(())

        n_min = max(n_valid // 16, 1)
        # Filter to supervised pixels only so ignored pixels (loss=0) don't affect OHEM selection.
        loss = self.criterion(preds, labels).view(-1)
        loss = loss[valid.view(-1)]

        # Select hard pixels (loss > thresh); if too few, take top-k.
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return loss_hard.mean()

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class Dice(nn.Module):
    def __init__(self, delta: float = 0.5, aux_weights: list = [1, 0.4, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_classes = preds.shape[1]
        labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels*preds, dim=(2, 3))
        fn = torch.sum(labels*(1-preds), dim=(2, 3))
        fp = torch.sum((1-labels)*preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)

def compute_eraft_flow_loss(flow_preds, flow_gt, valid=None, gamma=0.8, max_flow=400.0):
    """RAFT-style sequence loss for optical flow.

    Args:
        flow_preds: list[Tensor] or Tensor, each (B, 2, H, W)
        flow_gt: Tensor (B, 2, H, W)
        valid: optional Tensor mask (B, H, W) or (B, 1, H, W)
        gamma: discount factor for early iterations
        max_flow: ignore GT pixels with magnitude >= max_flow
    """
    if torch.is_tensor(flow_preds):
        flow_preds = [flow_preds]

    if len(flow_preds) == 0:
        raise ValueError("flow_preds must be a Tensor or a non-empty list of Tensors")

    if flow_gt.ndim != 4 or flow_gt.shape[1] != 2:
        raise ValueError(f"flow_gt must have shape (B, 2, H, W), got {tuple(flow_gt.shape)}")

    # Base validity mask (optionally provided by dataset).
    if valid is None:
        valid_mask = torch.ones_like(flow_gt[:, 0], dtype=torch.bool)
    else:
        if valid.ndim == 4 and valid.shape[1] == 1:
            valid = valid[:, 0]
        if valid.ndim != 3:
            raise ValueError(f"valid must have shape (B, H, W) or (B, 1, H, W), got {tuple(valid.shape)}")
        valid_mask = valid >= 0.5

    # Filter non-finite GT and extreme flows.
    finite_mask = torch.isfinite(flow_gt).all(dim=1)  # (B, H, W)
    mag = torch.sqrt(torch.sum(flow_gt ** 2, dim=1))  # (B, H, W)
    valid_mask = valid_mask & finite_mask & (mag < max_flow)

    valid_f = valid_mask[:, None].float()
    denom = (valid_f.sum().clamp_min(1.0) * flow_gt.shape[1])

    n_predictions = len(flow_preds)
    loss = flow_gt.new_zeros(())
    for i, pred in enumerate(flow_preds):
        weight = gamma ** (n_predictions - i - 1)
        diff = (pred - flow_gt).abs()
        loss = loss + weight * (valid_f * diff).sum() / denom

    return loss

__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice']


def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return Dice()
    return eval(loss_fn_name)(ignore_label, cls_weights)


if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 480, 640), dtype=torch.float)
    label = torch.randint(0, 19, (2, 480, 640), dtype=torch.long)
    loss_fn = Dice()
    y = loss_fn(pred, label)
    print(y)
