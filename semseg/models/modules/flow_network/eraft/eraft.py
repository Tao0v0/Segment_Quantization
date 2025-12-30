import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock
from .utils import coords_grid, upflowX
from argparse import Namespace
from .image_utils import ImagePadder

try:
    from torch.cuda.amp import autocast
    # autocast = torch.amp.autocast
except:
    pass

    # # dummy autocast for PyTorch < 1.6
    # class autocast:
    #     def __init__(self, enabled):
    #         pass
    #     def __enter__(self):
    #         pass
    #     def __exit__(self, *args):
    #         pass


def get_args():
    # This is an adapter function that converts the arguments given in out config file to the format, which the ERAFT
    # expects.
    args = Namespace(small=False,
                     dropout=False,
                     mixed_precision=False,
                     clip=1.0)
    return args



class ERAFT(nn.Module):
    def __init__(self, n_first_channels):
        # args:
        super(ERAFT, self).__init__()
        args = get_args()       # 当输入python eraft.py --lr 0.0005 --batch_size 4 时，会返回指定的参数 lr 和 batchsize
        self.args = args
        self.raft_type = 'large'
        self.image_padder = ImagePadder(min_size=32, mode="replicate")
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

        if self.raft_type == 'large':
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4
            self.iters = 12

            # feature network, context network, and update block
            self.fnet = BasicEncoder(dims=(64, 64, 96, 128, 256), norm_fn='instance', dropout=0,
                                        n_first_channels=n_first_channels)
            self.cnet = BasicEncoder(dims=(64, 64, 96, 128, 256), norm_fn='batch', dropout=0,
                                        n_first_channels=n_first_channels)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim, raft_type=self.raft_type)
        elif self.raft_type == 'small':
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
            self.iters = 12

            self.fnet = BasicEncoder(dims=(32, 32, 64, 96, 128), norm_fn='instance', dropout=0,
                                        n_first_channels=n_first_channels, raft_type=self.raft_type)
            self.cnet = BasicEncoder(dims=(32, 32, 64, 96, 160), norm_fn='none', dropout=0,
                                        n_first_channels=n_first_channels, raft_type=self.raft_type)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim, raft_type=self.raft_type)
        self.iters = 12

        # Fixed 3x3 grouped conv that is equivalent to `F.unfold(..., 3x3, padding=1)` for 2-channel flow.
        # This is used inside `upsample_flow_4d()` to keep all intermediate tensors strictly 4D (B,C,H,W).
        self.unfold_conv = nn.Conv2d(2, 18, kernel_size=3, stride=1, padding=1, groups=2, bias=False)
        with torch.no_grad():
            self.unfold_conv.weight.zero_()
            out_idx = 0
            for _ in range(2):  # two flow channels (u,v) handled via groups=2
                for ky in range(3):
                    for kx in range(3):
                        self.unfold_conv.weight[out_idx, 0, ky, kx] = 1.0
                        out_idx += 1
        self.unfold_conv.weight.requires_grad_(False)
    # def freeze_bn(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)  # 输出形状：[B, 2, ht, wd]  ，第一个通道装的每个特网格的x坐标，第二个通道装的y坐标
        coords1 = coords_grid(N, H//8, W//8).to(img.device)  

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)   # 含义：对每个低分辨率像素 (H,W) 的每个 8×8 子像素位置，都准备了 9 个权重（对应 3×3 的邻域点）。
        mask = torch.softmax(mask, dim=2)       # 在“9 个邻域权重”上做 softmax，确保每组权重非负且总和为 1（凸组合）

        up_flow = F.unfold(8 * flow, [3,3], padding=1)  # 先解释 为什么乘 8：低分辨率上的光流是以特征网格为单位（步长=8个原图像素）；要变回原图单位，需要把位移放大 8 倍。
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)     # F.unfold(x, [3,3], padding=1)：对 x ∈ [N, C, H, W] 提取滑动 3×3 块（步长=1），输出形状 [N, C*9, H*W]。
                                                        # 这里 x = 8*flow, C=2，所以得到 [N, 18, H*W]，其中每个位置收集了3×3 邻域的 2 通道光流。
        up_flow = torch.sum(mask * up_flow, dim=2)      # 用预测的权重mask 对 9 个邻域光流进行加权求和，得到每个高分辨率像素的光流估计，形状 [N, 2, 1, 1, H, W]
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def upsample_flow_4d(self, flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Convex upsampling that keeps all intermediate tensors strictly 4D (B,C,H,W).

        Equivalent to `upsample_flow()` but avoids reshaping to >4D/3D, which can help
        some deployment backends that only support 4D tensors.
        """
        # mask: (B, 64*9, H, W) -> (B, 9, H*8, W*8)
        mask_hr = F.pixel_shuffle(mask, 8)
        mask_hr = torch.softmax(mask_hr, dim=1)

        # flow: (B, 2, H, W) -> (B, 18, H, W) via fixed grouped conv (3x3 "unfold"), then to (B, 18, H*8, W*8).
        patches_lr = self.unfold_conv(8.0 * flow)
        patches_hr = F.interpolate(patches_lr, scale_factor=8, mode="nearest")

        # Weighted sum over 9 neighbors (keepdim=True keeps everything 4D).
        patches_u = patches_hr[:, :9]
        patches_v = patches_hr[:, 9:]
        flow_u = (mask_hr * patches_u).sum(dim=1, keepdim=True)
        flow_v = (mask_hr * patches_v).sum(dim=1, keepdim=True)
        return torch.cat([flow_u, flow_v], dim=1)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # `flow_unfold3x3.weight` is a fixed (non-trainable) parameter that is constant by construction.
        # Allow loading older checkpoints that do not contain this key even when `strict=True`.
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        fixed_missing_keys = {
            prefix + "unfold_conv.weight",
        }
        fixed_unexpected_keys = {
            prefix + "flow_unfold3x3.weight",  # backward-compat if a checkpoint was saved from an older revision
        }
        missing_keys[:] = [k for k in missing_keys if k not in fixed_missing_keys]
        unexpected_keys[:] = [k for k in unexpected_keys if k not in fixed_unexpected_keys]

    def forward(self, event1, event2, flow_init=None, upsample=True):
        """ Estimate optical flow between pair of frames """

        # Pad Image (for flawless up&downsampling)
        event1 = self.image_padder.pad(event1)          # 将分辨率补齐到32的倍数
        event2 = self.image_padder.pad(event2)

        event1 = event1.contiguous()                    # 把 event1 这个张量在内存里变成“连续存储”的布局（contiguous）
        event2 = event2.contiguous()


        hdim = self.hidden_dim  # 128
        cdim = self.context_dim # 128

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):       # 混合精度训练，但实际不会用混合精度
            fmap1, fmap2 = self.fnet([event1, event2])          # (B, 256, H//8, W//8)  两张图像的特征图
        
        fmap1 = fmap1.float()       # 没开混合精度就是多余的，但为了保险起见，还是转成float32
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):       # 实际不会用混合精度
            cnet = self.cnet(event2)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)   # 在dim =1 上，按照hdim 和 cdim长度进行切分 net：[B, hdim, H, W]作为GRU的更新块  inp：[B, cdim, H, W]作为GRU的上下文输入
            net = self.tanh(net)
            inp = self.gelu(inp)

        # 这里初始化的coords0 和 coords1 是 1/8 分辨率的坐标网格,现在是一样的
        coords0, coords1 = self.initialize_flow(event1)    # 没有对event做改变，只是利用event1的形状（N,C,H,W）生成两张相同的坐标网格，1/8分辨率，每个坐标shape(N,2,H/8,W/8)

        if flow_init is not None:       # 不执行
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(self.iters):       # self/iters = 12
            coords1 = coords1.detach()

            # 就是根据两个光流的特征值的相关性，给每个像素提供k*k范围的权重，让光流可以找到正确的移动位置，真正纠正光流的操作在update_block里
            corr = corr_fn(coords1) # 返回四层相关性  （B,4*K*K,H,W)  K = 2r+1，纠正光流的移动位置，

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)  # net来自cnet的hidden，inp来自cnet的context，corr是相关特征，flow是当前光流

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow      # 真正更新光流的位置

            # up_mask 不是None,要执行 upsample_flow_4d
            if up_mask is None:      
                if self.raft_type == 'small':
                    flow_up = upflowX(coords1 - coords0, X=8)
            else:   
                flow_up = self.upsample_flow_4d(coords1 - coords0, up_mask) #mask shape: (B, 64*9, H/8, W/8)，flow的结果: (B, 2, H, W)
                # flow_up = upflowX(coords1 - coords0, X=8)

            flow_predictions.append(self.image_padder.unpad(flow_up))   #unpad：把多余的边界裁剪掉，恢复到原图大小

        return coords1 - coords0, flow_predictions      
        # 返回低分辨1/8分辨率光流（N,2,H/8,W/8） 和 高分辨率光流列表 [(N,2,H,W), ...]12个元素，最后一项是全分辨率输出
