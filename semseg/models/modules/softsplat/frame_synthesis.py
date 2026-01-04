import torch
import torch.nn as nn
from typing import Union
from .softsplat import softsplat
import numpy as np
# from scipy.ndimage import gaussian_filter
import numbers
from einops import rearrange
    
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (SE Block)"""
    def __init__(self, channels, reduction=4):
        """
        Args:
            channels: Number of input channels.
            reduction: Reduction ratio for the bottleneck in the SE block.
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling

        # Two fully connected (1x1 convolution) layers for the squeeze and excitation operations
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global pooling: squeeze operation
        scale = self.avg_pool(x)

        # Excitation operation: FC1 -> ReLU -> FC2 -> Sigmoid
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)

        # Scale input tensor by SE attention
        return x * scale

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        """
        Args:
            channels: Number of input channels.
            reduction: Reduction ratio for the channel attention module.
            kernel_size: Convolution kernel size for the spatial attention module.
        """
        super(CBAM, self).__init__()

        # Channel Attention Module
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling

        # Shared MLP: FC1 -> ReLU -> FC2 for channel attention
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention Module
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # 1. Channel Attention Module
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_attention = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attention  # Apply channel attention

        # 2. Spatial Attention Module
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))

        return x * spatial_attention  # Apply spatial attention

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):           # 卷积版前馈网络，目的是在不改分辨率的前提下，提升表达力、非线性与局部感受野，同时保持高效。
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)      # 1x1卷积核，升维，升2*hiddens是为了后面一分为二做门控
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # chunk按2*hiddens通道维度一分为二给x1和x2
        x = torch.nn.functional.gelu(x1) * x2    # 让gelu(x1)当gate，x2当value，两者互相点乘
        x = self.project_out(x)                 # 回到原始通道数
        return x


class Attention(nn.Module):
    # Restormer (CVPR 2022) transposed-attnetion block
    # original source code: https://github.com/swz30/Restormer
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_conv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, f):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(f))  # 卷积
        kv = self.kv_dwconv(self.kv_conv(x)) # 卷积
        k, v = kv.chunk(2, dim=1)       # 按维度切成2份

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)   # 多head 划分通道
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)        #对q的每一行做norm，让之后q@k的点积变成了余弦相似度，数值更稳定
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature #  k.transpose -> (B,H,N,C) -> (B,H,C,N) 就是q的(N,C) * k的 (C,N)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)    # 再卷积一次

        return out


class MultiAttentionBlock(torch.nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type, ffn_expansion_factor, bias, is_DA=False):
        super(MultiAttentionBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.co_attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)

        if is_DA:
            self.norm3 = LayerNorm(dim, LayerNorm_type)
            self.da_attn = Attention(dim, num_heads, bias)
            self.norm4 = LayerNorm(dim, LayerNorm_type)
            self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, Fw, F0_c=None, Kd=None):
        if F0_c is not None:
            Fw = Fw + self.co_attn(self.norm1(Fw), F0_c)
        else:
            Fw = Fw + self.co_attn(self.norm1(Fw), Fw)
        Fw = Fw + self.ffn1(self.norm2(Fw))

        if Kd is not None:
            Fw = Fw + self.da_attn(self.norm3(Fw), Kd)
            Fw = Fw + self.ffn2(self.norm4(Fw))

        return Fw



class Synthesis(torch.nn.Module):
    """Modified from the synthesis model in Softmax Splatting (https://github.com/sniklaus/softmax-splatting). Modifications:
    1) Warping only one frame with forward flow;
    2) Estimating the importance metric from the input frame and forward flow."""
    
    def __init__(self, feature_dims, activation='GELU'):
        super().__init__()

        # 根据传入的激活函数名称选择激活函数，并处理不同的初始化参数
        if activation == 'PReLU':
            self.activation_layer = lambda out_channels: nn.GELU()
        elif activation == 'GELU':
            self.activation_layer = lambda out_channels: nn.GELU()
        elif activation == 'LeakyReLU':
            self.activation_layer = lambda out_channels: nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'ELU':
            self.activation_layer = lambda out_channels: nn.ELU(alpha=1.0)
        elif activation == 'Mish':
            self.activation_layer = lambda out_channels: nn.Mish()
        elif activation == 'ReLU':
            self.activation_layer = lambda out_channels: nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        class Basic(torch.nn.Module):
            def __init__(self, strType, intChannels, boolSkip, skip_type='residual', attention_type='cbam', activation_layer=None):
                super().__init__()
                self.strType = strType
                self.activation_layer = activation_layer
                if strType == 'relu-conv-relu-conv':
                    self.netMain = torch.nn.Sequential(
                        self.activation_layer(intChannels[0]),
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, padding=1, bias=False)
                    )
                elif strType == 'conv-relu-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )
                elif strType == 'conv-relu-conv-sep':
                    self.netMain = torch.nn.Sequential(
                        nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[0], kernel_size=3, stride=1, padding=1, groups=intChannels[0], bias=False),
                        nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, groups=intChannels[1], bias=False),
                        nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=1, bias=False)
                    )
                elif strType == 'more-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )
                elif strType == 'more-more-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1]//4, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//4),
                        torch.nn.Conv2d(in_channels=intChannels[1]//4, out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )
                elif strType == 'more-more-conv-sep':
                    self.netMain = torch.nn.Sequential(
                        # Depthwise separable convolution
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[0], kernel_size=3, stride=1, padding=1, groups=intChannels[0], bias=False),
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, groups=intChannels[1], bias=False),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, groups=intChannels[1]//2, bias=False),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1]//4, kernel_size=1, bias=False),
                        self.activation_layer(intChannels[1]//4),
                        
                        torch.nn.Conv2d(in_channels=intChannels[1]//4, out_channels=intChannels[1]//4, kernel_size=3, stride=1, padding=1, groups=intChannels[1]//4, bias=False),
                        torch.nn.Conv2d(in_channels=intChannels[1]//4, out_channels=intChannels[1]//2, kernel_size=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, groups=intChannels[1]//2, bias=False),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, groups=intChannels[1], bias=False),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=1, bias=False)
                    )
                elif strType == 'more-more-conv-k5':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=5, stride=1, padding=2, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=5, stride=1, padding=2, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1]//4, kernel_size=5, stride=1, padding=2, bias=False),
                        self.activation_layer(intChannels[1]//4),
                        torch.nn.Conv2d(in_channels=intChannels[1]//4, out_channels=intChannels[1]//2, kernel_size=5, stride=1, padding=2, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=5, stride=1, padding=2, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=5, stride=1, padding=2, bias=False)
                    )
                elif strType == 'dilation-more-more-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1]//4, kernel_size=3, stride=1, padding=5, dilation=5, bias=False),
                        self.activation_layer(intChannels[1]//4),
                        torch.nn.Conv2d(in_channels=intChannels[1]//4, out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
                    )

                elif strType == 'multi-attention':
                    dim = intChannels[2]
                    num_multi_attn = 1
                    num_heads = 16
                    LayerNorm_type = 'WithBias'
                    ffn_expansion_factor = 2.66
                    bias = False
                    self.is_DA=False
                    self.netMain = torch.nn.Sequential(
                        *[MultiAttentionBlock(dim, num_heads, LayerNorm_type, ffn_expansion_factor, bias, self.is_DA) 
                        for _ in range(num_multi_attn)]
                    )
                # end

                self.boolSkip = boolSkip

                if boolSkip == True:
                    if intChannels[0] == intChannels[2]:
                        self.netShortcut = None

                    elif intChannels[0] != intChannels[2]:
                        self.netShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0, bias=False)

                self.skip_type = skip_type
                print("skip_type: ", self.skip_type)
                self.attention_type = attention_type
                # 使用一个卷积层来确保不同跳跃连接输出的形状与预期一致
                if self.skip_type == 'dense':
                    self.matchChannels = torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0, bias=False)
                elif self.skip_type == 'attention':
                    if self.attention_type == 'se':
                        self.se_block = SEBlock(intChannels[2])
                    elif self.attention_type == 'cbam':
                        self.cbam_block = CBAM(intChannels[2])

                if self.strType == 'conv-relu-conv-sep':
                    for m in self.modules():
                        if isinstance(m, nn.Conv2d):
                            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                            if m.weight is not None:
                                nn.init.constant_(m.weight, 1)
                            if m.bias is not None:
                                nn.init.constant_(m.bias, 0)
                        m.initialized = True
                    # end
                # end
            # end


            def forward(self, tenInput, in1=None, in2=None):
                # Standard path through the main network
                if self.strType == 'multi-attention':
                    # tenMain = tenInput[:, :-1]
                    # if in1 is not None:
                    #     tenMetric = in1.repeat(1, tenMain.shape[1], 1, 1)
                    # else:
                    #     tenMetric = tenInput[:, -1:].repeat(1, tenMain.shape[1], 1, 1)
                    if in1 is None:
                        in1 = tenInput
                    for layer in self.netMain:
                        # tenMain = layer(tenMain, in1, in2)
                        tenMain = layer(tenMain, in1, None)
                    return tenMain
                tenMain = self.netMain(tenInput)

                if self.boolSkip == False:
                    return tenMain

                if self.skip_type == 'residual':
                    # Residual connections: output shape matches input shape
                    if self.netShortcut is None:
                        return tenMain + tenInput
                    else:
                        return tenMain + self.netShortcut(tenInput)

                elif self.skip_type == 'dense':
                    # Dense connections: concatenate input and output
                    return self.dense_forward(tenInput)

                elif self.skip_type == 'attention':
                    # Attention-based skip connections
                    if self.attention_type == 'se':
                        if self.netShortcut is None:
                            return self.se_block(tenMain) + tenInput
                        else:
                            return self.se_block(tenMain) + self.netShortcut(tenInput)
                    elif self.attention_type == 'cbam':
                        if self.netShortcut is None:
                            return self.cbam_block(tenMain) + tenInput
                        else:
                            return self.cbam_block(tenMain) + self.netShortcut(tenInput)

            def dense_forward(self, tenInput):
                """ Dense forward pass with DenseNet-style concatenation. """
                features = [tenInput]  # Initialize the feature list with the input
                for layer in self.netMain:
                    new_feature = layer(torch.cat(features, dim=1))  # Concatenate all previous features
                    features.append(new_feature)
                # After passing through all layers, concatenate all features
                print(torch.cat(features, dim=1).shape)
                return self.matchChannels(torch.cat(features, dim=1))

                # end
            # end
        # end

        class Downsample(torch.nn.Module):
            def __init__(self, intChannels, activation_layer=None):
                super().__init__()
                self.activation_layer = activation_layer

                self.netMain = torch.nn.Sequential(
                    self.activation_layer(intChannels[0]),
                    torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1, bias=False),
                    self.activation_layer(intChannels[1]),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        class Upsample(torch.nn.Module):
            def __init__(self, intChannels, activation_layer=None):
                super().__init__()
                self.activation_layer = activation_layer
                self.netMain = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    self.activation_layer(intChannels[0]),
                    torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                    self.activation_layer(intChannels[1]),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        class Upsample_CrossScale(torch.nn.Module):
            def __init__(self, intChannels, upsample_mode='bilinear'):
                super().__init__()
                
                # 调整高分辨率特征图的通道数
                self.conv_high_res = torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[0], kernel_size=1, stride=1, padding=0, bias=False)
                # 上采样层，将低分辨率特征图上采样到与高分辨率特征图相同的空间尺寸
                self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False)
                self.netMain = torch.nn.Sequential(
                    # torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    self.activation_layer(intChannels[0]*2),
                    torch.nn.Conv2d(in_channels=intChannels[0]*2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                    self.activation_layer(intChannels[1]),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )
            # end

            def forward(self, high_res_feature, low_res_feature):
                # 对高分辨率特征图应用 1x1 卷积，调整通道数
                high_res_feature = self.conv_high_res(high_res_feature)

                # 对低分辨率特征图进行上采样
                low_res_feature_upsampled = self.upsample(low_res_feature)

                tenInput = torch.cat([high_res_feature, low_res_feature_upsampled], 1)
                return self.netMain(tenInput)
            # end
        # end

        class Softmetric(torch.nn.Module):
            def __init__(self, skip_type, in_ch, out_ch, attention_type=None, activation_layer=None):
                super().__init__()
                # embed_dim = [32, 64, 128, 256]
                embed_dim = [16, 32, 64, 96]

                # self.netEventInput = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
                self.netEventInput = torch.nn.Conv2d(in_channels=in_ch[0], out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
                # self.netRGBInput = torch.nn.Conv2d(in_channels=in_ch[2], out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
                self.netFlow = torch.nn.Conv2d(in_channels=in_ch[1], out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
                for intRow, intFeatures in enumerate(embed_dim):
                    self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True, skip_type, attention_type, activation_layer))
                # end

                for intCol in [0]:
                    self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([embed_dim[0], embed_dim[1], embed_dim[1]], activation_layer=activation_layer))
                    self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([embed_dim[1], embed_dim[2], embed_dim[2]], activation_layer=activation_layer))
                    self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([embed_dim[2], embed_dim[3], embed_dim[3]], activation_layer=activation_layer))
                # end

                for intCol in [1]:
                    self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([embed_dim[3], embed_dim[2], embed_dim[2]], activation_layer=activation_layer))
                    self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([embed_dim[2], embed_dim[1], embed_dim[1]], activation_layer=activation_layer))
                    self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([embed_dim[1], embed_dim[0], embed_dim[0]], activation_layer=activation_layer))
                    # self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample_CrossScale([embed_dim[3], embed_dim[2], embed_dim[2]]))
                    # self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample_CrossScale([embed_dim[2], embed_dim[1], embed_dim[1]]))
                    # self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample_CrossScale([embed_dim[1], embed_dim[0], embed_dim[0]]))
                # end

                self.netOutput = Basic('conv-relu-conv', [embed_dim[0], embed_dim[0], out_ch], True, activation_layer=activation_layer)
                # self.netOutput = Basic('more-more-conv', [embed_dim[0], embed_dim[0], 1], True)
            # end

            def forward(self, event, flow, rgb):
                tenColumn = [None, None, None, None]

                tenColumn[0] = torch.cat([
                    self.netEventInput(event),
                    self.netFlow(flow),
                    # self.netRGBInput(rgb)
                ], 1)
                # tenColumn[0]  = self.netEventInput(event_voxel)
                tenColumn[1] = self._modules['0x0 - 1x0'](tenColumn[0])
                tenColumn[2] = self._modules['1x0 - 2x0'](tenColumn[1])
                tenColumn[3] = self._modules['2x0 - 3x0'](tenColumn[2])

                # skip residual
                intColumn = 1
                for intRow in range(len(tenColumn) -1, -1, -1):
                    # '3x0 - 3x1'
                    tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
                    if intRow != len(tenColumn) - 1:
                        tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])
                        if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                        if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)

                        tenColumn[intRow] = tenColumn[intRow] + tenUp
                    # end
                # end

                return self.netOutput(tenColumn[0])
                # return torch.sigmoid(self.netOutput(tenColumn[0]))
            # end
        # end

        class Warp(torch.nn.Module):
            def __init__(self, embed_dim, activation_layer=None):
                super().__init__()
                self.refine_type = ['conv-relu-conv' if embed_dim[i] == 3 else 'conv-relu-conv' for i in range(len(embed_dim))]
                # self.refine_type = ['conv-relu-conv' if embed_dim[i] == 3 else 'conv-relu-conv-sep' for i in range(len(embed_dim))]
                # self.refine_type = ['conv-relu-conv' if embed_dim[i] == 3 else 'more-more-conv' for i in range(len(embed_dim))]
                # self.refine_type = ['conv-relu-conv' for i in range(len(embed_dim))]
                # self.netsOne = Basic(self.refine_type[0], [embed_dim[0]+1, embed_dim[0], embed_dim[0]], True, activation_layer=activation_layer)
                # self.netsTwo = Basic(self.refine_type[1], [embed_dim[1]+1, embed_dim[1], embed_dim[1]], True, activation_layer=activation_layer)
                # self.netsThree = Basic(self.refine_type[2], [embed_dim[2]+1, embed_dim[2], embed_dim[2]], True, activation_layer=activation_layer)
                # self.netsFour = Basic(self.refine_type[3], [embed_dim[3]+1, embed_dim[3], embed_dim[3]], True, activation_layer=activation_layer)
                self.nets = nn.ModuleList([
                    # nn.Sequential(
                    # Basic('conv-relu-conv', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True)
                    # Basic('more-conv', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True)
                    # Basic('more-conv', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True, activation_layer=activation_layer)
                    # Basic('more-more-conv', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True, activation_layer=activation_layer)
                    Basic(self.refine_type[i], [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True, activation_layer=activation_layer)
                    # Basic('more-more-conv-k5', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True, activation_layer=activation_layer)
                    # Basic('dilation-more-more-conv', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True, activation_layer=activation_layer)
                    # Basic('multi-attention', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True, activation_layer=activation_layer)
                    # SELayer(embed_dim[i]),
                    # )
                    for i in range(len(embed_dim))
                ])
            # end

            def forward(self, tenEncone, tenMetricone, tenForward):
                tenOutput = []
                # tenMid = []
                # tenFlow = []

                for intLevel in range(len(tenEncone)):
                    tenMetricone = torch.nn.functional.interpolate(input=tenMetricone, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False)
                    
                    tenForward = torch.nn.functional.interpolate(input=tenForward, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False) * (float(tenEncone[intLevel].shape[3]) / float(tenForward.shape[3]))
                    # event_voxel = torch.nn.functional.interpolate(input=event_voxel, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False)
                    # rgb = torch.nn.functional.interpolate(input=rgb, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False)
                    # tenScale = torch.nn.functional.interpolate(input=tenScale, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False)
                    # tenFlow.append(tenForward)
                    tenIn=torch.cat([tenEncone[intLevel], tenMetricone], 1)
                    # tenMask = torch.ones_like(tenMetricone)
                    # tenIn = tenEncone[intLevel]
                    tenWarp = softsplat(tenIn=tenIn, tenFlow=tenForward, tenMetric=tenMetricone, strMode='soft')
                    # tenMaskWarp = softsplat(tenIn=tenMask, tenFlow=tenForward, tenMetric=tenMetricone, strMode='soft')
                    # tenMaskWarp = tenMaskWarp.expand(-1, tenIn.shape[1], -1, -1)
                    # print(tenWarp.shape, tenMaskWarp.shape, tenIn.shape)
                    # print((tenMaskWarp > 0).shape)
                    # tenWarp = tenWarp[tenMaskWarp > 0] + tenIn[tenMaskWarp == 0]
                    # tenMid.append(tenWarp)
                    tenOutput.append(
                        # [self.netsOne, self.netsTwo, self.netsThree, self.netsFour][intLevel](
                        self.nets[intLevel](
                            # torch.cat([tenEncone[intLevel], tenEncone_event[intLevel], tenWarp], 1)
                            # torch.cat([tenEncone[intLevel], tenWarp], 1)
                            tenWarp
                            # tenWarp, event_voxel, rgb
                            # tenWarp, tenMetricone
                            # tenWarp+tenIn
                            # tenWarp + tenEncone[intLevel]
                            # tenWarp + tenEncone_event[intLevel]
                            # tenWarp + tenEncone[intLevel] + tenEncone_event[intLevel]
                        )
                    )
                # end
                return tenOutput
                # return tenOutput, tenMid, tenFlow
            # end
        # end
        skip_type = 'residual'
        # skip_type = 'dense'
        # skip_type = 'attention'

        # attention_type = 'se'
        attention_type = 'cbam'

        # self.netFlow = Softmetric(skip_type, in_ch=[4,3], out_ch=2, attention_type=attention_type, activation_layer=self.activation_layer)
        self.netSoftmetric = Softmetric(skip_type, in_ch=[4,2,3], out_ch=1, attention_type=attention_type, activation_layer=self.activation_layer)
        # self.netScale = Softmetric()

        self.netWarp = Warp(feature_dims, activation_layer=self.activation_layer)
        # self.netWarp_img = Warp([3], activation_layer=self.activation_layer)



    def forward(self, tenEncone, tenForward, event_voxel, rgb=None):        # 输入特征金字塔，光流，以及事件
        # 如果tenForward和event_vexel 和tenEncone[0]的shape 不一样 需要进行插值
        if tenForward.shape[2:] != tenEncone[0].shape[2:]:
            tenForward = torch.nn.functional.interpolate(
                input=tenForward,
                size=(tenEncone[0].shape[2], tenEncone[0].shape[3]),
                mode='bilinear',
                align_corners=False,
            ) * (float(tenEncone[0].shape[3]) / float(tenForward.shape[3]))

        # Numerical safety: ensure flow is finite and bounded before warping kernels.
        tenForward = torch.nan_to_num(tenForward, nan=0.0, posinf=0.0, neginf=0.0)
        tenForward = tenForward.clamp(-512.0, 512.0)
        if event_voxel.shape[2:] != tenEncone[0].shape[2:]:
            event_voxel = torch.nn.functional.interpolate(input=event_voxel, size=(tenEncone[0].shape[2], tenEncone[0].shape[3]), mode='bilinear', align_corners=False)
        if rgb is not None and rgb.shape[2:] != tenEncone[0].shape[2:]:
            rgb = torch.nn.functional.interpolate(input=rgb, size=(tenEncone[0].shape[2], tenEncone[0].shape[3]), mode='bilinear', align_corners=False)
        # print(tenEncone[0].shape, tenForward.shape, event_voxel.shape)
        if event_voxel.mean() == 0:
            B, C, H, W = event_voxel.shape
            tenMetricone = torch.zeros(B, 1, H, W).to(event_voxel.device)
        else:
            tenMetricone = self.netSoftmetric(event_voxel, tenForward, rgb) * 2.0

        # Numerical safety: metric is used as both a feature channel and an exponent weight in softsplat.
        tenMetricone = torch.nan_to_num(tenMetricone, nan=0.0, posinf=0.0, neginf=0.0)
        tenMetricone = tenMetricone.clamp(-20.0, 20.0)
        # print(tenMetricone.mean())
        tenWarp = self.netWarp(tenEncone, tenMetricone, tenForward)
        return tenWarp

@torch.no_grad()
def predict_tensor(src_frame: torch.Tensor, flow: torch.Tensor, model: Synthesis, batch_size: int = 32):
    # src_frame and flow should be normalized tensors
    out_frames = []
    for i in range(0, flow.shape[0], batch_size):
        bs = min(batch_size, flow.shape[0] - i)
        out_frames.append(model(src_frame.repeat(bs, 1, 1, 1), flow[i:i + bs]))
    out_frames = torch.cat(out_frames, dim=0)
    
    return out_frames

@torch.no_grad()
def softsplat_tensor(src_frame: torch.Tensor, flow: torch.Tensor, transforms, weight_type: Union[None, str] = None, return_tensor: bool = True):
    # src_frame and flow should be normalized tensors
    if weight_type == "flow_mag":
        weight = torch.sqrt(torch.square(flow[:, 0, :, :] + flow[:, 1, :, :])).unsqueeze(1)
        mode = "soft"
    else:
        weight = None
        mode = "avg"
    out_frames = softsplat(tenIn=src_frame.repeat(flow.shape[0], 1, 1, 1), tenFlow=flow, tenMetric=weight, strMode=mode)
    if return_tensor:
        return transforms.denormalize_frame(out_frames)
    else:
        return transforms.deprocess_frame(out_frames)
