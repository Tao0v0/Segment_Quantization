import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.GELU()

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = self.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128, raft_type='large'):
        super(SepConvGRU, self).__init__()

        self.raft_type = raft_type
        self.tanh = nn.Tanh()
        if self.raft_type == 'large':
            self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
            self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
            self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

            self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
            self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
            self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

        elif self.raft_type == 'small':
            self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
            self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
            self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = self.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        if self.raft_type == 'large':
            # vertical
            hx = torch.cat([h, x], dim=1)
            z = torch.sigmoid(self.convz2(hx))
            r = torch.sigmoid(self.convr2(hx))
            q = self.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
            h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args, raft_type='large'):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.raft_type = raft_type
        self.gelu = nn.GELU()

        if self.raft_type == 'large':
            self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
            self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
            self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
            self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
            self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)
        elif self.raft_type == 'small':
            self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
            self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
            self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
            self.conv = nn.Conv2d(32+96, 82-2, 3, padding=1)
        

    def forward(self, flow, corr):
        cor = self.gelu(self.convc1(corr))
        if self.raft_type == 'large':
            cor = self.gelu(self.convc2(cor))
        flo = self.gelu(self.convf1(flow))
        flo = self.gelu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.gelu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, raft_type='large'):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.raft_type = raft_type
        self.encoder = BasicMotionEncoder(args, self.raft_type)

        if self.raft_type == 'large':
            self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim, raft_type=self.raft_type)
            self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
            self.mask = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(256, 64*9, 1, padding=0))
        elif self.raft_type == 'small':
            self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
            self.flow_head = FlowHead(hidden_dim, hidden_dim=128)



    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)          # corr: conv-gelu-conv-gelu  flow: conv-gelu-conv-gelu    cat-conv-gelu  cat
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)            # 正常  cat(net, inp) - conv -sigmoid - conv -sigmoid -cat - conv -tanh    shape:[B, hidden_dim, H, W]
        delta_flow = self.flow_head(net)    # 正常 conv-<gelu>-conv     shape:[B, 2, H, W]

        if self.raft_type == 'large':
            # scale mask to balance gradients
            mask = .25 * self.mask(net)     # mask： conv-gelu-conv   shape:[B, 64*9, H/8, W/8]是个权重图对每个低分辨率像素 (h,w)，对应一个 8×8 的高分辨率小块；这个小块里每个子像素位置都有 9 个权重
        elif self.raft_type == 'small':
            mask = None
        return net, mask, delta_flow


