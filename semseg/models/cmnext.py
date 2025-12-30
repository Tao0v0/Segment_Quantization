import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
# from semseg.models.heads import LightHamHead
# from semseg.models.heads import UPerHead
# from semseg.models.modules.flow_network import unet
# from semseg.models.modules.flow_network.FRMA.modified_frma import EventFlowEstimator
# from semseg.models.modules.flow_network.FRMA.model import flow_network
# from semseg.models.modules.flow_network.FRMA.config import Config
from semseg.models.modules.flow_network.eraft.eraft import ERAFT
# from semseg.models.modules.flow_network.bflow.raft_spline.raft import RAFTSpline
from semseg.models.modules.softsplat.frame_synthesis import *
from semseg.models.modules.softsplat.softsplat import *
# from semseg.models.modules.memory.memory_encoder import *
# from semseg.utils.pac import SupervisedGaussKernel2d
# from semseg.losses import calc_photometric_loss, reduce_photometric_loss, LapLoss, VGGLoss, outlier_penalty_loss
# from fvcore.nn import flop_count_table, FlopCountAnalysis
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import moviepy.editor
    
class CMNeXt(BaseModel):     # 主干网络名称          # 检测类别数              #  模态                                          # 是否只返回骨干特征                 # 是否启动光流网络                               # 是否启用逐步细化预测
    def __init__(self, backbone: str = 'CMNeXt-B0', num_classes: int = 25, modals: list = ['img', 'depth', 'event', 'lidar'], backbone_flag: bool=False, flow_net_flag: bool=False, dataset_type: str=None, anytime_flag: bool=False) -> None:
        super().__init__(backbone, num_classes, modals, with_events=False,backbone_flag=backbone_flag,  flow_net_flag=flow_net_flag, dataset_type=dataset_type, anytime_flag=anytime_flag)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512, num_classes)
        if self.flow_net_flag:
            self.n_first_channels = 4
            self.flow_net = ERAFT(n_first_channels=self.n_first_channels)
            # self.flow_net = RAFTSpline()

        if not self.backbone_flag: # 实际取False，执行这段
        # if True:
            feature_dims = [64, 128, 320, 512]
            # feature_dims = [3]
            self.softsplat_net = Synthesis(feature_dims, activation='GELU')
            
            # self.fusion_attens = MultiAttentionBlock(
            #                         dim=feature_dims[-1],
            #                         num_heads=8,
            #                         LayerNorm_type='WithBias',
            #                         ffn_expansion_factor=2.66,
            #                         bias=False,
            #                         is_DA=True
            #                     )

        self.apply(self._init_weights)

    def forward(self, x: list, rgb_next: Tensor=None, lookup_timestamps: list=[0.5, 1.0],seq_names=None, seq_index=None) -> list:
        feature_init = self.backbone(x)
        y = []
        #######################
        if len(x) != 1:
            flows_split = []
            tenMetricones = []
            event_voxel = x[1]
            event_voxel_before = x[2]
            # event_voxel = torch.nn.functional.interpolate(x[1], scale_factor=0.5, mode='bilinear', align_corners=False)
            # event_voxel_before = torch.nn.functional.interpolate(x[2], scale_factor=0.5, mode='bilinear', align_corners=False)
            # event_voxel_total = x[1]
            B, C, H, W = event_voxel.shape
            # n_it = event_voxel_total.shape[1]//20
            # event_voxel = event_voxel_total.view(-1, 20, H, W)
            if not self.flow_net_flag:  # 实际取True 不管这段
                ### for 50 ms test ###
                bin = 5
                event_voxel = torch.cat([x[1][:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(4)], dim=1)
                flow = x[-1]
                feature_after = self.softsplat_net(tenEncone=feature_init, tenForward=flow, event_voxel=event_voxel, rgb=x[0])
                y_mid = self.decode_head(feature_after)
                y.append(F.interpolate(y_mid, size=x[0].shape[2:], mode='bilinear', align_corners=False))
                return y


            else:
                feature_after = feature_init

                if not self.anytime_flag: # 实际为True ，不执行
                    # import ipdb; ipdb.set_trace()
                    event_voxel_after = x[3]
                    # event_voxel_after = torch.nn.functional.interpolate(x[3], scale_factor=0.5, mode='bilinear', align_corners=False)

                    bin = 5
                    ev_t0_t1 = torch.cat([event_voxel[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                    ev_before = torch.cat([event_voxel_before[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                    flow_t0_t1 = self.flow_net(ev_before, ev_t0_t1)[-1]

                    ev_t1_t2 = torch.cat([event_voxel_after[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                    flow_t1_t2 = self.flow_net(ev_t0_t1, ev_t1_t2)[-1]
                    # flow_t1_t2 = torch.zeros(flow_t0_t1.shape).to(x[0].device)

                else:   # anytime tau
                    # 输入是ev_before -50 0 bin20 和 event_voxel 0 50 bin 20
                    assert event_voxel_before.shape[1] == 20
                    assert  event_voxel.shape[1] == 20
                    tau = 50        # 用于事件数据分段的时间窗口  -50→0ms 的 event_voxel_before 和 0→+50ms 的 event_voxel 因此每个 bin ≈ 2.5ms（50ms/20）


                    # event_voxel_before = event_voxel_before[:, -index:]    # TODO can be set to -index: ?
                    if tau <= 50:
                        index = tau//10*4    # interframe [0, 4, 8, 12, 16, 20] 0ms 10ms 20ms 30ms 40ms 50ms
                        bin = tau//10
                        # event_voxel_before = event_voxel_before[:, -index:]    # TODO can be set to -tau: ?
                        event_voxel = event_voxel[:, :index]
                        ev_t0_t1 = torch.cat([event_voxel[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(4)], dim=1)               # 把20bins压成4bins，稀疏压的更密集
                        ev_before = torch.cat([event_voxel_before[:, 5*i:5*(i+1)].mean(1).unsqueeze(1) for i in range(4)], dim=1)
                        # ev_before = torch.cat([event_voxel_before[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(4)], dim=1)
                        flow_save, flow_t0_t1s = self.flow_net(ev_before, ev_t0_t1)
                        flow_t0_t1 = flow_t0_t1s[-1]

                        ev_t1_t2 = torch.zeros(ev_t0_t1.shape).to(x[0].device)
                        flow_t1_t2 = torch.zeros(flow_t0_t1.shape).to(x[0].device)

                    elif tau > 50:
                        bin = (tau-50)//10
                        index = bin*4
                        event_voxel_after = x[3][:,:index]
                        ev_t0_t1 = torch.cat([event_voxel, event_voxel_after], dim=1)
                        bin = 5 + bin
                        ev_t0_t1 = torch.cat([ev_t0_t1[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(4)], dim=1)
                        bin = 5
                        ev_before = torch.cat([event_voxel_before[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                        flow_save, flow_t0_t1s = self.flow_net(ev_before, ev_t0_t1)
                        flow_t0_t1 = flow_t0_t1s[-1]


                        ev_t1_t2 = torch.cat([event_voxel_after[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(4)], dim=1)
                        # flow_t1_t2 = self.flow_net(ev_t0_t1, ev_t1_t2)[-1]
                        ev_t1_t2 = torch.zeros(ev_t0_t1.shape).to(x[0].device)
                        flow_t1_t2 = torch.zeros(flow_t0_t1.shape).to(x[0].device)

                    else:  # 不管
                        B, _, H, W = event_voxel.shape
                        ev_t0_t1 = torch.zeros(B, 4, H, W).to(x[0].device)
                        flow_t0_t1 = torch.zeros(B, 2, H, W).to(x[0].device)
                        ev_t1_t2 = torch.zeros(ev_t0_t1.shape).to(x[0].device)
                        flow_t1_t2 = torch.zeros(flow_t0_t1.shape).to(x[0].device)

                self.memory_bank = [feature_init[-1]]

                
                # t0 → t1
                feature_t1 = self.softsplat_net(tenEncone=feature_init, tenForward=flow_t0_t1, event_voxel=ev_t0_t1, rgb=x[0])  # 输出 warp+fine 后的特征

                # feature_t1[-1] = self.fusion_attens(Fw=feature_t1[-1], F0_c=self.memory_bank[0])  # 输入warp后的特征，和memory里的第一个特征（刚开始是金字塔的最顶层，分辨率最小的那个）
                #                                   把Fw当做Q在内部，F0_c当做KV，最后输出处理后的Fw特征给features_t1的最上层
                # self.memory_bank.append(feature_t1[-1])

                # t1 → t2
                feature_t2 = self.softsplat_net(tenEncone=feature_t1, tenForward=flow_t1_t2, event_voxel=ev_t1_t2, rgb=x[0])  # 还是一个金字塔，最上层的被改了

                # feature_t2[-1] = self.fusion_attens(Fw=feature_t2[-1], F0_c=self.memory_bank[0], Kd=self.memory_bank[1]) # 第一次是Fw做Q，FC做KV，得到Fw，然后再Fw做Q，Kd做KV，得到Fw

                y_t2 = self.decode_head(feature_t2)  # 解码金字塔，用的segformer的,返回每个像素的语义类别分割概率

                y.append(F.interpolate(y_t2, size=x[0].shape[2:], mode='bilinear', align_corners=False)) # y
                # exit(0)
                return y



        elif len(x) == 1: # 不执行
            feature_after = feature_init
            ## decoder
            y_mid = self.decode_head(feature_after)
            y.append(F.interpolate(y_mid, size=x[0].shape[2:], mode='bilinear', align_corners=False))
            return y

        ## visualization
        # import ipdb; ipdb.set_trace()
        # self.visualize_all([x[0]]+feature_before, [rgb_next]+feature_mid, [rgb_next]+feature_after, [flow]+interFlow)
        # self.visualize_features_all(feature_after)
        # self.visualize_all([x[0]]+feature_before, feature_after, [rgb_next]+feature_next, interFlow)
        # exit(0)  

    def visualize_feature(self, name, feature, save_path="feature.png"):
        """
        可视化特征图并保存为图像文件。

        参数：
            name (str): 特征名称，用于标题（现已移除）。
            feature (Tensor): 要可视化的特征张量。
            save_path (str): 保存图像的路径。
        """
        # 取第一个样本的特征图
        feature = feature[0]
        feature_np = feature.detach().cpu().numpy()
        
        # 对多通道特征进行平均以简化可视化
        if feature_np.shape[0] > 1:
            feature_to_plot = np.mean(feature_np, axis=0)
        else:
            feature_to_plot = feature_np[0]

        # 创建图形，设置大小为6x6英寸
        plt.figure(figsize=(6, 6))
        
        # 显示特征图，使用'viridis'颜色映射
        plt.imshow(feature_to_plot, cmap='viridis')
        
        # 移除标题
        # plt.title(name)
        
        # 关闭坐标轴
        plt.axis('off')
        
        # 保存图像，去除边框和额外空白
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        
        # 关闭图形以释放内存
        plt.close()

    def visualize_features_all(self, features):
        total_features = sum(f.shape[1] for f in features)
        print(f'Total number of features: {total_features}')
        num_cols = int(total_features ** 0.5)
        num_rows = (total_features + num_cols - 1) // num_cols

        fig = plt.figure(figsize=(100, 60))
        gs = GridSpec(num_rows, num_cols, figure=fig, wspace=0.1, hspace=0.1)

        idx = 0
        for i, feature in enumerate(features):
            # 取第一个 batch 的特征图
            feature_map = feature[0].detach().cpu().numpy()
            for j in range(len(feature_map)):
                if idx >= num_rows * num_cols:
                    break
                h, w = feature_map[j].shape
                ax = fig.add_subplot(gs[idx])
                im = ax.imshow(feature_map[j], cmap='viridis', extent=[0, w, 0, h])
                ax.axis('off')
                idx += 1

        # # 添加颜色条
        # for j in range(num_cols):
        #     img = fig.axes[j].images[0]
        #     fig.colorbar(img, ax=fig.axes[j::num_cols], orientation='vertical', fraction=0.046, pad=0.04)

        plt.savefig('features_all.png', dpi=150)
        plt.show()

    def visualize_features(self, features, axes, title_prefix):
        for i, feature in enumerate(features):
            # 取第一个batch的特征图
            feature_map = feature[0].detach().cpu().numpy()
            # 取平均值以减少通道维度
            feature_map = feature_map.mean(0)
            h, w = feature_map.shape
            axes[i].imshow(feature_map, cmap='viridis', extent=[0, w, 0, h])
            axes[i].set_title(f'{title_prefix} Scale {i+1}')
            axes[i].axis('off')

    def visualize_flow(self, flows, axes, title_prefix):
        for i, flow in enumerate(flows):
            flow_map = flow[0].detach().cpu().numpy()
            flow_magnitude = (flow_map ** 2).sum(axis=0) ** 0.5
            axes[i].imshow(flow_magnitude, cmap='plasma')
            axes[i].set_title(f'{title_prefix} {i+1}', fontsize=10)
            axes[i].axis('off')

    def visualize_all(self, feature_before, feature_after, feature_next, interFlow):
        num_features = len(feature_before)
        fig, axes = plt.subplots(4, num_features, figsize=(20, 12))

        # 可视化特征图（处理前）
        self.visualize_features(feature_before, axes[0], 'Before')

        # 可视化特征图（处理后）
        self.visualize_features(feature_after, axes[1], 'After')

        # 可视化特征图（下一步）
        self.visualize_features(feature_next, axes[2], 'Next')

        # 可视化光流
        self.visualize_flow(interFlow, axes[3], 'Flow')
        # 添加颜色条
        for i in range(4):
            for j in range(num_features):
                img = axes[i, j].images[0]
                fig.colorbar(img, ax=axes[i, j], orientation='vertical', fraction=0.046, pad=0.04)

        plt.tight_layout(pad=2.0)
        plt.savefig('features_and_flow.png', dpi=150)
        plt.show()

    def init_pretrained(self, pretrained: str = None) -> None:
        # for name, module in self.named_modules():
        #     if name == 'softsplat_net.netWarp.netsFour.netShortcut':
        #         print('before, mean is ', module.weight.mean())
        #         print('before, std is ', module.weight.std())
        if pretrained:
            if self.backbone.num_modals > 0:
                load_dualpath_model(self.backbone, pretrained)
            else:
                try:
                    checkpoint = torch.load(pretrained, map_location='cpu', weights_only=True)
                except TypeError:
                    checkpoint = torch.load(pretrained, map_location='cpu')
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    checkpoint = checkpoint['model']
                if not isinstance(checkpoint, dict):
                    raise TypeError(f'Unsupported checkpoint format: {type(checkpoint)}')
                checkpoint = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in checkpoint.items()}
                # NOTE
                if self.backbone_flag:
                    if any(k.startswith('backbone.') for k in checkpoint):
                        checkpoint = {k.replace('backbone.', '', 1): v for k, v in checkpoint.items() if k.startswith('backbone.')}
                     ### for carla
                    # checkpoints 去掉backbone前缀
                    # checkpoint = {k.replace('backbone.', ''): v for k, v in checkpoint.items()}
                    msg = self.backbone.load_state_dict(checkpoint, strict=False)
                    # msg = self.load_state_dict(checkpoint, strict=False)
                    ### for dsec
                    # msg = self.backbone.load_state_dict(checkpoint, strict=False)
                else:
                    if not any(k.startswith('backbone.') for k in checkpoint):
                        if any(('patch_embed' in k) or ('block' in k) or ('norm' in k) or ('layers' in k) for k in checkpoint):
                            checkpoint = {f'backbone.{k}': v for k, v in checkpoint.items()}
                    msg = self.load_state_dict(checkpoint, strict=False)
                print("load from: ", pretrained)
                print("init_pretrained message: ", msg)
                # exit(0)
        # for name, module in self.named_modules():
        #     if name == 'softsplat_net.netWarp.netsFour.netShortcut':
        #         print('after, mean is ', module.weight.mean())
        #         print('after, std is ', module.weight.std())
    def viz2(self, flow, x, rgb_next):
        # 可视化光流
        tenFlow = flow[0].unsqueeze(0)  # .detach().cpu().numpy()
        flow_magnitude = (tenFlow ** 2).sum(axis=0) ** 0.5  # 计算光流的大小
        tenFirst = x[0][0].unsqueeze(0)
        tenSecond = rgb_next[0].unsqueeze(0)
        # end
        # 1 1 H W
        tenMetric_L1 = torch.nn.functional.l1_loss(input=tenFirst, target=backwarp(tenIn=tenSecond, tenFlow=tenFlow), reduction='none').mean(1, True)
        tenMetric_flow_mag = torch.sqrt(torch.square(tenFlow[:, 0, :, :] + tenFlow[:, 1, :, :])).unsqueeze(1)
        Z = tenMetric_flow_mag

        tenOutputs_softsplat = [softsplat(tenIn=tenFirst, tenFlow=tenFlow * fltTime, tenMetric=-tenMetric_L1, strMode='soft') for fltTime in np.linspace(0.0, 1.0, 11).tolist()]
        npyOutputs_softsplat = [(tenOutput_softsplat[0, :, :, :].cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0.0, 255.0).astype(np.uint8) for tenOutput_softsplat in tenOutputs_softsplat + list(reversed(tenOutputs_softsplat[1:-1]))]
        preds = [np.concatenate([
            # npyOutputs_sum[i][:, :, ::-1],
            # npyOutputs_avg[i][:, :, ::-1],
            # npyOutputs_linear[i][:, :, ::-1], 
            npyOutputs_softsplat[i][:, :, ::-1],
            ], axis=1) for i in range(len(npyOutputs_softsplat))]
        video = moviepy.editor.ImageSequenceClip(sequence=preds, fps=5)
        video.write_gif('./out.gif')

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # 绘制 flow_magnitude
        axes[0, 0].imshow(flow_magnitude[0].cpu().numpy(), cmap='plasma')
        axes[0, 0].set_title('Flow Magnitude')
        axes[0, 0].axis('off')

        # 绘制 tenFirst
        axes[0, 1].imshow(tenFirst[0].cpu().numpy().transpose(1, 2, 0))
        axes[0, 1].set_title('First Frame')
        axes[0, 1].axis('off')

        # 绘制 tenSecond
        axes[1, 0].imshow(tenSecond[0].cpu().numpy().transpose(1, 2, 0))
        axes[1, 0].set_title('Second Frame')
        axes[1, 0].axis('off')

        # 绘制 npyOutputs_softsplat
        axes[1, 1].imshow(npyOutputs_softsplat[0])
        axes[1, 1].set_title('Softsplat Output')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig('flow_and_softsplat.png', dpi=150)
        plt.close()

def load_dualpath_model(model, model_file):
    extra_pretrained = None
    if isinstance(extra_pretrained, str):
        raw_state_dict_ext = torch.load(extra_pretrained, map_location=torch.device('cpu'))
        if 'state_dict' in raw_state_dict_ext.keys():
            raw_state_dict_ext = raw_state_dict_ext['state_dict']
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys(): 
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    
    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v

    if isinstance(extra_pretrained, str):
        for k, v in raw_state_dict_ext.items():
            if k.find('patch_embed1.proj') >= 0:
                state_dict[k.replace('patch_embed1.proj', 'extra_downsample_layers.0.proj.module')] = v 
            if k.find('patch_embed2.proj') >= 0:
                state_dict[k.replace('patch_embed2.proj', 'extra_downsample_layers.1.proj.module')] = v 
            if k.find('patch_embed3.proj') >= 0:
                state_dict[k.replace('patch_embed3.proj', 'extra_downsample_layers.2.proj.module')] = v 
            if k.find('patch_embed4.proj') >= 0:
                state_dict[k.replace('patch_embed4.proj', 'extra_downsample_layers.3.proj.module')] = v 
            
            if k.find('patch_embed1.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed1.norm', 'extra_downsample_layers.0.norm.ln_{}'.format(i))] = v 
            if k.find('patch_embed2.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed2.norm', 'extra_downsample_layers.1.norm.ln_{}'.format(i))] = v 
            if k.find('patch_embed3.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed3.norm', 'extra_downsample_layers.2.norm.ln_{}'.format(i))] = v 
            if k.find('patch_embed4.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed4.norm', 'extra_downsample_layers.3.norm.ln_{}'.format(i))] = v 
            elif k.find('block') >= 0:
                state_dict[k.replace('block', 'extra_block')] = v
            elif k.find('norm') >= 0:
                state_dict[k.replace('norm', 'extra_norm')] = v


    msg = model.load_state_dict(state_dict, strict=False)
    del state_dict

if __name__ == '__main__':
    modals = ['img', 'depth', 'event', 'lidar']
    model = CMNeXt('CMNeXt-B2', 25, modals)
    model.init_pretrained('checkpoints/pretrained/segformer/mit_b2.pth')
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    y = model(x)
    print(y.shape)
