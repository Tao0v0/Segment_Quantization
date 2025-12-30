import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from semseg.models.cmnext import CMNeXt

def main():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 配置
    num_classes = 11
    H, W = 1920, 1080           # 保持 8 的倍数，ERAFT/金字塔更稳

    # === 构建模型：开启 ERAFT + anytime ===
    model = CMNeXt(
        backbone='CMNeXt-B2',
        num_classes=num_classes,
        modals=['img'],   # 两模态：RGB + Event
        backbone_flag=False,
        flow_net_flag=True,       # ★ 开 ERAFT（事件→光流，返回 Tensor）
        dataset_type='dsec',
        anytime_flag=True         # ★ 开 anytime（会取 x[2] = event_before）
    ).to(device)
    model.eval()

    # === 构造 dummy 输入 ===
    # RGB：当前时刻 t 的图像
    rgb = torch.randn(1, 3, H, W, device=device)

    # Event voxel：
    #   event         表示 [t, t+δt] 的事件体素（例如 4 个时间 bin）
    #   event_before  表示 [t-δt, t] 的事件体素
    bins = 20
    event         = torch.randn(1, bins, H, W, device=device)   # t→t+δt
    event_before  = torch.randn(1, bins, H, W, device=device)   # t-δt→t

    # ★ 必须传三个元素，顺序与 forward 中的索引一致
    sample = [rgb, event, event_before]

    with torch.no_grad():
        out = model(sample)   # 期望: [1, num_classes, H, W]
    out = out[0]

    print('Model output shape:', tuple(out.shape))

    # === 语义图后处理并保存 ===
    seg_map = F.softmax(out, dim=1).argmax(dim=1)[0].cpu().numpy()

    # 固定随机调色板
    rng = np.random.default_rng(42)
    palette = rng.integers(0, 256, size=(num_classes, 3), dtype=np.uint8)
    seg_rgb = palette[seg_map]
    Image.fromarray(seg_rgb).save('dummy_result.png')
    print('Saved: dummy_result.png')

if __name__ == '__main__':
    main()
