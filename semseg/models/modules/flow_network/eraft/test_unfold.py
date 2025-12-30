import torch
import torch.nn as nn
import torch.nn.functional as F

def unfold_replacement_demo():
    # 1. 定义输入数据
    # 形状: (Batch=1, Channel=2, Height=34, Width=34)
    N, C, H, W = 1, 2, 34, 34
    x = torch.randn(N, C, H, W)

    # ==========================================
    # 方法 A: 使用官方 F.unfold (硬件不支持)
    # ==========================================
    # kernel_size=3, padding=1, 默认 stride=1
    target_output = F.unfold(x, kernel_size=3, padding=1)
    
    print(f"Target Unfold Shape: {target_output.shape}")
    # 预期输出: (1, 18, 1156)  其中 1156 = 34*34

    # ==========================================
    # 方法 B: 使用 Conv2d 实现 (硬件支持)
    # ==========================================
    kernel_size = 3
    padding = 1
    stride = 1
    
    # 计算输出通道数: C * K * K = 2 * 3 * 3 = 18
    out_channels = C * kernel_size * kernel_size
    
    # 1. 初始化权重: (Out, In, K, K) -> (18, 2, 3, 3)
    # 必须在特定设备上创建 (如 cuda)
    weight = torch.zeros((out_channels, C, kernel_size, kernel_size), device=x.device)
    
    # 2. 构造 "One-Hot" 权重
    # Unfold 的展平顺序是: 先铺满一个 patch 的 Channel 0，再铺满 Channel 1...
    # 或者是: (Channel 0 的 (0,0), (0,1)...) 接 (Channel 1 的 (0,0)...)
    # PyTorch 顺序: dim 1 (channel) 变化最慢，空间维度变化快
    
    idx = 0
    for c in range(C):              # 遍历输入通道 (0, 1)
        for i in range(kernel_size):    # 遍历核高 (0, 1, 2)
            for j in range(kernel_size): # 遍历核宽 (0, 1, 2)
                # 将对应位置设为 1，其余为 0
                weight[idx, c, i, j] = 1.0
                idx += 1
    
    # 3. 执行卷积
    # bias 必须为 None (或者 0)
    conv_out = F.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    
    # 4. 维度调整 (Reshape)
    # Conv 输出: (1, 18, 34, 34)
    # Unfold 需要: (1, 18, 34*34) = (1, 18, 1156)
    my_output = conv_out.view(N, out_channels, -1)

    print(f"Conv2d Impl Shape:   {my_output.shape}")

    # ==========================================
    # 验证结果
    # ==========================================
    # 检查最大误差
    diff = (target_output - my_output).abs().max()
    print(f"Max Difference: {diff.item()}")
    
    if diff < 1e-6:
        print("✅ 成功：Conv2d 实现与 Unfold 结果完全一致！")
    else:
        print("❌ 失败：结果不一致。")

if __name__ == "__main__":
    unfold_replacement_demo()