import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from semseg.models.cmnext import CMNeXt
import semseg.models.modules.softsplat.frame_synthesis as fs

# =================================================================
# 1. Monkey Patch A: 修复 Softsplat 断图
#用 grid_sample 欺骗导出器，让数据流能从 FlowNet 传到 Backbone 特征
# =================================================================
def dummy_softsplat(tenIn, tenFlow, tenMetric, strMode):
    # 确保尺寸匹配
    if tenFlow.shape[2:] != tenIn.shape[2:]:
        tenFlow = torch.nn.functional.interpolate(tenFlow, size=tenIn.shape[2:], mode='bilinear')
    
    # 构造采样网格 [B, H, W, 2]
    grid = tenFlow.permute(0, 2, 3, 1)
    
    # 使用标准算子 grid_sample 替代，打通连接
    return torch.nn.functional.grid_sample(tenIn, grid, align_corners=True, padding_mode='border')

fs.softsplat = dummy_softsplat
print(">> [Patch] Softsplat 替换成功 (图已连通)")

# =================================================================
# 2. Monkey Patch B: 强制显示 LayerNorm
# 把手写的数学公式替换为 nn.LayerNorm，让 Netron 显示清爽的绿色节点
# =================================================================
class FakeLayerNorm(nn.Module):
    def __init__(self, normalized_shape, *args, **kwargs):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, x):
        return self.ln(x)

# 替换源码中的类定义
fs.WithBias_LayerNorm = FakeLayerNorm
fs.BiasFree_LayerNorm = FakeLayerNorm
print(">> [Patch] Norm 层替换成功 (显示为 LayerNormalization)")

# =================================================================
# 3. 包装器：统一接口
# =================================================================
class CMNeXtUnifiedWrapper(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, rgb, event, event_before):
        # 模拟原项目的输入列表格式
        # x = [rgb, event, event_before]
        out_list = self.core([rgb, event, event_before])
        return out_list[0] # 返回 logits

# =================================================================
# 4. 主程序
# =================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">> Using device: {device}")

    # 配置固定尺寸 (Static Shape)
    H, W = 256, 256
    bins = 20
    
    # 初始化模型
    model_core = CMNeXt(
        backbone='CMNeXt-B2',
        num_classes=11,
        modals=['img'],
        backbone_flag=False,
        flow_net_flag=True,      
        dataset_type='dsec',
        anytime_flag=True
    ).to(device).eval()

    # ★★★ 关键优化：减少 FlowNet 循环次数 ★★★
    # 原本是 12 次循环，导出后图会超级长。
    # 为了看结构，改成 1 次循环，逻辑结构不变，图缩小 90%。
    model_core.flow_net.iters = 1
    print(">> [Optimization] FlowNet iters 设置为 1 (图结构大幅简化)")

    # 准备 Dummy 输入
    rgb = torch.randn(1, 3, H, W, device=device)
    event = torch.randn(1, bins, H, W, device=device)
    event_before = torch.randn(1, bins, H, W, device=device)

    # 包装模型
    wrapper = CMNeXtUnifiedWrapper(model_core).to(device).eval()
    
    # 导出文件名
    onnx_file = "cmnext_unified_static.onnx"
    
    print(f"\n>> 开始导出完整静态模型: {onnx_file} ...")
    
    # ★★★ 导出设置 (Static) ★★★
    torch.onnx.export(
        wrapper,
        (rgb, event, event_before),
        onnx_file,
        input_names=['rgb', 'event', 'event_before'],
        output_names=['logits'],
        opset_version=20,          # 17 支持 LayerNorm
        do_constant_folding=True,  # 开启常量折叠
        # dynamic_axes=None        # ★ 不传这个参数，即默认为静态形状
    )
    
    # ★★★ Simplify (Static) ★★★
    print(">> 正在执行 Simplify (Static Mode)...")
    try:
        model_onnx = onnx.load(onnx_file)
        # 不传 dynamic_input_shape，让 simplify 尽情折叠常量
        model_simp, check = simplify(model_onnx)
        
        if check:
            onnx.save(model_simp, onnx_file)
            print(f"   Success! 最终文件已保存: {onnx_file}")
            print("   说明: 该模型包含 Backbone -> FlowNet(iter=1) -> Softsplat -> Fusion -> Head")
            print("   请使用 Netron 查看，图结构应该非常清晰且没有多余的 Shape/Gather 节点。")
        else:
            print("   Simplify 校验失败 (但文件已保存)")
            
    except Exception as e:
        print(f"   Simplify 出错: {e}")

if __name__ == '__main__':
    main()
