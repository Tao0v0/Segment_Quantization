import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

import onnx
from onnxsim import simplify

from semseg.models.cmnext import CMNeXt


# ---- 关键改动：包装器，把3个独立输入转成list喂给原模型 ----
class CMNeXtWrapper(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, rgb: torch.Tensor, event: torch.Tensor, event_before: torch.Tensor):
        # 原模型forward期望一个长度为3的list：[rgb, event, event_before]
        out_list = self.core([rgb, event, event_before])
        # 约定返回第一个张量作为logits: [B, num_classes, H, W]
        return out_list[0]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 配置
    num_classes = 11
    H, W = 256, 256
    bins = 20

    # === 构建模型：开启 ERAFT + anytime ===
    model_core = CMNeXt(
        backbone='CMNeXt-B2',
        num_classes=num_classes,
        modals=['img'],
        backbone_flag=False,
        flow_net_flag=True,
        dataset_type='dsec',
        anytime_flag=True
    ).to(device)
    model_core.eval()

    # === dummy 输入 ===
    rgb = torch.randn(1, 3, H, W, device=device)
    event = torch.randn(1, bins, H, W, device=device)         # t→t+δt
    event_before = torch.randn(1, bins, H, W, device=device)  # t-δt→t

    # === 导出 ONNX（raw） ===
    wrapper = CMNeXtWrapper(model_core).to(device).eval()

    onnx_raw = "cmnext_rgb_event.onnx"
    onnx_simp = "cmnext_rgb_event_simple.onnx"

    print("Starting export with opset_version=20...")
    
    torch.onnx.export(
        wrapper,
        (rgb, event, event_before),
        onnx_raw,
        input_names=['rgb', 'event', 'event_before'],
        output_names=['logits'],
        opset_version=20,  # <--- 已修改为 20
        do_constant_folding=True,
        dynamic_axes={
            'rgb': {0: 'N', 2: 'H', 3: 'W'},
            'event': {0: 'N', 2: 'H', 3: 'W'},
            'event_before': {0: 'N', 2: 'H', 3: 'W'},
            'logits': {0: 'N', 2: 'H', 3: 'W'}
        }
    )
    print(f"Exported raw ONNX: {onnx_raw}")

    # === simplify 并保存 ===
    # 注意：onnxsim 对过高版本的 opset 支持可能滞后，如果此处报错，尝试更新 onnxsim 或回退 opset
    onnx_model = onnx.load(onnx_raw)

    # 有 dynamic_axes 的情况下，建议把当前 dummy shape 喂给 simplify，成功率更高
    onnx_model_simp, check = simplify(
        onnx_model,
        input_shapes={
            "rgb": list(rgb.shape),
            "event": list(event.shape),
            "event_before": list(event_before.shape),
        },
        dynamic_input_shape=True
    )

    if not check:
        raise RuntimeError("onnxsim simplify 校验失败（check=False），请检查导出算子/动态维度支持。")

    onnx.save(onnx_model_simp, onnx_simp)
    print(f"ONNX simplified saved as: {onnx_simp}")


if __name__ == '__main__':
    main()
