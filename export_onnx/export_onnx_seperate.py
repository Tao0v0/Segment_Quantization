import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import onnx
from onnxsim import simplify

from semseg.models.cmnext import CMNeXt
import semseg.models.modules.softsplat.frame_synthesis as fs


def _apply_patches() -> None:
    # Patch A: make softsplat ONNX-exportable (structure viewing only).
    def dummy_softsplat(tenIn, tenFlow, tenMetric, strMode):
        if tenFlow.shape[2:] != tenIn.shape[2:]:
            tenFlow = F.interpolate(tenFlow, size=tenIn.shape[2:], mode="bilinear", align_corners=False)
        grid = tenFlow.permute(0, 2, 3, 1)
        return F.grid_sample(tenIn, grid, align_corners=True, padding_mode="border")

    fs.softsplat = dummy_softsplat

    # Patch B: force a clean LayerNorm module in the exported graph (structure viewing only).
    class FakeLayerNorm(nn.Module):
        def __init__(self, normalized_shape, *args, **kwargs):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.ln = nn.LayerNorm(normalized_shape)

        def forward(self, x):
            return self.ln(x)

    fs.WithBias_LayerNorm = FakeLayerNorm
    fs.BiasFree_LayerNorm = FakeLayerNorm


class BackbonePart(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone([x])
        return f1, f2, f3, f4


class FlowNetPart(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.flow_net = model.flow_net
        # Keep the exported graph concise for Netron (not for deployment correctness).
        self.flow_net.iters = 1

    def forward(self, event_prev, event_curr):
        flow_low, flow_preds = self.flow_net(event_prev, event_curr)
        return flow_preds[-1]


class SoftsplatPart(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.softsplat = model.softsplat_net

    def forward(self, f1, f2, f3, f4, flow, event_voxel, rgb):
        out = self.softsplat([f1, f2, f3, f4], flow, event_voxel, rgb)
        return tuple(out) if isinstance(out, (list, tuple)) else out


class HeadPart(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.head = model.decode_head

    def forward(self, f1, f2, f3, f4):
        return self.head([f1, f2, f3, f4])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CMNeXt/ERAFT/Softsplat/Decoder to separate ONNX files (for Netron structure viewing)."
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable onnxsim.simplify() (keeps raw subgraphs; graph will be larger).",
    )
    parser.add_argument("--opset", type=int, default=20, help="ONNX opset version.")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory for ONNX files.")
    args = parser.parse_args()

    _apply_patches()
    print(">> Patches applied.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Using device: {device}")

    # Dummy input sizes (only for exporting structure).
    H, W = 440, 640
    bins = 20

    model = CMNeXt(
        backbone="CMNeXt-B2",
        num_classes=11,
        modals=["img"],
        flow_net_flag=True,
        anytime_flag=True,
    ).to(device).eval()
    # Keep the exported graph concise for Netron (not for deployment correctness).
    model.flow_net.iters = 1

    print(">> Capturing intermediate tensors with dummy inputs...")
    rgb = torch.randn(1, 3, H, W, device=device)
    event = torch.randn(1, bins, H, W, device=device)
    event_before = torch.randn(1, bins, H, W, device=device)

    with torch.no_grad():
        feature_init = model.backbone([rgb])  # list[Tensor] len=4

        # Match CMNeXt runtime: compress 20 bins -> 4 bins by averaging each 5-bin chunk.
        ev_t0_t1 = torch.cat([event[:, 5 * i : 5 * (i + 1)].mean(1, keepdim=True) for i in range(4)], dim=1)
        ev_before = torch.cat(
            [event_before[:, 5 * i : 5 * (i + 1)].mean(1, keepdim=True) for i in range(4)], dim=1
        )

        _, flow_preds = model.flow_net(ev_before, ev_t0_t1)
        flow_out = flow_preds[-1]

        warped_features = model.softsplat_net(feature_init, flow_out, ev_t0_t1, rgb)

    export_configs = [
        {
            "name": "01_cmnext_backbone",
            "module": BackbonePart(model),
            "inputs": (rgb,),
            "input_names": ["rgb"],
            "output_names": ["feat_s1", "feat_s2", "feat_s3", "feat_s4"],
        },
        {
            "name": "02_eraft",
            "module": FlowNetPart(model),
            "inputs": (ev_before, ev_t0_t1),
            "input_names": ["event_prev", "event_curr"],
            "output_names": ["flow_final"],
        },
        {
            "name": "03_softsplat",
            "module": SoftsplatPart(model),
            "inputs": (*feature_init, flow_out, ev_t0_t1, rgb),
            "input_names": ["feat_s1", "feat_s2", "feat_s3", "feat_s4", "flow", "event_voxel", "rgb"],
            "output_names": ["warped_s1", "warped_s2", "warped_s3", "warped_s4"],
        },
        {
            "name": "04_decoder",
            "module": HeadPart(model),
            "inputs": tuple(warped_features),
            "input_names": ["warped_s1", "warped_s2", "warped_s3", "warped_s4"],
            "output_names": ["logits"],
        },
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(">> Exporting 4 ONNX modules...")
    for cfg in export_configs:
        filename = str((out_dir / f"{cfg['name']}.onnx").resolve())
        print(f"   Exporting: {filename}")

        torch.onnx.export(
            cfg["module"].to(device).eval(),
            cfg["inputs"],
            filename,
            input_names=cfg["input_names"],
            output_names=cfg["output_names"],
            opset_version=int(args.opset),
            do_constant_folding=True,
        )

        if args.no_simplify:
            print("     -> simplify skipped (--no-simplify)")
            continue

        try:
            model_onnx = onnx.load(filename)
            model_simp, check = simplify(model_onnx)
            if check:
                onnx.save(model_simp, filename)
                print("     -> simplify ok")
            else:
                print("     -> simplify check failed (kept original)")
        except Exception as exc:
            print(f"     -> simplify error: {exc}")

    print(">> Done.")


if __name__ == "__main__":
    main()
