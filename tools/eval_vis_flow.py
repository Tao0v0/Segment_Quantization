import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib import colors
from PIL import Image
from typing import List, Optional

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None  # type: ignore

from semseg.datasets import DSEC_Flow
from semseg.metrics import compute_epe, compute_npe


def _build_flow_net(flow_net_type: str, n_first_channels: int):
    flow_net_type = str(flow_net_type).lower()
    if flow_net_type in ("eraft",):
        from semseg.models.modules.flow_network.eraft.eraft import ERAFT

        return ERAFT(n_first_channels=n_first_channels)

    if flow_net_type in ("eraft_original", "original"):
        repo_root = Path(__file__).resolve().parents[1]
        flow_root = repo_root / "semseg" / "models" / "modules" / "flow_network"
        candidates = (
            flow_root / "ERAFT_original",
            flow_root / "E-RAFT_original",
            flow_root / "R-RAFT_original",
        )
        eraft_root = next((p for p in candidates if (p / "model" / "eraft.py").is_file()), None)
        if eraft_root is None:
            raise FileNotFoundError(
                "Could not locate upstream ERAFT code. Expected one of:\n"
                + "\n".join(f"  - {c}" for c in candidates)
            )
        if str(eraft_root) not in sys.path:
            sys.path.insert(0, str(eraft_root))
        from model.eraft import ERAFT as ERAFT_ORIG

        return ERAFT_ORIG(config={"subtype": "standard"}, n_first_channels=n_first_channels)

    raise ValueError(f"Unsupported flow net type: {flow_net_type!r}")

def _load_state_dict(path: str, keys=("model_state_dict", "state_dict", "model")):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in keys:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def flow_to_rgb(flow_2hw: np.ndarray, scaling: Optional[float] = 10.0) -> np.ndarray:
    """
    flow_2hw: numpy array (2, H, W)
    returns: uint8 RGB image (H, W, 3)
    """
    if flow_2hw.ndim != 3 or flow_2hw.shape[0] != 2:
        raise ValueError(f"Expected flow shape (2,H,W), got {flow_2hw.shape}")

    flow = np.transpose(flow_2hw, (1, 2, 0)).astype(np.float32)
    flow[np.isinf(flow)] = 0

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2) ** 0.5
    ang = np.arctan2(flow[..., 1], flow[..., 0])
    ang[ang < 0] += 2 * np.pi

    hsv[..., 0] = ang / (2 * np.pi)
    hsv[..., 1] = 1.0
    if scaling is None:
        denom = float((mag - mag.min()).max())
        hsv[..., 2] = 0.0 if denom <= 1e-12 else (mag - mag.min()) / denom
    else:
        mag = np.clip(mag, 0.0, float(scaling))
        hsv[..., 2] = mag / float(scaling)

    rgb = colors.hsv_to_rgb(hsv)
    return (rgb * 255.0).clip(0, 255).astype(np.uint8)


def encode_flow_kitti_png16(flow_2hw: np.ndarray, valid_hw: Optional[np.ndarray] = None) -> np.ndarray:
    """Encode optical flow to KITTI/DSEC 16-bit PNG format (u, v, valid).

    - flow_2hw: float32 array (2, H, W) in pixel units
    - returns: uint16 array (H, W, 3) storing (u, v, valid)

    Encoding matches the decoder in `semseg/datasets/dsec_flow.py::_load_flow`:
      flow = (png[:2] - 2**15) / 128.0
    """
    if flow_2hw.ndim != 3 or flow_2hw.shape[0] != 2:
        raise ValueError(f"Expected flow shape (2,H,W), got {flow_2hw.shape}")

    flow = flow_2hw.astype(np.float32, copy=True)
    finite_hw = np.isfinite(flow).all(axis=0)
    flow[~np.isfinite(flow)] = 0.0

    if valid_hw is None:
        valid_hw = finite_hw
    else:
        valid_hw = valid_hw.astype(bool) & finite_hw

    u = flow[0]
    v = flow[1]

    u_enc = np.round(u * 128.0 + 32768.0).astype(np.int64)
    v_enc = np.round(v * 128.0 + 32768.0).astype(np.int64)
    u_enc = np.clip(u_enc, 0, 65535).astype(np.uint16)
    v_enc = np.clip(v_enc, 0, 65535).astype(np.uint16)
    valid_enc = valid_hw.astype(np.uint16)

    return np.stack([u_enc, v_enc, valid_enc], axis=-1)


def write_png16(path: Path, rgb_u16: np.ndarray) -> None:
    """Write a uint16 HxWx3 PNG (RGB channel order)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if rgb_u16.ndim != 3 or rgb_u16.shape[2] != 3 or rgb_u16.dtype != np.uint16:
        raise ValueError(f"Expected uint16 HxWx3 array, got {rgb_u16.dtype} {rgb_u16.shape}")

    if cv2 is not None:
        # OpenCV expects BGR; convert from RGB -> BGR for correct on-disk channel order.
        bgr = rgb_u16[..., ::-1]
        ok = cv2.imwrite(str(path), bgr)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {path}")
        return

    if imageio is not None:
        imageio.imwrite(str(path), rgb_u16)
        return

    raise RuntimeError("Neither cv2 nor imageio is available to write 16-bit RGB PNG files.")


def _reduce_event_bins(event_bchw: torch.Tensor, group: int = 5) -> torch.Tensor:
    """
    Reduce event voxel bins by averaging every `group` bins.
    Input:  (B, C, H, W)  (e.g., C=20)
    Output: (B, C/group, H, W) (e.g., 4)
    """
    if event_bchw.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(event_bchw.shape)}")
    c = int(event_bchw.shape[1])
    if c % group != 0:
        raise ValueError(f"Bins ({c}) must be divisible by group ({group}).")
    return torch.cat(
        [event_bchw[:, group * i : group * (i + 1)].mean(1, keepdim=True) for i in range(c // group)], dim=1
    )

def _collect_event_pairs(
    root: Path, split: str, bins: int, seqs: Optional[List[str]] = None
) -> List[tuple]:
    event_dir = root / "event_t0_t1" / f"event_{bins}" / split
    before_dir = root / "event_t-1_t0" / f"event_{bins}" / split
    if not event_dir.is_dir():
        raise FileNotFoundError(f"Missing event directory: {event_dir}")
    if not before_dir.is_dir():
        raise FileNotFoundError(f"Missing event_before directory: {before_dir}")

    available = sorted([p.name for p in event_dir.iterdir() if p.is_dir()])
    if seqs:
        available = [s for s in available if s in set(seqs)]

    pairs: List[tuple] = []
    for seq_name in available:
        ev_seq = event_dir / seq_name
        bf_seq = before_dir / seq_name
        if not bf_seq.is_dir():
            continue
        for ev_path in sorted(ev_seq.glob("*.npy")):
            stem = ev_path.stem
            if not stem.isdigit():
                continue
            bf_path = bf_seq / ev_path.name
            if not bf_path.is_file():
                continue
            pairs.append((seq_name, stem, ev_path, bf_path))
    return pairs


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Quick ERAFT inference: EPE + flow visualization.")
    parser.add_argument("--cfg", type=str, required=True, help="Config yaml used for training.")
    parser.add_argument("--weights", type=str, required=True, help="ERAFT weights (.pth) or checkpoint.")
    parser.add_argument("--root", type=str, default=None, help="Override dataset root (processed folder).")
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "test"), help="Dataset split.")
    parser.add_argument("--duration", type=int, default=100, help="Flow GT duration in ms (DSEC is usually 100).")
    parser.add_argument("--classes", type=int, default=11, help="Dummy class count (kept for dataset signature).")
    parser.add_argument("--num-samples", type=int, default=1, help="How many random samples to run.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for sampling.")
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="Optional fixed dataset indices.")
    parser.add_argument("--all", action="store_true", help="Run on all available samples (ignores random sampling).")
    parser.add_argument("--seq", type=str, nargs="*", default=None, help="Optional sequence name filter(s).")
    parser.add_argument("--bins", type=int, default=20, help="Event voxel bins folder to read (event_{bins}).")
    parser.add_argument("--reduce-group", type=int, default=5, help="Reduce bins by averaging every N bins.")
    parser.add_argument(
        "--spatial-scale",
        type=float,
        default=0.5,
        help="Spatial scale factor for event inputs (and GT flow). Use 1.0 for full-resolution inference.",
    )
    parser.add_argument(
        "--save-flow-png16",
        action="store_true",
        help="Also save predicted flow as KITTI/DSEC 16-bit PNG (u,v,valid).",
    )
    parser.add_argument(
        "--flow-png-layout",
        type=str,
        default="dsec",
        choices=("dsec", "seq", "flat"),
        help="Directory layout for --save-flow-png16 outputs.",
    )
    parser.add_argument(
        "--export-spatial-scale",
        type=float,
        default=None,
        help=(
            "If set, rescale predicted flow to this spatial scale before exporting (e.g., 1.0). "
            "Useful when inference runs with --spatial-scale 0.5 but you want full-res flow files."
        ),
    )
    parser.add_argument("--no-gt", action="store_true", help="Run inference without GT flow (no EPE).")
    parser.add_argument("--save-flow-npy", action="store_true", help="Also save predicted flow as .npy.")
    parser.add_argument("--no-vis", action="store_true", help="Disable saving visualization PNGs.")
    parser.add_argument("--out-dir", type=str, default=None, help="Where to save visualizations/metrics.")
    parser.add_argument("--vis-scaling", type=float, default=10.0, help="Fixed scaling for flow colors.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu.")
    args = parser.parse_args()

    if not (args.spatial_scale > 0):
        raise ValueError(f"--spatial-scale must be > 0, got {args.spatial_scale}")

    if args.export_spatial_scale is not None and not (args.export_spatial_scale > 0):
        raise ValueError(f"--export-spatial-scale must be > 0, got {args.export_spatial_scale}")

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = str(cfg["DATASET"]["ROOT"]).replace("${DURATION}", str(args.duration))
    if args.root:
        root = args.root
    root_path = Path(root)

    if args.bins % args.reduce_group != 0:
        raise ValueError(f"--bins ({args.bins}) must be divisible by --reduce-group ({args.reduce_group})")
    n_first_channels = args.bins // args.reduce_group
    flow_net_type = str(cfg.get("MODEL", {}).get("FLOW_NET", "eraft")).lower()
    model = _build_flow_net(flow_net_type, n_first_channels=n_first_channels)
    state_dict = _load_state_dict(args.weights)
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    if flow_net_type == "eraft":
        # Backward-compat: older checkpoints (before `unfold_conv`) won't contain this fixed parameter.
        # We inject the expected constant weight so `strict=True` can still be used.
        if "unfold_conv.weight" not in state_dict and hasattr(model, "unfold_conv"):
            state_dict = dict(state_dict)
            state_dict["unfold_conv.weight"] = model.unfold_conv.weight.detach().clone()
        # Backward-compat: remove legacy experimental key if present.
        if "flow_unfold3x3.weight" in state_dict:
            state_dict = dict(state_dict)
            state_dict.pop("flow_unfold3x3.weight", None)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.weights).resolve().parent / "quick_vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    metrics_path = out_dir / "metrics.txt"
    epe_list: List[float] = []
    time_window = int(args.duration) // 50

    def _flow_png_path(seq_name: str, seq_id: str) -> Path:
        if args.flow_png_layout == "flat":
            return out_dir / f"{seq_name}_{seq_id}_pred_flow.png"
        if args.flow_png_layout == "seq":
            return out_dir / "pred_flow_png16" / seq_name / f"{seq_id}.png"
        if args.flow_png_layout == "dsec":
            return out_dir / f"flow_t0_t{time_window}" / args.split / seq_name / f"{seq_id}.png"
        raise ValueError(f"Unsupported --flow-png-layout: {args.flow_png_layout}")

    if args.no_gt:
        pairs = _collect_event_pairs(root_path, args.split, args.bins, seqs=args.seq)
        if not pairs:
            raise RuntimeError(f"No paired events found under {root_path} (split={args.split}).")
        if args.all:
            chosen = pairs
        elif args.indices and len(args.indices) > 0:
            chosen = [pairs[i] for i in args.indices if 0 <= i < len(pairs)]
        else:
            chosen = [pairs[rng.randrange(len(pairs))] for _ in range(args.num_samples)]

        for i, (seq_name, seq_id, ev_path, bf_path) in enumerate(chosen):
            event = torch.from_numpy(np.load(ev_path, allow_pickle=True)[:, :440]).unsqueeze(0).to(device).float()
            event_before = (
                torch.from_numpy(np.load(bf_path, allow_pickle=True)[:, :440]).unsqueeze(0).to(device).float()
            )

            ev_t0_t1 = _reduce_event_bins(event, group=args.reduce_group)
            ev_before = _reduce_event_bins(event_before, group=args.reduce_group)

            if args.spatial_scale != 1.0:
                ev_t0_t1 = F.interpolate(
                    ev_t0_t1, scale_factor=args.spatial_scale, mode="bilinear", align_corners=False
                )
                ev_before = F.interpolate(
                    ev_before, scale_factor=args.spatial_scale, mode="bilinear", align_corners=False
                )

            pred_out = model(ev_before, ev_t0_t1)
            if isinstance(pred_out, tuple):
                pred = pred_out[1][-1]
            elif torch.is_tensor(pred_out):
                pred = pred_out
            else:
                pred = pred_out[-1]

            pred_np = pred[0].detach().cpu().numpy()
            pred_rgb = flow_to_rgb(pred_np, scaling=args.vis_scaling)
            stem = f"{seq_name}_{seq_id}"
            if not args.no_vis:
                Image.fromarray(pred_rgb).save(out_dir / f"{stem}_pred.png")
            if args.save_flow_npy:
                np.save(out_dir / f"{stem}_pred.npy", pred_np)
            if args.save_flow_png16:
                export_scale = args.export_spatial_scale if args.export_spatial_scale is not None else args.spatial_scale
                scale_ratio = float(export_scale) / float(args.spatial_scale)
                pred_export = pred
                if scale_ratio != 1.0:
                    pred_export = F.interpolate(pred_export, scale_factor=scale_ratio, mode="bilinear", align_corners=False)
                    pred_export = pred_export * scale_ratio
                flow_png = encode_flow_kitti_png16(pred_export[0].detach().cpu().numpy())
                write_png16(_flow_png_path(seq_name, seq_id), flow_png)

            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(f"[{i}] sample={stem} (no-gt)\n")

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(f"\nSaved to: {out_dir}\n")

        print(f"Saved visualizations to: {out_dir}")
        return

    if args.split == "test":
        raise ValueError("--split test has no GT flow. Use --no-gt to run events-only inference.")

    dataset = DSEC_Flow(
        root=str(root_path),
        split=args.split,
        n_classes=args.classes,
        transform=None,
        modals=cfg["DATASET"]["MODALS"],
        duration=args.duration,
        flow_net_flag=True,
        dataset_type=cfg["DATASET"]["TYPE"],
    )

    if args.indices is None or len(args.indices) == 0:
        if len(dataset) == 0:
            raise RuntimeError("Dataset is empty.")
        if args.all:
            indices = list(range(len(dataset)))
        else:
            indices = [rng.randrange(len(dataset)) for _ in range(args.num_samples)]
    else:
        indices = list(args.indices)

    for i, dataset_idx in enumerate(indices):
        seq_name, seq_id, sample_list = dataset[int(dataset_idx)]
        event = sample_list[0].unsqueeze(0).to(device).float()
        event_before = sample_list[1].unsqueeze(0).to(device).float()
        flow_gt_raw = sample_list[-1].unsqueeze(0).to(device).float()
        valid = None
        if flow_gt_raw.ndim == 4 and flow_gt_raw.shape[1] == 3:
            valid = flow_gt_raw[:, 2]
            flow_gt = flow_gt_raw[:, :2]
        else:
            flow_gt = flow_gt_raw

        ev_t0_t1 = _reduce_event_bins(event, group=args.reduce_group)
        ev_before = _reduce_event_bins(event_before, group=args.reduce_group)

        if args.spatial_scale != 1.0:
            ev_t0_t1 = F.interpolate(ev_t0_t1, scale_factor=args.spatial_scale, mode="bilinear", align_corners=False)
            ev_before = F.interpolate(ev_before, scale_factor=args.spatial_scale, mode="bilinear", align_corners=False)
            flow_gt = F.interpolate(flow_gt, scale_factor=args.spatial_scale, mode="bilinear", align_corners=False)
            flow_gt = flow_gt * args.spatial_scale
            if valid is not None:
                valid = F.interpolate(valid[:, None], scale_factor=args.spatial_scale, mode="nearest")[:, 0]

        pred_out = model(ev_before, ev_t0_t1)
        if isinstance(pred_out, tuple):
            pred = pred_out[1][-1]
        elif torch.is_tensor(pred_out):
            pred = pred_out
        else:
            pred = pred_out[-1]

        epe = float(compute_epe(pred, flow_gt, valid=valid).item())
        n1pe, n2pe, n3pe = compute_npe(pred, flow_gt, valid=valid)
        epe_list.append(epe)

        pred_rgb = flow_to_rgb(pred[0].detach().cpu().numpy(), scaling=args.vis_scaling)
        gt_rgb = flow_to_rgb(flow_gt[0].detach().cpu().numpy(), scaling=args.vis_scaling)
        concat = np.concatenate([gt_rgb, pred_rgb], axis=1)

        stem = f"{seq_name}_{seq_id}"
        if not args.no_vis:
            Image.fromarray(gt_rgb).save(out_dir / f"{stem}_gt.png")
            Image.fromarray(pred_rgb).save(out_dir / f"{stem}_pred.png")
            Image.fromarray(concat).save(out_dir / f"{stem}_gt_pred.png")
        if args.save_flow_npy:
            np.save(out_dir / f"{stem}_pred.npy", pred[0].detach().cpu().numpy())
        if args.save_flow_png16:
            export_scale = args.export_spatial_scale if args.export_spatial_scale is not None else args.spatial_scale
            scale_ratio = float(export_scale) / float(args.spatial_scale)
            pred_export = pred
            if scale_ratio != 1.0:
                pred_export = F.interpolate(pred_export, scale_factor=scale_ratio, mode="bilinear", align_corners=False)
                pred_export = pred_export * scale_ratio
            flow_png = encode_flow_kitti_png16(pred_export[0].detach().cpu().numpy())
            write_png16(_flow_png_path(seq_name, seq_id), flow_png)

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{i}] idx={dataset_idx} sample={stem} EPE={epe:.4f} "
                f"N1PE={n1pe:.2f}% N2PE={n2pe:.2f}% N3PE={n3pe:.2f}%\n"
            )

    mean_epe = float(np.mean(epe_list)) if epe_list else 0.0
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(f"\nMean EPE over {len(epe_list)} samples: {mean_epe:.4f}\n")
        f.write(f"Saved to: {out_dir}\n")

    print(f"Saved visualizations to: {out_dir}")
    print(f"Mean EPE ({len(epe_list)} samples): {mean_epe:.4f}")


if __name__ == "__main__":
    main()

"""
Example (with GT flow, compute EPE + visualize GT/Pred):        duration多少，只是告诉dsec_flow去读哪一套GT目录，--duration 100 → time_window=2 → 读取 flow_t0_t2/...（对应你 100ms 的 GT）
你如果不传或传错（比如 50）→ 会去找 flow_t0_t1/...，要么找不到、要么读到另一套 GT

PYTHONPATH=. python tools/eval_vis_flow.py \
  --cfg configs/dsec_rgb_day_flow.yaml \
  --weights output/DSEC_Flow_CMNeXt-B2/your_model.pth \
  --split val \
  --duration 100 \                              
  --num-samples 5 \
  --out-dir output/DSEC_Flow吧_CMNeXt-B2/quick_vis
"""

"""
Example (no GT flow, events-only inference + visualize Pred):   每个npy样本是多少间隔，那么推理就自动适应多少间隔。

PYTHONPATH=. python tools/eval_vis_flow.py \
  --cfg configs/dsec_rgb_day_flow.yaml \
  --weights output/DSEC_Flow_CMNeXt-B2/your_model.pth \
  --root dataset/DSEC_no_flow_processed \
  --split train \
  --bins 20 \
  --reduce-group 5 \
  --no-gt \
  --seq interlaken_00_d \
  --num-samples 5 \
  --out-dir output/no_flow_vis


3. 全分辨率用1.0 二分之一用0.5
  PYTHONPATH=. python tools/eval_vis_flow.py   --cfg configs/dsec_rgb_day_flow.yaml   --weights output_all_LNorm/DSEC_Flow_CMNeXt-B2_i/model_thun_00_a_11_CMNeXt_CMNeXt-B2_DSEC_Flow_epoch31_0.7229367727186622.pth   
  --split val   --duration 100 --spatial-scale 1.0   --num-samples 40 --out-dir inference_vis_result/20251220_all_LN  
"""



"""
4. 提交DSEC 官方测试
(want_cmnext) [sme-wangzr@ae2u07g segment_anytime_v1]$ PYTHONPATH=. python tools/eval_vis_flow.py   --cfg configs/dsec_rgb_day_flow.yaml   --weights output_all_LNorm/DSEC_Flow_CMNeXt-B2_i/model_thun_00_a_11_CMNeXt_CMNeXt-B2_DSEC_Flow_epoch31_0.7229367727186622.pth   --root dataset/DSEC_no_flow_processed   --split train   --bins 20   --reduce-group 5   --no-g
t   --all   --spatial-scale 0.5   --export-spatial-scale 1.0   --save-flow-png16   --flow-png-layout dsec   --out-dir submission_output

"""
