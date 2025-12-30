# val_eraft.py
import os

# ---- 修复 HDF5 plugin 目录报错 ----
if os.environ.get("HDF5_PLUGIN_PATH") == "/usr/local/hdf5/lib/plugin":
    os.environ.pop("HDF5_PLUGIN_PATH", None)

# ---- 让 h5py 能读 DSEC events.h5 常见 blosc/zstd 过滤器 ----
try:
    import hdf5plugin  # noqa: F401
except ImportError as e:
    raise RuntimeError(
        "缺少 hdf5plugin，无法读取 DSEC 的 events.h5（压缩过滤器需要）。\n"
        "安装方式二选一：\n"
        "  pip install -U hdf5plugin\n"
        "  conda install -c conda-forge hdf5plugin\n"
    ) from e

import csv
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from semseg.models.modules.flow_network.eraft.eraft import ERAFT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1) Load ERAFT checkpoint (strict=True 失败就 strict=False，跟你 dummy 一致)
# ============================================================
def load_eraft_dsec(ckpt_path: str):
    ckpt_raw = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt_raw.get("model", ckpt_raw.get("state_dict", ckpt_raw))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    if "fnet.conv1.weight" not in sd:
        raise KeyError("state_dict 里找不到 fnet.conv1.weight，检查 checkpoint 是否正确")
    cin = sd["fnet.conv1.weight"].shape[1]
    print(f"[ckpt] expected input channels Cin = {cin}")

    net = ERAFT(n_first_channels=cin).to(device)

    try:
        net.load_state_dict(sd, strict=True)
        print("[ckpt] loaded with strict=True ✅")
    except RuntimeError as e:
        print("[ckpt] strict=True failed, fallback to strict=False (跟你 dummy 一样)")
        print(str(e)[:300], "...\n")
        msg = net.load_state_dict(sd, strict=False)
        print("[ckpt] missing keys:", msg.missing_keys[:5], "..." if len(msg.missing_keys) > 5 else "")
        print("[ckpt] unexpected keys:", msg.unexpected_keys[:5], "..." if len(msg.unexpected_keys) > 5 else "")

    net.eval()
    return net, cin


# ============================================================
# 2) CSV timestamps
# ============================================================
def read_timestamp_pairs(csv_path):
    pairs = []
    with open(csv_path, "r") as f:
        rows = list(csv.reader(f))
    start = 0
    if len(rows) and (not rows[0][0].strip().lstrip("-").replace(".", "").isdigit()):
        start = 1
    for r in rows[start:]:
        if len(r) < 2:
            continue
        t0 = int(float(r[0]))
        t1 = int(float(r[1]))
        pairs.append((t0, t1))
    return pairs


# ============================================================
# 3) Find events datasets
# ============================================================
def find_event_group(h5: h5py.File):
    if all(k in h5.keys() for k in ["x", "y", "t", "p"]):
        return h5, "x", "y", "t", "p"
    if "events" in h5 and isinstance(h5["events"], h5py.Group):
        g = h5["events"]
        t_key = "t" if "t" in g.keys() else ("ts" if "ts" in g.keys() else None)
        if t_key and all(k in g.keys() for k in ["x", "y", "p"]):
            return g, "x", "y", t_key, "p"
    for k in h5.keys():
        if isinstance(h5[k], h5py.Group):
            g = h5[k]
            t_key = "t" if "t" in g.keys() else ("ts" if "ts" in g.keys() else None)
            if t_key and all(kk in g.keys() for kk in ["x", "y", "p"]):
                return g, "x", "y", t_key, "p"
    raise KeyError(f"events.h5 里没找到 events 结构。keys={list(h5.keys())}")


# ============================================================
# 4) ms_to_idx + time-unit 推断（去掉 offset，用 duration 推）
# ============================================================
def get_ms_to_idx(g: h5py.Group):
    if "ms_to_idx" in g:
        return g["ms_to_idx"]
    if "ms_to_idx" in g.file:
        return g.file["ms_to_idx"]
    raise KeyError("找不到 ms_to_idx")


def infer_ms_div_from_duration(t_first: int, t_last: int, n_ms: int):
    duration = max(t_last - t_first, 1)
    ms_div_est = duration / max(n_ms, 1)

    cand = [1000, 1_000_000]  # us / ns
    best = min(cand, key=lambda c: abs(ms_div_est - c))
    if abs(ms_div_est - best) / max(best, 1) > 0.2:
        return int(round(ms_div_est))
    return int(best)


def choose_csv_offset(pairs, t_first, t_last):
    t0_first = pairs[0][0]
    candidates = [
        0,
        t_first - t0_first,   # 让第一对 t0 对齐到 t_first
        t_first,              # 如果 csv 从 0 开始
        -t_first,
    ]

    def score(off):
        ok = 0
        K = min(10, len(pairs))
        for i in range(K):
            a, b = pairs[i]
            a2, b2 = a + off, b + off
            if (t_first <= a2 <= t_last) and (t_first <= b2 <= t_last):
                ok += 1
        return ok

    scores = [(score(off), off) for off in candidates]
    scores.sort(reverse=True, key=lambda x: x[0])
    best_ok, best_off = scores[0]
    return best_off, scores


# ============================================================
# 5) Slice events (ms index 用相对 t_first)
# ============================================================
def slice_events(g, kx, ky, kt, kp, t0_abs, t1_abs, ms_to_idx, ms_div, t_base, t_first, t_last):
    if t1_abs <= t0_abs:
        return (np.empty((0,), np.int32),
                np.empty((0,), np.int32),
                np.empty((0,), np.int64),
                np.empty((0,), np.int8))

    # clamp time to event range
    t0_abs = max(int(t0_abs), int(t_first))
    t1_abs = min(int(t1_abs), int(t_last))
    if t1_abs <= t0_abs:
        return (np.empty((0,), np.int32),
                np.empty((0,), np.int32),
                np.empty((0,), np.int64),
                np.empty((0,), np.int8))

    rel0 = max(int(t0_abs - t_base), 0)
    rel1 = max(int(t1_abs - t_base), 0)

    max_ms = len(ms_to_idx) - 1
    ms0 = int(rel0 // ms_div)
    ms1 = int(rel1 // ms_div) + 1
    ms0 = max(0, min(ms0, max_ms))
    ms1 = max(0, min(ms1, max_ms))

    i0 = int(ms_to_idx[ms0])
    i1 = int(ms_to_idx[ms1])

    x = g[kx][i0:i1]
    y = g[ky][i0:i1]
    t = g[kt][i0:i1]
    p = g[kp][i0:i1]

    m = (t >= t0_abs) & (t < t1_abs)
    return x[m], y[m], t[m], p[m]


# ============================================================
# 6) Rectify map（关键修复：保持长度不变，无效点置 -1）
# ============================================================
def load_rectify_map(rectify_h5_path: str):
    if not os.path.exists(rectify_h5_path):
        return None, None
    with h5py.File(rectify_h5_path, "r") as f:
        for kx, ky in [("map_x", "map_y"), ("rectify_map_x", "rectify_map_y"), ("x_map", "y_map")]:
            if kx in f and ky in f:
                return f[kx][:].astype(np.float32), f[ky][:].astype(np.float32)
        for k in f.keys():
            arr = f[k][:]
            if arr.ndim == 3 and (arr.shape[-1] == 2 or arr.shape[0] == 2):
                if arr.shape[-1] == 2:
                    return arr[..., 0].astype(np.float32), arr[..., 1].astype(np.float32)
                if arr.shape[0] == 2:
                    return arr[0].astype(np.float32), arr[1].astype(np.float32)
    return None, None


def apply_rectify_keep_len(x, y, map_x, map_y):
    """
    输入 x,y 长度 N
    输出 xr,yr 仍然长度 N
    无效点 -> -1（后面 voxelize 自己用 mask 丢掉）
    """
    if map_x is None or map_y is None:
        return x.astype(np.int32), y.astype(np.int32)

    H, W = map_x.shape
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    N = x.shape[0]

    xr = np.full((N,), -1, dtype=np.int32)
    yr = np.full((N,), -1, dtype=np.int32)

    m0 = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not np.any(m0):
        return xr, yr

    idx0 = np.nonzero(m0)[0]
    x0 = x[idx0]
    y0 = y[idx0]

    rx = map_x[y0, x0]
    ry = map_y[y0, x0]

    m1 = np.isfinite(rx) & np.isfinite(ry)
    if not np.any(m1):
        return xr, yr

    idx1 = idx0[m1]
    rx1 = np.rint(rx[m1]).astype(np.int32)
    ry1 = np.rint(ry[m1]).astype(np.int32)

    m2 = (rx1 >= 0) & (rx1 < W) & (ry1 >= 0) & (ry1 < H)
    if not np.any(m2):
        return xr, yr

    idx2 = idx1[m2]
    xr[idx2] = rx1[m2]
    yr[idx2] = ry1[m2]
    return xr, yr


# ============================================================
# 7) Voxelize (Cin bins) —— 现在保证 x/y/t/p 长度一致就不会炸
# ============================================================
def events_to_voxel_bincount(x, y, t, p, H, W, C, t0, t1, normalize=True):
    voxel = np.zeros((C, H, W), dtype=np.float32)
    if len(t) == 0:
        return torch.from_numpy(voxel[None]).to(device)

    x = x.astype(np.int32)
    y = y.astype(np.int32)
    t = t.astype(np.int64)

    p = p.astype(np.int16)
    pol = np.where(p > 0, 1.0, -1.0).astype(np.float32)

    denom = float(max(t1 - t0, 1))
    tn = (t.astype(np.float64) - float(t0)) / denom
    b = np.floor(tn * C).astype(np.int32)
    b = np.clip(b, 0, C - 1)

    # 这里会自动把 rectify 后 -1 的点丢掉
    m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, b, pol = x[m], y[m], b[m], pol[m]

    HW = H * W
    idx = b * HW + y * W + x
    acc = np.bincount(idx, weights=pol, minlength=C * HW).astype(np.float32)
    voxel = acc.reshape(C, H, W)

    if normalize:
        mx = np.max(np.abs(voxel))
        if mx > 0:
            voxel /= mx

    return torch.from_numpy(voxel[None]).to(device)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    DSEC_TEST_ROOT = "/work/sme-wangzr/wan_tao/projects/segment_anytime_v1/dataset/DSEC/test"
    SEQ_NAME = "interlaken_00_b"
    CKPT = "checkpoints/pretrained/flownet/dsec.tar"

    N_RUN = 10
    INPUT_SCALE = 1.0
    USE_RECTIFY = True
    DT_MS = 50  # 50ms window

    seq_dir = os.path.join(DSEC_TEST_ROOT, SEQ_NAME)
    events_h5 = os.path.join(seq_dir, "events.h5")
    rectify_h5 = os.path.join(seq_dir, "rectify_map.h5")
    ts_csv = os.path.join(seq_dir, "test_forward_flow_timestamps.csv")

    assert os.path.exists(events_h5), f"missing: {events_h5}"
    assert os.path.exists(ts_csv), f"missing: {ts_csv}"

    flow_net, Cin = load_eraft_dsec(CKPT)

    pairs = read_timestamp_pairs(ts_csv)
    print(f"[data] timestamp pairs = {len(pairs)} from {ts_csv}")

    map_x, map_y = (None, None)
    if USE_RECTIFY:
        map_x, map_y = load_rectify_map(rectify_h5)

    if map_x is not None:
        H, W = map_x.shape
        print(f"[rectify] loaded rectify_map: H={H}, W={W}")
    else:
        H, W = 480, 640
        print(f"[rectify] no rectify_map, fallback H={H}, W={W}")

    out_dir = os.path.join("outputs_eraft", SEQ_NAME)
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(events_h5, "r") as h5, torch.no_grad():
        g, kx, ky, kt, kp = find_event_group(h5)

        ms_to_idx = get_ms_to_idx(g)
        t_first = int(g[kt][0])
        t_last = int(g[kt][-1])
        n_ms = len(ms_to_idx) - 1

        ms_div = infer_ms_div_from_duration(t_first, t_last, n_ms)
        dt = DT_MS * ms_div

        off, scores = choose_csv_offset(pairs, t_first, t_last)
        print(f"[time] events t_first={t_first}, t_last={t_last}, duration={t_last-t_first}")
        print(f"[time] ms_to_idx len={len(ms_to_idx)} => inferred ms_div={ms_div}, dt({DT_MS}ms)={dt}")
        print(f"[time] chosen csv offset={off}, candidates(scores)={scores}")

        for i, (t0_csv, t1_csv) in enumerate(pairs[:N_RUN]):
            t0 = t0_csv + off
            t1 = t1_csv + off

            a0, a1 = t0 - dt, t0
            b0, b1 = t1 - dt, t1

            x0, y0, tt0, p0 = slice_events(g, kx, ky, kt, kp, a0, a1, ms_to_idx, ms_div, t_first, t_first, t_last)
            x1, y1, tt1, p1 = slice_events(g, kx, ky, kt, kp, b0, b1, ms_to_idx, ms_div, t_first, t_first, t_last)

            raw0, raw1 = len(tt0), len(tt1)

            if USE_RECTIFY and (map_x is not None):
                x0r, y0r = apply_rectify_keep_len(x0, y0, map_x, map_y)
                x1r, y1r = apply_rectify_keep_len(x1, y1, map_x, map_y)
            else:
                x0r, y0r = x0.astype(np.int32), y0.astype(np.int32)
                x1r, y1r = x1.astype(np.int32), y1.astype(np.int32)

            # voxelize（注意：这里 t0/t1 用 clamp 后的窗口边界更稳）
            a0c, a1c = max(a0, t_first), min(a1, t_last)
            b0c, b1c = max(b0, t_first), min(b1, t_last)

            ev0 = events_to_voxel_bincount(x0r, y0r, tt0, p0, H, W, Cin, a0c, a1c, normalize=True)
            ev1 = events_to_voxel_bincount(x1r, y1r, tt1, p1, H, W, Cin, b0c, b1c, normalize=True)

            print(
                f"[{i}] csvΔt={t1_csv-t0_csv} absΔt={t1-t0} "
                f"raw_events=({raw0},{raw1}) "
                f"ev_absmax=({ev0.abs().max().item():.3f},{ev1.abs().max().item():.3f}) "
                f"nonzero=({(ev0!=0).sum().item()},{(ev1!=0).sum().item()})"
            )

            # ERAFT inference
            if INPUT_SCALE != 1.0:
                ev0s = F.interpolate(ev0, scale_factor=INPUT_SCALE, mode="bilinear", align_corners=False)
                ev1s = F.interpolate(ev1, scale_factor=INPUT_SCALE, mode="bilinear", align_corners=False)
                _, flow_preds = flow_net(ev0s, ev1s)
                flow = flow_preds[-1]
                flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False) * (1.0 / INPUT_SCALE)
            else:
                _, flow_preds = flow_net(ev0, ev1)
                flow = flow_preds[-1]

            print(f"[{i}] flow={tuple(flow.shape)} mean={flow.mean().item():.4f} std={flow.std().item():.4f}")

            flow_np = flow[0].permute(1, 2, 0).float().cpu().numpy()
            np.save(os.path.join(out_dir, f"{i:06d}_t0_{t0_csv}_t1_{t1_csv}.npy"), flow_np)

    print("done.")
