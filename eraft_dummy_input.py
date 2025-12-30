import torch
import torch.nn.functional as F
from semseg.models.modules.flow_network.eraft.eraft import ERAFT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_eraft_dsec(ckpt_path: str):
    ckpt_raw = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt_raw.get("model", ckpt_raw.get("state_dict", ckpt_raw))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # checkpoint 期望的输入通道数（别猜，直接从权重读）
    if "fnet.conv1.weight" not in sd:
        raise KeyError("state_dict 里找不到 fnet.conv1.weight，检查你加载的是不是 ERAFT 的 dsec.tar")
    cin = sd["fnet.conv1.weight"].shape[1]
    print(f"[ckpt] expected input channels Cin = {cin}")

    net = ERAFT(n_first_channels=cin).to(device)

    # 尽量严格加载；如果真不匹配，把 missing/unexpected 打出来
    try:
        net.load_state_dict(sd, strict=True)
        print("[ckpt] loaded with strict=True ✅")
    except RuntimeError as e:
        print("[ckpt] strict=True failed, fallback to strict=False")
        msg = str(e)
        print(msg)
        ret = net.load_state_dict(sd, strict=False)
        print("[ckpt] missing keys:", ret.missing_keys)
        print("[ckpt] unexpected keys:", ret.unexpected_keys)

    net.eval()
    return net, cin


def adapt_event_bins(ev: torch.Tensor, target_c: int) -> torch.Tensor:
    """
    ev: [B, bins, H, W]
    返回: [B, target_c, H, W]
    说明：
      - 真正推理/训练：最好在 voxelization 直接生成 target_c 个 bins
      - 这里只是为了“dummy 能跑通 + 通道数匹配权重”
    """
    B, bins, H, W = ev.shape
    if bins == target_c:
        return ev

    # 优先：能整除就分组平均（最合理的无参数降维）
    if bins % target_c == 0:
        g = bins // target_c
        ev = ev.view(B, target_c, g, H, W).mean(dim=2)
        return ev

    # 退一步：用 1D 插值把 bins 拉到 target_c（只是为了跑通，不保证语义对齐）
    # [B, bins, H, W] -> [B*H*W, 1, bins] -> interpolate -> [B, target_c, H, W]
    x = ev.permute(0, 2, 3, 1).contiguous().view(B * H * W, 1, bins)
    x = F.interpolate(x, size=target_c, mode="linear", align_corners=False)
    x = x.view(B, H, W, target_c).permute(0, 3, 1, 2).contiguous()
    return x


# ========= main =========
flow_net, cin = load_eraft_dsec("checkpoints/pretrained/flownet/dsec.tar")

# dummy 事件体素：你换成真实数据就行
B, H, W = 1, 440, 640
bins = 20
ev1 = torch.randn(B, bins, H, W, device=device)
ev2 = torch.randn(B, bins, H, W, device=device)

# 让输入通道数匹配 checkpoint
ev1 = adapt_event_bins(ev1, cin)
ev2 = adapt_event_bins(ev2, cin)

with torch.no_grad():
    flow_low, flow_preds = flow_net(ev1, ev2)
    flow_pred = flow_preds[-1]

print("flow_low shape:", flow_low.shape)      # 通常 [B,2,H/8,W/8]（未 upsample 的内部网格流）
print("flow_pred shape:", flow_pred.shape)    # 通常 [B,2,H,W]（每次迭代 upsample 的预测）
print("num preds:", len(flow_preds))
print("flow_pred stats:", flow_pred.mean().item(), flow_pred.std().item())
