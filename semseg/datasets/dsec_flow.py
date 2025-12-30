import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF 
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm import get_train_augmentation
import re
import random
from PIL import Image
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None  # type: ignore

from scipy.ndimage import gaussian_filter

def backwarp(tenIn, tenFlow):
    tenHor = torch.linspace(start=-1.0, end=1.0, steps=tenFlow.shape[2], dtype=tenFlow.dtype).view(1, 1, -1).repeat(1, tenFlow.shape[1], 1)
    tenVer = torch.linspace(start=-1.0, end=1.0, steps=tenFlow.shape[1], dtype=tenFlow.dtype).view(1, -1, 1).repeat(1, 1, tenFlow.shape[2])
    tenGrid = torch.cat([tenHor, tenVer], 0)

    tenFlow = torch.cat([tenFlow[0:1, :, :] / ((tenIn.shape[2] - 1.0) / 2.0), tenFlow[1:2, :, :] / ((tenIn.shape[1] - 1.0) / 2.0)] , 0)

    return torch.nn.functional.grid_sample(input=tenIn.unsqueeze(0), grid=(tenGrid + tenFlow).permute(1,2,0).unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(0)
# end

def compute_photometric_consistency(I0, I1, F0to1):
    """计算光度一致性 ψ_photo"""
    warped_I1 = backwarp(I1, F0to1)
    diff = I0 - warped_I1
    psi_photo = torch.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)
    return psi_photo

def compute_flow_consistency(F0to1, F1to0):
    """计算光流一致性 ψ_flow"""
    # 反向映射光流 F1to0
    warped_F1to0 = backwarp(F1to0, F0to1)
    # 计算一致性
    diff = F0to1 - warped_F1to0
    psi_flow = torch.sqrt(diff[0]**2+diff[1]**2)
    return psi_flow

def compute_flow_variance(F0to1):
    """计算光流方差 ψ_varia"""
    F_squared = F0to1 ** 2
    G_F_squared = gaussian_filter(F_squared, sigma=1)
    
    G_F = gaussian_filter(F0to1, sigma=1)
    
    variance = torch.from_numpy(G_F_squared - (G_F ** 2))
    
    psi_varia = torch.sqrt(variance[0]+variance[1])
    return psi_varia

def get_new_name(filepath, idx_diff):
    # 正则表达式匹配文件名中的编号部分
    filename = os.path.basename(filepath)
    pattern = re.compile(r'(\d+)_gtFine_labelTrainIds')

    match = pattern.match(filename)
    if match:
        # 提取编号部分
        number = match.group(1)
        # 构建新的文件名
        new_filename = f'{int(number)+idx_diff:06d}_gtFine_labelTrainIds'
        # 构建完整的路径名
        new_filepath = filepath.replace(f'{number}_gtFine_labelTrainIds', new_filename)
        return new_filepath
    return None

class DSEC_Flow(Dataset):           # 核心任务：配对时间戳，确保“当前时刻的事件”、“上一时刻的事件”和“当前时刻的光流真值”能够正确对应。
    # 定义类别和调色板的字典
    SEGMENTATION_CONFIGS = {
        9: {
            "CLASSES": [
                "background", "building", "person", "pole",
                "road", "sidewalk", "vegetation", "car",
                "traffic sign",
            ],
            # "PALETTE": torch.tensor([
            #     [0, 0, 0], [70, 70, 70], [220, 20, 60], [153, 153, 153], 
            #     [128, 64, 128], [244, 35, 232], [107, 142, 35], [0, 0, 142]
            #     [220, 220, 0],
            # ])
            # dict palette
            "PALETTE": dict(
                {0: [0, 0, 0], 1: [70, 70, 70], 2: [220, 20, 60], 3: [153, 153, 153],
                4: [128, 64, 128], 5: [244, 35, 232], 6: [107, 142, 35], 7: [0, 0, 142],
                8: [220, 220, 0]}
            )
        },
        10: {
            "CLASSES": [
                "background", "building", "fence", "person", "pole",
                "road", "sidewalk", "vegetation", "car", 
                "traffic sign",
            ],
            # "PALETTE": torch.tensor([
            #     [0, 0, 0], [70, 70, 70], [190, 153, 153], [220, 20, 60], [153, 153, 153], 
            #     [128, 64, 128], [244, 35, 232], [107, 142, 35], [0, 0, 142], [102, 102, 156], 
            #     [220, 220, 0],
            # ])
            # dict palette
            "PALETTE": dict(
                {0: [0, 0, 0], 1: [70, 70, 70], 2: [190, 153, 153], 3: [220, 20, 60], 4: [153, 153, 153],
                5: [128, 64, 128], 6: [244, 35, 232], 7: [107, 142, 35], 8: [0, 0, 142],
                9: [220, 220, 0]}
            )
        },
        11: {
            "CLASSES": [
                "background", "building", "fence", "person", "pole",
                "road", "sidewalk", "vegetation", "car", "wall",
                "traffic sign",
            ],
            # "PALETTE": torch.tensor([
            #     [0, 0, 0], [70, 70, 70], [190, 153, 153], [220, 20, 60], [153, 153, 153], 
            #     [128, 64, 128], [244, 35, 232], [107, 142, 35], [0, 0, 142], [102, 102, 156], 
            #     [220, 220, 0],
            # ])
            # dict palette
            "PALETTE": dict(
                {0: [0, 0, 0], 1: [70, 70, 70], 2: [190, 153, 153], 3: [220, 20, 60], 4: [153, 153, 153],
                5: [128, 64, 128], 6: [244, 35, 232], 7: [107, 142, 35], 8: [0, 0, 142], 9: [102, 102, 156],
                10: [220, 220, 0], 11: [55, 90, 80], 12: [157, 234, 50]}
            )
        },
        12: {
            "CLASSES": [
                "background", "building", "fence", "person", "pole",
                "road", "sidewalk", "vegetation", "car", "wall",
                "traffic sign", "curb",
            ],
            "PALETTE": torch.tensor([
                [0, 0, 0], [70, 70, 70], [190, 153, 153], [220, 20, 60], [153, 153, 153], 
                [128, 64, 128], [244, 35, 232], [107, 142, 35], [0, 0, 142], [102, 102, 156], 
                [220, 220, 0], [255, 170, 255],
            ])
        },
       13: {
            "CLASSES": [
                "background", "building", "fence", "person", "pole",
                "road", "sidewalk", "vegetation", "car", "wall",
                "traffic sign", "other", "roadline"
            ],
            # "PALETTE": torch.tensor([
            #     [0, 0, 0], [70, 70, 70], [190, 153, 153], [220, 20, 60], [153, 153, 153], 
            #     [128, 64, 128], [244, 35, 232], [107, 142, 35], [0, 0, 142], [102, 102, 156], 
            #     [220, 220, 0], [55, 90, 80], [157, 234, 50]
            # ])
            # dict palette
            "PALETTE": dict(
                {0: [0, 0, 0], 1: [70, 70, 70], 2: [190, 153, 153], 3: [220, 20, 60], 4: [153, 153, 153],
                5: [128, 64, 128], 6: [244, 35, 232], 7: [107, 142, 35], 8: [0, 0, 142], 9: [102, 102, 156],
                10: [220, 220, 0], 11: [55, 90, 80], 12: [157, 234, 50], 255: [255, 255, 255]}
            )
        },
        19: {
            "CLASSES": [
                'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 
                'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
            ],
            "PALETTE": torch.tensor([
                [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], 
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
            ])
        },


    }
                # root: 数据集在硬盘上的根目录， # split: 'train' 或 'val' # n_classes: 分类任务的类别数 光流训练不用
    def __init__(self, root: str = 'data/DSEC', split: str = 'train', n_classes: int = 11, transform = None, modals = ['img', 'event'], case = None, duration: int=None, flow_net_flag: bool=False, dataset_type: str=None) -> None:
        super().__init__()
        self.root = root
        self.split = split
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = n_classes
        self.ignore_label = 255
        self.modals = modals
        self.case = case            # 场景类型
        self.train_flow = True      # 在train_mm_flow 被设置为true

        self.duration = duration
        self.time_window = duration//50
        if dataset_type == 'sdsec':
            print(f"Loading SDSEC dataset with {duration}ms duration.")
            self.index_window = self.duration//10
            self.bin = 40
        elif dataset_type == 'dsec':
            print(f"Loading DSEC dataset with {duration}ms duration.")
            # self.index_window = self.duration//50
            self.index_window = self.duration//10
            self.bin = 20                               # 确定切片的时间长度为20
    
        self.flow_net_flag = flow_net_flag
        self.dataset_type = dataset_type
        self.iterframe_test = False
        # self.seg_gt_dirname = f'/gtFine_t1_interpolation'
        # self.seg_gt_dirname = f'/gtFine_t1'
        # NOTE: For flow-only training we only need:
        # - event voxels:   event_t0_t1/event_{bin}/<split>/<scene>/<id>.npy
        # - prev voxels:    event_t-1_t0/event_{bin}/<split>/<scene>/<id_prev>.npy
        # - flow GT:        flow_t0_t{time_window}/<split>/<scene>/<id>.(npy|png)
        self.flow_gt_dirname = f'flow_t0_t{self.time_window}'               # 例如 flow_t0_t1
        self.event_t0_t1_dirname = f'event_t0_t1/event_{self.bin}'          # 当前时刻的事件
        self.event_tminus1_t0_dirname = f'event_t-1_t0/event_{self.bin}'   # 上一个时刻的事件
        # dt = 1
        # self.seg_gt_dirname = f'/gtFine_t{self.time_window}_dt{dt}'
        print("Root: ", self.root)
        flow_split_dir = Path(self.root) / self.flow_gt_dirname / split
        event_split_dir = Path(self.root) / self.event_t0_t1_dirname / split
        event_before_split_dir = Path(self.root) / self.event_tminus1_t0_dirname / split

        if not flow_split_dir.is_dir():
            raise FileNotFoundError(f"Missing flow directory: {flow_split_dir}")
        if not event_split_dir.is_dir():
            raise FileNotFoundError(f"Missing event voxel directory: {event_split_dir}")
        if not event_before_split_dir.is_dir():
            raise FileNotFoundError(f"Missing previous event voxel directory: {event_before_split_dir}")

        prev_offset = 0
        if self.time_window != 0:
            prev_offset = self.index_window // self.time_window

        self.files = []
        scenes = sorted([p for p in flow_split_dir.iterdir() if p.is_dir()])
        for scene_dir in scenes:
            scene = scene_dir.name
            # Prefer .npy flow GT; fallback to .png if needed.  优先找npy格式的光流文件
            flow_paths = sorted(scene_dir.glob('*.npy'))
            if not flow_paths:
                flow_paths = sorted(scene_dir.glob('*.png'))

            for flow_path in flow_paths:
                stem = flow_path.stem
                if not stem.isdigit():
                    continue
                flow_id = int(stem)
                ev_path = event_split_dir / scene / f"{flow_id:06d}.npy"
                if not ev_path.is_file():
                    continue

                # Prefer matching IDs for "before" events, but keep backward-compatibility
                # with datasets where the previous window is stored with an offset ID.
                ev_before_path = event_before_split_dir / scene / f"{flow_id:06d}.npy"
                if not ev_before_path.is_file():
                    prev_id = flow_id - prev_offset
                    if prev_id < 0:
                        continue
                    ev_before_path = event_before_split_dir / scene / f"{prev_id:06d}.npy"
                    if not ev_before_path.is_file():
                        continue

                # If we indexed by PNG but a corresponding NPY exists, prefer NPY.
                if flow_path.suffix.lower() == '.png':
                    flow_npy = flow_path.with_suffix('.npy')
                    if flow_npy.is_file():
                        flow_path = flow_npy

                self.files.append((scene, f"{flow_id:06d}", str(ev_path), str(ev_before_path), str(flow_path)))

        self.files.sort(key=lambda x: (x[0], x[1]))
        print(f"Found {len(self.files)} {split} flow samples.")


    def __len__(self) -> int:
        return len(self.files)
    
    def _load_flow(self, flow_path: str) -> np.ndarray: # 专门用于读取光流文件
        suffix = Path(flow_path).suffix.lower()
        if suffix == '.npy':
            flow_arr = np.load(flow_path, allow_pickle=True)
        elif suffix == '.png':
            flow_png = None
            if cv2 is not None:
                flow_png = cv2.imread(str(flow_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
                if flow_png is not None and flow_png.ndim == 3 and flow_png.shape[2] >= 3:
                    # OpenCV loads color images as BGR; DSEC/KITTI store (u,v,valid) in RGB.
                    flow_png = flow_png[..., ::-1]
            if flow_png is None and imageio is not None:
                flow_png = imageio.imread(str(flow_path))
            if flow_png is None:
                flow_png = np.array(Image.open(flow_path))
            if flow_png.ndim != 3 or flow_png.shape[2] < 2:
                raise ValueError(f"Unsupported flow PNG format: {flow_path} shape={flow_png.shape}")
            if flow_png.dtype != np.uint16:
                raise ValueError(
                    f"Expected uint16 DSEC/KITTI flow PNG, got dtype={flow_png.dtype} for {flow_path}. "
                    f"Make sure to read with cv2.IMREAD_ANYDEPTH|ANYCOLOR (or IMREAD_UNCHANGED)."
                )
            # KITTI-style 16-bit PNG encoding: (u, v, valid)
            flow_uv = flow_png[..., :2].astype(np.float32)
            flow_uv = (flow_uv - 2**15) / 128.0
            if flow_png.shape[2] >= 3:
                valid = (flow_png[..., 2] > 0).astype(np.float32)
            else:
                valid = np.ones(flow_uv.shape[:2], dtype=np.float32)
            flow_uv[valid == 0] = 0.0
            flow_2hw = np.transpose(flow_uv, (2, 0, 1))  # (2, H, W)
            flow = np.concatenate([flow_2hw, valid[None]], axis=0)  # (3, H, W) with valid mask
        else:
            raise ValueError(f"Unsupported flow file: {flow_path}")

        if suffix == '.npy':
            if flow_arr.ndim != 3:
                raise ValueError(f"Unsupported flow NPY format: {flow_path} shape={flow_arr.shape}")
            # Support common layouts:
            # - (2, H, W) or (H, W, 2): flow only
            # - (3, H, W) or (H, W, 3): flow + valid mask
            if flow_arr.shape[0] == 3:
                flow = flow_arr.astype(np.float32, copy=False)
            elif flow_arr.shape[0] == 2:
                flow_2hw = flow_arr.astype(np.float32, copy=False)
                valid = np.isfinite(flow_2hw).all(axis=0).astype(np.float32)
                flow_2hw = np.nan_to_num(flow_2hw, nan=0.0, posinf=0.0, neginf=0.0)
                flow = np.concatenate([flow_2hw, valid[None]], axis=0)
            elif flow_arr.shape[-1] == 3:
                flow_2hw = np.transpose(flow_arr[..., :2], (2, 0, 1)).astype(np.float32, copy=False)
                valid = (flow_arr[..., 2] > 0).astype(np.float32)
                flow_2hw[:, valid == 0] = 0.0
                flow = np.concatenate([flow_2hw, valid[None]], axis=0)
            elif flow_arr.shape[-1] == 2:
                flow_2hw = np.transpose(flow_arr, (2, 0, 1)).astype(np.float32, copy=False)
                valid = np.isfinite(flow_2hw).all(axis=0).astype(np.float32)
                flow_2hw = np.nan_to_num(flow_2hw, nan=0.0, posinf=0.0, neginf=0.0)
                flow = np.concatenate([flow_2hw, valid[None]], axis=0)
            else:
                raise ValueError(f"Unsupported flow NPY layout: {flow_path} shape={flow_arr.shape}")

        if flow.ndim != 3 or flow.shape[0] != 3:
            raise ValueError(f"Flow must have shape (3, H, W), got {flow.shape} for {flow_path}")
        return flow

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        seq_name, seq_idx, event_path, event_before_path, flow_path = self.files[index]

        sample = {}
        if self.time_window != 0:
            event_voxel = np.load(event_path, allow_pickle=True)                # 加载当前时刻的事件体素
            event_voxel_before = np.load(event_before_path, allow_pickle=True)  # 加载上一个时刻的事件体素
            flow = self._load_flow(flow_path)                                   # 加载光流GT

            sample['event'] = torch.from_numpy(event_voxel[:, :440])            # 原始 DSEC 事件是 640x480，但通常标签只有 640x440。
            sample['event_before'] = torch.from_numpy(event_voxel_before[:, :440])  # 因此这里裁剪到 440
            sample['flow'] = torch.from_numpy(flow[:, :440])                   

        if self.transform:      # 数据增强，如果有
            sample = self.transform(sample)

        flow = sample.pop('flow')
        if self.time_window != 0:
            event_voxel = sample.pop('event')
            event_voxel_before = sample.pop('event_before')

        sample_list = []
        if self.time_window != 0:
            sample_list.append(event_voxel)
            sample_list.append(event_voxel_before)
            sample_list.append(flow)
        return seq_name, seq_idx, sample_list  # 返回场景名，帧编号，[当前事件，过去事件，光流真值]

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)

class ExtendedDSEC_FLOW(DSEC_Flow):
    def __init__(self, original_dataset, target_length):
        """
        original_dataset: 原始的 DSEC 数据集实例
        target_length: 需要扩展到的目标长度
        """
        # 调用 DSEC 的初始化
        super().__init__(
            root=original_dataset.root,
            split=original_dataset.split,
            n_classes=original_dataset.n_classes,
            transform=original_dataset.transform,
            modals=original_dataset.modals,
            case=original_dataset.case,
            duration=original_dataset.duration,
            flow_net_flag=original_dataset.flow_net_flag,
            dataset_type=original_dataset.dataset_type
        )
        
        # 保存原始数据集的长度和目标长度
        self.original_dataset = original_dataset
        self.original_length = len(original_dataset)
        self.target_length = target_length

    def __len__(self):
        # 返回扩展后的目标长度
        return self.target_length

    def __getitem__(self, index):
        # 如果 idx 超过原始数据长度，则循环使用原始数据，比如108 % 105 = 3，去取原始数据集的第3个样本
        return super().__getitem__(index % self.original_length)
    
if __name__ == '__main__':
    cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    traintransform = get_train_augmentation((1024, 1024), seg_fill=255)
    for case in cases:

        trainset = DELIVER(transform=traintransform, split='val', case=case)
        trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=False, pin_memory=False)

        for i, (sample, lbl) in enumerate(trainloader):
            print(torch.unique(lbl))
