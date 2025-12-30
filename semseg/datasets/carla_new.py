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

class CarlaNew(Dataset):
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

    def __init__(self, root: str = 'data/DSEC', split: str = 'train', n_classes: int = 11, transform = None, modals = ['img', 'event'], case = None, duration: int=None, flow_net_flag: bool=False, dataset_type: str=None) -> None:
        super().__init__()
        self.root = root
        self.split = split
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = n_classes
        self.ignore_label = 255
        self.modals = modals
        self.case = case

        self.duration = duration
        print(f"Loading SDSEC dataset with {duration}ms duration.")
        self.index_window = self.duration//10
        self.bin = 20
    
        self.flow_net_flag = flow_net_flag
        self.dataset_type = dataset_type
        self.iterframe_test = False

        print("Root: ", self.root)
        self.files = sorted(glob.glob(os.path.join(*[root, split, '*', 'image', '*.png'])))
        self.files = self.files[::10]  # 每隔十个文件取一个

        print(f"Found {len(self.files)} {split} images.")


    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sample = {}
        dt = self.index_window
        # dt = 5  # for interframe test
        # /mnt/sdc/lxy/datasets/carla_new_100_N/train/Town01/image/00499401.png
        seq_idx = int(self.files[index].split('/')[-1].split('.')[0])
        seq_name = self.files[index].split('/')[-3]
        # print('seq_name:', seq_name)
        rgb_path = str(self.files[index])
        seg_path = os.path.join(self.root, self.split, seq_name, 'label', f'{seq_idx+dt:08d}.png')
        sample['img'] = io.read_image(rgb_path)[:3, ...][:, :440]
        sample['mask'] = io.read_image(seg_path)[0,...].unsqueeze(0)[:, :440]
        if self.duration != 0:
            event_path = os.path.join(self.root, self.split, seq_name, 'event', f'{seq_idx:08d}.npy')
            event_path_before = os.path.join(self.root, self.split, seq_name, 'event', f'{seq_idx-5:08d}.npy')
            event_path_after = os.path.join(self.root, self.split, seq_name, 'event', f'{seq_idx+5:08d}.npy')
            # sample['flow'] = torch.zeros((2, 440, 640))
            # for i in range(5):
            #     flow_path = os.path.join(self.root, self.split, seq_name, 'flow', f'{seq_idx+i:08d}.npy')
            #     if not os.path.exists(flow_path):
            #         continue
            #     sample['flow'] += torch.from_numpy(
            #         np.load(flow_path, allow_pickle=True)[:, :440]
            #     )
            event_voxel = torch.from_numpy(np.load(event_path, allow_pickle=True)[:, :440])
            # event_voxel = event_voxel[:4*dt]
            # event_voxel = torch.cat([event_voxel[dt*i:dt*(i+1)].mean(0).unsqueeze(0) for i in range(4)], dim=0)
            # event_voxel = event_voxel.repeat(5, 1, 1)

            event_voxel_before = np.load(event_path_before, allow_pickle=True)[:, :440]
            event_voxel_after = np.load(event_path_after, allow_pickle=True)[:, :440]
            sample['event'] = event_voxel
            sample['event_before'] = torch.from_numpy(event_voxel_before)
            sample['event_after'] = torch.from_numpy(event_voxel_after)

        for k in sample:
            if sample[k].shape[1] != 440 or sample[k].shape[2] != 640:
                print('k:', k, sample[k].shape)
        if self.transform:
            sample = self.transform(sample)

        label = sample['mask']
        del sample['mask']
        label = [self.encode(label.squeeze().numpy()).long()]

        # if not self.flow_net_flag and self.time_window!=0:
        #     flow = sample['flow']
        #     del sample['flow']

        if self.duration != 0:
            event_voxel = sample['event']
            del sample['event']
            event_voxel_before = sample['event_before']
            del sample['event_before']
            if self.index_window == 10:
                event_voxel_after = sample['event_after']
                del sample['event_after']
        # img_next = sample['img_next']
        # del sample['img_next']

        # flow = sample['flow']
        # del sample['flow']

        sample = [sample[k] for k in self.modals]
        if self.duration != 0:
            sample.append(event_voxel)
            sample.append(event_voxel_before)
            sample.append(event_voxel_after)
            # sample.append(flow)
        return seq_name, seq_idx, sample, label

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

class ExtendedCarlaNew(CarlaNew):
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
        # 如果 idx 超过原始数据长度，则循环使用原始数据
        return super().__getitem__(index % self.original_length)
    
if __name__ == '__main__':
    cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    traintransform = get_train_augmentation((1024, 1024), seg_fill=255)
    for case in cases:

        trainset = DELIVER(transform=traintransform, split='val', case=case)
        trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=False, pin_memory=False)

        for i, (sample, lbl) in enumerate(trainloader):
            print(torch.unique(lbl))