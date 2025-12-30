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
    pattern = re.compile(r'(\d+)_gtFine_labelTrainIds11\.png')

    match = pattern.match(filename)
    if match:
        # 提取编号部分
        number = match.group(1)
        # 构建新的文件名
        new_filename = f'{int(number)+idx_diff:06d}_gtFine_labelTrainIds11.png'
        # 构建完整的路径名
        new_filepath = filepath.replace(f'{number}_gtFine_labelTrainIds11.png', new_filename)
        return new_filepath
    return None

class DSEC(Dataset):
    # 定义类别和调色板的字典
    SEGMENTATION_CONFIGS = {
        11: {
            "CLASSES": [
                "background", "building", "fence", "person", "pole",
                "road", "sidewalk", "vegetation", "car", "wall",
                "traffic sign",
            ],
            "PALETTE": torch.tensor([
                [0, 0, 0], [70, 70, 70], [190, 153, 153], [220, 20, 60], [153, 153, 153], 
                [128, 64, 128], [244, 35, 232], [107, 142, 35], [0, 0, 142], [102, 102, 156], 
                [220, 220, 0],
            ])
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
        19: {
            "CLASSES": [
                'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 
                'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
            ],
            "PALETTE": torch.tensor([
                [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], 
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
            ]),
            "ID2TRAINID": {
                0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 
                20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, 34: 2, 35: 4, 36: 255, 37: 5, 38: 255, 39: 255, 
                40: 255, 41: 255, 42: 255, 43: 255, 44: 255, -1: 255
            }
        },
        23: {
            "CLASSES": [
                "None", "Building", "Fences", "Other", "Pedestrian",
                "Pole", "RoadLines", "Road", "Sidewalk", "Vegetation",
                "Vehicle", "Wall", "TrafficSign", "Sky", "Ground",
                "Bridge", "RailTrack", "GuardRail", "TrafficLight", "Static",
                "Dynamic", "Water", "Terrain"
            ],
            "PALETTE": torch.tensor([
                [255, 255, 255], [70, 70, 70], [100, 40, 40], [55, 90, 80], [220, 20, 60],
                [153, 153, 153], [157, 234, 50], [128, 64, 128], [244, 35, 232], [107, 142, 35],
                [0, 0, 142], [102, 102, 156], [220, 220, 0], [70, 130, 180], [81, 0, 81],
                [150, 100, 100], [230, 150, 140], [180, 165, 180], [250, 170, 30], [110, 190, 160],
                [170, 120, 50], [45, 60, 150], [145, 170, 100]
            ]),
            "ID2TRAINID": {
                0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 
                20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, 34: 2, 35: 4, 36: 255, 37: 5, 38: 255, 39: 255, 
                40: 255, 41: 255, 42: 255, 43: 255, 44: 255, -1: 255
            }
        }
    }

    def __init__(self, root: str = 'data/DSEC', split: str = 'train', n_classes: int = 11, transform = None, modals = ['img', 'event'], case = None, duration: int=0, flow_net_flag: bool=False) -> None:
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
        self.time_window = duration//50
        self.flow_net_flag = flow_net_flag
        self.iterframe_test = False
        print(f"Loading DSEC dataset with {duration}ms duration.")
        self.seg_gt_dirname = f'/gtFine_t{self.time_window}'
        # self.files = sorted(glob.glob(os.path.join(*[root, 'leftImg8bit', split, '*', '*.png'])))
        self.files = sorted(glob.glob(os.path.join(*[root, self.seg_gt_dirname[1:], split, '*', '*_gtFine_labelTrainIds11.png'])))
        # self.files = sorted(glob.glob(os.path.join(*[root, 'sample', split, '*', '*.npy'])))
        # --- debug
        # self.files = sorted(glob.glob(os.path.join(*[root, 'img', '*', split, '*', '*.png'])))[:100]
        print(f"Found {len(self.files)} {split} {case} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sample = {}
        lbl_path = str(self.files[index])
        if self.time_window != 0:
            bin = 40
            if self.iterframe_test:
                if bin==20:
                    start_t = 1
                    rgb_path = get_new_name(lbl_path, idx_diff=start_t-self.time_window).replace(self.seg_gt_dirname, f'/leftImg8bit_t{start_t}').replace('_gtFine_labelTrainIds11.png', '.png')
                    ### event ###
                    event_path = get_new_name(lbl_path, idx_diff=0-self.time_window).replace(self.seg_gt_dirname, f'/event_t0_t{self.time_window}/event_40').replace('_gtFine_labelTrainIds11.png', '.npy')
                    event_voxel = np.load(event_path, allow_pickle=True)
                    if start_t == 1:
                        event_voxel = event_voxel[20:]
                    sample['event'] = torch.from_numpy(event_voxel[:, :440])
                    ### flow ###
                    if not self.flow_net_flag:
                        flow_path_t0_t1 = get_new_name(lbl_path, idx_diff=0-self.time_window).replace(self.seg_gt_dirname, f'/flow_t0_t1').replace('_gtFine_labelTrainIds11.png', '.npy')
                        flow_path_t1_t2 = get_new_name(lbl_path, idx_diff=1-self.time_window).replace(self.seg_gt_dirname, f'/flow_t1_t2').replace('_gtFine_labelTrainIds11.png', '.npy')
                        flow_t0_t1 = np.load(flow_path_t0_t1, allow_pickle=True)
                        flow_t1_t2 = np.load(flow_path_t1_t2, allow_pickle=True)
                        if start_t == 1:
                            flow = flow_t1_t2
                        elif start_t == 0:
                            flow = np.concatenate([flow_t0_t1, flow_t1_t2], axis=0)
                        sample['flow'] = torch.from_numpy(flow[:, :440])
            else:
                rgb_path = get_new_name(lbl_path, idx_diff=0-self.time_window).replace(self.seg_gt_dirname, f'/leftImg8bit_t0').replace('_gtFine_labelTrainIds11.png', '.png')
                ### event ###
                event_path = get_new_name(lbl_path, idx_diff=-0-self.time_window).replace(self.seg_gt_dirname, f'/event_t0_t{self.time_window}/event_{bin}').replace('_gtFine_labelTrainIds11.png', '.npy')
                event_voxel = np.load(event_path, allow_pickle=True)
                sample['event'] = torch.from_numpy(event_voxel[:, :440])
                ### flow ###
                if not self.flow_net_flag:
                    flow_path = rgb_path.replace('/leftImg8bit_t0', f'/flow_t0_t{self.time_window}').replace('.png', '.npy')
                    flow = np.load(flow_path, allow_pickle=True)
                    sample['flow'] = torch.from_numpy(flow[:, :440])
        else:
            rgb_path = lbl_path.replace(self.seg_gt_dirname, '/leftImg8bit_t0').replace('_gtFine_labelTrainIds11.png', '.png')

        # lbl_path_t0 = get_new_name(lbl_path, idx_diff=-self.time_window).replace(self.seg_gt_dirname, '/gtFine_t0')
        # rgb_ref = lbl_path.replace(self.seg_gt_dirname, '/leftImg8bit_next').replace('_gtFine_labelTrainIds11.png', '.png')
        # flow_inverse = rgb_ref.replace('/leftImg8bit_next', '/flow_reverse').replace('.png', '.npy')

        if self.n_classes == 12:
            lbl_path = lbl_path.replace('_gtFine_labelTrainIds11.png', '_gtFine_labelTrainIds12.png')
        elif self.n_classes == 19:
            lbl_path = lbl_path.replace('_gtFine_labelTrainIds11.png', '_gtFine_labelTrainIds.png')
        # lbl_path = lbl_path.split('.')[0]  # 获取文件名的基础部分（去掉扩展名）
        # lbl_path = f"{lbl_path}_gtFine_labelTrainIds11.png"  # 添加后缀并重新组合
        seq_name = Path(rgb_path).parts[-2]
        seq_idx = Path(rgb_path).parts[-1].split('_')[0]

        sample['img'] = io.read_image(rgb_path)[:3, ...][:, :440]
        # H, W = sample['img'].shape[1:]
        # sample['img_next'] = io.read_image(rgb_ref)[:3, ...][:, :440]
        label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        # label_ref = io.read_image(lbl_path_t0)[0,...].unsqueeze(0)
        sample['mask'] = label[:, :440]
        # sample['mask_cur'] = label_ref[:, :440]

        # # # save dict
        # # np.save(event_path.replace(f'/event_t0_t{self.time_window}/event_20', '/sample'), sample)
        # sample_path = str(self.files[index])
        # sample = np.load(sample_path, allow_pickle=True).item()
        # # dict_keys(['img', 'img_next', 'mask', 'mask_cur', 'event', 'flow', 'flow_inverse'])
        # seq_name = Path(sample_path).parts[-2]
        # seq_idx = Path(sample_path).parts[-1].split('_')[0]
        # # bin = 5
        # # sample['event'] = torch.cat([sample['event'][bin*i:bin*(i+1)].mean(0).unsqueeze(0) for i in range(20//bin)], dim=0)

        if self.transform:
            sample = self.transform(sample)

        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        # label_ref = sample['mask_cur']
        # del sample['mask_cur']
        # label_ref = self.encode(label_ref.squeeze().numpy()).long()

        if not self.flow_net_flag and self.time_window!=0:
            flow = sample['flow']
            del sample['flow']

        if self.time_window != 0:
            event_voxel = sample['event']
            del sample['event']
        # img_next = sample['img_next']
        # del sample['img_next']

        # flow_inverse = sample['flow_inverse']
        # del sample['flow_inverse']

        sample = [sample[k] for k in self.modals]
        if self.time_window != 0:
            sample.append(event_voxel)
            if not self.flow_net_flag:
                sample.append(flow)
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

class ExtendedDSEC(DSEC):
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
            flow_net_flag=original_dataset.flow_net_flag
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