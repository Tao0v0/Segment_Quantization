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
import math

from scipy.ndimage import gaussian_filter


def _is_rank0() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    try:
        return torch.distributed.get_rank() == 0
    except Exception:
        return True

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

class DSEC(Dataset):
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
        self.root_path = Path(root)
        self.split = split
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = n_classes
        self.ignore_label = 255
        self.modals = modals
        self.case = case

        self.duration = duration
        self.time_window = duration//50
        if dataset_type == 'sdsec':
            if _is_rank0():
                print(f"Loading SDSEC dataset with {duration}ms duration.")
            self.index_window = self.duration//10
            self.bin = 20
        elif dataset_type == 'dsec':
            if _is_rank0():
                print(f"Loading DSEC dataset with {duration}ms duration.")
            self.index_window = self.duration//50
            self.bin = 20
    
        self.flow_net_flag = flow_net_flag
        self.dataset_type = dataset_type
        self.iterframe_test = False

        # Layout detection:
        # - legacy preprocessed layout: leftImg8bit_t*/gtFine_t*/event_*
        # - DSEC semantic export layout: images/<split>/<seq>/left/rectified + semantic/<split>/<seq>/<N>classes
        self.semantic_layout = (self.root_path / "images").is_dir() and (self.root_path / "semantic").is_dir()
        self.semantic_class_dirname = f"{self.n_classes}classes"
        # self.seg_gt_dirname = f'/gtFine_t1_interpolation'
        # self.seg_gt_dirname = f'/gtFine_t2'
        self.seg_gt_dirname = '/semantic' if self.semantic_layout else f'/gtFine_t{self.time_window}'
        # dt = 1
        # self.seg_gt_dirname = f'/gtFine_t1'
        # self.seg_gt_dirname = f'/gtFine_t{self.time_window}_dt{dt}'
        if _is_rank0():
            print("Root: ", self.root)
            print(f"Loading {self.seg_gt_dirname} segmentation ground truth.")
        # self.files = sorted(glob.glob(os.path.join(*[root, 'leftImg8bit', split, '*', '*.png'])))
        # self.n_classes = 13
        if self.semantic_layout:
            label_glob = os.path.join(
                *[root, "semantic", split, "*", self.semantic_class_dirname, "*.png"]
            )
            def _is_valid_label_path(p: str) -> bool:
                stem = Path(p).stem
                if not stem.isdigit():
                    return False
                # Need at least one preceding frame for event_t-1_t0.
                return int(stem) >= (self.time_window + 1)

            self.files = sorted(p for p in glob.glob(label_glob) if _is_valid_label_path(p))
        else:
            self.files = sorted(
                file for file in glob.glob(
                    os.path.join(*[root, self.seg_gt_dirname[1:], split, '*', f'*_gtFine_labelTrainIds{self.n_classes}.png'])
                )
                if not os.path.basename(file).startswith("000002")
            )
        # self.files = sorted(glob.glob(os.path.join(*[root, 'sample', split, '*', '*.npy'])))
        # --- debug
        # self.files = sorted(glob.glob(os.path.join(*[root, 'img', '*', split, '*', '*.png'])))[:100]

        # self.sample_dirname = 'processed'
        # self.files = sorted(glob.glob(os.path.join(*[root, self.sample_dirname, split, '*', '*.npy'])))
        if _is_rank0():
            print(f"Found {len(self.files)} {split} images.")


    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sample = {}
        lbl_path = str(self.files[index])
        if self.semantic_layout:
            # DSEC semantic export layout.
            if not self.flow_net_flag and self.time_window != 0:
                raise NotImplementedError(
                    "DSEC semantic layout currently expects FLOW_NET_FLAG=True (no precomputed flow GT support)."
                )
            label_idx = int(Path(lbl_path).stem)  # target label at t_{time_window}
            t0_idx = label_idx - self.time_window
            seq_name = Path(lbl_path).parents[1].name

            rgb_path = self.root_path / "images" / self.split / seq_name / "left" / "rectified" / f"{t0_idx:06d}.png"

            event_path = self.root_path / "event_t0_t1" / f"event_{self.bin}" / self.split / seq_name / f"{t0_idx:06d}.npy"
            event_path_before = self.root_path / "event_t-1_t0" / f"event_{self.bin}" / self.split / seq_name / f"{t0_idx - 1:06d}.npy"

            if self.time_window == 2:
                event_path_after = self.root_path / "event_t1_t2" / f"event_{self.bin}" / self.split / seq_name / f"{t0_idx + 1:06d}.npy"
                event_voxel_after = np.load(str(event_path_after), allow_pickle=True)
                sample['event_after'] = torch.from_numpy(event_voxel_after[:, :440])

                lbl_path_after = self.root_path / "semantic" / self.split / seq_name / self.semantic_class_dirname / f"{t0_idx + 1:06d}.png"
                label_after = io.read_image(str(lbl_path_after))[0, ...].unsqueeze(0)
                sample['mask_after'] = label_after[:, :440]

            event_voxel = np.load(str(event_path), allow_pickle=True)
            event_voxel_before = np.load(str(event_path_before), allow_pickle=True)
            sample['event'] = torch.from_numpy(event_voxel[:, :440])
            sample['event_before'] = torch.from_numpy(event_voxel_before[:, :440])

            # Use the event voxel resolution as the reference for all modalities.
            target_h, target_w = sample['event'].shape[-2:]

            img = io.read_image(str(rgb_path))[:3, ...]
            if img.shape[-2:] != (target_h, target_w):
                # DSEC rectified RGB frames can be 1440x1080 while semantic labels/events are 640x440.
                # Match the semantic/event resolution by resizing (keep aspect) then cropping to (target_h, target_w).
                h_img, w_img = img.shape[-2:]
                scale = max(target_h / h_img, target_w / w_img)
                new_h = int(math.ceil(h_img * scale))
                new_w = int(math.ceil(w_img * scale))
                img = TF.resize(img, [new_h, new_w], TF.InterpolationMode.BILINEAR)
                img = img[:, :target_h, :target_w]
            sample['img'] = img

            label = io.read_image(lbl_path)[0, ...].unsqueeze(0)
            if label.shape[-2:] != (target_h, target_w):
                label = TF.resize(label, [target_h, target_w], TF.InterpolationMode.NEAREST)
            sample['mask'] = label

            if self.time_window == 2:
                if sample['mask_after'].shape[-2:] != (target_h, target_w):
                    sample['mask_after'] = TF.resize(sample['mask_after'], [target_h, target_w], TF.InterpolationMode.NEAREST)
                if sample['event_after'].shape[-2:] != (target_h, target_w):
                    sample['event_after'] = TF.resize(sample['event_after'], [target_h, target_w], TF.InterpolationMode.BILINEAR)

            seq_idx = f"{t0_idx:06d}"

        elif self.time_window != 0:
            if self.iterframe_test:
                if self.bin==20:
                    start_t = 0
                    rgb_path = get_new_name(lbl_path, idx_diff=start_t-self.index_window).replace(self.seg_gt_dirname, f'/leftImg8bit_t{start_t}').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.png')
                    ### event ###
                    event_path = get_new_name(lbl_path, idx_diff=0-self.index_window).replace(self.seg_gt_dirname, f'/event_t0_t{self.time_window}/event_40').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.npy')
                    event_voxel = np.load(event_path, allow_pickle=True)
                    if start_t == 1:
                        event_voxel = event_voxel[20:]
                    sample['event'] = torch.from_numpy(event_voxel[:, :440])
                    ### flow ###
                    if not self.flow_net_flag:
                        flow_path_t0_t1 = get_new_name(lbl_path, idx_diff=0-self.index_window).replace(self.seg_gt_dirname, f'/flow_t0_t1').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.npy')
                        flow_path_t1_t2 = get_new_name(lbl_path, idx_diff=1-self.index_window).replace(self.seg_gt_dirname, f'/flow_t1_t2').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.npy')
                        flow_t0_t1 = np.load(flow_path_t0_t1, allow_pickle=True)
                        flow_t1_t2 = np.load(flow_path_t1_t2, allow_pickle=True)
                        if start_t == 1:
                            flow = flow_t1_t2
                        elif start_t == 0:
                            flow = np.concatenate([flow_t0_t1, flow_t1_t2], axis=0)
                        sample['flow'] = torch.from_numpy(flow[:, :440])
            else:
                rgb_path = get_new_name(lbl_path, idx_diff=0-self.index_window).replace(self.seg_gt_dirname, f'/leftImg8bit_t0').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.png')
                ### event ###
                # event_path = get_new_name(lbl_path, idx_diff=0-self.index_window).replace(self.seg_gt_dirname, f'/event_t0_t{self.time_window}/event_{self.bin}').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.npy')
                # event_path_before = get_new_name(lbl_path, idx_diff=0-2*self.index_window).replace(self.seg_gt_dirname, f'/event_t-{self.time_window}_t0/event_{self.bin}').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.npy')
                event_path = get_new_name(lbl_path, idx_diff=0-self.index_window).replace(self.seg_gt_dirname, f'/event_t0_t1/event_{self.bin}').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.npy')
                event_path_before = get_new_name(lbl_path, idx_diff=0-self.index_window-self.index_window//self.time_window).replace(self.seg_gt_dirname, f'/event_t-1_t0/event_{self.bin}').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.npy')
                if self.time_window == 2:
                    event_path_after = get_new_name(lbl_path, idx_diff=0-self.index_window+self.index_window//self.time_window).replace(self.seg_gt_dirname, f'/event_t1_t2/event_{self.bin}').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.npy')
                    event_voxel_after = np.load(event_path_after, allow_pickle=True)
                    sample['event_after'] = torch.from_numpy(event_voxel_after[:, :440])
                    lbl_path_after = get_new_name(lbl_path, idx_diff=0-self.index_window+self.index_window//self.time_window).replace(self.seg_gt_dirname, f'/gtFine_t1')
                    label_after = io.read_image(lbl_path_after)[0,...].unsqueeze(0)
                    sample['mask_after'] = label_after[:, :440]
                event_voxel = np.load(event_path, allow_pickle=True)
                event_voxel_before = np.load(event_path_before, allow_pickle=True)
                sample['event'] = torch.from_numpy(event_voxel[:, :440])
                sample['event_before'] = torch.from_numpy(event_voxel_before[:, :440])
                ### flow ###
                if not self.flow_net_flag:
                    flow_path = rgb_path.replace('/leftImg8bit_t0', f'/flow_t0_t{self.time_window}').replace('.png', '.npy')
                    flow = np.load(flow_path, allow_pickle=True)
                    sample['flow'] = torch.from_numpy(flow[:, :440])
        else:
            # rgb_path = get_new_name(lbl_path, idx_diff=0).replace(self.seg_gt_dirname, f'/leftImg8bit_t2_timelens').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.png')
            # rgb_path = get_new_name(lbl_path, idx_diff=0).replace(self.seg_gt_dirname, f'/leftImg8bit_t2').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.png')
            # rgb_path = get_new_name(lbl_path, idx_diff=0).replace(self.seg_gt_dirname, f'/leftImg8bit_t1_timelensxl_finetune').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.png')
            # rgb_path = get_new_name(lbl_path, idx_diff=0).replace('gtFine_t0', f'leftImg8bit_t0_timelensxl_finetune').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.png')
            # rgb_path = get_new_name(lbl_path, idx_diff=0).replace('gtFine_t2', f'leftImg8bit_t2').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.png')
            # rgb_path = lbl_path.replace(self.seg_gt_dirname, '/leftImg8bit_t0').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.png')
            # rgb_path = lbl_path.replace(self.seg_gt_dirname, '/leftImg8bit_t0_dt5').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.png')
            rgb_path = get_new_name(lbl_path, idx_diff=-10).replace(self.seg_gt_dirname, '/leftImg8bit_t0').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.png')
            # rgb_path = lbl_path.replace(self.seg_gt_dirname, '/leftImg8bit_t1_interpolation').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.png')
            # rgb_path = get_new_name(lbl_path, idx_diff=-1).replace(self.seg_gt_dirname, '/leftImg8bit_t0').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.png')

        # lbl_path_t0 = get_new_name(lbl_path, idx_diff=-self.index_window).replace(self.seg_gt_dirname, '/gtFine_t0')
        # rgb_ref = lbl_path.replace(self.seg_gt_dirname, '/leftImg8bit_next').replace(f'_gtFine_labelTrainIds{self.n_classes}.png', '.png')
        # flow_inverse = rgb_ref.replace('/leftImg8bit_next', '/flow_reverse').replace('.png', '.npy')

        # lbl_path = lbl_path.split('.')[0]  # 获取文件名的基础部分（去掉扩展名）
        # lbl_path = f"{lbl_path}_gtFine_labelTrainIds{self.n_classes}.png"  # 添加后缀并重新组合
        if not self.semantic_layout:
            seq_name = Path(rgb_path).parts[-2]
            seq_idx = Path(rgb_path).stem.split('_')[0]

        if not self.semantic_layout:
            sample['img'] = io.read_image(rgb_path)[:3, ...][:, :440]
        # H, W = sample['img'].shape[1:]
        # sample['img_next'] = io.read_image(rgb_ref)[:3, ...][:, :440]
        # lbl_path = get_new_name(lbl_path, idx_diff=0-0).replace(self.seg_gt_dirname, f'/gtFine_t2')
        # lbl_path = get_new_name(lbl_path, idx_diff=0-1).replace(self.seg_gt_dirname, f'/gtFine_t0_dt9')
        # lbl_path = get_new_name(lbl_path, idx_diff=0-2).replace(self.seg_gt_dirname, f'/gtFine_t0_dt8')
        # lbl_path = get_new_name(lbl_path, idx_diff=0-3).replace(self.seg_gt_dirname, f'/gtFine_t0_dt7')
        # lbl_path = get_new_name(lbl_path, idx_diff=0-4).replace(self.seg_gt_dirname, f'/gtFine_t0_dt6')
        # lbl_path = get_new_name(lbl_path, idx_diff=0-5).replace(self.seg_gt_dirname, f'/gtFine_t0_dt5')
        # lbl_path = get_new_name(lbl_path, idx_diff=0-6).replace(self.seg_gt_dirname, f'/gtFine_t0_dt4')
        # lbl_path = get_new_name(lbl_path, idx_diff=0-7).replace(self.seg_gt_dirname, f'/gtFine_t0_dt3')
        # lbl_path = get_new_name(lbl_path, idx_diff=0-8).replace(self.seg_gt_dirname, f'/gtFine_t0_dt2')
        # lbl_path = get_new_name(lbl_path, idx_diff=0-9).replace(self.seg_gt_dirname, f'/gtFine_t0_dt1')
        if not self.semantic_layout:
            label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        # label_ref = io.read_image(lbl_path_t0)[0,...].unsqueeze(0)
        if not self.semantic_layout:
            sample['mask'] = label[:, :440]
        ## for interpolation image that must have shape as multiple of 32
        if sample['img'].shape[-2:] != sample['mask'].shape[-2:]:
            h, w = sample['mask'].shape[1:]
            hn, wn = (h // 32 - 1) * 32, (w // 32 - 1) * 32
            hleft = (h - hn) // 2
            wleft = (w - wn) // 2
            sample['mask'] = sample['mask'][:, hleft:hleft + hn, wleft:wleft + wn]
            assert sample['img'].shape[1:] == sample['mask'].shape[1:], f"{sample['img'].shape} != {sample['mask'].shape}"
        # sample['mask_cur'] = label_ref[:, :440]

        # # # save dict
        # # np.save(event_path.replace(f'/event_t0_t{self.time_window}/event_20', '/sample'), sample)
        # sample_path = str(self.files[index])
        # sample = np.load(sample_path, allow_pickle=True).item()
        # # dict_keys(['img', 'img_next', 'mask', 'mask_cur', 'event', 'flow', 'flow_inverse'])
        # seq_name = Path(sample_path).parts[-2]
        # seq_idx = Path(sample_path).parts[-1].split('_')[0]

        # # save dict
        # lbl_path = str(self.files[index])
        # seq_name = Path(lbl_path).parts[-2]
        # seq_idx = Path(lbl_path).parts[-1].split('.')[0]
        # # sample_path = f'/mnt/sdc/lxy/datasets/DSEC/DSEC/dsec_anytime_{self.duration}_N/day/processed/{self.split}/{seq_name}/{seq_idx}.npy'
        # # # 如果文件夹不存在，则创建
        # # if not os.path.exists(os.path.dirname(sample_path)):
        # #     os.makedirs(os.path.dirname(sample_path))
        # # np.save(sample_path, sample)
        # # return seq_name, seq_idx, sample, label

        # sample = np.load(lbl_path, allow_pickle=True).item()

        if self.transform:
            sample = self.transform(sample)

        label = sample['mask']

        del sample['mask']
        label = [self.encode(label.squeeze().numpy()).long()]
        if self.time_window == 2:
            label_after = sample['mask_after']
            del sample['mask_after']
            label_after = self.encode(label_after.squeeze().numpy()).long()
            label.append(label_after)
        # label_ref = sample['mask_cur']
        # del sample['mask_cur']
        # label_ref = self.encode(label_ref.squeeze().numpy()).long()

        if not self.flow_net_flag and self.time_window!=0:
            flow = sample['flow']
            del sample['flow']

        if self.time_window != 0:
            event_voxel = sample['event']
            del sample['event']
            event_voxel_before = sample['event_before']
            del sample['event_before']
            if self.time_window == 2:
                event_voxel_after = sample['event_after']
                del sample['event_after']
        # img_next = sample['img_next']
        # del sample['img_next']

        # flow_inverse = sample['flow_inverse']
        # del sample['flow_inverse']

        sample = [sample[k] for k in self.modals]
        if self.time_window != 0:
            sample.append(event_voxel)
            sample.append(event_voxel_before)
            if self.time_window == 2:
                sample.append(event_voxel_after)
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
