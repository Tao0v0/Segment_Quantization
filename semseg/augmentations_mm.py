import torchvision.transforms.functional as TF 
import random
import math
import torch
from torch import Tensor
from typing import Tuple, List, Union, Tuple, Optional


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: list) -> list:
        # img, mask = sample['img'], sample['mask']
        # if mask.ndim == 2:
        #     assert img.shape[1:] == mask.shape
        # else:
        #     assert img.shape[1:] == mask.shape[1:]

        for transform in self.transforms:
            sample = transform(sample)

        return sample


class Normalize:
    def __init__(self, mean: list = (0.485, 0.456, 0.406), std: list = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        self.event_mean = [-0.0005]
        self.event_std = [0.5128]

    def __call__(self, sample: list) -> list:
        for k, v in sample.items():
            # if k in ['mask', 'flow', 'event']: 
            if k in ['mask', 'mask_after', 'flow']: 
                continue
            # elif k == 'event' or k=='event_before' or k=='event_after':  # 处理事件数据
            #     sample[k] = sample[k].float()  # 转换为浮点数
            #     # 对每一channel 进行归一化 mean和std都是event_mean 和 event_std
            #     event_mean = [self.event_mean for _ in range(sample[k].shape[1])]
            #     event_std = [self.event_std for _ in range(sample[k].shape[1])]
            #     sample[k] = TF.normalize(sample[k], event_mean, event_std)
            elif k == 'img' or k == 'img_next':
                sample[k] = sample[k].float()
                sample[k] /= 255
                sample[k] = TF.normalize(sample[k], self.mean, self.std)
            # else:
            #     sample[k] = sample[k].float()
            #     sample[k] /= 255
        
        return sample


class RandomColorJitter:
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            self.brightness = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_brightness(sample['img'], self.brightness)
            self.contrast = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_contrast(sample['img'], self.contrast)
            self.saturation = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_saturation(sample['img'], self.saturation)
            if 'img_next' in sample:
                sample['img_next'] = TF.adjust_brightness(sample['img_next'], self.brightness)
                sample['img_next'] = TF.adjust_contrast(sample['img_next'], self.contrast)
                sample['img_next'] = TF.adjust_saturation(sample['img_next'], self.saturation)
        return sample


class AdjustGamma:
    def __init__(self, gamma: float, gain: float = 1) -> None:
        """
        Args:
            gamma: Non-negative real number. gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
            gain: constant multiplier
        """
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.adjust_gamma(img, self.gamma, self.gain), mask


class RandomAdjustSharpness:
    def __init__(self, sharpness_factor: float, p: float = 0.5) -> None:
        self.sharpness = sharpness_factor
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            sample['img'] = TF.adjust_sharpness(sample['img'], self.sharpness)
        return sample


class RandomAutoContrast:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            sample['img'] = TF.autocontrast(sample['img'])
        return sample


class RandomGaussianBlur:
    def __init__(self, kernel_size: int = 3, p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            sample['img'] = TF.gaussian_blur(sample['img'], self.kernel_size)
            # img = TF.gaussian_blur(img, self.kernel_size)
            if 'img_next' in sample:
                sample['img_next'] = TF.gaussian_blur(sample['img_next'], self.kernel_size)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            for k, v in sample.items():
                if k == 'flow':
                    # 水平翻转光流图像
                    v = TF.hflip(v)
                    # 反转水平分量的符号
                    v[0, :, :] = -v[0, :, :]
                    sample[k] = v
                else:
                    sample[k] = TF.hflip(v)
        return sample


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            return TF.vflip(img), TF.vflip(mask)
        return img, mask


class RandomGrayscale:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.rgb_to_grayscale(img, 3)
        return img, mask


class Equalize:
    def __call__(self, image, label):
        return TF.equalize(image), label


class Posterize:
    def __init__(self, bits=2):
        self.bits = bits # 0-8
        
    def __call__(self, image, label):
        return TF.posterize(image, self.bits), label


class Affine:
    def __init__(self, angle=0, translate=[0, 0], scale=1.0, shear=[0, 0], seg_fill=0):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.seg_fill = seg_fill
        
    def __call__(self, img, label):
        return TF.affine(img, self.angle, self.translate, self.scale, self.shear, TF.InterpolationMode.BILINEAR, 0), TF.affine(label, self.angle, self.translate, self.scale, self.shear, TF.InterpolationMode.NEAREST, self.seg_fill) 


class RandomRotation:
    def __init__(self, degrees: float = 10.0, p: float = 0.2, seg_fill: int = 0, expand: bool = False) -> None:
        """Rotate the image, segmentation mask, and flow by a random angle between -degrees and degrees with probability p.

        Args:
            degrees: Maximum rotation angle in degrees, counter-clockwise.
            p: Probability of applying the rotation.
            seg_fill: Fill value for segmentation mask when rotated.
            expand: If true, expands the output to fit the entire rotated image; otherwise, keeps the original size.
        """
        self.degrees = degrees
        self.p = p
        self.seg_fill = seg_fill
        self.expand = expand

    def __call__(self, sample: dict) -> dict:
        """Apply random rotation to the sample dictionary containing image, mask, and/or flow.

        Args:
            sample: Dictionary with keys 'image', 'mask', and/or 'flow'.

        Returns:
            Dictionary with rotated tensors.
        """
        if random.random() < self.p:
            # Generate a single random angle for all items to ensure consistency
            angle = random.uniform(-self.degrees, self.degrees)

            for key, value in sample.items():
                if key == 'image':
                    # Rotate image with bilinear interpolation
                    sample[key] = TF.rotate(value, angle, TF.InterpolationMode.BILINEAR, self.expand, fill=0)
                elif key == 'mask':
                    # Rotate segmentation mask with nearest interpolation
                    sample[key] = TF.rotate(value, angle, TF.InterpolationMode.NEAREST, self.expand, fill=self.seg_fill)
                elif key == 'flow':
                    # Rotate flow image with bilinear interpolation
                    flow_rotated = TF.rotate(value, angle, TF.InterpolationMode.BILINEAR, self.expand, fill=0)
                    # Rotate flow vectors (u, v components)
                    theta = torch.tensor(angle * np.pi / 180.0)  # Convert degrees to radians
                    rotation_matrix = torch.tensor([
                        [torch.cos(theta), -torch.sin(theta)],
                        [torch.sin(theta), torch.cos(theta)]
                    ]).to(flow_rotated.device)
                    # Assume flow is [2, H, W] with u, v channels
                    flow_vectors = flow_rotated.permute(1, 2, 0)  # [H, W, 2]
                    flow_vectors_rotated = torch.matmul(flow_vectors.reshape(-1, 2), rotation_matrix.T).reshape(flow_vectors.shape)
                    sample[key] = flow_vectors_rotated.permute(2, 0, 1)  # Back to [2, H, W]
        return sample
    

class CenterCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]]) -> None:
        """Crops the image at the center

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.center_crop(img, self.size), TF.center_crop(mask, self.size)


class RandomCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]], p: float = 0.5) -> None:
        """Randomly Crops the image.

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        if random.random() < self.p:
            margin_h = max(H - tH, 0)
            margin_w = max(W - tW, 0)
            y1 = random.randint(0, margin_h+1)
            x1 = random.randint(0, margin_w+1)
            y2 = y1 + tH
            x2 = x1 + tW
            img = img[:, y1:y2, x1:x2]
            mask = mask[:, y1:y2, x1:x2]
        return img, mask

class RandomCrop_cat_max_ratio:
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could occupy.
        ignore_index (int): The label index to ignore.
    """

    def __init__(self, crop_size: Tuple[int, int], cat_max_ratio: float = 1.0, ignore_index: int = 255) -> None:
        assert crop_size[0] > 0 and crop_size[1] > 0, "Crop size must be positive."
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img: Tensor) -> Tuple[int, int, int, int]:
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[1] - self.crop_size[0], 0)
        margin_w = max(img.shape[2] - self.crop_size[1], 0)
        offset_h = random.randint(0, margin_h + 1)
        offset_w = random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img: Tensor, crop_bbox: Tuple[int, int, int, int]) -> Tensor:
        """Crop from `img`."""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        return img

    def __call__(self, sample:list) -> list:
    # def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Call function to randomly crop images and semantic segmentation maps.

        Args:
            img (Tensor): Image tensor.
            mask (Tensor): Mask tensor.

        Returns:
            Tuple[Tensor, Tensor]: Randomly cropped image and mask.
        """
        img, mask = sample['img'], sample['mask']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.0:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(mask, crop_bbox)
                labels, cnt = torch.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and torch.max(cnt).item() / torch.sum(cnt).item() < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # Crop the sample
        for k, v in sample.items():
            sample[k] = self.crop(v, crop_bbox)

        return sample
    
class Pad:
    def __init__(self, size: Union[List[int], Tuple[int], int], seg_fill: int = 0) -> None:
        """Pad the given image on all sides with the given "pad" value.
        Args:
            size: expected output image size (h, w)
            fill: Pixel fill value for constant fill. Default is 0. This value is only used when the padding mode is constant.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, sample:list) -> list:
    # def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        img = sample['img']
        padding = (0, 0, self.size[1]-img.shape[2], self.size[0]-img.shape[1])
        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.pad(v, padding, fill=self.seg_fill)
            else:
                sample[k] = TF.pad(v, padding, fill=0)
        return sample


class ResizePad:
    def __init__(self, size: Union[int, Tuple[int], List[int]], seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size. 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        # scale the image 
        scale_factor = min(tH/H, tW/W) if W > H else max(tH/H, tW/W)
        # nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

        # pad the image
        padding = [0, 0, tW - nW, tH - nH]
        img = TF.pad(img, padding, fill=0)
        mask = TF.pad(mask, padding, fill=self.seg_fill)
        return img, mask 


class Resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]], scale: Optional[Tuple[float, float]] = None) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size. 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size
        self.scale = scale

    def __call__(self, sample:list) -> list:
        if self.scale is not None:
            # img, mask = sample['img'], sample['mask']
            H, W = sample['img'].shape[1:]
            tH, tW = self.size

            # get the scale
            ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
            # ratio = random.uniform(min(self.scale), max(self.scale))
            scale = int(tH*ratio), int(tW*4*ratio)
            # scale the image 
            scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
            nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
            # nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
                else:
                    sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)
            return sample
        else:
            H, W = sample['img'].shape[1:]

            # scale the image 
            scale_factor = self.size[0] / min(H, W)
            nH, nW = round(H*scale_factor), round(W*scale_factor)
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
                else:
                    sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)
            # img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
            # mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

            # make the image divisible by stride
            alignH, alignW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
            
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.NEAREST)
                else:
                    sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.BILINEAR)
            # img = TF.resize(img, (alignH, alignW), TF.InterpolationMode.BILINEAR)
            # mask = TF.resize(mask, (alignH, alignW), TF.InterpolationMode.NEAREST)
            return sample


class RandomResizedCrop:
    def __init__(self, size: Union[int, Tuple[int], List[int]], scale: Tuple[float, float] = (0.5, 2.0), seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        """
        self.size = size
        self.scale = scale
        self.seg_fill = seg_fill

    def __call__(self, sample: list) -> list:
        # img, mask = sample['img'], sample['mask']
        H, W = sample['img'].shape[1:]
        tH, tW = self.size

        # get the scale
        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        # ratio = random.uniform(min(self.scale), max(self.scale))
        scale = int(tH*ratio), int(tW*4*ratio)
        # scale the image 
        scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        # nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
            elif k == 'flow':
                # Resize flow and scale its values by scale_factor
                flow_resized = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)
                sample[k] = flow_resized * scale_factor  # Scale the flow vectors by the resize factor
            else:
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)
        # random crop
        margin_h = max(sample['img'].shape[1] - tH, 0)
        margin_w = max(sample['img'].shape[2] - tW, 0)
        y1 = random.randint(0, margin_h+1)
        x1 = random.randint(0, margin_w+1)
        y2 = y1 + tH
        x2 = x1 + tW
        for k, v in sample.items():
            sample[k] = v[:, y1:y2, x1:x2]

        # pad the image
        if sample['img'].shape[1:] != self.size:
            padding = [0, 0, tW - sample['img'].shape[2], tH - sample['img'].shape[1]]
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.pad(v, padding, fill=self.seg_fill)
                else:
                    sample[k] = TF.pad(v, padding, fill=0)

        return sample



def get_train_augmentation(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
    return Compose([
        RandomColorJitter(p=0.2), # 
        RandomHorizontalFlip(p=0.5), #
        RandomGaussianBlur((3, 3), p=0.2), #
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill), #
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def get_train_augmentation_flow(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
    return Compose([
        RandomHorizontalFlip(p=0.5), #
        RandomResizedCrop(size=(288,384), scale=(1.0, 1.0), seg_fill=seg_fill), #
    ])

# def get_train_augmentation(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
#     return Compose([
#         Resize(size, scale=(0.5, 2.0)),
#         RandomCrop_cat_max_ratio(size, cat_max_ratio=0.75),
#         RandomHorizontalFlip(p=0.5), #
#         Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         Pad(size, seg_fill=seg_fill)
#     ])

def get_val_augmentation(size: Union[int, Tuple[int], List[int]]):
    return Compose([
        # Resize(size),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


if __name__ == '__main__':
    h = 230
    w = 420
    sample = {}
    sample['img'] = torch.randn(3, h, w)
    sample['depth'] = torch.randn(3, h, w)
    sample['lidar'] = torch.randn(3, h, w)
    sample['event'] = torch.randn(3, h, w)
    sample['mask'] = torch.randn(1, h, w)
    aug = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomResizedCrop((512, 512)),
        Resize((224, 224)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    sample = aug(sample)
    for k, v in sample.items():
        print(k, v.shape)