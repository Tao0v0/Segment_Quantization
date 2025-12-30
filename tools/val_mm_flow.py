import torch
import argparse
import yaml
import math
import os
import time
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics, compute_epe, compute_npe
from semseg.utils.utils import setup_cudnn
from math import ceil
import numpy as np
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
# import Image
from PIL import Image

def concatenate_images(images, direction='horizontal'):
    if not images:
        raise ValueError("The images list should not be empty")

    if direction == 'horizontal':
        # 水平拼接
        total_width = sum(image.width for image in images)
        max_height = max(image.height for image in images)
        new_image = Image.new('RGB', (total_width, max_height))

        current_x = 0
        for image in images:
            new_image.paste(image, (current_x, 0))
            current_x += image.width

    elif direction == 'vertical':
        # 垂直拼接
        max_width = max(image.width for image in images)
        total_height = sum(image.height for image in images)
        new_image = Image.new('RGB', (max_width, total_height))

        current_y = 0
        for image in images:
            new_image.paste(image, (0, current_y))
            current_y += image.height

    else:
        raise ValueError("Direction should be 'horizontal' or 'vertical'")

    return new_image

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

@torch.no_grad()
def sliding_predict(model, image, num_classes, flip=True):
    image_size = image[0].shape
    tile_size = (int(ceil(image_size[2]*1)), int(ceil(image_size[3]*1)))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = torch.zeros((num_classes, image_size[2], image_size[3]), device=torch.device('cuda'))
    count_predictions = torch.zeros((image_size[2], image_size[3]), device=torch.device('cuda'))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = [modal[:, :, y_min:y_max, x_min:x_max] for modal in image]
            padded_img = [pad_image(modal, tile_size) for modal in img]
            tile_counter += 1
            padded_prediction = model(padded_img)
            if flip:
                fliped_img = [padded_modal.flip(-1) for padded_modal in padded_img]
                fliped_predictions = model(fliped_img)
                padded_prediction += fliped_predictions.flip(-1)
            predictions = padded_prediction[:, :, :img[0].shape[2], :img[0].shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.squeeze(0)

    return total_predictions.unsqueeze(0)

def denormalize(image, mean: list = (0.485, 0.456, 0.406), std: list = (0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return image * std + mean

@torch.no_grad()
def evaluate(model, dataloader, device, save_dir=None, palette=None, spatial_scale: float = 0.5):
    print('Evaluating...')

    model.eval()
    if spatial_scale <= 0:
        raise ValueError(f"spatial_scale must be > 0, got {spatial_scale}")
    total_epe = 0.0
    total_n1pe = 0.0
    total_n2pe = 0.0
    total_n3pe = 0.0
    num_batches = 0
    for seq_names, seq_index, images in tqdm(dataloader):
        images = [x.to(device).float() for x in images]
        bin = 5
        ev_t0_t1 = torch.cat([images[0][:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
        ev_before = torch.cat([images[1][:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
        if spatial_scale != 1.0:
            ev_t0_t1 = torch.nn.functional.interpolate(
                ev_t0_t1, scale_factor=spatial_scale, mode='bilinear', align_corners=False
            )
            ev_before = torch.nn.functional.interpolate(
                ev_before, scale_factor=spatial_scale, mode='bilinear', align_corners=False
            )
        pred_out = model(ev_before, ev_t0_t1)
        if isinstance(pred_out, tuple):
            preds = pred_out[1][-1]
        elif torch.is_tensor(pred_out):
            preds = pred_out
        else:
            preds = pred_out[-1]
        
        gt_raw = images[-1]
        valid = None
        if gt_raw.ndim == 4 and gt_raw.shape[1] == 3:
            valid = gt_raw[:, 2]
            gt = gt_raw[:, :2]
        else:
            gt = gt_raw

        if spatial_scale != 1.0:
            gt = (
                torch.nn.functional.interpolate(gt, scale_factor=spatial_scale, mode='bilinear', align_corners=False)
                * spatial_scale
            )
            if valid is not None:
                valid = torch.nn.functional.interpolate(
                    valid[:, None], scale_factor=spatial_scale, mode='nearest'
                )[:, 0]

        epe = compute_epe(preds, gt, valid=valid)
        n1pe, n2pe, n3pe = compute_npe(preds, gt, valid=valid)

        total_epe += float(epe.item() if hasattr(epe, "item") else epe)
        total_n1pe += float(n1pe)
        total_n2pe += float(n2pe)
        total_n3pe += float(n3pe)
        num_batches += 1
    
    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        total_epe / num_batches,
        total_n1pe / num_batches,
        total_n2pe / num_batches,
        total_n3pe / num_batches,
    )


@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = [F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=True) for img in images]
            scaled_images = [scaled_img.to(device) for scaled_img in scaled_images]
            logits = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = [torch.flip(scaled_img, dims=(3,)) for scaled_img in scaled_images]
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)
    
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou


def main(cfg, scene, classes, model_path, duration):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    # cases = ['cloud', 'fog', 'night', 'rain', 'sun']
    # cases = ['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    cases = [None] # all
    
    model_path = Path(model_path)
    if not model_path.exists(): 
        print(model_path)
        raise FileNotFoundError
    print(f"Evaluating {model_path}...")

    exp_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    save_dir  = os.path.join(os.path.dirname(model_path), '{}_eval_{}'.format(scene, exp_time))
    eval_path = os.path.join(os.path.dirname(model_path), '{}_eval_{}.txt'.format(scene, exp_time))

    for case in cases:
        dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'].replace("${DURATION}", str(duration)), 'val', classes, transform, cfg['DATASET']['MODALS'], case, duration=duration, flow_net_flag=cfg['MODEL']['FLOW_NET_FLAG'], dataset_type=cfg['DATASET']['TYPE'])
        # dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'].replace("${DURATION}", str(duration)), 'train', classes, transform, cfg['DATASET']['MODALS'], case, duration=duration, flow_net_flag=cfg['MODEL']['FLOW_NET_FLAG'], dataset_type=cfg['DATASET']['TYPE'])
        # --- test set
        # dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'].replace("${DURATION}", str(duration)), 'test', transform, cfg['DATASET']['MODALS'], case)

        model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes, cfg['DATASET']['MODALS'], cfg['MODEL']['BACKBONE_FLAG'], cfg['MODEL']['FLOW_NET_FLAG'], dataset_type=cfg['DATASET']['TYPE'], anytime_flag=True)
        # model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], 11, cfg['DATASET']['MODALS'], cfg['MODEL']['BACKBONE_FLAG'], cfg['MODEL']['FLOW_NET_FLAG'])
        state_dict = torch.load(str(model_path), map_location="cpu")
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        # Backward-compat: older checkpoints (before `unfold_conv`) won't contain this fixed parameter.
        if "unfold_conv.weight" not in state_dict and hasattr(model, "unfold_conv"):
            state_dict = dict(state_dict)
            state_dict["unfold_conv.weight"] = model.unfold_conv.weight.detach().clone()
        if "flow_unfold3x3.weight" in state_dict:
            state_dict = dict(state_dict)
            state_dict.pop("flow_unfold3x3.weight", None)
        msg = model.load_state_dict(state_dict, strict=True)
        print(msg)
        model = model.to(device)

        logger.info('================== model complexity =====================')
        cal_flops(model, cfg['DATASET']['MODALS'], logger)
        exit(0)

        sampler_val = None
        dataloader = DataLoader(dataset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=eval_cfg['BATCH_SIZE'], pin_memory=False, sampler=sampler_val)
        if True:
            if eval_cfg['MSF']['ENABLE']:
                acc, macc, f1, mf1, ious, miou = evaluate_msf(model, dataloader, device, eval_cfg['MSF']['SCALES'], eval_cfg['MSF']['FLIP'])
            else:
                acc, macc, f1, mf1, ious, miou = evaluate(model, dataloader, device, save_dir, palette=dataset.SEGMENTATION_CONFIGS[classes]["PALETTE"])

            table = {
                'Class': list(dataset.SEGMENTATION_CONFIGS[classes]["CLASSES"]) + ['Mean'],
                'IoU': ious + [miou],
                'F1': f1 + [mf1],
                'Acc': acc + [macc]
            }
            print("mIoU : {}".format(miou))
            print("Results saved in {}".format(model_path))

        with open(eval_path, 'a+') as f:
            f.writelines(str(model_path))
            f.write("\n============== Eval on {} {} images =================\n".format(case, len(dataset)))
            f.write("\n")
            print(tabulate(table, headers='keys'), file=f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/DELIVER.yaml')
    parser.add_argument('--scene', type=str, default='night')
    parser.add_argument('--model_path', type=str, default='night')
    parser.add_argument('--classes', type=int, default=11)
    parser.add_argument('--input_type', type=str, default='rgbe')
    parser.add_argument('--duration', type=int, default=50)
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    # gpu = setup_ddp()
    # main(cfg, gpu)

    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    exp_name = '_'.join(['inference', cfg['DATASET']['NAME'], model, modals])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    if os.path.isfile(cfg['MODEL']['RESUME']):
        save_dir =  Path(os.path.dirname(cfg['MODEL']['RESUME']))
    os.makedirs(save_dir, exist_ok=True)
    time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_logger(save_dir / f'{args.input_type}_{args.scene}_{args.classes}_{time_}_val.log')

    main(cfg, args.scene, args.classes, args.model_path, args.duration)
