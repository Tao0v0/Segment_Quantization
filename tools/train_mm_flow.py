import os
import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
# PyTorch AMP API compatibility (torch 1.12 uses torch.cuda.amp; newer uses torch.amp).
try:
    from torch.amp import autocast as _autocast  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    from torch.cuda.amp import autocast as _autocast  # type: ignore[assignment]

try:
    from torch.amp import GradScaler  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler  # type: ignore[assignment]


def autocast_ctx(enabled: bool):
    try:
        return _autocast(device_type="cuda", enabled=enabled)
    except TypeError:
        return _autocast(enabled=enabled)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation, get_train_augmentation_flow
from semseg.losses import get_loss, compute_eraft_flow_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from semseg.metrics import Metrics
from val_mm_flow import evaluate
import numpy as np
import math
# import Image
from PIL import Image
from torchviz import make_dot

def _load_state_dict(path: str, keys=("model", "state_dict")):
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    if isinstance(ckpt, dict):
        for k in keys:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def main(cfg, scene, classes, gpu, save_dir, duration):
    start = time.time()
    best_epe = 1e8
    best_epoch = 0
    num_workers = 4
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    grad_clip = float(train_cfg.get('GRAD_CLIP', 1.0))
    flow_spatial_scale = float(train_cfg.get('FLOW_SPATIAL_SCALE', 0.5))
    eval_flow_spatial_scale = float(eval_cfg.get('FLOW_SPATIAL_SCALE', flow_spatial_scale))
    if flow_spatial_scale <= 0:
        raise ValueError(f"TRAIN.FLOW_SPATIAL_SCALE must be > 0, got {flow_spatial_scale}")
    if eval_flow_spatial_scale <= 0:
        raise ValueError(f"EVAL.FLOW_SPATIAL_SCALE must be > 0, got {eval_flow_spatial_scale}")
    resume_path = cfg['MODEL']['RESUME']
    gpus = int(os.environ['WORLD_SIZE'])

    # traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    # traintransform = get_train_augmentation_flow(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    traintransform = None
    # valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    valtransform = None
    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'].replace("${DURATION}", str(duration)), 'train', classes, traintransform, dataset_cfg['MODALS'], duration=duration, flow_net_flag=model_cfg['FLOW_NET_FLAG'], dataset_type=dataset_cfg['TYPE'])
    # 计算补齐后的目标长度
    if len(trainset) % train_cfg['BATCH_SIZE'] != 0:
        num_batches = math.ceil(len(trainset) / train_cfg['BATCH_SIZE'])
        target_length = num_batches * train_cfg['BATCH_SIZE']
        trainset = ExtendedDSEC_FLOW(trainset, target_length)
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'].replace("${DURATION}", str(duration)), 'val', classes, valtransform, dataset_cfg['MODALS'], duration=duration, flow_net_flag=model_cfg['FLOW_NET_FLAG'], dataset_type=dataset_cfg['TYPE'])
    # valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'].replace("${DURATION}", str(duration)), 'train', classes, valtransform, dataset_cfg['MODALS'], duration=duration, flow_net_flag=model_cfg['FLOW_NET_FLAG'], dataset_type=dataset_cfg['TYPE'])
    class_names = trainset.SEGMENTATION_CONFIGS[classes]["CLASSES"]

    # NOTE: For flow-only training, instantiate the flow network directly.
    # Building CMNeXt and taking `.flow_net` would apply CMNeXt weight init on ERAFT,
    # which can make the flow head initialization overly aggressive (fan_out-based init).
    flow_net_type = str(model_cfg.get('FLOW_NET', 'eraft')).lower()
    if flow_net_type == 'eraft':
        from semseg.models.modules.flow_network.eraft.eraft import ERAFT

        model = ERAFT(n_first_channels=4)
    elif flow_net_type == 'eraft_original':
        # Upstream ERAFT expects `model.*` absolute imports. Add the upstream root to `sys.path`
        # so `from model.eraft import ERAFT` works regardless of how the folder is named.
        import sys

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
        sys.path.insert(0, str(eraft_root))
        from model.eraft import ERAFT as ERAFT_ORIG

        model = ERAFT_ORIG(config={"subtype": "standard"}, n_first_channels=4)
    else:
        model = eval(model_cfg['NAME'])(
            model_cfg['BACKBONE'],
            trainset.n_classes,
            dataset_cfg['MODALS'],
            model_cfg['BACKBONE_FLAG'],
            model_cfg['FLOW_NET_FLAG'],
            dataset_type=dataset_cfg['TYPE'],
            anytime_flag=False,
        ).flow_net
    for name, param in model.named_parameters():
        print(name)
    print("model: ", model_cfg['FLOW_NET_FLAG'])
    print(model_cfg['RESUME_FLOWNET'])
    print(os.path.isfile(model_cfg['RESUME_FLOWNET']))
    if model_cfg['FLOW_NET_FLAG'] and os.path.isfile(model_cfg['RESUME_FLOWNET']):
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            print('Loading flownet model...')
        flow_net_type = str(model_cfg['FLOW_NET']).lower()
        resume_flownet_path = model_cfg['RESUME_FLOWNET']
        print("flow_net_type: ", flow_net_type)
        print("dataset_cfg['TYPE']: ", dataset_cfg['TYPE'])

        if flow_net_type in ('eraft', 'raft_small', 'eraft_original'):
            ## for eraft
            # if dataset_cfg['TYPE'] == 'dsec':
            if dataset_cfg['TYPE'] == 'dsec_':
                flownet_checkpoint = _load_state_dict(resume_flownet_path)
                # flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'), weights_only=True)['model']
            elif dataset_cfg['TYPE'] == 'dsec':
            # elif dataset_cfg['TYPE'] == 'sdsec':
                # flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'))
                flownet_checkpoint = _load_state_dict(resume_flownet_path)   # for dsec.tar
                if any(k.startswith('module.') for k in flownet_checkpoint.keys()):
                    flownet_checkpoint = {k.replace('module.', '', 1): v for k, v in flownet_checkpoint.items()}
                if flow_net_type == 'raft_small':
                    print("Loading raft_small")
                # # 筛选出以 'flownet' 为前缀的键
                # flownet_checkpoint = {
                #     key: value for key, value in flownet_checkpoint.items() if key.startswith('flow_net')
                # }
                # # 给所有key去掉前缀 'flow_net.'
                # flownet_checkpoint = {k.replace('flow_net.', ''): v for k, v in flownet_checkpoint.items()}
            # Drop only mismatched-shape keys (e.g. when loading RGB-RAFT weights into event-RAFT).
            model_state = model.state_dict()
            mismatched = []
            for k, v in list(flownet_checkpoint.items()):
                if k in model_state and hasattr(v, "shape") and v.shape != model_state[k].shape:
                    mismatched.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                    flownet_checkpoint.pop(k, None)
            if mismatched and ((train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP'])):
                print("Dropping mismatched flownet keys:")
                for k, src, dst in mismatched[:20]:
                    print(f"  {k}: ckpt{src} != model{dst}")
                if len(mismatched) > 20:
                    print(f"  ... and {len(mismatched) - 20} more")
        elif flow_net_type == 'bflow':
            # for bflow
            flownet_checkpoint = _load_state_dict(resume_flownet_path, keys=("state_dict", "model"))
            # 过滤掉 'flow_network.' 前缀
            # flownet_checkpoint = {k.replace('flow_network.', ''): v for k, v in flownet_checkpoint.items()}
            # 过滤掉 'net.' 前缀
            flownet_checkpoint = {k.replace('net.', ''): v for k, v in flownet_checkpoint.items()}
            if 'fnet_ev.conv1.weight' in flownet_checkpoint:
                # delete weights of the first layer
                flownet_checkpoint.pop('fnet_ev.conv1.weight')
                flownet_checkpoint.pop('fnet_ev.conv1.bias')
            if 'update_block.encoder.convc1.weight' in flownet_checkpoint:
                # delete weights of the first layer
                flownet_checkpoint.pop('update_block.encoder.convc1.weight')
                flownet_checkpoint.pop('update_block.encoder.convc1.bias') 
            # if 'cnet.conv1.weight' in flownet_checkpoint:
            #     # delete weights of the second layer
            #     flownet_checkpoint.pop('cnet.conv1.weight')
            #     flownet_checkpoint.pop('cnet.conv1.bias')

        flownet_msg = model.load_state_dict(flownet_checkpoint, strict=False)
        print("flownet_checkpoint msg: ", flownet_msg)
        logger.info(flownet_msg)
        # exit(0)

    model = model.to(device)

    start_epoch = 0
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])

    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = None
        model = DDP(model, device_ids=[gpu], output_device=0, find_unused_parameters=True)
    else:
        sampler = RandomSampler(trainset)
        sampler_val = None

    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=sampler, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    # trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=0, drop_last=True, pin_memory=True, sampler=sampler, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=True, sampler=sampler_val, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    # valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=0, pin_memory=True, sampler=sampler_val, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    iters_per_epoch = len(trainloader)
    print("iters_per_epoch: ", iters_per_epoch)
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, int((epochs+1)*iters_per_epoch), sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])

    scaler = GradScaler(enabled=train_cfg['AMP'])
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer = SummaryWriter(str(save_dir))
        # logger.info('================== model complexity =====================')
        # cal_flops(model, dataset_cfg['MODALS'], logger)
        logger.info('================== model structure =====================')
        # logger.info(flownet_msg)
        logger.info(model)
        logger.info('================== training config =====================')
        logger.info(cfg)

        # exit(0)

    for epoch in range(start_epoch, epochs):
        model.train()
        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0
        lr = scheduler.get_lr()
        lr = sum(lr) / len(lr)
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        for iter, (seq_names, seq_index, sample) in pbar:
            optimizer.zero_grad(set_to_none=True)
            sample = [x.to(device).float() for x in sample]
            
            with autocast_ctx(train_cfg['AMP']):
                bin = 5
                ev_t0_t1 = torch.cat([sample[0][:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                ev_before = torch.cat([sample[1][:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                if flow_spatial_scale != 1.0:
                    ev_t0_t1 = torch.nn.functional.interpolate(
                        ev_t0_t1, scale_factor=flow_spatial_scale, mode='bilinear', align_corners=False
                    )
                    ev_before = torch.nn.functional.interpolate(
                        ev_before, scale_factor=flow_spatial_scale, mode='bilinear', align_corners=False
                    )
                pred_out = model(ev_before, ev_t0_t1)
                if isinstance(pred_out, tuple):
                    predict_flows = pred_out[1]
                elif torch.is_tensor(pred_out):
                    predict_flows = [pred_out]
                else:
                    predict_flows = pred_out
                flow_gt_raw = sample[-1]
                valid = None
                if flow_gt_raw.ndim == 4 and flow_gt_raw.shape[1] == 3:
                    valid = flow_gt_raw[:, 2]
                    flow_gt = flow_gt_raw[:, :2]
                else:
                    flow_gt = flow_gt_raw

                if flow_spatial_scale != 1.0:
                    flow_gt = (
                        torch.nn.functional.interpolate(
                            flow_gt, scale_factor=flow_spatial_scale, mode='bilinear', align_corners=False
                        )
                        * flow_spatial_scale
                    )
                    if valid is not None:
                        valid = torch.nn.functional.interpolate(
                            valid[:, None], scale_factor=flow_spatial_scale, mode='nearest'
                        )[:, 0]

                loss = compute_eraft_flow_loss(predict_flows, flow_gt, valid=valid)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            if lr <= 1e-8:
                lr = 1e-8 # minimum of lr
            train_loss += loss.item()
            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        train_loss /= iter+1
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.add_scalar('train/loss', train_loss, epoch)
        # torch.cuda.empty_cache()

            if ((epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch+1)>train_cfg['EVAL_START']) or (epoch+1) == epochs:
                if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                    epe, n1pe, n2pe, n3pe = evaluate(model, valloader, device, spatial_scale=eval_flow_spatial_scale)
                    writer.add_scalar('val/EPE', epe, epoch)

                if epe < best_epe:
                    prev_best_ckp = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_epe}_checkpoint.pth"
                    prev_best = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_epe}.pth"
                    if os.path.isfile(prev_best): os.remove(prev_best)
                    if os.path.isfile(prev_best_ckp): os.remove(prev_best_ckp)
                    best_epe = epe
                    best_epoch = epoch+1
                    cur_best_ckp = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_epe}_checkpoint.pth"
                    cur_best = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_epe}.pth"
                    torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), cur_best)
                    # --- 
                    torch.save({'epoch': best_epoch,
                                'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_epe': best_epe,
                                '1pe': n1pe,
                                '2pe': n2pe,
                                '3pe': n3pe,
                                }, cur_best_ckp)
                    logger.info(f"EPE: {epe: 2f}, 1PE: {n1pe: 2f}, 2PE: {n2pe: 2f}, 3PE: {n3pe: 2f}")
                    logger.info(f"Best model saved at epoch {best_epoch}")
                logger.info(f"Current epoch:{epoch} EPE: {epe} Best EPE: {best_epe}")

    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best EPE', f"{best_epe:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logger.info(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/deliver_rgbdel.yaml', help='Configuration file to use')
    parser.add_argument('--scene', type=str, default='night')
    parser.add_argument('--input_type', type=str, default='rgbe')
    parser.add_argument('--classes', type=int, default=11)
    parser.add_argument('--duration', type=int, default=50)
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    exp_name = '_'.join([cfg['DATASET']['NAME'], model, modals])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    if os.path.isfile(cfg['MODEL']['RESUME']):
        save_dir =  Path(os.path.dirname(cfg['MODEL']['RESUME']))
    os.makedirs(save_dir, exist_ok=True)
    time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_logger(save_dir / f'{args.input_type}_{args.scene}_{args.classes}_{time_}_train.log')
    main(cfg, args.scene, args.classes, gpu, save_dir, args.duration)
    cleanup_ddp()


"""
 CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. torchrun --nproc_per_node=2 --master_port 29501 tools/train_mm_flow.py --cfg configs/dsec_rgb_day_flow.yaml --duration 100 --scene thun_00_a

"""
