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
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou, calculate_class_weights, cal_event_meanandstd
from semseg.metrics import Metrics
from val_mm import evaluate
import numpy as np
import math
# import Image
from PIL import Image
from torchviz import make_dot
from torch.profiler import profile, record_function, ProfilerActivity

def main(cfg, scene, classes, gpu, save_dir, duration):
    start = time.time()
    best_mIoU = 0.0
    best_epoch = 0
    num_workers = 8
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    resume_path = cfg['MODEL']['RESUME']
    gpus = int(os.environ.get('WORLD_SIZE', '1'))

    if train_cfg.get('DDP', False) and not dist.is_initialized():
        print("[warn] TRAIN.DDP=True but torch.distributed is not initialized; falling back to single-process training. "
              "Use torchrun/torch.distributed.run to enable DDP.")
        train_cfg['DDP'] = False

    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    # trainset = eval('CarlaNew')("/mnt/sdc/lxy/datasets/carla_new_100_N", 'train', classes, traintransform, dataset_cfg['MODALS'], duration=duration, flow_net_flag=model_cfg['FLOW_NET_FLAG'], dataset_type=dataset_cfg['TYPE'])
    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'].replace("${DURATION}", str(duration)), 'train', classes, traintransform, dataset_cfg['MODALS'], duration=duration, flow_net_flag=model_cfg['FLOW_NET_FLAG'], dataset_type=dataset_cfg['TYPE'])
    # 计算补齐后的目标长度
    if len(trainset) % train_cfg['BATCH_SIZE'] != 0:
        num_batches = math.ceil(len(trainset) / train_cfg['BATCH_SIZE'])
        target_length = num_batches * train_cfg['BATCH_SIZE']
        trainset = ExtendedDSEC(trainset, target_length)
    # NOTE for carla_newd
    # valset = eval('CarlaNew')("/mnt/sdc/lxy/datasets/carla_new_100_N", 'val', classes, valtransform, dataset_cfg['MODALS'], duration=duration, flow_net_flag=model_cfg['FLOW_NET_FLAG'], dataset_type=dataset_cfg['TYPE'])
    # NOTE for sdsec
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'].replace("${DURATION}", str(duration)), 'val', classes, valtransform, dataset_cfg['MODALS'], duration=duration, flow_net_flag=model_cfg['FLOW_NET_FLAG'], dataset_type=dataset_cfg['TYPE'])
    # valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'].replace("${DURATION}", str(duration)), 'train', classes, valtransform, dataset_cfg['MODALS'], duration=duration, flow_net_flag=model_cfg['FLOW_NET_FLAG'], dataset_type=dataset_cfg['TYPE'])
    class_names = trainset.SEGMENTATION_CONFIGS[classes]["CLASSES"]

    anytime_flag = bool(model_cfg.get('ANYTIME_FLAG', False))
    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes, dataset_cfg['MODALS'], model_cfg['BACKBONE_FLAG'], model_cfg['FLOW_NET_FLAG'], dataset_type=dataset_cfg['TYPE'], anytime_flag=anytime_flag)
    resume_checkpoint = None
    if os.path.isfile(resume_path):
        resume_checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        msg = model.load_state_dict(resume_checkpoint, strict=False)
        # print("resume_checkpoint msg: ", msg)
        logger.info(msg)
    else:
        if model_cfg.get('BACKBONE_FLAG', False):
            candidate_pretrained = [model_cfg.get('PRETRAINED_BACKBONE'), model_cfg.get('PRETRAINED')]
        else:
            candidate_pretrained = [model_cfg.get('PRETRAINED'), model_cfg.get('PRETRAINED_BACKBONE')]

        candidate_pretrained = [p for p in candidate_pretrained if isinstance(p, str) and p]
        pretrained = next((p for p in candidate_pretrained if os.path.isfile(p)), None)

        if pretrained:
            if candidate_pretrained and pretrained != candidate_pretrained[0]:
                logger.warning(f"Pretrained checkpoint not found: {candidate_pretrained[0]!r}; falling back to: {pretrained!r}")
            model.init_pretrained(pretrained)
        elif candidate_pretrained:
            raise FileNotFoundError(f"No pretrained checkpoint found. Tried: {candidate_pretrained}")
    
    if model_cfg['FLOW_NET_FLAG'] and os.path.isfile(model_cfg['RESUME_FLOWNET']):
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            print('Loading flownet model...')
        flow_net_type = model_cfg['FLOW_NET']
        resume_flownet_path = model_cfg['RESUME_FLOWNET']

        if flow_net_type == 'eraft':
            ## for eraft
            if 'dsec' in resume_flownet_path:
            # if dataset_cfg['TYPE'] == 'dsec_':
                # flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'))  # for self trained flownet
                flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'))['model'] # for dsec.tar
                # flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'), weights_only=True)['model']
            # elif dataset_cfg['TYPE'] == 'dsec':
            else:
                flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'))  # for self trained flownet
                # flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'), weights_only=True)
                # 筛选出以 'flownet' 为前缀的键
                flownet_checkpoint = {
                    key: value for key, value in flownet_checkpoint.items() if key.startswith('flow_net')
                }
                # 给所有key去掉前缀 'flow_net.'
                flownet_checkpoint = {k.replace('flow_net.', ''): v for k, v in flownet_checkpoint.items()}
            if 'fnet.conv1.weight' in flownet_checkpoint:
                # delete weights of the first layer
                flownet_checkpoint.pop('fnet.conv1.weight')
                flownet_checkpoint.pop('fnet.conv1.bias')
            if 'cnet.conv1.weight' in flownet_checkpoint:
                # delete weights of the second layer
                flownet_checkpoint.pop('cnet.conv1.weight')
                flownet_checkpoint.pop('cnet.conv1.bias')
        elif flow_net_type == 'raft_small':
            flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'))
        elif flow_net_type == 'bflow':
            # for bflow
            flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'))['state_dict']
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

        flownet_msg = model.flow_net.load_state_dict(flownet_checkpoint, strict=False)
        print("flownet_checkpoint msg: ", flownet_msg)
        logger.info(flownet_msg)
        # exit(0)

    model = model.to(device)

    if dataset_cfg['TYPE'] == 'dsec':
        train_cls_weights = torch.tensor([0.8977, 0.8722, 0.9863, 0.9992, 0.9899, 0.7580, 0.9377, 0.7867, 0.9666, 0.9805, 0.9968])
    elif dataset_cfg['TYPE'] == 'sdsec':
        # train_cls_weights = torch.tensor([0.8026, 0.8507, 0.9960, 0.9993, 0.9820, 0.7291, 0.9429, 0.9069, 0.9895, 0.9913, 0.9988])
        train_cls_weights = torch.tensor([0.8026, 0.8507, 0.9960, 0.9993, 0.9820, 0.7291, 0.9429, 0.9069, 0.9895, 0.9913, 0.9999])
        # train_cls_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0])
    # train_cls_weights = torch.tensor([0.9106, 0.8747, 0.9862, 0.9990, 0.9881, 0.7629, 0.9402, 0.7753, 0.9594, 0.9789, 0.9964])
    # test_cls_weights = torch.tensor([0.8663, 0.8616, 0.9879, 0.9984, 0.9911, 0.7544, 0.9410, 0.8237, 0.9534, 0.9931, 0.9961])
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, train_cls_weights.to(device))
    start_epoch = 0
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])

    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = None
        model = DDP(model, device_ids=[gpu], output_device=0, find_unused_parameters=True)
    else:
        sampler = RandomSampler(trainset)
        sampler_val = None
    
    # if resume_checkpoint:
    #     start_epoch = resume_checkpoint['epoch'] - 1
    #     optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
    #     scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
    #     loss = resume_checkpoint['loss']        
    #     best_mIoU = resume_checkpoint['best_miou']
    
    # NOTE
    if not model_cfg['BACKBONE_FLAG']:
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            print('Freezing backbone...')
        # # 冻结除 flow_net 和 softsplat_net 之外的所有层
        for name, param in model.named_parameters():
            param.requires_grad = True
            if 'backbone' in name:
                param.requires_grad = False
        # for name, param in model.named_parameters():
        #     if not 'flow_net' in name and not 'softsplat_net' in name:
        #     # if not 'softsplat_net' in name:
        #         param.requires_grad = False
        # for name, param in model.named_parameters():
        #     if 'decode_head' in name:
        #         param.requires_grad = True
        # # 检查哪些参数被冻结了
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
        # # end
    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=sampler, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    # trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=0, drop_last=True, pin_memory=True, sampler=sampler, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=True, sampler=sampler_val, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    # valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=0, pin_memory=True, sampler=sampler_val, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    # train_cls_weights = calculate_class_weights(trainloader, num_classes=11)
    # print('train_cls_weights:', train_cls_weights)
    # exit(0)
    # event_mean, event_std = cal_event_meanandstd(trainloader)
    # print('train event mean and std:', event_mean, event_std)   # -0.0005 0.5128
    # event_mean, event_std = cal_event_meanandstd(valloader) 
    # print('val event mean and std:', event_mean, event_std) # 0.0001 0.4652
    # train_cls_weights = calculate_class_weights(trainloader, num_classes=11)
    # print('train_cls_weights:', train_cls_weights)
    # exit(0)
    iters_per_epoch = len(trainloader)
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, int((epochs+1)*iters_per_epoch), sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])

    scaler = GradScaler(enabled=train_cfg['AMP'])
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer = SummaryWriter(str(save_dir))
        logger.info('================== model complexity =====================')
        cal_flops(model, dataset_cfg['MODALS'], logger)
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
        amp_enabled = bool(train_cfg.get('AMP', False))
        for iter, (seq_names, seq_index, sample, lbls) in pbar:
            optimizer.zero_grad(set_to_none=True)
            sample = [x.to(device) for x in sample]
            if not amp_enabled:
                sample = [x.float() if x.is_floating_point() else x for x in sample]
            lbls = [lbl.to(device) for lbl in lbls]
            # event_voxel = sample[1].to(device)
            # rgb_next = sample[2].to(device)
            # flow = sample[3].to(device)
            # sample = [sample[0]]
            
            with autocast_ctx(amp_enabled):
                # logits = model(sample, event_voxel, rgb_next, flow)
                # logits, feature_loss = model(sample, event_voxel)
                logits = model(sample)
                # logits, feature_loss = model(sample, event_voxel, rgb_next, flow)
                loss = loss_fn(logits[-1], lbls[0])
                # if len(logits) == 2:
                #     # print("Mid Supervised!")
                #     loss = loss + loss_fn(logits[0], lbls[1])
                # loss = loss_fn(logits, lbl) + 0.5*feature_loss + 0.5*consistent_loss

            if not torch.isfinite(loss).item():
                print(f"[error] Non-finite loss detected at epoch={epoch+1} iter={iter+1}")
                try:
                    print("seq_names:", seq_names)
                    print("seq_index:", seq_index)
                except Exception:
                    pass
                try:
                    lbl = lbls[0]
                    ignore = int(dataset_cfg.get('IGNORE_LABEL', 255))
                    valid = (lbl != ignore).sum().item()
                    uniq = torch.unique(lbl).detach().cpu().tolist()
                    print(f"label valid_pixels={valid} unique={uniq[:64]}{'...' if len(uniq) > 64 else ''}")
                except Exception as e:
                    print("label debug failed:", repr(e))
                try:
                    out = logits[-1]
                    finite = torch.isfinite(out).all().item()
                    out_min = out.detach().float().min().item()
                    out_max = out.detach().float().max().item()
                    print(f"logits[-1] finite={finite} min={out_min:.6g} max={out_max:.6g} shape={tuple(out.shape)} dtype={out.dtype}")
                except Exception as e:
                    print("logits debug failed:", repr(e))
                for si, sx in enumerate(sample):
                    if torch.is_tensor(sx) and sx.is_floating_point():
                        finite = torch.isfinite(sx).all().item()
                        smin = sx.detach().float().min().item()
                        smax = sx.detach().float().max().item()
                        print(f"sample[{si}] finite={finite} min={smin:.6g} max={smax:.6g} shape={tuple(sx.shape)} dtype={sx.dtype}")
                raise FloatingPointError("Non-finite loss (NaN/Inf). See debug prints above.")

            scaler.scale(loss).backward()
            # scaler.scale(loss).backward(retain_graph=True)
            # # 可视化计算图
            # dot = make_dot(loss, params=dict(model.named_parameters()))
            # dot.format = 'png'
            # dot.render('cmnext_computation_graph')

            # # 检查 netSoftmetric 和 netWarp 的梯度
            # print(model.module.softsplat_net.netSoftmetric.netEventInput.weight.grad)
            # print(model.module.softsplat_net.netWarp.nets[0].netMain[1].weight.grad)
            # exit(0)
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            # 在优化器步骤之前，我们使用梯度裁剪
            # # 对于模型的每个参数，计算其梯度的L2范数
            # for param in model.parameters():
            #     if param.grad is not None:
            #         grad_norm = torch.norm(param.grad, p=2)
            #         print(grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
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
        torch.cuda.empty_cache()

        if ((epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch+1)>train_cfg['EVAL_START']) or (epoch+1) == epochs:
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                acc, macc, _, _, ious, miou = evaluate(model, valloader, device)
                writer.add_scalar('val/mIoU', miou, epoch)

                if miou > best_mIoU:
                    prev_best_ckp = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    prev_best = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    if os.path.isfile(prev_best): os.remove(prev_best)
                    if os.path.isfile(prev_best_ckp): os.remove(prev_best_ckp)
                    best_mIoU = miou
                    best_epoch = epoch+1
                    cur_best_ckp = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    cur_best = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), cur_best)
                    # --- 
                    torch.save({'epoch': best_epoch,
                                'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_miou': best_mIoU,
                                }, cur_best_ckp)
                    logger.info(print_iou(epoch, ious, miou, acc, macc, class_names))
                logger.info(f"Current epoch:{epoch} mIoU: {miou} Best mIoU: {best_mIoU}")

    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
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
