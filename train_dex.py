import cv2
import torch
import os
from basicsr.utils import scandir, get_time_str, get_root_logger
from ldm.data.dataset_reg import dataset_dex
import argparse
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.modules.encoders.adapter import CoAdapter
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import os.path as osp
from basicsr.utils.options import copy_opt_file, dict2str
import logging
from dist_util import init_dist, master_only, get_bare_model, get_dist_info
from ldm.modules.midas.api import MiDaSInference
from tqdm import tqdm
from ldm.util import load_model_from_config, read_state_dict
import torchvision.utils as tvu

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = read_state_dict(ckpt)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(pl_sd, strict=False)
    if len(m) > 0:
        print("missing keys:")
        print(m)
    if len(u) > 0:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model

@master_only
def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(experiments_root, 'models'))
    os.makedirs(osp.join(experiments_root, 'training_states'))
    os.makedirs(osp.join(experiments_root, 'visualization'))

def load_resume_state(opt):
    resume_state_path = None
    if opt.auto_resume:
        state_path = osp.join('experiments', opt.name, 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt.resume_state_path = resume_state_path
    # else:
    #     if opt['path'].get('resume_state'):
    #         resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location='cpu')
        # check_resume(opt, resume_state['iter'])
        # lambda storage, loc: storage.cuda(device_id)
    return resume_state

def load_adapter_state(opt): # add the adapter and sd states
    resume_state_path = None
    resume_sd_path = None
    if opt.adapter_ckpt:
        resume_state_path = opt.adapter_ckpt
        opt.resume_model_path = resume_state_path
    elif opt.auto_resume:
        state_path = osp.join('experiments', opt.name, 'training_states')
        model_path = osp.join('experiments', opt.name, 'models')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(model_path, f'model_ad_{max(states):.0f}.pth')
                opt.resume_model_path = resume_state_path
                if not opt.sd_lock:
                    resume_sd_path = osp.join(model_path, f'model_sd_{max(states):.0f}.ckpt')

    if resume_sd_path is None: 
        resume_sd_state = None
    else:
        print(f"Loading from {resume_sd_path}")
        sd_dict = read_state_dict(resume_sd_path, device = 'cpu')
        new_state_dict = {}
        for k, v in sd_dict.items():
            new_state_dict['module.'+k] = v
        resume_sd_state = new_state_dict

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        print(f"Loading from {resume_state_path}")
        state_dict = read_state_dict(resume_state_path, device = 'cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('adapter.'):
                new_state_dict[k[len('adapter.'):]] = v
            else:
                new_state_dict['module.'+k] = v
        resume_state = new_state_dict

    return resume_state, resume_sd_state
        
def depth2norm(depth_tensor, bg_th=0.1, a=np.pi * 2.0):
    depth_pt = depth_tensor.clone()
    depth_min, depth_max = torch.amin(depth_pt, dim=[1, 2, 3], keepdim=True), torch.amax(depth_pt, dim=[1, 2, 3], keepdim=True)
    depth_pt = (depth_pt - depth_min) / (depth_max - depth_min)
    depth_pt = depth_pt.cpu().numpy()

    batch_size = depth_pt.shape[0]
    normals = []
    for i in range(batch_size):
        depth_np = depth_tensor[i,0].cpu().numpy()
        x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
        y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
        z = np.ones_like(x) * a
        x[depth_pt[i,0] < bg_th] = 0
        y[depth_pt[i,0] < bg_th] = 0
        normal = np.stack([x, y, z], axis=2)
        normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
        normal_image = ( ((normal+1)/2).transpose(2,0,1).clip(0, 1) )
        normals.append(torch.tensor(normal_image).unsqueeze(0).float())
    normals = torch.cat(normals, dim=0)
    return normals

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="the path to the data file"
)
parser.add_argument(
    "--reg_data",
    type=str,
    default="",
    help="the path to the regularization data file"
)
parser.add_argument(
    "--bsize",
    type=int,
    default=8,
    help="batch size"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10000,
    help="total training epoch numbers"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    help="number of workers"
)
parser.add_argument(
    "--use_shuffle",
    type=bool,
    default=True,
    help="whether use shuffle in data loading"
)
parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
)
parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
)
parser.add_argument(
        "--auto_resume",
        action='store_true',
        help="auto resume training",
)
parser.add_argument(
        "--ckpt",
        type=str,
        default="ckp/sd-v1-4.ckpt",
        help="path to checkpoint of sd backbone model",
)
parser.add_argument(
        "--adapter_ckpt",
        type=str,
        default="",
        help="path to checkpoint of adapter condition model",
)
parser.add_argument(
        "--config",
        type=str,
        default="configs/train_dex.yaml",
        help="path to config which constructs model",
)
parser.add_argument(
        "--print_fq",
        type=int,
        default=100,
        help="print log frequency",
)
parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--bg_th",
    type=float,
    default=0.1,
    help="normalization background threshold",
)
parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
)
parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)

parser.add_argument(
        '--local_rank', 
        default=0, 
        type=int,
        help='node rank for distributed training'
)
parser.add_argument(
        '--launcher', 
        default='pytorch', 
        type=str,
        help='node rank for distributed training'
)
parser.add_argument(
        "--sd_lock",
        action="store_true",
        help="Lock the decoder or not",
)
parser.add_argument(
        '--name', 
        default="train_dex", 
        type=str,
        help='experiment path name'
)
parser.add_argument(
        '--lr', 
        default=1e-5, 
        type=float,
        help='learning rate'
)
parser.add_argument(
        '--save_freq', 
        default=5000, 
        type=int,
        help='model save frequency'
)
parser.add_argument(
        '--reg_prob', 
        default=0, 
        type=float,
        help='regularization data probability'
)
opt = parser.parse_args()

if __name__ == '__main__':
    config = OmegaConf.load(f"{opt.config}")

    # distributed setting
    init_dist(opt.launcher)
    torch.backends.cudnn.benchmark = True
    device='cuda'
    torch.cuda.set_device(opt.local_rank)

    # dataset,ignore validation temporarily to faster the training
    path_json_train = opt.data #'/data/mez005/data/oakink/odh_prompt.csv'
    # reg_json = "/data/mez005/data/reg/prompt.csv"
    if opt.reg_prob > 0 :
        assert opt.reg_data != "", "Regularization data file path should be provided when regularization prob > 0"
    reg_json = opt.reg_data
    train_dataset = dataset_dex(path_json_train, reg_json)
    print("=========Dataset Length========")
    print(len(train_dataset))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.bsize,
            shuffle=(train_sampler is None),
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=train_sampler)
    
    # depth generator
    midas = MiDaSInference(model_type="dpt_hybrid")
    midas.cuda()
    # Initialize sd model and adapter
    model = load_model_from_config(config, f"{opt.ckpt}").to(device)
    model_ad = CoAdapter(w1 = 1, w2 = 1, w3 = 1).to(device) 
    # to gpus
    model_ad = torch.nn.parallel.DistributedDataParallel(
        model_ad,
        device_ids=[opt.local_rank], 
        output_device=opt.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[opt.local_rank], 
        output_device=opt.local_rank)
    midas = torch.nn.parallel.DistributedDataParallel(
        midas,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)
    
    # optimizer
    config['training']['lr'] = opt.lr
    if opt.save_freq != 0:
        config['training']['save_freq'] = opt.save_freq

    params = list(model_ad.parameters())
    if not opt.sd_lock:
       print("====Unlock SD, add SD parameters====")
       params += list(model.module.model.diffusion_model.output_blocks.parameters())
       params += list(model.module.model.diffusion_model.out.parameters())
    optimizer = torch.optim.AdamW(params, lr=config['training']['lr'])

    experiments_root = osp.join('experiments', opt.name)

    # resume state
    resume_state = load_resume_state(opt)
    resume_model, resume_sd_model = load_adapter_state(opt)

    if resume_model is not None:
        print("Resuming Adapter model parameters")
        m, u = model_ad.load_state_dict(resume_model, strict=False)
        if len(m) > 0 :
            print("missing adapter key:")
            print(m)
        if len(u) > 0:
            print("unexpected adapter key:")
            print(u)
    
    if resume_sd_model is not None:
        m, u = model.load_state_dict(resume_sd_model, strict=False)
        if len(m) > 0 :
            print("missing new sd key:")
            print(m)
        if len(u) > 0:
            print("unexpected new sd key:")
            print(u)

    if resume_state is None:
        mkdir_and_rename(experiments_root)
        start_epoch = 0
        current_iter = 0
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='HOI_Train', log_level=logging.INFO, log_file=log_file)
        logger.info(dict2str(config))
    else:
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='HOI_Train', log_level=logging.INFO, log_file=log_file)
        logger.info(dict2str(config))
        resume_optimizers = resume_state['optimizers']
        optimizer.load_state_dict(resume_optimizers)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        
    del resume_state
    del resume_sd_model
    del resume_model

    # copy the yml file to the experiment root
    copy_opt_file(opt.config, experiments_root)

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    for epoch in tqdm(range(start_epoch, opt.epochs)):
        train_dataloader.sampler.set_epoch(epoch)
        # train
        for _, data in tqdm(enumerate(train_dataloader)):
            current_iter += 1
            # import torchvision.utils as tvu
            # import random
            # tvu.save_image(data["im"],f"./debug/{random.random()}.jpg")
            prob = random.random()
            if prob > opt.reg_prob: # normal training mode
                data = data["data"]
                with torch.no_grad():
                    bs = data["depth"].shape[0]
                    # Depth_image
                    normal_batch = []
                    cc = midas(data["depth"].cuda(non_blocking=True))
                    normals = depth2norm(cc, opt.bg_th).cuda(non_blocking=True)
                    normals = torch.nn.functional.interpolate(
                        normals,
                        size=(512, 512),
                        mode="bicubic",
                        align_corners=False,
                    )
                    cc = normals * (data["seg"].cuda(non_blocking=True))
                normal = cc.float()
                skeleton = data["skeleton"]
                mask = data["mask"].float()
            else:
                data = data["reg_data"]
                normal = torch.zeros_like(data['im']).float()
                skeleton = torch.zeros_like(data['im']).float()
                mask = torch.zeros_like(data['im']).float()
            
            with torch.no_grad():
                c = model.module.get_learned_conditioning(data['sentence'])
                z = model.module.encode_first_stage((data['im']*2-1.).cuda(non_blocking=True))
                z = model.module.get_first_stage_encoding(z)

            optimizer.zero_grad()
            model.zero_grad()
            features_adapter = model_ad(skeleton, normal, mask)
            l_pixel, loss_dict = model(z, c=c, features_adapter = features_adapter)
            l_pixel.backward()
            optimizer.step()

            if (current_iter+1)%opt.print_fq == 0:
                logger.info(loss_dict)
            
            # save checkpoint
            rank, _ = get_dist_info()
            
            if (rank==0) and ((current_iter+1)%config['training']['save_freq'] == 0):
                save_filename = f'model_ad_{current_iter+1}.pth'
                save_path = os.path.join(experiments_root, 'models', save_filename)
                save_dict = {}
                model_ad_bare = get_bare_model(model_ad)
                state_dict = model_ad_bare.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    save_dict[key] = param.cpu()
                print("===Saving Adapter model====")
                torch.save(save_dict, save_path)

                # save SD model
                if not opt.sd_lock:
                    print("===Saving New SD model===")
                    save_sd_filename = f'model_sd_{current_iter+1}.ckpt'
                    save_sd_path = os.path.join(experiments_root, 'models', save_sd_filename)
                    save_dict = {}
                    model_bare = get_bare_model(model)
                    state_dict = model_bare.state_dict()
                    for key, param in state_dict.items():
                        if key.startswith('module.'):  # remove unnecessary 'module.'
                            key = key[7:]
                        save_dict[key] = param.cpu()
                    torch.save(save_dict, save_sd_path)

                # save state
                print("===Saving Training Status===")
                state = {'epoch': epoch, 'iter': current_iter+1, 'optimizers': optimizer.state_dict()}
                save_filename = f'{current_iter+1}.state'
                save_path = os.path.join(experiments_root, 'training_states', save_filename)
                torch.save(state, save_path)