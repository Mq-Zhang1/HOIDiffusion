import os
import cv2
import torch
from tqdm import tqdm
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast
from dist_util import init_dist, master_only, get_bare_model, get_dist_info
from ldm.inference_base import (diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)
from ldm.data.dataset_grabnet import dataset_grabnet

torch.set_grad_enabled(False)


def main():
    supported_cond = [e.name for e in ExtraCondition]
    parser = get_base_argument_parser()
    parser.add_argument(
        '--which_cond',
        type=str,
        required=True,
        choices=supported_cond,
        help='which condition modality you want to test',
    )
    parser.add_argument(
        "--bs", #num_images
        type=int,
        default=8,
        help="batch size for testing"
    )
    parser.add_argument( 
        '--input', #obj
        type=str,
        required=True,
        help='the path to the test condition image data folder'
    )
    parser.add_argument(
        '--file', 
        default='test.csv', 
        type=str,
        help='testing data file'
    )
    parser.add_argument(
        "--bg_th",
        type=float,
        default=0.1,
        help="background threshold"
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
    opt = parser.parse_args()

    # distributed setting
    init_dist(opt.launcher)
    torch.cuda.set_device(opt.local_rank)
    which_cond = opt.which_cond
    if opt.outdir is None:
        opt.outdir = os.path.join(opt.input, "render")
    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)
    if opt.resize_short_edge is None:
        print(f"you don't specify the resize_shot_edge, so the maximum resolution is set to {opt.max_resolution}")
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Get model conditioning
    cond_model = None
    if opt.cond_inp_type == 'image':
        cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond)) # in our case, it is depth estimated data
    process_cond_module = getattr(api, f'get_cond_{which_cond}')

    # prepare models
    sd_model, sampler = get_sd_models(opt)
    adapter = get_adapters(opt, getattr(ExtraCondition, which_cond))
    sd_model = torch.nn.parallel.DistributedDataParallel(
        sd_model,
        device_ids=[opt.local_rank], 
        output_device=opt.local_rank)
    adapter["model"] = torch.nn.parallel.DistributedDataParallel(
        adapter["model"],
        device_ids=[opt.local_rank], 
        output_device=opt.local_rank)
    cond_model = torch.nn.parallel.DistributedDataParallel(
        cond_model,
        device_ids=[opt.local_rank], 
        output_device=opt.local_rank)

    # Prepare test data, temporarily obtain from training data
    path_json_test = os.path.join(opt.input, opt.file)
    use_depth_image = False
    train_dataset = dataset_grabnet(path_json_test)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.bs,
            shuffle=(train_sampler is None),
            num_workers=4,
            pin_memory=True,
            sampler=train_sampler)
    
    # inference
    seed_everything(opt.seed)
    with torch.inference_mode(), \
            torch.no_grad(),\
            autocast('cuda'):
        for _, batch in tqdm(enumerate(train_dataloader)):
            cond_mask = batch["seg"]
            depth = process_cond_module(opt, batch["depth"], cond_seg = None, cond_model = cond_model, cond_mask = cond_mask)
            adapter_features, append_to_context = get_adapter_feature(batch["skeleton"].half().cuda(), depth.half().cuda(), batch["mask"].half().cuda(), adapter)
            opt.prompt = batch["sentence"]
            result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context)
            for i in range(result.shape[0]):
                cv2.imwrite(os.path.join(opt.outdir, f'{batch["name"][i]}.jpg'), tensor2img(result[i:i+1]))


if __name__ == '__main__':
    main()
