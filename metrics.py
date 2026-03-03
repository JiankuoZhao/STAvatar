#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim, l1_loss
from kiui.lpips import LPIPS
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, specify_iteration=[]):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    lpips_loss = LPIPS(net='alex').to('cuda')
    lpips_loss.requires_grad_(False)

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            if len(specify_iteration) != 0:
                methods = []
                for method in os.listdir(test_dir):
                    parts = method.split('_')
                    iteration = int(parts[1])
                    if iteration in specify_iteration:
                        methods.append(method)
            else:
                methods = os.listdir(test_dir)

            for method in methods:
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                l1s = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_loss(renders[idx], gts[idx]))
                    l1s.append(l1_loss(renders[idx], gts[idx]))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  L1 Loss: {:>12.7f}".format(torch.tensor(l1s).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({
                    "SSIM": round(torch.tensor(ssims).mean().item(), 4),
                    "PSNR": round(torch.tensor(psnrs).mean().item(), 2),
                    "LPIPS": round(torch.tensor(lpipss).mean().item(), 4),
                    "L1 Loss": round(torch.tensor(l1s).mean().item(), 4)
                })
                per_view_dict[scene_dir][method].update({"SSIM": {name: round(ssim, 4) for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: round(psnr, 2) for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: round(lp, 4) for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                        "L1 Loss": {name: round(l1, 4) for l1, name in zip(torch.tensor(l1s).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--specify_iteration', type=str, default=[], help='specify what iterations to test')
    
    args = parser.parse_args()
    iterations = [int(i) for i in args.specify_iteration[0].split(',')] if args.specify_iteration else []
    evaluate(args.model_paths, iterations)