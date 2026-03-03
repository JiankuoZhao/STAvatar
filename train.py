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

import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.loss_utils import (
    l1_loss,
    compute_per_gs_error,
    compute_ssim_map,
)
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render
import sys
from scene import Scene, FlameGaussianModel
from utils.general_utils import safe_state
import uuid
import yaml
import json
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from kiui.lpips import LPIPS
from PIL import Image
from scene.cluster_loader import create_clustered_dataloaders
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from networks.dual_branch import DualBranchUNet

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    debug_from,
):
    """
    Main training loop.

    This function initializes the Gaussian model and DualBranchUNet,
    performs clustered training with densification,
    and periodically evaluates and saves checkpoints.
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = FlameGaussianModel(
        dataset.sh_degree,
        dataset.disable_flame_static_offset,
        dataset.not_finetune_flame_params,
    )

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    camera_dataset = scene.getTrainCameras()
    train_data_len = len(camera_dataset)

    #========== prepare input for prediction ==========
    position_map = gaussians.get_position_map()
    uv_mask = gaussians.get_uv_mask()
    uv_coords = gaussians.uvcoords_sample()
    ref_image_path = os.path.join(dataset.source_path, dataset.images, "00000_00.png")
    if not os.path.exists(ref_image_path):
        raise FileNotFoundError(f"Reference image not found at: {ref_image_path}")
    reference_image = Image.open(ref_image_path).convert("RGB")
    
    #========== Initialize the network (DualBranchUNet) ==========
    dual_branch_net = DualBranchUNet(
        device="cuda",
        uv_sample_coords=uv_coords,
        uv_mask=uv_mask,
        reference_image=reference_image,
        position_map=position_map,
    )
    dual_branch_net.to("cuda")
    dual_branch_net.train()
    optimizer_net = torch.optim.Adam(dual_branch_net.parameters(), lr=opt.dual_branch_lr)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    #========== Cluster Sample ==========
    cluster_indices, opt.num_clusters = gaussians.cluster_flame_conditions(train_data_len)
    clustered_dataloaders = create_clustered_dataloaders(
        camera_dataset,
        cluster_indices
    )
    cluster_epochs = [0] * opt.num_clusters
    cluster_indices = list(range(opt.num_clusters))
    cluster_pointer = 0
    cluster_id = cluster_indices[cluster_pointer]
    loader = clustered_dataloaders[cluster_id]
    loader_iter = iter(loader)
    class_densify = False
    cluster_densify_done = False
    cluster_epoch_limit = opt.epochs
    opt.iterations = opt.epochs * train_data_len

    if opt.final_refine_epoch:
        cluster_epoch_limit = opt.epochs - 1
        full_loader = DataLoader(
            camera_dataset,
            batch_size=None,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

    # construct test iterations and save iteraions
    testing_iterations.extend([opt.iterations // 2, opt.iterations])
    saving_iterations.extend([opt.iterations // 2, opt.iterations])

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    lpips_loss = LPIPS(net="vgg").to("cuda")
    lpips_loss.requires_grad_(False)

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        #=========== Cluster training ===========
        try:
            viewpoint_cam = next(loader_iter)
        except StopIteration:
            cluster_epochs[cluster_id] += 1

            if cluster_epochs[cluster_id] == 2:
                class_densify = True

            # finish current clusters
            if cluster_epochs[cluster_id] >= cluster_epoch_limit:
                cluster_pointer = cluster_pointer + 1
                # finish all clusters
                if cluster_pointer >= opt.num_clusters:
                    if opt.final_refine_epoch:
                        loader_iter = iter(full_loader)
                        viewpoint_cam = next(loader_iter)
                    else:
                        break
                else:
                    # change to next cluster
                    cluster_densify_done = False
                    cluster_id = cluster_indices[cluster_pointer]
                    loader = clustered_dataloaders[cluster_id]
                    loader_iter = iter(loader)
                    viewpoint_cam = next(loader_iter)
            # change to current cluster's next epoch
            else:
                loader_iter = iter(loader)
                viewpoint_cam = next(loader_iter)

        gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)
        displacement_map = gaussians.get_vertex_displace_map()

        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Forward pass through dual branch network
        gt_image = viewpoint_cam.original_image.cuda()
        offset = dual_branch_net(gaussians.flame_param, viewpoint_cam.timestep, displacement_map)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, offset=offset)

        (image, viewspace_point_tensor, visibility_filter, radii) = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"]
        )

        # Loss
        losses = {}
        losses["l1"] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
        losses["ssim"] = (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))) * opt.lambda_dssim
        if iteration > opt.iterations / 2:
            losses["lpips"] = lpips_loss(image, gt_image) * opt.lambda_lpips
        losses["offset_scale_reg"] = (torch.abs(offset[0, :, 3:6] - 1).mean() * opt.lambda_offset_scale_reg)
        losses["offset_color_reg"] = (torch.abs(offset[0, :, 10:13]).mean() * opt.lambda_offset_color_reg)

        if opt.lambda_scale != 0:
            losses["scale"] = (F.relu(torch.exp(gaussians._scaling[visibility_filter]) - 
            opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale)
        
        losses["total"] = sum([v for _, v in losses.items()])
        losses["total"].backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * losses["total"].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if "scale" in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration > opt.iterations:
                progress_bar.close()
            
            if not cluster_densify_done:
                # keep track every gs image error
                l1_map = torch.abs(image - gt_image).mean(dim=0)
                ssim_map = torch.abs(compute_ssim_map(image, gt_image))
                error_map = (1 - opt.lambda_fpe) * l1_map + opt.lambda_fpe * ssim_map
                gs_image_error = compute_per_gs_error(
                    error_map,
                    render_pkg["gs_centers"],
                    render_pkg["accum_alpha_per_gs"],
                    render_pkg["pixels_per_gs"],
                )

                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, gs_image_error)

                if class_densify:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, None)
                    # resample uv coords after densify
                    uv_coords = gaussians.uvcoords_sample()
                    dual_branch_net.update_uv_coords(uv_coords)

                    class_densify = False
                    cluster_densify_done = True

            if iteration <= opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                optimizer_net.step()
                optimizer_net.zero_grad(set_to_none=True)

            # Log and save
            training_report(
                tb_writer,
                iteration,
                losses,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                dual_branch_net,
                dataset.model_path,
            )

            if iteration in saving_iterations:
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                print("[ITER {}] Saving Network and Uv sample coordinates".format(iteration))
                save_dir = os.path.join(scene.model_path, f"param/iteration_{iteration}")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    dual_branch_net.state_dict(),
                    os.path.join(save_dir, f"dual_branch.pth"),
                )
                torch.save(uv_coords, os.path.join(save_dir, f"uv_coords.pt"))

def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") if os.getenv("OAR_JOB_ID") else str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f: f.write(str(Namespace(**vars(args))))

    return SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None


def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene, renderFunc, renderArgs, dual_branch_net, save_path):
    """
    Log training statistics and run validation.

    Computes L1, PSNR, SSIM and LPIPS metrics on validation and test sets,
    and writes results to TensorBoard and evaluation.json.
    """
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", losses["l1"].item(), iteration)
        tb_writer.add_scalar("train_loss_patches/ssim_loss", losses["ssim"].item(), iteration)
        if "scale" in losses:
            tb_writer.add_scalar("train_loss_patches/scale_loss", losses["scale"].item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", losses["total"].item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "val", "cameras": scene.getValCameras()},
            {"name": "test", "cameras": scene.getTestCameras()},
        )
        lpips_loss = LPIPS(net="alex").to("cuda")
        lpips_loss.requires_grad_(False)

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(
                    enumerate(DataLoader(config["cameras"],shuffle=False,batch_size=None,num_workers=8)),
                    total=len(config["cameras"]),
                ):

                    scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    vertex_displace_map = scene.gaussians.get_vertex_displace_map()
                    offset = dual_branch_net(scene.gaussians.flame_param, viewpoint.timestep, vertex_displace_map)

                    image = torch.clamp(
                        renderFunc(
                            viewpoint,
                            scene.gaussians,
                            *renderArgs,
                            offset=offset
                        )["render"],
                        0.0,
                        1.0,
                    )

                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (
                        idx % (len(config["cameras"]) // num_vis_img) == 0
                    ):
                        tb_writer.add_images(
                            config["name"] + "_{}/render".format(vis_ct),
                            image[None],
                            global_step=iteration,
                        )
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(
                            config["name"] + "_{}/error".format(vis_ct),
                            error_image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + "_{}/ground_truth".format(vis_ct),
                                gt_image[None],
                                global_step=iteration,
                            )
                        vis_ct += 1
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += (fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double())

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config["cameras"]) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips_loss(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                lpips_test /= len(config["cameras"])
                ssim_test /= len(config["cameras"])

                print(
                    "[ITER {}] Evaluating {}: l1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(
                        iteration,
                        config["name"],
                        l1_test,
                        psnr_test,
                        ssim_test,
                        lpips_test,
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - lpips",lpips_test,iteration)

                # Create a dictionary to store results for current iteration
                result_entry = {
                    str(iteration): {
                        "PSNR": round(psnr_test.item(), 2),
                        "SSIM": round(ssim_test.item(), 4),
                        "LPIPS": round(lpips_test.item(), 4),
                        "L1 Loss": round(l1_test.item(), 4),
                    }
                }

                results_path = os.path.join(save_path, "evaluation.json")
                if os.path.exists(results_path):
                    with open(results_path, "r") as f:
                        all_results = json.load(f)
                else:
                    all_results = {}

                all_results.update(result_entry)
                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=4)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def save_config_to_yml(args, lp, op, pp, output_dir):
    """save config.yml"""
    config_dict = vars(args)

    lp_params = vars(lp)
    op_params = vars(op)
    pp_params = vars(pp)

    config_dict.update(
        {
            "ModelParams": lp_params,
            "OptimizationParams": op_params,
            "PipelineParams": pp_params,
        }
    )

    config_path = os.path.join(output_dir, "config.yml")

    with open(config_path, "w") as config_file:
        yaml.dump(config_dict, config_file, default_flow_style=False)
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--interval", type=int, default=-1, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[], help="the iteration to save checkpoint")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--ptvsd", action="store_true", help="whether to debug")
    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.cuda.set_device(torch.device("cuda:0"))

    # save config.yml
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path, exist_ok=True)
    save_config_to_yml(args, lp, op, pp, args.model_path)

    # configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.debug_from,
    )

    print("\nTraining complete.")