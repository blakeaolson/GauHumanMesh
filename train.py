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
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import imageio
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

import time
import torch.nn.functional as F
from utils.general_utils import build_rotation

def compute_sugar_regularization_canonical(gaussians, iteration, cache, num_samples=5000, k=16, update_interval=500):
    """
    Computes both the SuGaR density loss and normal alignment loss.
    """
    means = gaussians.get_xyz 
    opacities = gaussians.get_opacity
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    
    num_gaussians = means.shape[0]
    num_samples = min(num_samples, num_gaussians)
    
    if cache.get('knn_idx') is None or (iteration % update_interval == 0):
        dists = torch.cdist(means, means)
        _, idx_all = torch.topk(dists, k, dim=-1, largest=False)
        cache['knn_idx'] = idx_all
        
    knn_idx_all = cache['knn_idx']
    
    indices = torch.randint(0, num_gaussians, (num_samples,), device="cuda")
    R_all = build_rotation(rotations) 
    sampled_means = means[indices]
    sampled_scales = scales[indices]
    sampled_R = R_all[indices]
    
    z = torch.randn((num_samples, 3, 1), device="cuda")
    Sz = sampled_scales.unsqueeze(-1) * z 
    RSz = torch.bmm(sampled_R, Sz).squeeze(-1) 
    p = sampled_means + RSz
    
    idx = knn_idx_all[indices]  
    
    nn_means = means[idx]       
    nn_alphas = opacities[idx]  
    nn_scales = scales[idx]     
    nn_R = R_all[idx]           
    
    diff = p.unsqueeze(1) - nn_means 
    diff = diff.unsqueeze(-1)        
    
    inv_sq_scales = 1.0 / (nn_scales ** 2 + 1e-6) 
    v = torch.matmul(nn_R.transpose(-1, -2), diff).squeeze(-1) 
    power = torch.sum((v ** 2) * inv_sq_scales, dim=-1) 
    
    # Actual Density d(p)
    exp_term = nn_alphas.squeeze(-1) * torch.exp(-0.5 * power)
    d_p = torch.sum(exp_term, dim=-1)
    
    # --- NEW: Analytical Gradient of Density for R_Norm ---
    # Gradient = sum( alpha * exp(...) * (-Sigma^-1 * (p - mu)) )
    sigma_inv_diff = torch.matmul(nn_R, (v * inv_sq_scales).unsqueeze(-1)).squeeze(-1)
    grad_d = torch.sum(exp_term.unsqueeze(-1) * (-sigma_inv_diff), dim=1) 
    
    # Normalize gradient to get the smooth surface normal
    surface_normals = F.normalize(grad_d, p=2, dim=-1) 
    # ------------------------------------------------------
    
    g_star_idx = idx[:, 0]
    g_star_means = means[g_star_idx]
    g_star_scales = scales[g_star_idx]
    g_star_R = R_all[g_star_idx]
    
    min_scales, min_idx = torch.min(g_star_scales, dim=-1)
    batch_indices = torch.arange(num_samples, device="cuda")
    
    # n_g is the normal vector of the Gaussian's flat face
    n_g = g_star_R[batch_indices, :, min_idx] 
    
    dot_prod = torch.sum((p - g_star_means) * n_g, dim=-1)
    
    power_bar = (dot_prod ** 2) / (min_scales ** 2 + 1e-6)
    d_bar_p = torch.exp(-0.5 * power_bar) 
    
    # 1. Density Loss
    density_loss = torch.mean(torch.abs(d_p - d_bar_p))
    
    # 2. Normal Alignment Loss (R_Norm)
    # We penalize 1.0 - absolute dot product to handle sign ambiguity 
    # (n_g and -n_g represent the same physical Gaussian orientation)
    alignment = torch.sum(surface_normals * n_g, dim=-1)
    normal_loss = torch.mean(1.0 - torch.abs(alignment))
    
    return density_loss, normal_loss

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    Ll1_loss_for_log = 0.0
    mask_loss_for_log = 0.0
    ssim_loss_for_log = 0.0
    lpips_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    sugar_cache = {'knn_idx': None}
    # lpips_test_lst = []

    elapsed_time = 0
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Start timer
        start_time = time.time()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        bkgd_mask = viewpoint_cam.bkgd_mask.cuda()
        bound_mask = viewpoint_cam.bound_mask.cuda()
        Ll1 = l1_loss(image.permute(1,2,0)[bound_mask[0]==1], gt_image.permute(1,2,0)[bound_mask[0]==1])
        mask_loss = l2_loss(alpha[bound_mask==1], bkgd_mask[bound_mask==1])

        # crop the object region
        x, y, w, h = cv2.boundingRect(bound_mask[0].cpu().numpy().astype(np.uint8))
        img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
        img_gt = gt_image[:, y:y + h, x:x + w].unsqueeze(0)
        # ssim loss
        ssim_loss = ssim(img_pred, img_gt)
        # lipis loss
        lpips_loss = loss_fn_vgg(img_pred, img_gt).reshape(-1)

        loss = Ll1 + 0.1 * mask_loss + 0.01 * (1.0 - ssim_loss) + 0.01 * lpips_loss

        # density regularization SuGaR loss
        sugar_start_iter = 1200
        
        if iteration == sugar_start_iter:
            print(f"\n[ITER {iteration}] Initiating SuGaR Fine-Tuning.")
            prune_mask = (gaussians.get_opacity > 0.5).squeeze()
            gaussians.prune_points(prune_mask)
            sugar_cache['knn_idx'] = None
            
        if iteration >= sugar_start_iter:
            density_loss, normal_loss = compute_sugar_regularization_canonical(
                gaussians, 
                iteration=iteration, 
                cache=sugar_cache, 
                num_samples=5000, 
                k=16, 
                update_interval=500
            )            
            loss += 0.1 * density_loss
            if iteration % 500 == 0:
                print("[ITER {}] SuGaR Density Loss: {:.6f}, Normal Loss: {:.6f}".format(iteration, density_loss.item(), normal_loss.item()))

        loss.backward()

        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += (end_time - start_time)

        if (iteration in testing_iterations):
            print("[Elapsed time]: ", elapsed_time) 

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            Ll1_loss_for_log = 0.4 * Ll1.item() + 0.6 * Ll1_loss_for_log
            mask_loss_for_log = 0.4 * mask_loss.item() + 0.6 * mask_loss_for_log
            ssim_loss_for_log = 0.4 * ssim_loss.item() + 0.6 * ssim_loss_for_log
            lpips_loss_for_log = 0.4 * lpips_loss.item() + 0.6 * lpips_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"#pts": gaussians._xyz.shape[0], "Ll1 Loss": f"{Ll1_loss_for_log:.{3}f}", "mask Loss": f"{mask_loss_for_log:.{2}f}",
                                          "ssim": f"{ssim_loss_for_log:.{2}f}", "lpips": f"{lpips_loss_for_log:.{2}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Start timer
            start_time = time.time()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, kl_threshold=0.4, t_vertices=viewpoint_cam.big_pose_world_vertex, iter=iteration)
                    # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, 1)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # end time
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time += (end_time - start_time)

            # if (iteration in checkpoint_iterations):
            if (iteration in testing_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
def prepare_output_and_logger(args):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        # args.model_path = os.path.join("./output/", unique_str[0:10])
        args.model_path = os.path.join("./output/", args.exp_name)

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
        #                       {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        smpl_rot = {}
        smpl_rot['train'], smpl_rot['test'] = {}, {}
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0: 
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    smpl_rot[config['name']][viewpoint.pose_id] = {}
                    render_output = renderFunc(viewpoint, scene.gaussians, *renderArgs, return_smpl_rot=True)
                    image = torch.clamp(render_output["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    bound_mask = viewpoint.bound_mask
                    image.permute(1,2,0)[bound_mask[0]==0] = 0 if renderArgs[1].sum().item() == 0 else 1 
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += loss_fn_vgg(image, gt_image).mean().double()

                    smpl_rot[config['name']][viewpoint.pose_id]['transforms'] = render_output['transforms']
                    smpl_rot[config['name']][viewpoint.pose_id]['translation'] = render_output['translation']

                l1_test /= len(config['cameras']) 
                psnr_test /= len(config['cameras'])   
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])      
                print("\n[ITER {}] Evaluating {} #{}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], len(config['cameras']), l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        # Store data (serialize)
        save_path = os.path.join(scene.model_path, 'smpl_rot', f'iteration_{iteration}')
        os.makedirs(save_path, exist_ok=True)
        with open(save_path+"/smpl_rot.pickle", 'wb') as handle:
            pickle.dump(smpl_rot, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_200, 2_000, 3_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_200, 2_000, 3_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
