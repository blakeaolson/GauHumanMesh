import torch
import os
import time
import pickle
import trimesh
import numpy as np
import nvdiffrast.torch as dr
from tqdm import tqdm
from os import makedirs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
import lpips
import torch.nn.functional as F

# Initialize LPIPS
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

def load_mesh(mesh_path, device):
    """Loads an OBJ mesh and prepares it for nvdiffrast."""
    print(f"Loading mesh from: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
    
    if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
        colors = torch.tensor(mesh.visual.vertex_colors[:, :3] / 255.0, dtype=torch.float32, device=device)
    else:
        colors = torch.ones_like(vertices)

    return vertices, faces, colors


def get_canonical_t_params(smpl_neutral):
    """Constructs the 'Big Pose' (Canonical) parameters."""
    bs = 1
    device = smpl_neutral['v_template'].device
    return {
        'poses': torch.zeros((bs, 72), device=device),
        'shapes': torch.zeros((bs, 10), device=device),
        'R': torch.eye(3, device=device).unsqueeze(0),
        'Th': torch.zeros((bs, 1, 3), device=device)
    }

def deform_mesh_gauhuman(vertices, gaussians, view):
    """Applies GauHuman's specific deformation logic to mesh vertices."""
    # 1. Prepare Inputs from View
    try:
        params = {
            'poses': view.smpl_param['poses'].clone().detach().to("cuda").unsqueeze(0).float(),
            'shapes': view.smpl_param['shapes'].clone().detach().to("cuda").unsqueeze(0).float(),
            'R': view.smpl_param['R'].clone().detach().to("cuda").unsqueeze(0).float(),
            'Th': view.smpl_param['Th'].clone().detach().to("cuda").unsqueeze(0).float()
        }
    except AttributeError:
        print(f"Warning: Could not find SMPL params for view {view.image_name}, skipping deformation.")
        return vertices
    except Exception:
        params = {
            'poses': torch.tensor(view.smpl_param['poses'], device="cuda").unsqueeze(0).float(),
            'shapes': torch.tensor(view.smpl_param['shapes'], device="cuda").unsqueeze(0).float(),
            'R': torch.tensor(view.smpl_param['R'], device="cuda").unsqueeze(0).float(),
            'Th': torch.tensor(view.smpl_param['Th'], device="cuda").unsqueeze(0).float()
        }

    big_pose_params = view.big_pose_smpl_param
    big_pose_verts = view.big_pose_world_vertex.cuda().unsqueeze(0)

    # 2. Compute Offsets & Corrections
    if gaussians.motion_offset_flag:
        # LBS Offset Network
        lbs_weight_offsets = gaussians.lweight_offset_decoder(vertices.unsqueeze(0)).permute(0, 2, 1)
        # Pose Correction
        pose_input = params['poses'].squeeze(1)[:, 3:]
        correct_Rs = gaussians.pose_decoder(pose_input)['Rs']
    else:
        lbs_weight_offsets = None
        correct_Rs = None

    # 3. Apply Deformation
    with torch.no_grad():
        _, world_verts, _, _, _ = gaussians.coarse_deform_c2source(
            query_pts=vertices.unsqueeze(0),
            params=params,
            t_params=big_pose_params,
            t_vertices=big_pose_verts,
            lbs_weights=lbs_weight_offsets,
            correct_Rs=correct_Rs,
            return_transl=False
        )
    
    return world_verts.squeeze(0)

def render_set(model_path, name, iteration, views, mesh_data, gaussians, glctx):
    vertices_base, faces, colors_base = mesh_data
    
    render_path = os.path.join(model_path, name, "ours_mesh_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_mesh_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    rgbs = []
    rgbs_gt = []
    elapsed_time = 0

    for _, view in enumerate(tqdm(views, desc=f"Rendering {name} progress")):
        gt = view.original_image[0:3, :, :].cuda()
        full_proj = view.full_proj_transform
        
        start_time = time.time()

        if gaussians.motion_offset_flag:
             vertices_world = deform_mesh_gauhuman(vertices_base, gaussians, view)
        else:
             vertices_world = vertices_base

        v_hom = torch.cat((vertices_world, torch.ones_like(vertices_world[:, :1])), dim=1)
        r_pos = v_hom @ full_proj
        
        h, w = view.image_height, view.image_width
        rast_out, _ = dr.rasterize(glctx, r_pos[None, ...], faces, resolution=[h, w])
        out_img, _ = dr.interpolate(colors_base[None, ...], rast_out, faces)
        rendering = out_img[0].permute(2, 0, 1)
        
        mask = rast_out[0, :, :, 3] > 0
        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda") 
        rendering[:, ~mask] = bg_color[:, None]

        end_time = time.time()
        elapsed_time += end_time - start_time

        rgbs.append(rendering)
        rgbs_gt.append(gt)

    print("Elapsed time: ", elapsed_time, " FPS: ", len(views)/elapsed_time) 

    psnrs = 0.0
    ssims = 0.0
    lpipss = 0.0

    for id in range(len(views)):
        rendering = rgbs[id]
        gt = rgbs_gt[id]
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(id) + ".png"))

        psnrs += psnr(rendering, gt).mean().double()
        ssims += ssim(rendering, gt).mean().double()
        lpipss += loss_fn_vgg(rendering, gt).mean().double()

    psnrs /= len(views)   
    ssims /= len(views)
    lpipss /= len(views)  

    print("\n[MESH EVAL] Evaluating {} #{}: PSNR {} SSIM {} LPIPS {}".format(name, len(views), psnrs, ssims, lpipss))

def render_sets(dataset, iteration, pipeline, opt, skip_train, skip_test, mesh_path):
    with torch.no_grad():
        glctx = dr.RasterizeGLContext()
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        checkpoint_path = os.path.join(dataset.model_path, "chkpnt" + str(iteration) + ".pth")
        if os.path.exists(checkpoint_path):
            (model_params, first_iter) = torch.load(checkpoint_path)
            gaussians.restore(model_params, opt)
            print(f"Restored Gaussian Model & Decoders from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found! Rendering with random weights.")

        mesh_data = load_mesh(mesh_path, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), mesh_data, gaussians, glctx)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), mesh_data, gaussians, glctx)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to the .obj file")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    
    print("Rendering Mesh: " + args.mesh_path)
    safe_state(args.quiet)
    
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), opt.extract(args), args.skip_train, args.skip_test, args.mesh_path)