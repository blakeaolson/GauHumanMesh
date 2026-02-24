import numpy as np
from plyfile import PlyData
from skimage.measure import marching_cubes
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates
import trimesh
import sys
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def build_covariance_from_scaling_rotation(scaling, rotation):
    L = np.zeros((scaling.shape[0], 3, 3))
    R_mat = R.from_quat(rotation).as_matrix()
    np.einsum('ijj->ij', L)[:] = scaling
    M = R_mat @ L
    Sigma = M @ M.transpose(0, 2, 1)
    return Sigma

def get_rgb_from_sh(sh_dc):
    # SH factor for degree 0 (DC component)
    SH_C0 = 0.28209479177387814
    rgb = 0.5 + SH_C0 * sh_dc
    return np.clip(rgb, 0, 1)

def load_gaussians(ply_path):
    print(f"Loading {ply_path}...")
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']
    
    xyz = np.stack((v['x'], v['y'], v['z']), axis=1)
    opacities = sigmoid(v['opacity'])
    scales = np.exp(np.stack((v['scale_0'], v['scale_1'], v['scale_2']), axis=1))
    
    # Load Rotations (w, x, y, z) - note Ply order might vary, usually rot_0 is w
    rot = np.stack((v['rot_1'], v['rot_2'], v['rot_3'], v['rot_0']), axis=1)
    
    # Load Color (SH DC component)
    # 3DGS stores color as f_dc_0, f_dc_1, f_dc_2
    sh_dc = np.stack((v['f_dc_0'], v['f_dc_1'], v['f_dc_2']), axis=1)
    colors = get_rgb_from_sh(sh_dc)

    return xyz, scales, rot, opacities, colors

def evaluate_grid(xyz, scales, rots, opacities, colors, grid_res=256, truncation=3.0):
    # 1. Define Grid Bounds
    min_bound = xyz.min(axis=0) - 0.5
    max_bound = xyz.max(axis=0) + 0.5
    
    x = np.linspace(min_bound[0], max_bound[0], grid_res)
    y = np.linspace(min_bound[1], max_bound[1], grid_res)
    z = np.linspace(min_bound[2], max_bound[2], grid_res)
    voxel_size = (max_bound - min_bound) / grid_res
    
    # Volumes for density and color accumulation
    density_vol = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)
    color_vol = np.zeros((grid_res, grid_res, grid_res, 3), dtype=np.float32)
    weight_vol = np.zeros((grid_res, grid_res, grid_res, 1), dtype=np.float32) # For normalizing color

    print("Computing covariances...")
    covariances = build_covariance_from_scaling_rotation(scales, rots)
    inv_covariances = np.linalg.inv(covariances)
    
    print(f"Splatting {len(xyz)} Gaussians onto {grid_res}^3 grid...")
    
    for i in tqdm(range(len(xyz))):
        opacity = opacities[i]
        if opacity < 0.05: continue
        
        mu = xyz[i]
        sigma = covariances[i]
        inv_sigma = inv_covariances[i]
        color = colors[i]
        
        radius = truncation * np.sqrt(np.diag(sigma))
        
        # Convert world pos to grid index
        min_idx = np.clip(((mu - radius - min_bound) / voxel_size).astype(int), 0, grid_res)
        max_idx = np.clip(((mu + radius - min_bound) / voxel_size).astype(int) + 1, 0, grid_res)
        
        if np.any(min_idx >= max_idx): continue

        # Extract sub-grid
        gx = x[min_idx[0]:max_idx[0]]
        gy = y[min_idx[1]:max_idx[1]]
        gz = z[min_idx[2]:max_idx[2]]
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
        pts = np.stack([GX, GY, GZ], axis=-1)
        
        diff = pts - mu
        mahalanobis = np.einsum('...k,kl,...l->...', diff, inv_sigma, diff)
        
        # Gaussian weight
        weight = opacity * np.exp(-0.5 * mahalanobis)
        weight = weight[..., np.newaxis] # Expand for broadcasting (make it 4D)
        
        # Accumulate Density
        density_vol[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]] += weight[..., 0]
        
        # Accumulate Weighted Color (Weight * Color)
        # weight is (N,M,K,1) and color is (3,), creating (N,M,K,3)
        color_vol[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]] += weight * color
        
        # Accumulate Total Weight for Normalization
        weight_vol[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]] += weight

    # Normalize Color Volume (Average Color)
    mask_3d = weight_vol[..., 0] > 0.0001
    
    # 2. Extract valid weights. 
    valid_weights = weight_vol[mask_3d] 

    # 3. Divide safely
    color_vol[mask_3d] /= valid_weights
    
    return density_vol, color_vol, min_bound, max_bound, voxel_size
    
    return density_vol, color_vol, min_bound, max_bound, voxel_size

def extract_colored_mesh(ply_path, output_path, resolution=256, threshold=0.2):
    xyz, scales, rots, opacities, colors = load_gaussians(ply_path)
    
    # 1. Compute Grids
    den_vol, col_vol, min_b, max_b, vox_sz = evaluate_grid(xyz, scales, rots, opacities, colors, grid_res=resolution)
    
    print("Running Marching Cubes...")
    verts_grid, faces, normals, _ = marching_cubes(den_vol, level=threshold)
    
    # 2. Interpolate Color at Vertices
    # verts_grid contains coordinates in voxel indices (0..256)
    # We sample the color volume at these exact indices
    print("Mapping colors to vertices...")
    mesh_colors = np.zeros((verts_grid.shape[0], 3))
    for i in range(3): # R, G, B channels
        mesh_colors[:, i] = map_coordinates(col_vol[..., i], verts_grid.T, order=1)
    
    # 3. Transform vertices to world space
    world_verts = verts_grid * vox_sz + min_b
    
    # 4. Save
    print(f"Saving colored mesh to {output_path}...")
    # Create mesh with vertex colors
    mesh = trimesh.Trimesh(vertices=world_verts, faces=faces, vertex_normals=normals, vertex_colors=mesh_colors)
    mesh.export(output_path)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_color_mesh.py <input.ply> [output.obj]")
    else:
        ply_file = sys.argv[1]
        out_file = sys.argv[2] if len(sys.argv) > 2 else "colored_mesh.obj"
        # Increase resolution to 256 or 512 for better texture details
        extract_colored_mesh(ply_file, out_file, resolution=512, threshold=0.1)