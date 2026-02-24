import open3d as o3d
import numpy as np
import sys
import pandas as pd
from plyfile import PlyData

# Usage: python extract_canonical.py path/to/your_canonical.ply

def extract_mesh_from_ply(ply_path, output_path="canonical_mesh.obj"):
    print(f"Loading {ply_path}...")
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']
    
    # Extract positions and normals for Open3D
    points = np.stack([v['x'], v['y'], v['z']], axis=-1)
    
    # Convert SH DC coefficients to RGB
    sh_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1)
    colors = 0.5 + 0.28209 * sh_dc
    colors = np.clip(colors, 0, 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if 'nx' in v:
        normals = np.stack([v['nx'], v['ny'], v['nz']], axis=-1)
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    print("Removing statistical outliers...")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
    
    # 2. Estimate Normals (Crucial if your PLY doesn't have them)
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
    pcd.orient_normals_towards_camera_location(pcd.get_center())

    # 3. Poisson Reconstruction
    print("Running Poisson reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, 
        depth=8,
        width=0, 
        scale=1.1, 
        linear_fit=False
    )
    print("Painting mesh vertices...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    vertices = np.asarray(mesh.vertices)
    pcd_colors = np.asarray(pcd.colors)    
    indices = []
    for v in vertices:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(v, 1)
        indices.append(idx[0])
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_colors[indices])

    # 5. Save
    print(f"Saving to {output_path}...")
    o3d.io.write_triangle_mesh(output_path, mesh)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a .ply file path")
    else:
        extract_mesh_from_ply(sys.argv[1])