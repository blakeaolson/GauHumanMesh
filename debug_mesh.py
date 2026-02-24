import sys, numpy as np, torch
import render_mesh
from scene.gaussian_model import GaussianModel

mesh_path = sys.argv[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# init GaussianModel with motion offsets enabled to inspect decoders
g = GaussianModel(sh_degree=0, smpl_type='smpl', motion_offset_flag=True, actor_gender='neutral')

v,f,c = render_mesh.load_mesh(mesh_path, device=device)
render_mesh._debug_print_mesh_and_smpl(v,f,c,g)

# nearest-neighbour distances (mesh -> SMPL template)
try:
    t_v = g.SMPL_NEUTRAL['v_template'].to(device)
    print("SMPL template shape:", t_v.shape)
    if t_v.shape[0] != v.shape[0]:
        print("Computing NN distances (mesh -> SMPL template)...")
        dists = torch.cdist(v, t_v)
        min_dists, _ = dists.min(dim=1)
        print("NN dists (min, mean, max):", float(min_dists.min()), float(min_dists.mean()), float(min_dists.max()))
    else:
        print("Vertex counts match -> no NN needed.")
except Exception as e:
    print("NN check failed:", e)

# LBS weight offset stats
try:
    with torch.no_grad():
        w_off = g.lweight_offset_decoder(v.unsqueeze(0).to(device)).permute(0,2,1)[0]  # (N,24)
        norms = torch.norm(w_off, dim=1).cpu().numpy()
    ys = v[:,1].cpu().numpy()  # assume Y is up
    med = np.median(ys)
    top_mask = ys > med
    bottom_mask = ~top_mask
    print("LBS offset norms (overall min,mean,max):", norms.min(), norms.mean(), norms.max())
    print("Top mean norm:", norms[top_mask].mean() if top_mask.sum()>0 else float('nan'))
    print("Bottom mean norm:", norms[bottom_mask].mean() if bottom_mask.sum()>0 else float('nan'))
    print("Top count:", int(top_mask.sum()), "Bottom count:", int(bottom_mask.sum()))
except Exception as e:
    print("LBS offset stats failed:", e)

