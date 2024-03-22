import open3d as o3d
import numpy as np
import os 
import torch
from pointnet2_vec_ssg import get_model

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def run_abc(path, n_points):
    mesh = o3d.io.read_triangle_mesh(path)
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    array=np.asarray(pcd.points).astype(np.float32)
    array_norm = pc_normalize(array)
    point_set = array_norm[:, :3]
    point_set = point_set.T[None, ...]
    
    with torch.no_grad():
        a, b = model(torch.from_numpy(point_set))

    return b[..., 0]

def select_files(root_dir, num_files):
    files = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
            count += 1
            if count == num_files:
                 return files
    return (files)

# get model to cpu
device = torch.device("cpu")
model = get_model(num_class=40, normal_channel=False)
model.to(device)

# load model checkpoint
checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


files = select_files("data/abc_0000_obj_v00", 10)

model.eval()
features = [] 
n = 10000

for path in files:
    b = run_abc(path, n)

    features.append(b)
features = torch.cat(features, axis=0)

print(torch.round(features[0], decimals=3))