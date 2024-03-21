import copy
import open3d as o3d
import numpy as np

## gaussian noise o3d https://www.open3d.org/docs/0.12.0/tutorial/pipelines/robust_kernels.html

source = o3d.io.read_point_cloud('data/00000002_1ffb81a71e5b402e966b9341_trimesh_001_p1000000.ply') # read ply

mu, sigma = 0, 3.4  # mean and standard deviation

def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd

source_noisy = apply_noise(source, mu, sigma)

o3d.visualization.draw_geometries([source_noisy])
print(source)