import open3d as o3d

# obj -> ply
#mesh = o3d.io.read_triangle_mesh('data/alfa147.obj') # read obj 00000009_9b3d6a97e8de4aa193b81000_trimesh_001.
#pcd = mesh.sample_points_uniformly(number_of_points=1000000) # sample ply

#pcd = o3d.io.read_point_cloud('data/alfa147.ply')
#o3d.io.write_point_cloud('data/alfa147.pcd', pcd)

# pcd to bin
import numpy as np
from pypcd import pypcd

pcd_data = pypcd.PointCloud.from_path('data/alfa147_p10000.pcd') # read pcd as pypcd.PointCloud

points = np.zeros([pcd_data.width, 3], dtype=np.float32)
points[:, 0] = pcd_data.pc_data['x'].copy()
points[:, 1] = pcd_data.pc_data['y'].copy()
points[:, 2] = pcd_data.pc_data['z'].copy()
#points[:, 3] = pcd_data.pc_data['intensity'].copy().astype(np.float32)
with open('data/alfa147_p10000.bin', 'wb') as f:
    f.write(points.tobytes())

