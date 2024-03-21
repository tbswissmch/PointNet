import numpy as np
import open3d as o3d
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

pcd = o3d.io.read_point_cloud("data/00000002_1ffb81a71e5b402e966b9341_trimesh_001_p10000.pcd")

# Convert the point cloud to a NumPy array
pc_np = np.asarray(pcd.points)

# Normalize the point cloud
normalized_pc_np = pc_normalize(pc_np)

# Convert the normalized NumPy array back to an Open3D point cloud
normalized_pc_o3d = o3d.geometry.PointCloud()
normalized_pc_o3d.points = o3d.utility.Vector3dVector(normalized_pc_np)

# Write the normalized point cloud to a new file
o3d.io.write_point_cloud("data/00000002_1ffb81a71e5b402e966b9341_trimesh_001_p10000_normalized.pcd", normalized_pc_o3d)
