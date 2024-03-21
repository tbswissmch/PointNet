import numpy as np
import open3d as o3d

pcd= o3d.io.read_point_cloud("data/00000002_1ffb81a71e5b402e966b9341_trimesh_001_p10000_normalized.pcd")
array=np.asarray(pcd.points)
print(pcd.points)
with open("data/00000002_1ffb81a71e5b402e966b9341_trimesh_001_p10000_normalized.txt", mode='w') as f:  # I add the mode='w'
    for i in range(len(array)):
        f.write("%f,"%float(array[i][0].item()))
        f.write("%f,"%float(array[i][1].item()))
        f.write("%f\n"%float(array[i][2].item()))