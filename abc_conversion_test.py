import trimesh
import numpy as np
import open3d as o3d


## trimesh https://trimesh.org/

# obj -> ply
mesh = trimesh.load('data/00000002_1ffb81a71e5b402e966b9341_trimesh_001.obj')  # read obj
mesh.export('data/00000002_1ffb81a71e5b402e966b9341_trimesh_001.ply', file_type='ply') # write ply

print('path/file.ply')  # to see number of points sampled from trimesh

def to_ply(input_path, output_path, original_type):
    mesh = trimesh.load(input_path, file_type=original_type)  # read file
    mesh.export(output_path, file_type='ply')  # convert to ply

## o3d https://www.open3d.org/docs/0.12.0/index.html

# obj -> ply
mesh = o3d.io.read_triangle_mesh('data/00000002_1ffb81a71e5b402e966b9341_trimesh_001.obj') # read obj
pcd = mesh.sample_points_uniformly(number_of_points=1000000) # sample ply

# ply -> bin
pcd = o3d.io.read_point_cloud('data/00000002_1ffb81a71e5b402e966b9341_trimesh_001.ply') # read ply
array = np.asarray(pcd.points)  # convert to array
array.astype(np.float32).tofile('data/00000002_1ffb81a71e5b402e966b9341_trimesh_001.bin') # write array as bin

# ply -> pcd
ply = o3d.io.read_point_cloud('data/00000002_1ffb81a71e5b402e966b9341_trimesh_001.ply') # read ply
o3d.io.write_point_cloud('data/00000002_1ffb81a71e5b402e966b9341_trimesh_001.pcd', pcd) # write pcd

### mmdet3d docs

# convert bboxes objects to numpy dtype
        bboxes_3d = tensor2ndarray(bboxes_3d.tensor)

# convert ply to bin

import pandas as pd
from plyfile import PlyData

def convert_ply(input_path, output_path):
    plydata = PlyData.read(input_path)  # read file
    data = plydata.elements[0].data  # read data
    data_pd = pd.DataFrame(data)  # convert to DataFrame
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # initialize array to store data
    property_names = data[0].dtype.names  # read names of properties
    for i, name in enumerate(
            property_names):  # read data by property
        data_np[:, i] = data_pd[name]
    data_np.astype(np.float32).tofile(output_path)

# pcd to bin 
import numpy as np
from pypcd import pypcd

pcd_data = pypcd.PointCloud.from_path('point_cloud_data.pcd')
points = np.zeros([pcd_data.width, 4], dtype=np.float32)
points[:, 0] = pcd_data.pc_data['x'].copy()
points[:, 1] = pcd_data.pc_data['y'].copy()
points[:, 2] = pcd_data.pc_data['z'].copy()
points[:, 3] = pcd_data.pc_data['intensity'].copy().astype(np.float32)
with open('point_cloud_data.bin', 'wb') as f:
    f.write(points.tobytes())