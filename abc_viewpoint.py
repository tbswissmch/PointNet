import open3d as o3d
import numpy as np

source = o3d.io.read_point_cloud('data/lobster.ply') #00000002_1ffb81a71e5b402e966b9341_trimesh_001_p1000000.ply') # read ply

diameter = np.linalg.norm(
    np.asarray(source.get_max_bound()) - np.asarray(source.get_min_bound()))

camera =[diameter, 0, 0]
radius = diameter * 10000

_, pt_map = source.hidden_point_removal(camera, radius)

source_hidden = source.select_by_index(pt_map)
o3d.visualization.draw_geometries([source_hidden])
