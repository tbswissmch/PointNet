import open3d as o3d

source = o3d.io.read_point_cloud('data/lobster.ply') #00000002_1ffb81a71e5b402e966b9341_trimesh_001_p1000000.ply') # read ply

max_bound = source.get_max_bound() # max bounds for geometry coordinates
min_bound = source.get_min_bound() # min bounds

print(max_bound, min_bound)

bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bound[0]/2, min_bound[1]/4, min_bound[2]), max_bound=(max_bound[0], max_bound[1], max_bound[2]))

cropped_source = source.crop(bbox, invert=True)

o3d.visualization.draw_geometries([cropped_source])


## limitation: only crop on axis
