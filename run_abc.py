import open3d as o3d
import numpy as np
import os 
import torch
from pointnet2_vec_ssg import get_model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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
    point_set = pc_normalize(array)
    point_set = point_set.T[None, ...]
    
    with torch.no_grad():
        a, b = model(torch.from_numpy(point_set))

    return b[..., 0]

# read abc paths
def select_files(root_dir, n_files=None):
    files = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
            count += 1
            if n_files is not None and count == n_files:
                 return files
    return (files)

files = select_files("/home/tobias/data/abc/abc_0000_obj_v00", 100)

# model to cpu
device = torch.device("cpu")
model = get_model(num_class=40, normal_channel=False)
model.to(device)

# load model checkpoint
checkpoint = torch.load("/home/tobias/repos/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
features = [] 
n = 10000

for path in files:
    b = run_abc(path, n)

    features.append(b)
features = torch.cat(features, axis=0)

#tSNE
features_numpy = features.detach().numpy()
tsne = TSNE(n_components=3)
features_tsne = tsne.fit_transform(features_numpy)

# Visualize in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features_tsne[:, 0], features_tsne[:, 1], features_tsne[:, 2])

ax.set_title('t-SNE Visualization of Feature Vectors')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')

plt.show()

# knn
dist = torch.norm(features - features[0], dim=1, p=None)
knn = dist.topk(3, largest=False)

print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))

for index in knn.indices:
    print(files[index.item()], index)

'''
# Calculate pairwise distances between feature vectors
distances = torch.cdist(features, features, p=2)
distances_numpy = distances.numpy()

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(distances_numpy, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Distance')
plt.title('Pairwise Distance Heatmap')
plt.xlabel('Feature Vectors')
plt.ylabel('Feature Vectors')
plt.show()
'''