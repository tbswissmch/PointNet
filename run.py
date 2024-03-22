import os
import torch
import numpy as np
from pathlib import Path
from pointnet2_vec_ssg import get_model

# get model to cpu
device = torch.device("cpu")
model = get_model(num_class=40, normal_channel=False)
model.to(device)

# load model checkpoint
checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# run modelnet40 inference function
def run_modelnet40(path):
    point_set = np.loadtxt(path, delimiter=',').astype(np.float32)
    point_set = point_set[:, :3]
    point_set = point_set.T[None, ...]

    with torch.no_grad():
        a, b = model(torch.from_numpy(point_set))

    return b[..., 0]

# run abc inference function 


# collect data from modelnet40
doors = np.random.choice(list(Path(r"C:\Users\ZOTWISSM\data\modelnet40_normal_resampled\door").glob("*.txt")), 5)
cars = np.random.choice(list(Path(r"C:\Users\ZOTWISSM\data\modelnet40_normal_resampled\car").glob("*.txt")), 5)
objects = np.append(doors, cars)


# inference
model.eval()
features = []

for path in objects:
    b = run(path)

    features.append(b)
features = torch.cat(features, axis=0)

# knn
dist = torch.norm(features - features[1], dim=1, p=None)
knn = dist.topk(3, largest=False)

print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))