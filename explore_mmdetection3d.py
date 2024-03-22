import torch
import matplotlib
import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from mmdet3d.models.backbones.pointnet2_sa_ssg import PointNet2SASSG

matplotlib.use("TkAgg")


def load_mesh_and_sample_points(obj_path, vis=False):
    obj = o3d.io.read_triangle_mesh(str(obj_path))
    pcd = obj.sample_points_uniformly(number_of_points=8192)
    points_pyt = torch.tensor(np.array(pcd.points), dtype=torch.float32, device=device)[None, ...]
    if vis:
        pcd.points = o3d.utility.Vector3dVector(np.array(points_pyt[0].cpu()))
        pcd.paint_uniform_color((0.5, 0.5, 0.5))
        o3d.visualization.draw_geometries([pcd])
    return points_pyt


def prepare_model():
    checkpoint_file = "/home/chrisbe/repos/mmdetection3d/weights/pointnet2_ssg_xyz-only_16x2_cosine_200e_scannet_seg-3d-20class_20210514_143628-4e341a48.pth"

    model = PointNet2SASSG(in_channels=3,
                           num_points=(2048, 1024, 512, 256),
                           radius=(0.2, 0.4, 0.8, 1.2),
                           num_samples=(64, 32, 16, 16),
                           sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512)),
                           fp_channels=(),
                           norm_cfg=dict(type='BN2d'),
                           sa_cfg=dict(
                               type='PointSAModule',
                               pool_mod='max',
                               use_xyz=True,
                               normalize_xyz=True))

    # get state_dicts
    target_state_dict = model.state_dict()
    source_state_dict = torch.load(checkpoint_file)["state_dict"]

    # rename and remove unneeded keys
    source_state_dict = {".".join(k.split(".")[1:]): v for k, v in source_state_dict.items()}
    source_state_dict = {k: v for k, v in source_state_dict.items() if k in target_state_dict}

    model.load_state_dict(source_state_dict)
    model = model.to(device)
    return model

def compute_features(points_pyt):
    with torch.no_grad():
        out = model(points_pyt)
    features = out["sa_features"][-1] # output of last set abstraction module
    features = torch.max(features, 2)[0]  # maxpool; points are in last axis
    return features

def plot_scatter(ax, points_test):
    x, y, z = points_test[::20, :].T.cpu()
    mi = points_test.min().item()
    ma = points_test.max().item()
    ax.set_xlim([mi, ma])
    ax.set_ylim([mi, ma])
    ax.set_zlim([mi, ma])
    ax.scatter(x, y, z)

class KNearestNeighbors:
    def __init__(self, features_train, k=3):
        self.feature_train = features_train
        self.k = k

    def __call__(self, features_test):
        dist = torch.norm(features_train - features_test, dim=1, p=None)
        knn = dist.topk(self.k, largest=False)
        return knn


if __name__ == "__main__":
    device = "cuda:0"

    # prepare model
    model = prepare_model()
    model.eval()

    # prepare objects
    obj_paths = list(sorted(Path("/home/chrisbe/repos/cdetection/data/2023_09_29_Druckobjekte_decimated").glob("*.obj")))
    points_pyt = torch.cat([load_mesh_and_sample_points(path) for path in obj_paths], axis=0)

    # prepare features and classifier
    features_train = compute_features(points_pyt)
    classifier = KNearestNeighbors(features_train, k=3)

    for i, path in enumerate(obj_paths):
        # compute features and classify
        points_test = load_mesh_and_sample_points(str(path))
        features_test = compute_features(points_test)
        knn = classifier(features_test)

        # plot results
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(231, projection='3d')
        plot_scatter(ax, points_test[0])
        for i in range(3):
            ax = fig.add_subplot(2,3,4+i, projection='3d')
            plot_scatter(ax, points_pyt[knn[1][i]])
        plt.show()

