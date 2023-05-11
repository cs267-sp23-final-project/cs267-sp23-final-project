import numpy as np
import os
import cv2 as cv
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.zeros((3,4), dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose


def map_importance(normal_maps, pose_all, intrinsics_all):
    N, W, H, _ = normal_maps.shape
    importance_maps = np.empty((N, W, H))
    
    tx = np.arange(W)
    ty = np.arange(H)
    px, py = np.meshgrid(tx, ty, indexing='ij')
    p = np.stack([px, py, np.ones_like(py)], axis=-1) 
    for i in range(N):
        normals = normal_maps[i]
        pose = pose_all[i]
        intrinsic = intrinsics_all[i]
        p = (intrinsic[None, None, :, :] @ p[:, :, :, None]).squeeze() # W,H,3
        rays_d = p / np.linalg.norm(p, ord=2, axis=-1, keepdims=True)
        rays_d = (pose[None, None, :3, :3] @ rays_d[:, :, :, None]).squeeze() # W, H, 3
        importance_maps[i] = (normals[:, :, None, :] @ rays_d[:, :, :, None]).squeeze()
    return importance_maps


def accumulate_visibility(depth_maps, normal_maps, pose_all, intrinsics_all):
    N, W, H, _ = normal_maps.shape
    accumulated_visibility = np.zeros((N, W, H))
    
    tx = np.arange(W)
    ty = np.arange(H)
    px, py = np.meshgrid(tx, ty, indexing='ij')
    p = np.stack([px, py, np.ones_like(py)], axis=-1) 
    for i in range(N):
        pose = pose_all[i]
        intrinsic = intrinsics_all[i]
        depths = depth_maps[i]
        
        p = (intrinsic[None, None, :, :] @ p[:, :, :, None]).squeeze() # W,H,3
        rays_d = p / np.linalg.norm(p, ord=2, axis=-1, keepdims=True)
        rays_d = (pose[None, None, :3, :3] @ rays_d[:, :, :, None]).squeeze() # W, H, 3
        rays_o = np.tile(pose[:3, 3], (W, H, 1))
        points = rays_o + depths * rays_d
        for j in range(N):
            # TODO: retrive importance from other image
            visibility = np.random.randint(0, 2, size=importance_maps[j].shape)
            accumulated_visibility[i] += importance_maps[j] * visibility
        accumulated_visibility[i] /= accumulated_visibility[i].sum()
    return accumulated_visibility


camera_dict = np.load("thin_cube\cameras_sphere.npz")

NN = len(os.listdir("thin_cube\image"))
world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(NN)]
world_mats = world_mats + world_mats
scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(NN)]
scale_mats = scale_mats + scale_mats

intrinsics_all = []
pose_all = []

for scale_mat, world_mat in zip(scale_mats, world_mats):
    P = world_mat @ scale_mat
    P = P[:3, :4]
    intrinsics, pose = load_K_Rt_from_P(None, P)
    intrinsics_all.append(intrinsics)
    pose_all.append(pose)
pose_all = np.stack(pose_all)
intrinsics_all = np.stack(intrinsics_all)

W = 1980
H = 1480

from time import perf_counter

for n_images in range(2, 64):
    normal_maps = np.random.random((n_images, W, H, 3))
    depth_maps = np.random.random((n_images, W, H, 3))
    
    start = perf_counter()
    importance_maps = map_importance(normal_maps, pose_all[:n_images], intrinsics_all[:n_images])
    accumed = accumulate_visibility(depth_maps, normal_maps, pose_all[:n_images], intrinsics_all[:n_images])
    end = perf_counter()
    print(n_images, end - start)