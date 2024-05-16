import os
import numpy as np


# Pose
def extract_pose(E):
    W = np.mat([[0, -1, 0],[1, 0, 0],[0, 0, 1]], dtype=float)
    U, d, Vt = np.linalg.svd(E)
    assert np.linalg.det(U) >= 0
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    # print(f"ret:\n {ret}")
    return ret


# turn [[x, y]] -> [[x, y, 1]]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def normalize(K_inv, point):
    return np.dot(K_inv, add_ones(point).T).T[:, 0:2]


def denormalize(K, point):
    ret = np.dot(K, np.array([point[0], point[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))