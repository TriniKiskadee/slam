import os
import numpy as np

# Pose
def extract_pose(F):
    W = np.mat([[0, -1, 0],[1, 0, 0],[0, 0, 1]], dtype=float)
    U, d, Vt = np.linalg.svd(F)
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


def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]

    return ret