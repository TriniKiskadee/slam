import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform


# turn [[x, y]] -> [[x, y, 1]]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


# Identity matrix translation rotation
IRt = np.eye(4)


# Pose
def extract_pose(E):
    W = np.mat([[0, -1, 0],[1, 0, 0],[0, 0, 1]], dtype=float)
    U, d, Vt = np.linalg.svd(E)
    assert np.linalg.det(U) > 0
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


def normalize(K_inv, point):
    return np.dot(K_inv, add_ones(point).T).T[:, 0:2]


def denormalize(K, point):
    ret = np.dot(K, np.array([point[0], point[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))


def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.descriptors, f2.descriptors, k=2)

    # Feature Matching
    ret = []
    idx1, idx2 = [], []

    # Lowe's ratio test
    for m, n in matches:
        if m.distance < (0.7 * n.distance):
            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)

            kp1 = f1.points[m.queryIdx]
            kp2 = f2.points[m.trainIdx]
            ret.append((kp1, kp2))

    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # Fit matrix
    model, inliers = ransac(
        (ret[:, 0], ret[:, 1]),
        #FundamentalMatrixTransform,
        EssentialMatrixTransform,
        min_samples=10,
        residual_threshold=0.005,
        max_trials=100
    )

    # Filter outlires
    ret = ret[inliers]
    Rt = extract_pose(model.params)

    return idx1[inliers], idx2[inliers], Rt


def extract(image):
    orb = cv2.ORB_create()

    # Feature Detection
    points = cv2.goodFeaturesToTrack(
        np.mean(image, axis=2).astype(np.uint8),
        maxCorners=3000,
        qualityLevel=0.01,
        minDistance=3
    )

    # Feature Extraction
    keypoints = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in points]
    keypoints, descriptors = orb.compute(image, keypoints)

    # Return points and descriptors
    return np.array([(point.pt[0], point.pt[1]) for point in keypoints]), descriptors


class Frame(object):
    def __init__(self, image, K):
        self.K = K
        self.K_inv = np.linalg.inv(self.K)
        self.pose = IRt

        points, self.descriptors = extract(image)
        self.points = normalize(self.K_inv, points)
