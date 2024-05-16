import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform, FundamentalMatrixTransform

from helpers import extract_pose, normalize


# def match_frames(f1, f2):
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#     matches = bf.knnMatch(f1.descriptors, f2.descriptors, k=2)
#
#     # Feature Matching
#     ret = []
#     idx1, idx2 = [], []
#
#     # Lowe's ratio test
#     for m, n in matches:
#         if m.distance < (0.75 * n.distance):
#             idx1.append(m.queryIdx)
#             idx2.append(m.trainIdx)
#
#             kp1 = f1.points[m.queryIdx]
#             kp2 = f2.points[m.trainIdx]
#             ret.append((kp1, kp2))
#
#     assert len(ret) >= 8
#     ret = np.array(ret)
#     idx1 = np.array(idx1)
#     idx2 = np.array(idx2)
#
#     # Fit matrix
#     # print(f"ret[:, 0] :\n {ret[:, 0].shape}")
#     # print(f"ret[:, 1] :\n {ret[:, 1].shape}")
#     model, inliers = ransac(
#         (ret[:, 0], ret[:, 1]),
#         EssentialMatrixTransform,
#         min_samples=8,
#         residual_threshold=0.005,
#         max_trials=1000
#     )
#
#     # Filter outlires
#     ret = ret[inliers]
#     Rt = extract_pose(model.params)
#
#     return idx1[inliers], idx2[inliers], Rt

def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.descriptors, f2.descriptors, k=2)

    # Lowe's ratio test
    ret = []
    idx1, idx2 = [], []
    idx1s, idx2s = set(), set()

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            p1 = f1.points[m.queryIdx]
            p2 = f2.points[m.trainIdx]

            if m.distance < 32:
                # Keep around indices
                # TODO: refactor to not be O(N^2)
                if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    idx1s.add(m.queryIdx)
                    idx2s.add(m.trainIdx)
                    ret.append((p1, p2))

    assert(len(set(idx1)) == len(idx1))
    assert(len(set(idx2)) == len(idx2))

    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # Fit matrix
    model, inliers = ransac(
        (ret[:, 0], ret[:, 1]),
        # EssentialMatrixTransform,
        FundamentalMatrixTransform,
        min_samples=8,
        residual_threshold=0.005,
        max_trials=100,
    )

    print(f"Matches: {len(f1.descriptors)} -> {len(matches)}, -> {len(inliers)} -> {sum(inliers)}")
    return idx1[inliers], idx2[inliers], extract_pose(model.params)


def extract(image):
    orb = cv2.ORB_create()

    # Feature Detection
    points = cv2.goodFeaturesToTrack(
        np.mean(image, axis=2).astype(np.uint8),
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=10
    )

    # Feature Extraction
    keypoints = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in points]
    keypoints, descriptors = orb.compute(image, keypoints)

    # Return points and descriptors
    return np.array([(point.pt[0], point.pt[1]) for point in keypoints]), descriptors


class Frame(object):
    def __init__(self, mapp, image, K):
        self.K = K
        self.K_inv = np.linalg.inv(self.K)
        self.pose = np.eye(4)
        self.h, self.w = image.shape[0:2]

        self.points, self.descriptors = extract(image)
        self.points = normalize(self.K_inv, self.points)
        self.pts = [None]*len(self.points)

        self.id = len(mapp.frames)
        mapp.frames.append(self)

