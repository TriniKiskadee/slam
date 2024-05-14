import cv2
import numpy as np
import g2o

from display import Display
from frame import Frame, match_frames, denormalize, IRt


# Camera intrinsics
# Display dimensions
W, H = 1920//2, 1080//2

# Focal length
F = 270
# Intrinsic matrix
K = np.array(
    [
        [F, 0, W//2],
        [0, F, H//2],
        [0, 0, 1]
    ]
)

# Main classes
display = Display(W, H)


class Point(object):
    """
    A Point is a 3-D position in the video. Each point is observed in multiple frames
    """
    def __init__(self, location):
        self.location = location
        self.frames = []
        self.idxs = []

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)


def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T


frames = []


def process_frame(image):
    image = cv2.resize(image, (W, H))
    frame = Frame(image, K)
    frames.append(frame)
    if len(frames) <= 1:
        return

    idx1, idx2, Rt = match_frames(frames[-1], frames[-2])
    frames[-1].pose = np.dot(Rt, frames[-2].pose)

    # Homogenous 3-D coordinates
    pts4d = triangulate(frames[-1].pose, frames[-2].pose, frames[-1].points[idx1], frames[-2].points[idx2])
    pts4d /= pts4d[:, 3:]

    # Reject points without enough "parallax" and points behind the camera (z coordinate < 0)
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
    # pts4d = pts4d[good_pts4d]
    # print(sum(good_pts4d), len(good_pts4d))

    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(p)
        pt.add_observation(frames[-1], idx1[i])
        pt.add_observation(frames[-2], idx2[i])

    for pt1, pt2 in zip(frames[-1].points[idx1], frames[-2].points[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
        cv2.circle(image, (u1, v1), radius=3, color=(0, 255, 0))
        cv2.line(image, (u1, v1), (u2, v2), (255, 0, 0))

    display.paint(image)
