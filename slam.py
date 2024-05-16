import os

import cv2
import numpy as np

from display import Display
from frame import Frame, match_frames
from pointmap import Point, Map
from helpers import denormalize, triangulate


# Camera intrinsics
# Display dimensions
W, H = 1920//2, 1080//2

# Focal length
F = 700
# Intrinsic matrix
K = np.array(
    [
        [F, 0, W//2],
        [0, F, H//2],
        [0, 0, 1]
    ]
)
K_inv = np.linalg.inv(K)
# Main classes
mapp = Map()
# Display 3-D (aka the slam map)
D3D = True
if D3D:
    mapp.create_viewer()

# Display 2-D (aka the video output)
D2D = False
if D2D:
    display = Display(W, H)
else:
    display = None


def process_frame(image):
    image = cv2.resize(image, (W, H))
    frame = Frame(mapp, image, K)
    if frame.id == 0:
        return

    # Match with
    # frame.match(mapp.frames[-2])

    f1, f2 = mapp.frames[-1], mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)

    for i, idx in enumerate(idx2):
        if f2.pts[idx] is not None:
            f2.pts[idx].add_observation(f1, idx1[i])

    # Homogenous 3-D coordinates
    pts4d = triangulate(f1.pose, f2.pose, f1.points[idx1], f2.points[idx2])
    pts4d /= pts4d[:, 3:]

    unmatched_points = np.array([f1.points[i] is not None for i in idx1])
    print(f"\nAdding: {sum(unmatched_points)} points")

    # Reject bad points
    good_pts4d = (
            (np.abs(pts4d[:, 3]) > 0.005) &
            (pts4d[:, 2] > 0) &
            (unmatched_points)

    )
    # print(sum(good_pts4d), len(good_pts4d))

    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for pt1, pt2 in zip(f1.points[idx1], f2.points[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
        cv2.circle(image, (u1, v1), radius=3, color=(0, 255, 0))
        cv2.line(image, (u1, v1), (u2, v2), (255, 0, 0))

    # Video display
    display.paint(image) if display is not None else None

    # optimizer
    if frame.id > 3:
        mapp.optimizer()

    # 3-D map display
    mapp.display()
