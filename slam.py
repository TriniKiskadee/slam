import sys

import cv2
import numpy as np
from multiprocessing import Process, Queue

import g2o
import pypangolin as pangolin
import OpenGL.GL as gl

from display import Display
from frame import Frame, match_frames, denormalize


# Camera intrinsics
# Display dimensions
W, H = 1920//2, 1080//2

# Focal length
F = H
# Intrinsic matrix
K = np.array(
    [
        [F, 0, W//2],
        [0, F, H//2],
        [0, 0, 1]
    ]
)
K_inv = np.linalg.inv(K)


class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.q = Queue()
        p = Process(target=self.viewer_thread, args=(self.q, ))
        p.daemon = True
        p.start()

    def viewer_thread(self, q):
        self.viewer_init(w=1024, h=768)
        while 1:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind("SLAM Map", w, h, )
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 1000),
            pangolin.ModelViewLookAt(0, -10, -20,
                                     0, 0, 0,
                                     0, -1, 0)
        )
        self.handler = pangolin.Handler3D(self.scam)

        # Create interactive view in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(
            pangolin.Attach(0.0),
            pangolin.Attach(1.0),
            pangolin.Attach(0.0),
            pangolin.Attach(1.0),
            -w/h
        )
        self.dcam.SetHandler(self.handler)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        # Draw poses
        gl.glPointSize(10)
        gl.glColor3f(0.0, 1.0, 0.0)
        ppts = [p[:3, 3] for p in self.state[0]]
        pangolin.glDrawPoints(ppts)

        # Draw keypoints
        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.3, 0.0)
        pangolin.glDrawPoints(self.state[1][:, :3])

        pangolin.FinishFrame()

    def display(self):
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
        self.q.put((np.array(poses), np.array(pts)))


# Main classes
mapp = Map()
display = Display(W, H)


class Point(object):
    """
    A Point is a 3-D position in space (x, y, z) the video. Each point is observed in multiple frames
    """
    def __init__(self, mapp, location):
        self.pt = location
        self.frames = []
        self.idxs = []

        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)


def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T


def process_frame(image):
    image = cv2.resize(image, (W, H))
    frame = Frame(mapp, image, K)
    if frame.id == 0:
        return

    f1, f2 = mapp.frames[-1], mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)

    # Homogenous 3-D coordinates
    pts4d = triangulate(f1.pose, f2.pose, f1.points[idx1], f2.points[idx2])
    pts4d /= pts4d[:, 3:]

    # Reject points without enough "parallax" and points behind the camera (z coordinate < 0)
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
    # pts4d = pts4d[good_pts4d]
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
    display.paint(image)
    # 3-D map display
    mapp.display()
