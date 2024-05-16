import numpy as np

from multiprocessing import Process, Queue
import pypangolin as pangolin
import OpenGL.GL as gl


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

