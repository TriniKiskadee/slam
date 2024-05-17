from multiprocessing import Process, Queue
import numpy as np

import OpenGL.GL as gl
import g2o
import pypangolin as pangolin


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
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)


class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.q = None

    # Optimizer
    def optimizer(self):
        # init g2o optimizer
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)

        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

        # Add frames to graph
        for f in self.frames:
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(f.id)
            v_se3.set_estimate(g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[3, 0:3]))
            v_se3.set_fixed(f.id == 0)
            opt.add_vertex(v_se3)

        # Add points to frames
        for p in self.points:
            pt = g2o.BaseFixedSizedEdge_3_Vector3_VertexSBAPointXYZ_VertexSCam()

            pt.set_id(p.id + 0x10000)
            pt.set_estimate(p.pt[0:3])
            pt.set_marginalized(True)
            opt.add_vertex(pt)

            for f in p.frames:
                edge = g2o.EdgeSE3ProjectXYZ()
                edge.set_vertex(0, pt)
                edge.set_vertex(1, opt.vertex(f.id))
                edge.set_measurement(f.kps[f.pts.index(p)])
                edge.set_information(np.eye(2))
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)
                # edge_id += 1

        opt.optimize(20)

    # Viewer
    def create_viewer(self):
        self.q = Queue()
        self.vt = Process(target=self.viewer_thread, args=(self.q,))
        self.vt.daemon = True
        self.vt.start()

    def viewer_thread(self, q):
        self.viewer_init(w=1024, h=768)
        while 1:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind("SLAM Map", w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
            pangolin.ModelViewLookAt(
                0, -10, -8,
                0, 0, 0,
                0, -1, 0
            )
        )
        self.handler = pangolin.Handler3D(self.scam)

        # Create interactive view in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(
            pangolin.Attach(0.0),
            pangolin.Attach(1.0),
            pangolin.Attach(0.0),
            pangolin.Attach(1.0),
            w/h
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
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.glDrawPoints(self.state[1][:, :3])

        pangolin.FinishFrame()

    def display(self):
        if self.q is None:
            return

        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
        self.q.put((np.array(poses), np.array(pts)))

