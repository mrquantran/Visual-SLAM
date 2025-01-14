import cv2

# https://github.com/miquelmassot/g2o-python
import g2o
import numpy as np
from typing import Tuple

# check if CUDA is available
import torch

CUDA = torch.cuda.is_available()

if CUDA:
    from cv2 import cuda_ORB as cv2_ORB
else:
    from cv2 import ORB as cv2_ORB
    from cv2 import StereoSGBM as cv2_StereoSGBM

np.random.seed(0)

class VisualOdometry:
    """

    """
    def __init__(
        self, cx, cy, fx, baseline=1, initial_pose=np.eye(4), visualize=True, save_path=False
    ):
        # --------- Visualization ---------
        self.visualize = visualize
        self.save_path = save_path

        # --------- Camera Parameters and Matrices ---------
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.baseline = baseline  # Needed for stereo camera

        # Intrinsic matrix and its inverse
        self.K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)

        # --------- Feature Detectors and Matchers ---------
        self.orb = cv2_ORB.create(3000) # 3000 is the maximum number of keypoints

        # --------- FLANN based matcher ---------
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(
            indexParams=index_params, searchParams=search_params
        )

    def _convert_grayscale(self, img):
        if CUDA:
            if type(img) == np.ndarray or type(img) == cv2.Mat:
                img = cv2.cuda_GpuMat(img)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img_gray

    def _get_matches(
        self,
        descs_t_1,
        descs_t,
        ratio_threshold=0.7,
        distance_threshold=50.0,
    ):
        # FLANN based matcher
        try:
            matches = self.flann.knnMatch(descs_t_1, descs_t, k=2)
        except Exception as e:
            print(f"Error in FLANN matching: {e}")
            return []

        good_matches = []

        for m, n in matches:
            condition = m.distance < ratio_threshold * n.distance and m.distance < distance_threshold
            if condition:
                good_matches.append(m)

        return good_matches

    def _compute_orb(self, img_t) -> Tuple[cv2.KeyPoint, np.ndarray]:
        """
        Compute ORB keypoints and descriptors for a given image.
        - ORB detects keypoints in the image, which are points of interest such as edges, corners, or textured regions.
        - Each keypoint has a descriptor, a 32-byte vector that uniquely identifies its appearance.

        Parameters:
        - img_t (ndarray): Image at time t

        Returns:
        - kpts_t (tuple of cv2.Keypoint): Keypoints at time t (x, y)
        - descs_t (ndarray): Descriptors at time t

        Reference:
        - https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
        - https://gilscvblog.com/2013/08/18/a-short-introduction-to-descriptors/
        """
        if CUDA:
            kpts_t, descs_t = self.orb.detectAndComputeAsync(img_t, None)

            # Convert back to CPU
            if kpts_t.step == 0:
                print("Failed to process frame, No ORB keypoints found")
                return [], []

            kpts_t = self.orb.convert(kpts_t)
            descs_t = descs_t.download()
        else:
            kpts_t, descs_t = self.orb.detectAndCompute(img_t, None)

        return kpts_t, descs_t

    def _project_2d_kpts_to_3d(self, depth, matched_kpts):
        """
        Vectorized method to project 2D keypoints to 3D points using depth information.
        3D points may contain NaN values if the depth is NaN.

        To convert from 2D to 3D, we use the following equation:

        sp = KP

        where
        - p = [u v 1] is the 2D point as measured by the camera in homogeneous coordinates
        - K is the intrinsic matrix
        - P = [x y z] is the 3D point in camera frame
        - s is some scalar (equal to z in the 3D point P = [x y z] in camera frame)

        Therefore, we can solve for P by multiplying the inverse of K with the 2D keypoint:
        P = K^{-1} s p

        Parameters
        ----------
        depth (ndarray): Depth image
        matched_kpts (ndarray): 2D keypoints, assumed to be an array of shape (N, 2)

        Returns
        -------
        points_3d (ndarray): 3D points in camera frame, shape (N, 3)
        """
        # 1. Extract the inverse of the intrinsic matrix
        K_inv = self.K_inv

        # 2. Extract the depth values of 2D keypoints
        # Ensure indices are integers (pixel indices)
        depths = depth[matched_kpts[:, 1].astype(int), matched_kpts[:, 0].astype(int)]

        # Step 3: Scale the 2D keypoints by their respective depths
        # Add a column of ones to represent homogeneous coordinates [u, v, 1]
        homogeneous_2d = np.hstack((matched_kpts, np.ones((matched_kpts.shape[0], 1))))  # (N, 3)
        scaled_2d_points = homogeneous_2d * depths[:, np.newaxis]  # Scale [u, v, 1] by z (depth)

        # 4. Multiply by K^{-1} to transform into 3D camera coordinates
        # K^{-1} z p = [x y z]
        points_3d = (K_inv @ scaled_2d_points.T).T  # Transpose back to (N, 3)

        return points_3d

    def _minimize_reprojection_error(self, points_2d, points_3d):
        """
        Refer to SLAM textbook P3P for formulation. Solved with g2o.

        Parameters
        ----------
        points_2d (ndarray): 2D points
        points_3d (ndarray): 3D points

        Returns
        -------
        T: The transform old_T_new
        """
        assert len(points_2d) == len(points_3d)

        # Initialize the nonlinear optimizer
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)  #  Levenberg-Marquardt
        optimizer.set_algorithm(solver)

        # Set intrinsic camera parameters
        cam = g2o.CameraParameters(self.fx, (self.cx, self.cy), 0)
        cam.set_id(0)
        optimizer.add_parameter(cam)

        # Add camera pose
        pose = g2o.SE3Quat()
        vertex_pose = g2o.VertexSE3Expmap()
        vertex_pose.set_id(0)
        vertex_pose.set_estimate(pose)
        optimizer.add_vertex(vertex_pose)

        for i, point_2d in enumerate(points_2d):
            point_3d = points_3d[i]

            vp = g2o.VertexPointXYZ()
            vp.set_id(i + 1)
            vp.set_marginalized(True)
            vp.set_estimate(point_3d)
            optimizer.add_vertex(vp)

            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_vertex(0, vp)
            edge.set_vertex(1, optimizer.vertex(0))
            edge.set_measurement(point_2d)
            edge.set_information(np.identity(2))
            edge.set_robust_kernel(g2o.RobustKernelHuber())
            edge.set_parameter_id(0, 0)
            optimizer.add_edge(edge)

        optimizer.initialize_optimization()
        optimizer.optimize(10)

        T = vertex_pose.estimate().to_homogeneous_matrix()

        # old_T_new
        T = np.linalg.inv(T)
        return T
