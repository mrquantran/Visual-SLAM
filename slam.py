from typing import List
import numpy as np
import copy
import cv2

from utils.rotation import rotation_matrix_to_quaternion, project_points, angle_between

from vo import VisualOdometry
from datetime import datetime

class Frame:
    def __init__(self, timestamp, pose, kpts, descs: np.ndarray, points_3d, img=None):
        self.timestamp = timestamp
        self.pose = pose  # Homogeneous transformation matrix - 4x4 matrix

        # 2D feature point locations in image
        # (N,2) storing only the position because cv2.Keypoint is not pickleable directly
        self.kpts = kpts

        # Binary descriptors of the keypoints
        # (N, 32)
        self.descs = descs

        # [[X1,Y1,Z1], [X2,Y2,Z2], ...]
        # 3D coordinate of the keypoints in world frame
        self.points_3d = points_3d

        # Store image for debugging
        self.img = img

        assert len(kpts) == len(descs) == len(points_3d)

        # Format: [None, 1, None, 2, ...] where numbers are map point IDs
        # Associates keypoints with global map points
        # To be filled with *index* of associated map point for each keypoint.
        # None means no association
        self.map_point_ids = [None] * len(kpts)


class BundleAdjustment:
    def __init__(self, cx, cy, fx) -> None:
        # ------- Camera Parameters -------
        self.cx = cx
        self.cy = cy
        self.fx = fx

        self.verbose = True

    def optimize():
        pass


class MapPoint:
    """
    A map point is a specific 3D point in the environment that the SLAM system has identified and is tracking across multiple camera frames.
    """
    def __init__(
        self, position: List[float], desc, keyframe_id, keyframe_position
    ) -> None:
        self.position = position
        self.desc = desc

        # Mean viewing direction
        self.mean_viewing_dir = position - keyframe_position
        self.mean_viewing_dir /= np.linalg.norm(self.mean_viewing_dir)

        # List of keyframe IDs that observed this map point
        self.observed_keyframe_ids = [keyframe_id]

    def add_viewed_keyframe(self, keyframe_id: int, keyframes: List[Frame]):
        """
        Adding the keyframe computes the mean viewing direction, and selects the best keyframe
        """
        self.observed_keyframe_ids.append(keyframe_id)

        assert len(self.observed_keyframe_ids) == len(set(self.observed_keyframe_ids))

        # ------- Compute Mean Viewing Direction -------
        mean_dir = np.zeros(3) # [0.0, 0.0, 0.0]

        for keyframe_id in self.observed_keyframe_ids:
            keyframe = keyframes[keyframe_id]
            v = self.position - keyframe.pose[:3, 3]
            v /= np.linalg.norm(v) # Normalize
            mean_dir += v
            mean_dir /= np.linalg.norm(mean_dir)

        self.mean_viewing_dir = mean_dir

class Map:
    def __init__(self, cx, cy, fx) -> None:
        self.keyframes: List[Frame] = []
        self.map_points: List[MapPoint] = []
        self.ba = BundleAdjustment(cx, cy, fx)

    def optimize(self):
        if len(self.keyframes) < 5:
            return

        # @TODO: Implement Bundle Adjustment


class Tracking:
    def __init__(
        self,
        cx,
        cy,
        fx,
        baseline=1,
        initial_pose=np.eye(4),
        visualize=True,
        save_path=True,
    ) -> None:
        # --------- Visualization ---------
        self.visualize = visualize
        self.save_path = save_path

        created_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_path = f"./output/poses_{created_time}.txt"

        self.img_width = None
        self.img_height = None

        # ------- Camera Parameters -------
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.baseline = baseline

        self.initial_pose = initial_pose
        self.K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])

        # ------- Frames -------
        # a frame from self.frames is a keyframe if it has a timestamp match in self.keyframes
        self.frames: List[Frame] = []
        self.frames_elapsed_since_keyframe = 0

        # ------- Map -------
        self.keyframes: List[Frame] = []
        self.map = Map(cx, cy, fx)
        self.map_points: List[MapPoint] = []

        # 32x32 = 256-bit binary ORB descriptor
        self.map_point_descs = np.empty((0, 32), dtype=np.uint8)

        # ------- Visual Odometry -------
        self.vo = VisualOdometry(cx, cy, fx)

    def add_keyframe_tracking(self, curr_frame: Frame):
        """
        Function handles the addition of keyframes and map points during the tracking phase of SLAM
        """

        # After adding a new keyframe, the frame counter is reset to track the time since the last keyframe.
        self.frames_elapsed_since_keyframe = 0

        # Use a temporary copy of the keyframes list to avoid modifying the original list
        frame = copy.deepcopy(curr_frame)
        tmp_keyframes = self.keyframes.copy()  # Shallow copy is enough
        tmp_keyframes.append(frame)

        # ------- Add Map Points -------
        # Each feature in the current frame is processed to determine
        # if it corresponds to an existing map point or represents a new observation.
        for i, map_point_id in enumerate(frame.map_point_ids):
            if map_point_id is None:
                # The map point's ID is assigned in the frame
                frame.map_point_ids[i] = len(self.map.map_points)

                # Create a new map point
                new_map_point = MapPoint(
                    position=frame.points_3d[i],
                    desc=frame.descs[i],
                    keyframe_id=len(tmp_keyframes),
                    keyframe_position=frame.pose[:3, 3],  # Extract translation vector
                )
                self.map_points.append(new_map_point)

                self.map_point_descs = np.vstack(
                    [self.map_point_descs, frame.descs[i].reshape(1, -1)]
                )
            else:
                # ------ Update Mean Viewing Direction for Map Point ---------
                self.map_points[map_point_id].add_viewed_keyframe(
                    len(tmp_keyframes), tmp_keyframes
                )

        # ------- Add Keyframe to Map -------
        self.keyframes.append(frame)

        return frame

    def predict_pose(self, pose_t_2, pose_t_1):
        """
        Given two prior poses, predict the current pose. Use a constant velocity model.

        Derivation:
        Let 0_T_1 be the transformation from frame at time=1 to frame at time =0, and w_T_1 be the transformation from frame at time=1 to world frame.
        - pose_t_2 (pose at t=t-2) is w_T_1
        - pose_t_1 (pose at t=t-1) is w_T_2

        We want to find pose_t (pose at t=t), i.e. w_T_3

        We can approximate that 1_T_2 ~= 2_T_3 (velocity is constant)
        Thus, w_T_3 = w_T_2 @ 2_T_3
                    ~= w_T_2 @ 1_T_2
                    ~= w_T_2 @ (1_T_w @ w_T_2)

        """
        return pose_t_2 @ (np.linalg.inv(pose_t_1) @ pose_t_2)

    def match_map_points(self, prev_frame: Frame, curr_frame: Frame):
        # @TODO: Implement matching of map points between frames

        # 1. Identify Valid Map Points
        valid_mask = np.array([map_point is not None for map_point in prev_frame.map_point_ids])

        # 2. Projected into the image plane of the current frame
        map_points = np.array([(self.map_points[map_point_id].position if map_point_id is not None else [0, 0, 0]) for map_point_id in prev_frame.map_point_ids])
        projected_points = project_points(map_points, curr_frame.pose, self.K)

        # 3. Filter Points that fall outside the visible image region are excluded
        in_bounds_mask = (
            (0 <= projected_points[:, 0]) & (projected_points[:, 0] < self.img_width) &
            (0 <= projected_points[:, 1]) & (projected_points[:, 1] < self.img_height)
        )

        # 4. Filter Points with Poor Viewing Angles
        mean_viewing_angle = np.array(
            [
                (
                    angle_between(
                        self.map_points[map_point_id].mean_viewing_dir,
                        self.map_points[map_point_id].position - curr_frame.pose[:3, 3],
                    )
                    if map_point_id is not None
                    else 0
                )
                for map_point_id in prev_frame.map_point_ids
            ]
        )
        mean_viewing_angle_mask = mean_viewing_angle < np.pi / 3

        # 5. Combine Masks
        final_mask = valid_mask & in_bounds_mask & mean_viewing_angle_mask

        # 6. Match descriptors between Frames
        # @TODO: Implement matching of descriptors between frames
        good_matches = []

        if self.visualize:
            prev_frame_cv_kpts = [cv2.KeyPoint(kpt[0], kpt[1], 1) for kpt in prev_frame.kpts[final_mask]]
            curr_frame_cv_kpts = [cv2.KeyPoint(kpt[0], kpt[1], 1) for kpt in curr_frame.kpts]
            print(f"Number of prev frame keypoints: {len(prev_frame_cv_kpts)}")
            print(f"Number of curr frame keypoints: {len(curr_frame_cv_kpts)}")
            output_image = cv2.drawMatches(prev_frame.img, prev_frame_cv_kpts, curr_frame.img, curr_frame_cv_kpts, good_matches, None)
            cv2.imshow("Tracked ORB", output_image)
            cv2.waitKey(1)

        return curr_frame

    def track(self, img_t, depth_t, timestamp, gt_pose=None):
        self.img_width = img_t.shape[1]
        self.img_height = img_t.shape[0]

        # ------- Convert Image to Grayscale -------
        img_gray_t = self.vo._convert_grayscale(img_t)
        kpts_t, desc_t = self.vo._compute_orb(img_gray_t)
        kpts_t = np.array([k.pt for k in kpts_t])

        points_3d_k = self.vo._project_2d_kpts_to_3d(depth_t, kpts_t)  # (N, 3)

        # ------- Elminate features with invalid depth -------
        mask = np.isnan(points_3d_k).any(axis=1)
        kpts_t = np.array(kpts_t)[~mask]
        desc_t = desc_t[~mask]
        points_3d_k = points_3d_k[~mask]

        predicted_pose = self.initial_pose

        # ------- Predict Pose using constant velocity model -------
        if len(self.frames) >= 2:
            predicted_pose = self.predict_pose(
                self.frames[-2].pose, self.frames[-1].pose
            )

        # --------- Convert 3D points to world coordinate -----------
        points_3d_w = (
            predicted_pose @ np.hstack([points_3d_k, np.ones((len(points_3d_k), 1))]).T
        ).T[:, :3]

        # ------- Create Frame -------
        curr_frame = Frame(
            timestamp, predicted_pose, kpts_t, desc_t, points_3d_w, img_t
        )

        # ------- Initialize the Local Map -------
        if len(self.frames) == 0:
            keyframe = self.add_keyframe_tracking(curr_frame)
            self.frames.append(keyframe)

            # @TODO: Implement Bundle Adjustment

            return

        # ---------- Compute Visible Map Points in previous frame from current frame -----------
        prev_frame = self.frames[-1]
        curr_frame = self.match_map_points(prev_frame, curr_frame)

        # ------- Solve for Motion -------
        """
        Here, the p_2d represent where the 3D landmarks are seen in the current image.
        p_3d represent the 3D coordinates of the landmarks, as they are represented by the local map.
        """
        p_2d = curr_frame.kpts
        p_3d_w = np.array(
            [
                (self.map_points[map_point_id].position if map_point_id is not None else [0, 0, 0])
                for map_point_id in curr_frame.map_point_ids
            ]
        )

        # ---------- Filter out points that are not map points -----------
        assert len(p_2d) == len(p_3d_w)
        valid_pts_mask = np.array(
            [map_point is not None for map_point in curr_frame.map_point_ids]
        )
        p_2d = p_2d[valid_pts_mask]
        p_3d_w = p_3d_w[valid_pts_mask]
        k_T_w = np.linalg.inv(curr_frame.pose)
        p_3d_k = (k_T_w @ np.hstack([p_3d_w, np.ones((len(p_3d_w), 1))]).T).T[:, :3]

        print("Total map points tracked", len(p_2d))

        # ------- Estimate Camera Post by minimizing reprojection error -------
        T = self.vo._minimize_reprojection_error(p_2d, p_3d_k)
        estimated_pose = prev_frame.pose @ T
        curr_frame.pose = estimated_pose

        # ---------- Determine if is keyframe based on criteria -----------
        if len(p_2d) < 150 or self.frames_elapsed_since_keyframe > 20:
            keyframe = self.add_keyframe_tracking(curr_frame)
            curr_frame = keyframe

            # @TODO: Implement Bundle Adjustment

        self.frames.append(curr_frame)
        self.frames_elapsed_since_keyframe += 1

        if self.save_path:
            # Save the poses to a file

            with open(self.output_path, "a") as file:
                pose = self.frames[-1].pose
                position = pose[:3, 3]
                quaternion = rotation_matrix_to_quaternion(pose[:3, :3])

                pose_list = list(position) + list(quaternion)
                file.write(f"{timestamp} {' '.join(map(str, pose_list))}\n")
