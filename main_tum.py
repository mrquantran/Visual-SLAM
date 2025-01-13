import argparse
import cv2
import os
import numpy as np

from utils.rotation import quaternion_to_rotation_matrix
from eval.associate import read_file_list, associate

np.random.seed(0)

def main(dataset_name):
    # --------- Load data ---------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "data", dataset_name)

    # --------- Load Intrinsic Camera Calibration of the Kinect ---------
    # Taken from https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    factor = 5000  # for the 16-bit PNG files

    depth_image_paths = read_file_list(f"{dataset_path}/depth.txt")
    rgb_image_paths = read_file_list(f"{dataset_path}/rgb.txt")
    gt_paths = read_file_list(f"{dataset_path}/groundtruth.txt")

    # --------- Associate depth and rgb images ---------
    matches_list = associate(depth_image_paths, rgb_image_paths, 0.0, 0.02)
    gt_matches_list = associate(depth_image_paths, gt_paths, 0.0, 0.02)

    # Convert matches to dictionaries
    matches = {depth: rgb for depth, rgb in matches_list}
    gt_matches = {depth: gt for depth, gt in gt_matches_list}

    depth_images = []
    rgb_images = []
    timestamps = []
    gt_poses = []

    gt_dict = {}

    with open(f"{dataset_path}/groundtruth.txt") as file:
        # Each line contains: [timestamp tx ty tz qx qy qz qw]
        for line in file:
            if line[0] == "#":
                continue

            # 4x4 identity matrix
            # Creates: [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
            pose = np.eye(4)
            pose_list = list(map(float, line.split()))

            # Set translation vector [tx, ty, tz]
            pose[:3, 3] = pose_list[1:4]

            # Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix
            pose[:3, :3] = quaternion_to_rotation_matrix(pose_list[4:])

            # Store in dictionary with timestamp as key
            gt_dict[float(pose_list[0])] = pose

    # ----- Load depth and rgb images -----
    for depth_timestamp, rgb_timestamp in matches.items():
        if depth_timestamp not in gt_matches:
            continue

        depth_image_path = dataset_path + "/" + depth_image_paths[depth_timestamp][0]
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

        # Converts to meters by dividing by factor
        depth_image = depth_image / factor

        # Replaces 0 values with NaN (missing data)
        depth_image[depth_image == 0] = np.NaN
        depth_images.append(depth_image)

        rgb_image_path = dataset_path + "/" + rgb_image_paths[rgb_timestamp][0]
        rgb_image = cv2.imread(rgb_image_path)
        rgb_images.append(rgb_image)

        timestamps.append(depth_timestamp)
        gt_poses.append(gt_dict[gt_matches[depth_timestamp]])

    initial_pose = gt_poses[0]

    from slam import Tracking
    tracker = Tracking(cx, cy, fx, 1, initial_pose)

    for i, (rgb_image, depth_image, timestamp) in enumerate(zip(rgb_images, depth_images, timestamps)):
        print(f"Frame {i}")
        tracker.track(rgb_image, depth_image, timestamp)

        # Display the current RGB frame
        cv2.imshow('RGB Frame', rgb_image)

        # Wait for 1ms and check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="""This script takes two data files with timestamps and associates them""")
    parser.add_argument(
        "--dataset-name", help="dataset name", default="rgbd_dataset_freiburg1_desk"
    )

    args = parser.parse_args()
    main(dataset_name=args.dataset_name)
