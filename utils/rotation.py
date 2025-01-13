import numpy as np
from numpy.typing import NDArray


def quaternion_to_rotation_matrix(q) -> NDArray[np.float64]:
    """
    Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix.

    Parameters:
    q (array-like): Quaternion in format [qx, qy, qz, qw]

    Returns:
    NDArray[float64]: 3x3 rotation matrix
    """
    # Ensure input is numpy array and normalize quaternion
    q = np.array(q, dtype=np.float64)
    q = q / np.sqrt(np.sum(q * q))

    # Extract quaternion components
    x, y, z, w = q

    # Compute squared terms
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w

    # Compute cross terms
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    # Construct rotation matrix with explicit type
    R: NDArray[np.float64] = np.array(
        [
            [w2 + x2 - y2 - z2, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), w2 - x2 + y2 - z2, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), w2 - x2 - y2 + z2],
        ],
        dtype=np.float64,
    )

    return R

def rotation_matrix_to_quaternion(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw].

    Parameters:
    R (NDArray[float64]): 3x3 rotation matrix

    Returns:
    NDArray[float64]: Quaternion in format [qx, qy, qz, qw]
    """
    # Ensure input is a numpy array
    R = np.array(R, dtype=np.float64)

    # Compute the trace of the matrix
    trace = np.trace(R)

    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return np.array([qx, qy, qz, qw], dtype=np.float64)


def project_points(points, pose, K):
    """
    Perform 3D to 2D projection of points using camera intrinsic matrix and camera pose.
    - points: in world frame
    - pose: camera pose in world frame
    """
    # 1. Convert Points to Homogeneous Coordinates:
    point_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 4)

    # 2. Get Camera Transform Matrix
    k_T_w = np.linalg.inv(pose)

    # 3. Transform Points to Camera Frame
    # p_k = k_T_w @ p_w
    camera_homogeneous = (k_T_w @ point_homogeneous.T).T  # (N,4)

    # 4. Project Points to 2D
    projected = (K @ camera_homogeneous[:, :3].T).T  # (N,3)

    return projected[:, :2] / projected[:, 2, np.newaxis] # (N,2)


def angle_between(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(cos_angle)
