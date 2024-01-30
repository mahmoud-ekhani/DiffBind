import numpy as np

from constants import CA_C_DIST, N_CA_DIST, N_CA_C_ANGLE


def rotation_matrix(angle: np.ndarray, axis: int) -> np.ndarray:
    """
    Compute rotation matrices for given angles and axes.

    Args:
        angle (np.ndarray): An array of rotation angles in radians of shape (n,).
        axis (int): An integer representing the rotation axis:
            - 0: Rotation about the x-axis.
            - 1: Rotation about the y-axis.
            - 2: Rotation about the z-axis.

    Returns:
        np.ndarray: An array of rotation matrices of shape (n, 3, 3) corresponding
        to the rotations specified by the angles and axes.
    """
    n = len(angle)
    R = np.eye(3)[None, :, :].repeat(n, axis=0)

    axis = 2 - axis
    start = axis // 2
    step = axis % 2 + 1
    s = slice(start, start + step + 1, step)

    R[:, s, s] = np.array(
        [[np.cos(angle), (-1) ** (axis + 1) * np.sin(angle)],
         [(-1) ** axis * np.sin(angle), np.cos(angle)]]
    ).transpose(2, 0, 1)
    return R


def get_bb_transform(n_xyz: np.ndarray, ca_xyz: np.ndarray, c_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute translation and rotation of the canonical backbone frame (triangle N-Ca-C) from a position with
    Ca at the origin, N on the x-axis, and C in the xy-plane to the global position of the backbone frame.

    Args:
        n_xyz (np.ndarray): An array of shape (n, 3) representing the coordinates of N atoms.
        ca_xyz (np.ndarray): An array of shape (n, 3) representing the coordinates of Ca atoms.
        c_xyz (np.ndarray): An array of shape (n, 3) representing the coordinates of C atoms.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
        - quaternion (np.ndarray): An array of shape (n, 4) representing the rotation as quaternions.
        - translation (np.ndarray): An array of shape (n, 3) representing the translation vectors.
    """
    translation = ca_xyz
    n_xyz = n_xyz - translation
    c_xyz = c_xyz - translation

    # Find rotation matrix that aligns the coordinate systems
    #    rotate around y-axis to move N into the xy-plane
    theta_y = np.arctan2(n_xyz[:, 2], -n_xyz[:, 0])
    Ry = rotation_matrix(theta_y, 1)
    n_xyz = np.einsum('noi,ni->no', Ry.transpose(0, 2, 1), n_xyz)

    #    rotate around z-axis to move N onto the x-axis
    theta_z = np.arctan2(n_xyz[:, 1], n_xyz[:, 0])
    Rz = rotation_matrix(theta_z, 2)

    #    rotate around x-axis to move C into the xy-plane
    c_xyz = np.einsum('noj,nji,ni->no', Rz.transpose(0, 2, 1),
                      Ry.transpose(0, 2, 1), c_xyz)
    theta_x = np.arctan2(c_xyz[:, 2], c_xyz[:, 1])
    Rx = rotation_matrix(theta_x, 0)

    # Final rotation matrix
    R = np.einsum('nok,nkj,nji->noi', Ry, Rz, Rx)

    # Convert to quaternion
    # q = w + i*u_x + j*u_y + k * u_z
    quaternion = rotation_matrix_to_quaternion(R)

    return quaternion, translation


def rotation_matrix_to_quaternion(R):
    """
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    Args:
        R: (n, 3, 3)
    Returns:
        q: (n, 4)
    """

    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    r = np.sqrt(1 + t)
    w = 0.5 * r
    x = np.sign(R[:, 2, 1] - R[:, 1, 2]) * np.abs(
        0.5 * np.sqrt(1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]))
    y = np.sign(R[:, 0, 2] - R[:, 2, 0]) * np.abs(
        0.5 * np.sqrt(1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]))
    z = np.sign(R[:, 1, 0] - R[:, 0, 1]) * np.abs(
        0.5 * np.sqrt(1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]))

    return np.stack((w, x, y, z), axis=1)
