"""Check a toy shape-to-camera transform and single-view scale ambiguity.

Run: python sam3d-shape-layout.py
Requires NumPy. This is explanatory geometry, not a SAM 3D implementation.
"""
import numpy as np


def transform(points, rotation, scale, translation):
    """Accept an array with one column-vector point per row."""
    return (rotation @ (points * scale).T).T + translation


def project(points, focal_length=100.0):
    if np.any(points[:, 2] <= 0):
        raise ValueError("This example requires positive camera depth.")
    return focal_length * points[:, :2] / points[:, 2:3]


if __name__ == "__main__":
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    scale = np.array([2., 1., 1.])
    translation = np.array([1., 0., 4.])
    points = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.isclose(np.linalg.det(rotation), 1)
    camera_points = transform(points, rotation, scale, translation)
    assert np.allclose(camera_points[0], [1, 2, 4])
    assert np.allclose(project(camera_points)[0], [25, 50])
    larger = transform(points, rotation, 2 * scale, 2 * translation)
    assert np.allclose(larger, 2 * camera_points)
    assert np.allclose(project(larger), project(camera_points))
    print("Camera point:", camera_points[0])
    print("Image point:", project(camera_points)[0])
    print("Rotation, transformation, and global scale-ambiguity checks passed.")
