"""Coordinate illustration for Utonia. Requires NumPy; not a training reproduction."""
import numpy as np


def pairwise(x):
    return np.linalg.norm(x[:, None] - x[None, :], axis=-1)


def main():
    shape = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    a, b = 0.01 * shape, 0.1 * shape
    ga, gb = 0.01, 0.1
    na, nb = a / ga, b / gb
    np.testing.assert_allclose(na, nb)
    np.testing.assert_allclose(pairwise(na), pairwise(nb))
    shift = np.array([1.7, -2.3, 0.4])
    np.testing.assert_allclose((b + shift - shift) / gb, nb, atol=1e-12)
    np.testing.assert_allclose(np.cos(na[1, 0] - na[0, 0]), np.cos(nb[1, 0] - nb[0, 0]))
    with_far_point = np.vstack([a, [2.0, 0, 0]])
    old_box_edge = (a[1, 0] - a[0, 0]) / np.ptp(a[:, 0])
    new_box_edge = (a[1, 0] - a[0, 0]) / np.ptp(with_far_point[:, 0])
    assert new_box_edge < old_box_edge
    np.testing.assert_allclose((with_far_point[1] - with_far_point[0]) / ga, na[1] - na[0])
    print(f"Raw edge lengths: {pairwise(a)[0, 1]:.2f}, {pairwise(b)[0, 1]:.2f} meters")
    print(f"Aligned edge lengths: {pairwise(na)[0, 1]:.1f}, {pairwise(nb)[0, 1]:.1f}")
    print(f"One-plane rotary factor: {np.cos(1):.8f}")
    print(f"Bounding-box edge after adding distant point: {old_box_edge:.3f} -> {new_box_edge:.3f}")
    print("PASS: granularity scaling, recentering, and local-distance checks")


if __name__ == "__main__":
    main()
