"""A one-dimensional analogue of latent grid alignment.

This illustrates interpolation and channel conventions. It does not implement
or evaluate the paper's learned 4D reconstructor. Requires NumPy.
"""
import numpy as np


def align(z, out_times, projection):
    source_times = np.linspace(0., 1., len(z))
    resampled = np.column_stack([
        np.interp(out_times, source_times, z[:, c])
        for c in range(z.shape[1])
    ])
    return resampled @ projection


if __name__ == "__main__":
    z = np.array([[1., 0.], [0., 1.]])
    times = np.linspace(0., 1., 4)
    projection = np.array([[1., 0., -1.], [0., 1., 1.]])
    expected = np.array([[1., 0., -1.], [2/3, 1/3, -1/3],
                         [1/3, 2/3, 1/3], [0., 1., 1.]])
    q = align(z, times, projection)
    np.testing.assert_allclose(q, expected, atol=1e-14)
    swap = np.array([[0., 1.], [1., 0.]])
    incompatible = align(z @ swap, times, projection)
    corrected = align(z @ swap, times, swap @ projection)
    np.testing.assert_allclose(corrected, q)
    assert not np.allclose(incompatible, q)
    origin, direction, depth = np.array([1., 0., 0.]), np.array([0., 0., 1.]), 2.
    np.testing.assert_allclose(origin + depth * direction, [1., 0., 2.])
    print("Aligned tokens:\n", np.round(q, 6))
    print("Same-shaped, swapped-channel tokens:\n", np.round(incompatible, 6))
    print("Alignment and channel-convention checks passed.")
