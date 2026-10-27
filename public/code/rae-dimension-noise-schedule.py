"""Verify RAE's dimension shift and an illustrative repeated-signal model.

Run: python rae-dimension-noise-schedule.py
Requires NumPy. This script contains no benchmark or training experiment.
"""
import numpy as np


def shift(t, source_dim, target_dim):
    alpha = np.sqrt(target_dim / source_dim)
    return alpha * t / (1 + (alpha - 1) * t)


def inverse_shift(t, source_dim, target_dim):
    alpha = np.sqrt(target_dim / source_dim)
    return t / (alpha - (alpha - 1) * t)


def average_snr(dimension, time):
    return dimension * ((1 - time) / time) ** 2


if __name__ == "__main__":
    grid = np.linspace(0, 1, 1001)
    for target in [1024, 4096, 16384, 196608]:
        mapped = shift(grid, 4096, target)
        assert mapped[0] == 0 and np.isclose(mapped[-1], 1)
        assert np.all(np.diff(mapped) > 0)
        assert np.allclose(inverse_shift(mapped, 4096, target), grid)
        interior = grid[1:-1]
        mapped_interior = mapped[1:-1]
        alpha = np.sqrt(target / 4096)
        assert np.allclose(
            mapped_interior / (1 - mapped_interior),
            alpha * interior / (1 - interior),
        )
        assert np.allclose(
            average_snr(target, mapped_interior),
            average_snr(4096, interior),
        )
    midpoint = shift(0.5, 4096, 16384)
    assert np.isclose(midpoint, 2 / 3)
    print(f"Four-times dimension midpoint: {midpoint:.6f}")
    print(f"256 x 768 midpoint: {shift(0.5, 4096, 256 * 768):.6f}")
    print("Endpoint, monotonicity, inverse, odds, and toy SNR checks passed.")
