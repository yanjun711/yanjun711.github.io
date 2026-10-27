"""Check frozen residual mixing, not the full mHC network Jacobian.

Run: python mhc-sinkhorn-residual-mixing.py
Requires NumPy. All examples are illustrative.
"""
import numpy as np


def sinkhorn(logits, iterations=20):
    matrix = np.exp(logits - np.max(logits))
    for _ in range(iterations):
        matrix /= matrix.sum(axis=0, keepdims=True)
        matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


if __name__ == "__main__":
    mixing = sinkhorn(np.log([[3., 1.], [1., 3.]]))
    assert np.allclose(mixing, [[0.75, 0.25], [0.25, 0.75]])
    common = np.array([1., 1.])
    difference = np.array([1., -1.])
    assert np.allclose(mixing @ common, common)
    assert np.allclose(mixing @ difference, 0.5 * difference)
    assert np.allclose(np.linalg.matrix_power(mixing, 10) @ difference,
                       difference / 1024)
    assert np.isclose(np.linalg.norm(mixing, 2), 1)
    row_only = np.array([[1., 0.], [1., 0.]])
    assert np.isclose(np.linalg.norm(row_only, 2), np.sqrt(2))
    rng = np.random.default_rng(7)
    approximate = sinkhorn(4 * rng.normal(size=(4, 4)))
    row_error = np.max(np.abs(approximate.sum(axis=1) - 1))
    column_error = np.max(np.abs(approximate.sum(axis=0) - 1))
    # Final rows sum to one. The induced-norm inequality bounds the
    # spectral norm by the square root of the largest column sum.
    approximate_bound = np.sqrt(np.max(approximate.sum(axis=0)))
    assert np.linalg.norm(approximate, 2) <= approximate_bound + 1e-12
    product = mixing @ mixing
    assert np.allclose(product.sum(axis=0), 1)
    assert np.allclose(product.sum(axis=1), 1)
    print(f"20-step random-matrix row residual: {row_error:.3e}")
    print(f"20-step random-matrix column residual: {column_error:.3e}")
    print(f"Finite-iteration spectral-norm upper bound: {approximate_bound:.6f}")
    print("Toy eigenmode, spectral norm, and composition checks passed.")
