"""An eight-token MSA selector illustration, not a trained MSA layer.

Run with Python 3 and NumPy. Scores are supplied explicitly so that block
selection, restricted softmax, and the omitted-mass bound can be inspected.
"""
import numpy as np


def softmax(x):
    weights = np.exp(x - np.max(x))
    return weights / weights.sum()


def select_blocks(index_scores, block_size=2, budget=2):
    scores = np.asarray(index_scores).reshape(-1, block_size).max(axis=1)
    local = len(scores) - 1  # This example queries the final token.
    others = np.argsort(scores[:local])[-(budget - 1):]
    return np.sort(np.r_[others, local])


def main():
    index_scores = np.array([
        [3.0, 2.5, 1.0, 0.0, 0.0, 0.2, -1.0, -2.0],
        [0.1, 0.2, 2.5, 3.0, 0.2, 0.3, -1.0, -2.0],
    ])
    main_scores = np.array([
        [2., 1., -2., -2., -2., -2., 0., 0.],
        [-2., -2., 1., 2., -2., -2., 0., 0.],
    ])
    values = np.array([1., 1., -1., -1., 0.5, 0.5, 0., 0.])
    for group in range(2):
        blocks = select_blocks(index_scores[group])
        expected = [0, 3] if group == 0 else [1, 3]
        np.testing.assert_array_equal(blocks, expected)
        tokens = (2 * blocks[:, None] + np.arange(2)).reshape(-1)
        dense_prob = softmax(main_scores[group])
        sparse_prob = softmax(main_scores[group, tokens])
        mass = dense_prob[tokens].sum()
        np.testing.assert_allclose(sparse_prob, dense_prob[tokens] / mass)
        dense_output = dense_prob @ values
        sparse_output = sparse_prob @ values[tokens]
        error = abs(dense_output - sparse_output)
        bound = 2 * np.max(np.abs(values)) * (1 - mass)
        assert error <= bound + 1e-12
        print(f"Group {group + 1}: blocks {(blocks + 1).tolist()}")
        print(f"  retained dense mass={mass:.6f}")
        print(f"  dense={dense_output:.6f}, sparse={sparse_output:.6f}")
        print(f"  error={error:.6f}, bound={bound:.6f}")
    print("All selection and restricted-attention checks passed.")


if __name__ == "__main__":
    main()
