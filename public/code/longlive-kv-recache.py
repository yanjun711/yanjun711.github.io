"""Toy conditional KV cache; not a trained video generator.

Requires NumPy. Fixed visual inputs, prompt-dependent hidden representations,
and a fixed query isolate the effect of re-caching.
"""

import numpy as np


def cache(visual_features, prompt):
    hidden = visual_features.copy()
    hidden[:, 1] += prompt * visual_features[:, 0]
    return hidden.copy(), hidden.copy()  # Identity key and value projections.


def attend(query, keys, values):
    scores = query @ keys.T / np.sqrt(keys.shape[1])
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    return weights, weights @ values


def main():
    visual = np.array([[1., 0.], [0., 1.]])
    untouched_visual = visual.copy()
    query = np.array([1., 1.])
    old_cache = cache(visual, 0.)
    new_cache = cache(visual, 1.)
    old_weights, old_output = attend(query, *old_cache)
    new_weights, new_output = attend(query, *new_cache)
    assert np.array_equal(visual, untouched_visual)
    assert np.allclose(old_weights, [0.5, 0.5])
    assert np.allclose(old_output, [0.5, 0.5])
    assert np.allclose(new_weights, [0.66976155, 0.33023845])
    assert np.allclose(new_output, [0.66976155, 1.])
    assert not np.allclose(old_cache[0], new_cache[0])
    print("Stale cache:  weights", old_weights, "output", old_output)
    print("Re-cached:    weights", new_weights, "output", new_output)
    print("Visual inputs remain unchanged.")


if __name__ == "__main__":
    main()
