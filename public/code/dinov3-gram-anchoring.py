"""Three-patch illustration of DINOv3's Gram objective, not paper results.

Requires NumPy. Checks rotation invariance and the normalized-feature gradient.
"""

import numpy as np


def gram_loss(features, target_gram):
    return np.sum((features @ features.T - target_gram) ** 2)


def main():
    reference = np.array([[1., 0.], [0., 1.], [2**-0.5, 2**-0.5]])
    rotation = np.array([[0., -1.], [1., 0.]])
    rotated = reference @ rotation
    corrupted = reference.copy()
    corrupted[2] = reference[0]
    target = reference @ reference.T
    assert np.allclose(target, rotated @ rotated.T)
    assert abs(np.sum((rotated - reference) ** 2) - 6.) < 1e-12
    assert abs(gram_loss(corrupted, target) - (4. - 2. * np.sqrt(2.))) < 1e-12

    # Differentiate with respect to the already-normalized feature matrix.
    # A trainable, unnormalized feature matrix needs the normalization Jacobian.
    analytical = 4 * (corrupted @ corrupted.T - target) @ corrupted
    numerical = np.empty_like(corrupted)
    step = 1e-6
    for index in np.ndindex(corrupted.shape):
        plus, minus = corrupted.copy(), corrupted.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = (gram_loss(plus, target) - gram_loss(minus, target)) / (2 * step)
    assert np.max(np.abs(analytical - numerical)) < 1e-8
    print("Reference Gram matrix:\n", np.round(target, 4))
    print(f"Rotation: Gram loss {gram_loss(rotated, target):.6f}; feature loss 6.000000")
    print(f"Changed patch: Gram loss {gram_loss(corrupted, target):.6f}")
    print("Finite-difference gradient check passed.")


if __name__ == "__main__":
    main()
