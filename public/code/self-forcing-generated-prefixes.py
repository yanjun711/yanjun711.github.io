"""Scalar illustrations of generated prefixes and detached prefix gradients.

This is not a diffusion implementation or a reproduction of paper results.
Run with Python 3; no third-party dependencies are required.
"""


def rollout(b, a=0.9, steps=3):
    h = 1.0
    states = []
    for _ in range(steps):
        h = a * h + b
        states.append(h)
    return states


def loss(b):
    return sum(0.5 * (h - 1.0) ** 2 for h in rollout(b))


def main():
    a, b = 0.9, 0.15
    states = rollout(b, a)
    sensitivity = 0.0
    full_gradient = 0.0
    for h in states:
        sensitivity = a * sensitivity + 1.0
        full_gradient += (h - 1.0) * sensitivity
    detached_gradient = sum(h - 1.0 for h in states)
    delta = 1e-6
    numeric = (loss(b + delta) - loss(b - delta)) / (2 * delta)
    # A stop-gradient check holds the actual generated prefixes fixed.
    prefixes = [1.0] + states[:-1]

    def frozen_prefix_loss(test_b):
        return sum(0.5 * (a * p + test_b - 1.0) ** 2 for p in prefixes)

    frozen_numeric = (
        frozen_prefix_loss(b + delta) - frozen_prefix_loss(b - delta)
    ) / (2 * delta)
    assert abs(numeric - full_gradient) < 1e-9
    assert abs(frozen_numeric - detached_gradient) < 1e-9
    assert all(abs(x - y) < 1e-12 for x, y in zip(states, [1.05, 1.095, 1.1355]))
    assert abs(full_gradient - 0.597705) < 1e-12
    assert abs(detached_gradient - 0.2805) < 1e-12
    print("Generated states:", [round(h, 4) for h in states])
    print(f"Full gradient: {full_gradient:.6f}")
    print(f"Detached-prefix gradient: {detached_gradient:.6f}")
    print("Both finite-difference checks passed.")


if __name__ == "__main__":
    main()
