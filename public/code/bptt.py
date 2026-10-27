"""A tanh RNN, manual BPTT, and a central-difference gradient check.

Requires Python 3 and NumPy. Run: python bptt.py
The loss is a sum over time and output coordinates, with no batch dimension.
"""

import numpy as np


def forward(p, x, y):
    states = [np.zeros(p["W"].shape[0], dtype=x.dtype)]
    errors = []
    loss = 0.0
    for xt, yt in zip(x, y):
        h = np.tanh(p["W"] @ states[-1] + p["U"] @ xt + p["b"])
        e = p["V"] @ h + p["c"] - yt
        states.append(h)
        errors.append(e)
        loss += 0.5 * (e @ e)
    return float(loss), (states, errors)


def backward(p, x, cache):
    states, errors = cache
    grads = {name: np.zeros_like(value) for name, value in p.items()}
    dh_next = np.zeros_like(states[0])
    for t in reversed(range(len(x))):
        h_prev, h = states[t], states[t + 1]
        e = errors[t]
        dh = p["V"].T @ e + dh_next
        da = dh * (1.0 - h * h)
        grads["V"] += np.outer(e, h)
        grads["c"] += e
        grads["W"] += np.outer(da, h_prev)
        grads["U"] += np.outer(da, x[t])
        grads["b"] += da
        dh_next = p["W"].T @ da
    return grads


def numerical_gradients(p, x, y, epsilon=1e-6):
    result = {name: np.zeros_like(value) for name, value in p.items()}
    for name, value in p.items():
        for index in np.ndindex(value.shape):
            original = value[index]
            try:
                value[index] = original + epsilon
                positive, _ = forward(p, x, y)
                value[index] = original - epsilon
                negative, _ = forward(p, x, y)
            finally:
                value[index] = original
            result[name][index] = (positive - negative) / (2 * epsilon)
    return result


def main():
    rng = np.random.default_rng(7)
    p = {
        "W": 0.25 * rng.standard_normal((3, 3)),
        "U": 0.25 * rng.standard_normal((3, 2)),
        "b": np.zeros(3),
        "V": 0.25 * rng.standard_normal((2, 3)),
        "c": np.zeros(2),
    }
    x = rng.standard_normal((5, 2))
    y = rng.standard_normal((5, 2))
    loss, cache = forward(p, x, y)
    analytic = backward(p, x, cache)
    numerical = numerical_gradients(p, x, y)
    print(f"Loss: {loss:.6f}")
    for name in p:
        error = np.max(np.abs(analytic[name] - numerical[name]))
        np.testing.assert_allclose(analytic[name], numerical[name], rtol=1e-5, atol=1e-7)
        print(f"{name}: max absolute gradient error = {error:.2e}")
    print("All parameter gradients passed.")


if __name__ == "__main__":
    main()
