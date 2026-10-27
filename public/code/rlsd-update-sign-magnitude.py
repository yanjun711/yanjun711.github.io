"""RLSD coefficient and detached score-function illustration. Requires NumPy."""
import numpy as np


def token_advantages(advantage, teacher, student, width=0.2, mix=0.5):
    # These returned coefficients are held fixed during the policy-gradient check.
    delta = np.log(teacher) - np.log(student)
    weights = np.exp(np.sign(advantage) * delta)
    return advantage * ((1 - mix) + mix * np.clip(weights, 1 - width, 1 + width))


def log_softmax(z):
    shifted = z - z.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def main():
    rewards = np.array([1.0, 0.0])
    advantages = (rewards - rewards.mean()) / rewards.std()
    student = np.array([0.2, 0.4, 0.3])
    teacher = np.array([0.4, 0.2, 0.3])
    positive = token_advantages(advantages[0], teacher, student)
    negative = token_advantages(advantages[1], teacher, student)
    np.testing.assert_allclose(positive, [1.1, 0.9, 1])
    np.testing.assert_allclose(negative, [-0.9, -1.1, -1])
    assert np.all(positive > 0) and np.all(negative < 0)
    np.testing.assert_allclose(token_advantages(0, teacher, student), 0)
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(3, 4))
    chosen = np.array([0, 2, 1])
    for fixed in [positive, negative]:
        def loss(z):
            return -(fixed * log_softmax(z)[np.arange(3), chosen]).sum()
        probs = np.exp(log_softmax(logits))
        onehot = np.eye(4)[chosen]
        analytic = -fixed[:, None] * (onehot - probs)
        numerical = np.zeros_like(logits)
        step = 1e-5
        for index in np.ndindex(logits.shape):
            plus, minus = logits.copy(), logits.copy()
            plus[index] += step
            minus[index] -= step
            numerical[index] = (loss(plus) - loss(minus)) / (2 * step)
        np.testing.assert_allclose(analytic, numerical, atol=1e-9)
    print("Positive response:", positive)
    print("Negative response:", negative)
    print("PASS: coefficient signs, zero advantage, and detached loss finite differences")


if __name__ == "__main__":
    main()
