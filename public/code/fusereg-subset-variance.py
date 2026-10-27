"""Enumerate FuseReg subset statistics and a linear-loss identity.

Requires NumPy; the small example is exact enumeration, not a trained RAE.
"""
import itertools
import numpy as np


def enumerate_subsets(features, drop_rate):
    k = len(features)
    for bits in itertools.product([0, 1], repeat=k):
        mask = np.array(bits, dtype=bool)
        r = int(mask.sum())
        if not r:
            continue
        probability = (1 - drop_rate)**r * drop_rate**(k-r) / (1-drop_rate**k)
        yield probability, features[mask].mean(axis=0), r


def check(features, drop_rate, decoder, target):
    k = len(features)
    mean = features.mean(axis=0)
    centered = features - mean
    disagreement = centered.T @ centered / k
    rows = list(enumerate_subsets(features, drop_rate))
    mean_sample = sum(p*z for p, z, _ in rows)
    covariance = sum(p*np.outer(z-mean, z-mean) for p, z, _ in rows)
    coefficient = sum(p*(k-r)/(r*(k-1)) for p, _, r in rows)
    np.testing.assert_allclose(sum(p for p, _, _ in rows), 1.)
    np.testing.assert_allclose(mean_sample, mean, atol=1e-12)
    np.testing.assert_allclose(covariance, coefficient*disagreement, atol=1e-12)
    loss = sum(p*np.sum((decoder @ z-target)**2) for p, z, _ in rows)
    base = np.sum((decoder @ mean-target)**2)
    penalty = coefficient*np.trace(decoder @ disagreement @ decoder.T)
    np.testing.assert_allclose(loss, base+penalty, atol=1e-12)
    return coefficient, covariance, loss


if __name__ == "__main__":
    h = np.array([[1., 0.], [1., 1.], [1., -1.]])
    c, covariance, loss = check(h, .5, np.array([[0., 2.]]), np.array([0.]))
    np.testing.assert_allclose(c, 15/28)
    np.testing.assert_allclose(covariance[1, 1], 5/14)
    np.testing.assert_allclose(loss, 10/7)
    print(f"c_3(0.5) = {c:.9f}")
    print(f"disagreement-axis variance = {covariance[1, 1]:.9f}")
    print(f"linear expected loss = {loss:.9f}")
    rng = np.random.default_rng(21)
    for k in [2, 3, 5]:
        for p in [0., .2, .5, .9]:
            check(rng.normal(size=(k, 4)), p, rng.normal(size=(2, 4)), rng.normal(size=2))
    print("All exact subset-mean, covariance and linear-loss checks passed.")
