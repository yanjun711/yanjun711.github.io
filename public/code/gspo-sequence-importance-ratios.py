"""Check the GSPO ratio and a tiny surrogate objective in log-ratio space.

Independent log-ratio coordinates are a mathematical illustration, not an
implementation of a normalized language-model policy. Python 3, stdlib only.
"""

from math import exp, log, prod


def ratio(deltas):
    return exp(sum(deltas) / len(deltas))


def objective(deltas, advantage, epsilon=0.1):
    s = ratio(deltas)
    clipped = max(1.0 - epsilon, min(1.0 + epsilon, s))
    return min(s * advantage, clipped * advantage)


def finite_difference(deltas, advantage, position, step=1e-6):
    plus, minus = list(deltas), list(deltas)
    plus[position] += step
    minus[position] -= step
    return (objective(plus, advantage) - objective(minus, advantage)) / (2 * step)


def main():
    weights = [1.44, 0.81]
    deltas = [log(w) for w in weights]
    assert abs(prod(weights) - 1.1664) < 1e-12
    assert abs(ratio(deltas) - 1.08) < 1e-12
    for i in range(2):
        assert abs(finite_difference(deltas, 1.0, i) - 0.54) < 1e-9
    assert abs(ratio([log(4), log(0.25)]) - 1.0) < 1e-12
    for s, advantage in [(1.2, 1.0), (0.8, -1.0)]:
        point = [log(s), log(s)]
        assert abs(finite_difference(point, advantage, 0)) < 1e-9
    print(f"Joint ratio: {prod(weights):.4f}; GSPO ratio: {ratio(deltas):.2f}")
    print("Active log-ratio gradients:", [round(finite_difference(deltas, 1.0, i), 8) for i in range(2)])
    print("Positive upper-clip and negative lower-clip checks passed.")


if __name__ == "__main__":
    main()
