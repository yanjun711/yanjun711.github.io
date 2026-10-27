"""Analytical checks for the MOVA timing example; requires NumPy."""
import numpy as np


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def score(video_index, audio_index, video_rate=2.0, audio_rate=8.0, theta=2.0):
    pv = audio_rate / video_rate * video_index
    return np.cos((audio_index - pv) / theta)


def main():
    assert np.isclose(score(1, 4), 1.0)
    assert np.isclose(score(1, 1), np.cos(1.5))
    native_simultaneous = np.cos((4 - 1) / 2)
    native_early = np.cos((1 - 1) / 2)
    print(f"Native simultaneous: {native_simultaneous:.8f}; native early: {native_early:.8f}")
    print(f"Aligned simultaneous: {score(1, 4):.8f}; aligned early: {score(1, 1):.8f}")
    rng = np.random.default_rng(7)
    for _ in range(100):
        q, k = rng.normal(size=(2, 2))
        pv, pa, theta = rng.uniform(0.1, 8, size=3)
        actual = (rotation(pv / theta) @ q) @ (rotation(pa / theta) @ k)
        expected = q @ rotation((pa - pv) / theta) @ k
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    for shift in [0.125, 0.5, 1.75]:
        np.testing.assert_allclose(score(1 + 2 * shift, 3 + 8 * shift), score(1, 3))
    print("PASS: rotary identity and common-time-shift invariance")


if __name__ == "__main__":
    main()
