"""MolmoAct2-Think mask/buffer illustration. Requires NumPy, not model weights."""
import numpy as np


def update_mask(previous, current, threshold=0.996):
    norms = np.linalg.norm(previous, axis=1) * np.linalg.norm(current, axis=1)
    if np.any(norms == 0):
        raise ValueError("This cosine illustration requires nonzero patch vectors")
    similarity = (previous * current).sum(axis=1) / norms
    return similarity < threshold, similarity


def fill_prefix(cached, mask, predictions):
    # Replacement values stand in for causal model predictions at changed cells.
    if cached is None:
        return np.array(predictions, dtype=int), [('generate', len(predictions))]
    prefix = cached.copy()
    schedule = []
    i = 0
    while i < len(mask):
        if mask[i]:
            prefix[i] = predictions[i]
            schedule.append(('generate', 1))
            i += 1
        else:
            start = i
            while i < len(mask) and not mask[i]:
                i += 1
            schedule.append(('replay', i - start))
    return prefix, schedule


def main():
    # Two-dimensional patch vectors replace the actual 32x32x3 RGB patches.
    previous = np.tile([1.0, 0.0], (4, 1))
    angle = np.array([0.05, 0.10, 0.05, 0.10])
    current = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    mask, similarity = update_mask(previous, current)
    np.testing.assert_array_equal(mask, [False, True, False, True])
    old = np.array([12, 33, 7, 80])
    replacements = np.array([90, 35, 90, 81])
    prefix, schedule = fill_prefix(old, mask, replacements)
    np.testing.assert_array_equal(prefix, [12, 35, 7, 81])
    assert sum(n for _, n in schedule) == 4
    assert sum(n for mode, n in schedule if mode == 'generate') == 2
    _, span_schedule = fill_prefix(old, np.array([0, 0, 1, 1], dtype=bool), replacements)
    assert span_schedule == [('replay', 2), ('generate', 1), ('generate', 1)]
    reset_prefix, reset_schedule = fill_prefix(None, mask, replacements)
    np.testing.assert_array_equal(reset_prefix, replacements)
    assert reset_schedule == [('generate', 4)]
    bright_mask, bright_cosine = update_mask(previous, 1.7 * previous)
    assert not bright_mask.any()
    np.testing.assert_allclose(bright_cosine, 1)
    print('Cosine similarities:', np.round(similarity, 6))
    print('Update mask:', mask.astype(int))
    print('Filled prefix:', prefix)
    print('Schedule:', schedule)
    print('New predictions: 2; consumed depth codes: 4 (not a speed measurement)')
    print('PASS: threshold, buffer, replay spans, reset, and brightness scaling')


if __name__ == '__main__':
    main()
