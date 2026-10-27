"""Illustrate timestamp selection, not robot dynamics or task performance.

Run: python dynamicvla-action-time-alignment.py
Uses the Python standard library only.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    source: int
    ready: int
    actions: dict[int, tuple[int, int]]


def make_chunk(source, delay=2, count=6):
    return Chunk(source, source + delay,
                 {tick: (source, tick) for tick in range(source, source + count)})


def action_for_tick(chunks, tick):
    candidates = [chunk for chunk in chunks
                  if chunk.ready <= tick and tick in chunk.actions]
    if not candidates:
        return None
    newest = max(candidates, key=lambda chunk: chunk.source)
    return newest.actions[tick]


if __name__ == "__main__":
    chunks = [make_chunk(source) for source in (0, 2, 4)]
    selected = [action_for_tick(chunks, tick) for tick in range(2, 8)]
    assert selected == [(0, 2), (0, 3), (2, 4), (2, 5), (4, 6), (4, 7)]
    for tick, (source, action_tick) in zip(range(2, 8), selected):
        assert action_tick == tick
        print(f"tick {tick}: observation {source}, age {(tick-source)*50} ms")
    # The older observation arrives later, but must not replace the newer one.
    newer = make_chunk(2, delay=2)
    older_late = make_chunk(0, delay=5)
    assert action_for_tick([newer, older_late], 5) == (2, 5)
    # Six actions at ticks 0..5 have all expired by ready tick 6.
    expired = make_chunk(0, delay=6)
    assert action_for_tick([expired], 6) is None
    # n=4 > m=3 leaves a valid suffix, but not enough to reach the next chunk.
    short = [make_chunk(source, delay=3, count=5) for source in (0, 3)]
    assert action_for_tick(short, 5) is None
    assert action_for_tick(short, 6) == (3, 6)
    assert action_for_tick(chunks, 0) is None
    print("Timestamp, freshness, and expired-horizon checks passed.")
