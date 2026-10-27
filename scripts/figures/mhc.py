"""Render the exact two-stream mHC example; Python standard library only.

Run from any directory: python scripts/figures/mhc.py
The fixed doubly stochastic matrix is not a full-network Jacobian bound.
"""
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'public/images/blog/mhc-sinkhorn-residual-mixing'
INK, GRAY, PALE, ACCENT = '#282828', '#777777', '#d8d8d8', '#526b78'


def text(x, y, value, size=13, anchor='start', color=INK, weight='normal'):
    return (f'<text x="{x:g}" y="{y:g}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{color}" font-weight="{weight}">'
            f'{escape(value)}</text>')


def line(x1, y1, x2, y2, color=PALE, width=1, extra=''):
    return (f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" '
            f'y2="{y2:g}" stroke="{color}" stroke-width="{width}" {extra}/>')


def matrix(items, mobile):
    if mobile:
        left, right, top, bottom = 90, 222, 33, 112
        xs, ys, total_x, total_y = [124, 188], [60, 99], 283, 135
        items += [text(12, 17, 'Nonnegative mixing matrix', weight='bold'),
                  text(total_x, 29, 'Row sum', anchor='middle', color=GRAY),
                  text(35, 85, 'B =', size=19)]
    else:
        left, right, top, bottom = 64, 215, 66, 168
        xs, ys, total_x, total_y = [103, 177], [102, 149], 267, 199
        items += [text(12, 23, 'Nonnegative mixing matrix', size=14, weight='bold'),
                  text(total_x, 55, 'Row sum', anchor='middle', color=GRAY),
                  text(12, 128, 'B =', size=19)]
    # Matrix brackets, not a table or a surrounding card.
    for x, direction in [(left, 1), (right, -1)]:
        items += [line(x, top, x, bottom, INK),
                  line(x, top, x + direction * 7, top, INK),
                  line(x, bottom, x + direction * 7, bottom, INK)]
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            value = [[.75, .25], [.25, .75]][row][col]
            items.append(text(x, y, f'{value:.2f}', size=21,
                              anchor='middle', color=ACCENT if row == col else INK))
        items += [line(right + 13, y - 7, total_x - 16, y - 7),
                  text(total_x, y, '1', size=17, anchor='middle')]
    for x in xs:
        items += [line(x, bottom + 6, x, total_y - 18),
                  text(x, total_y, '1', size=17, anchor='middle')]
    if mobile:
        items.append(text(12, total_y, 'Column sum', color=GRAY))
    else:
        items.append(text(140, 224, 'Column sums', anchor='middle', color=GRAY))


def vector(items, left, right, y, values):
    """Two components shown as a mathematical column vector."""
    top, bottom = y - 25, y + 28
    for x, direction in [(left, 1), (right, -1)]:
        items += [line(x, top, x, bottom, INK),
                  line(x, top, x + direction * 5, top, INK),
                  line(x, bottom, x + direction * 5, bottom, INK)]
    for offset, value in [(-8, values[0]), (19, values[1])]:
        items.append(text((left + right) / 2, y + offset, value,
                          size=19, anchor='middle', color=ACCENT))


def modes(items, mobile):
    if mobile:
        for left, heading, values, note in [
            (12, 'Shared mode', ['1', '1'], 'Values unchanged'),
            (190, 'Difference mode', ['0.5', '−0.5'], 'Difference ×½'),
        ]:
            items.append(text(left, 164, heading, weight='bold'))
            items.append(text(left, 207, 'B', size=20))
            vector(items, left + 28, left + 62, 198, ['1', '1'] if left == 12 else ['1', '−1'])
            items.append(text(left + 73, 207, '=', size=20))
            vector(items, left + 93, left + (137 if left == 12 else 151), 198, values)
            items.append(text(left, 249, note, color=GRAY))
        items.append(text(12, 279, "Each feature's stream mean is conserved.", color=GRAY))
    else:
        items.append(text(330, 23, 'Action on two stream modes', size=14, weight='bold'))
        for center_y, heading, source, result in [
            (88, 'Shared mode', ['1', '1'], ['1', '1']),
            (180, 'Difference mode', ['1', '−1'], ['0.5', '−0.5']),
        ]:
            items.append(text(330, center_y - 38, heading, weight='bold'))
            items.append(text(330, center_y + 9, 'B', size=21))
            vector(items, 366, 406, center_y, source)
            items.append(text(424, center_y + 9, '=', size=21))
            vector(items, 451, 513, center_y, result)


def render(mobile=False):
    width, height = (360, 290) if mobile else (620, 238)
    items = []
    matrix(items, mobile)
    modes(items, mobile)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="Lato, Arial, sans-serif" '
            'role="img" aria-labelledby="title desc">'
            '<title id="title">Doubly stochastic mixing of two residual streams</title>'
            '<desc id="desc">Every row and column of the matrix sums to one. '
            'The shared vector (1,1) is unchanged; (1,-1) becomes (one half, minus one half). '
            'The equations show exact action on these two modes.</desc>'
            '<rect width="100%" height="100%" fill="white"/>'
            + ''.join(items) + '</svg>\n')


def check():
    b = [[.75, .25], [.25, .75]]
    assert all(sum(row) == 1 for row in b)
    assert all(sum(row[col] for row in b) == 1 for col in range(2))
    for vector, expected in [([1, 1], [1, 1]), ([1, -1], [.5, -.5])]:
        assert [sum(a * x for a, x in zip(row, vector)) for row in b] == expected


if __name__ == '__main__':
    check()
    TARGET.mkdir(parents=True, exist_ok=True)
    for mobile in [False, True]:
        name = 'mixing-depth' + ('-mobile' if mobile else '') + '.svg'
        (TARGET / name).write_text(render(mobile))
    print('mHC: verified row/column sums and both modes; wrote desktop/mobile SVGs.')
