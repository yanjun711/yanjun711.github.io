"""Render MOVA's shared temporal coordinates; Python standard library only.

Run from any directory: python scripts/figures/mova.py
Rates are illustrative latent-frame rates, not model hyperparameters.
"""
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'public/images/blog/mova-aligned-rope'
INK, GRAY, PALE, ACCENT = '#282828', '#777777', '#d8d8d8', '#526b78'


def text(x, y, value, size=13, anchor='start', color=INK, weight='normal'):
    return (f'<text x="{x:g}" y="{y:g}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{color}" font-weight="{weight}">'
            f'{escape(value)}</text>')


def line(x1, y1, x2, y2, color=PALE, width=1, extra=''):
    return (f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" '
            f'y2="{y2:g}" stroke="{color}" stroke-width="{width}" {extra}/>')


def token(x, y, chosen=False):
    radius = 4.6 if chosen else 2.5
    return (f'<circle cx="{x:g}" cy="{y:g}" r="{radius:g}" '
            f'fill="{ACCENT if chosen else "white"}" '
            f'stroke="{ACCENT if chosen else GRAY}" stroke-width="1"/>')


def panel(items, aligned, mobile):
    if mobile:
        offset = 0 if not aligned else 132
        left, start, end = 12, 82, 343
        heading, video_y, audio_y, axis_y = 43 + offset, 79 + offset, 111 + offset, 143 + offset
    else:
        offset = 0 if not aligned else 318
        left, start, end = 12 + offset, 82 + offset, 294 + offset
        heading, video_y, audio_y, axis_y = 52, 96, 150, 190
    def pos(value):
        return start + value * (end - start) / 8
    items.append(text(left, heading, 'Aligned positions' if aligned else 'Native indices',
                      size=14, weight='bold'))
    if mobile:
        items.append(text(end, heading, 'Δp = 0' if aligned else 'Δp = 3',
                          anchor='end', color=ACCENT))
    # Each row represents coordinates used by RoPE, not physical seconds.
    for y, name in [(video_y, 'Video'), (audio_y, 'Audio')]:
        items += [text(left, y + 4, name), line(start, y, end, y)]
    chosen_video = 4 if aligned else 1
    # The connector marks the same physical event in both streams.
    items.append(line(pos(chosen_video), video_y, pos(4), audio_y,
                      ACCENT, 1.1, 'stroke-dasharray="3 3"'))
    for index in range(3):
        value = 4 * index if aligned else index
        items.append(token(pos(value), video_y, index == 1))
    for index in range(9):
        items.append(token(pos(index), audio_y, index == 4))
    video_label = 'i = 1; p = 4' if aligned else 'i = 1; p = 1'
    items += [text(pos(chosen_video), video_y - 13, video_label, anchor='middle', color=ACCENT),
              text(pos(4), audio_y + 19, 'j = 4; p = 4', anchor='middle', color=ACCENT),
              line(start, axis_y, end, axis_y, GRAY, .8)]
    for value in [0, 1, 4, 8]:
        items += [line(pos(value), axis_y, pos(value), axis_y + 4, GRAY, .8),
                  text(pos(value), axis_y + 19, str(value), anchor='middle', color=GRAY)]
    items.append(text(start - 13, axis_y + 19, 'p', anchor='end', color=GRAY))
    if not mobile:
        items.append(text((start + end) / 2, 235,
                          'Δp = 0' if aligned else 'Δp = 3',
                          anchor='middle', color=ACCENT))


def render(mobile=False):
    width, height = (360, 320) if mobile else (620, 258)
    items = []
    if mobile:
        items.append(text(12, 17, 'Video 2 Hz · audio 8 Hz · event at 0.5 s'))
    else:
        items += [text(12, 20, 'One event at t = 0.5 s', size=14),
                  text(608, 20, 'Video 2 Hz · audio 8 Hz', anchor='end', color=GRAY)]
    panel(items, False, mobile)
    panel(items, True, mobile)
    if mobile:
        items.append(text(180, 315, 'Aligned: video p = 4i; audio p = j', anchor='middle'))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="Lato, Arial, sans-serif" '
            'role="img" aria-labelledby="title desc">'
            '<title id="title">MOVA temporal-coordinate alignment</title>'
            '<desc id="desc">At video rate two hertz and audio rate eight hertz, '
            'video index one and audio index four both occur at half a second. '
            'Native positions differ by three. Scaling the video position by four '
            'puts both at rotary position four.</desc>'
            '<rect width="100%" height="100%" fill="white"/>'
            + ''.join(items) + '</svg>\n')


def check():
    video_rate, audio_rate, video_index, audio_index = 2, 8, 1, 4
    assert video_index / video_rate == audio_index / audio_rate == .5
    assert audio_index - video_index == 3
    assert audio_rate / video_rate * video_index == audio_index == 4


if __name__ == '__main__':
    check()
    TARGET.mkdir(parents=True, exist_ok=True)
    for mobile in [False, True]:
        name = 'clocks' + ('-mobile' if mobile else '') + '.svg'
        (TARGET / name).write_text(render(mobile))
    print('MOVA: verified physical timestamps and both position systems; wrote desktop/mobile SVGs.')
