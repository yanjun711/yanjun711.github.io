"""Draw the Self Forcing conditioning/gradient figure using Python's stdlib.

Run from any directory: python scripts/figures/self_forcing.py
The two outputs replace the article's original analytical toy plots.
"""

from html import escape
from pathlib import Path


INK = "#282828"
GRAY = "#777777"
LIGHT = "#d8d8d8"
ACCENT = "#526b78"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "public/images/blog/self-forcing-generated-prefixes"


class Drawing:
    def __init__(self, width, height):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            '<title id="title">Generated context with a detached KV cache</title>',
            '<desc id="desc">Previous generated frames are encoded without gradients into a KV prefix. The selected denoising call reads this prefix and produces the current frame. Gradients reach the selected call but stop before entering the cached prefix.</desc>',
            '<rect width="100%" height="100%" fill="white"/>',
            '<defs>',
        ]
        for name, color in [("ink", INK), ("gray", GRAY), ("accent", ACCENT)]:
            self.parts.append(
                f'<marker id="arrow-{name}" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1 1L5 3L1 5" fill="none" stroke="{color}" stroke-width="1"/></marker>'
            )
        self.parts.extend(['</defs>', '<g font-family="Lato, Arial, sans-serif" fill="#282828" stroke-linecap="round" stroke-linejoin="round">'])

    def text(self, x, y, value, size=14, color=INK, anchor="middle", weight="400"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(value)}</text>')

    def path(self, d, color=INK, arrow=False, dashed=False):
        name = "accent" if color == ACCENT else "gray" if color == GRAY else "ink"
        marker = f' marker-end="url(#arrow-{name})"' if arrow else ""
        dash = ' stroke-dasharray="4 4"' if dashed else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.15"{marker}{dash}/>')

    def rect(self, x, y, width, height, stroke=LIGHT, fill="white"):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" stroke="{stroke}" fill="{fill}" stroke-width="1"/>')

    def frames(self, x, y, cell=36, gap=12):
        for index, label in enumerate(["x¹", "x²", "…", "xⁱ⁻¹"]):
            left = x + index * (cell + gap)
            if label != "…":
                self.rect(left, y, cell, 29, stroke=GRAY)
            self.text(left + cell / 2, y + 20, label, 15)

    def cache(self, x, y, width):
        # Two rows indicate keys and values; columns indicate past positions.
        self.text(x - 12, y + 12, "K", 13, GRAY)
        self.text(x - 12, y + 32, "V", 13, GRAY)
        cell = (width - 12) / 4
        for row in range(2):
            for col in range(4):
                self.rect(x + col * (cell + 4), y + row * 20, cell, 13, stroke=LIGHT, fill="white")

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text("\n".join(self.parts + ['</g>', '</svg>']) + "\n")


def desktop():
    d = Drawing(620, 260)
    d.text(128, 23, "Generated prefix", 15)
    d.text(440, 23, "Current frame", 15)
    d.frames(29, 43)
    d.path("M128 79V122", GRAY, arrow=True)
    d.text(143, 105, "encode (no grad)", 13, GRAY, anchor="start")
    d.cache(64, 132, 132)
    d.text(130, 191, "detached KV prefix", 13, GRAY)

    d.text(440, 66, "noisy xⁱ at tₛ", 16)
    d.path("M440 77V123", INK, arrow=True)
    d.text(440, 155, "Gθ", 24, ACCENT)
    d.text(490, 154, "gradients on", 13, ACCENT, anchor="start")
    d.text(490, 177, "selected call", 13, GRAY, anchor="start")

    d.path("M202 147H402", INK, arrow=True)
    d.text(304, 132, "conditions frame i", 13, GRAY)
    d.path("M403 168H239", ACCENT, arrow=True, dashed=True)
    d.path("M232 161V176", ACCENT)
    d.text(313, 192, "no gradient into KV", 13, ACCENT)

    d.path("M440 188V209", INK, arrow=True)
    d.rect(419, 216, 42, 30, stroke=GRAY)
    d.text(440, 237, "x̂ⁱ₀", 17)
    d.text(473, 236, "generated output", 13, GRAY, anchor="start")
    d.text(18, 232, "Earlier outputs keep their own loss paths.", 13, GRAY, anchor="start")
    d.save("prefix-drift.svg")


def mobile():
    d = Drawing(360, 305)
    d.text(88, 22, "Generated prefix", 15)
    d.text(270, 22, "Current frame", 15)
    d.frames(13, 43, cell=30, gap=10)
    d.path("M88 80V122", GRAY, arrow=True)
    d.text(100, 101, "encode", 13, GRAY, anchor="start")
    d.text(100, 116, "no grad", 13, GRAY, anchor="start")
    d.cache(38, 132, 102)
    d.text(88, 190, "stop-gradient KV", 13, ACCENT)

    d.text(270, 65, "noisy xⁱ at tₛ", 14)
    d.path("M270 77V124", INK, arrow=True)
    d.text(270, 156, "Gθ", 24, ACCENT)
    d.text(270, 181, "gradients on", 13, ACCENT)
    d.text(270, 199, "selected call", 13, GRAY)

    d.path("M146 147H232", INK, arrow=True)
    d.text(188, 133, "condition", 13, GRAY)
    d.path("M232 167H160", ACCENT, arrow=True, dashed=True)
    d.path("M153 160V174", ACCENT)

    d.path("M270 208V239", INK, arrow=True)
    d.rect(249, 246, 42, 30, stroke=GRAY)
    d.text(270, 267, "x̂ⁱ₀", 17)
    d.text(270, 297, "generated output", 13, GRAY)
    d.text(14, 260, "Earlier outputs keep", 13, GRAY, anchor="start")
    d.text(14, 278, "their own loss paths.", 13, GRAY, anchor="start")
    d.save("prefix-drift-mobile.svg")


if __name__ == "__main__":
    desktop()
    mobile()
    print("Wrote Self Forcing mechanism SVGs (620 × 260; 360 × 305).")
