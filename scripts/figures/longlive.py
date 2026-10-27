"""Draw LongLive's prompt-conditioned video-cache refresh with stdlib only.

Run from any directory: python scripts/figures/longlive.py
Both branches encode the same fixed visual prefix; the text condition differs.
"""

from html import escape
from pathlib import Path


INK = "#282828"
GRAY = "#777777"
LIGHT = "#d8d8d8"
ACCENT = "#526b78"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "public/images/blog/longlive-kv-recache"


class Drawing:
    def __init__(self, width, height):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            '<title id="title">Re-encode the same visual prefix under the new prompt</title>',
            '<desc id="desc">The same generated video prefix enters two encodings. Old-prompt cross-attention produces stale downstream video keys and values. New-prompt cross-attention requires recomputing those downstream keys and values before continuing. The visual prefix itself remains unchanged.</desc>',
            '<rect width="100%" height="100%" fill="white"/>',
            '<defs>',
        ]
        for name, color in [("ink", INK), ("gray", GRAY), ("accent", ACCENT)]:
            self.parts.append(f'<marker id="arrow-{name}" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1 1L5 3L1 5" fill="none" stroke="{color}" stroke-width="1"/></marker>')
        self.parts.extend(['</defs>', '<g font-family="Lato, Arial, sans-serif" fill="#282828" stroke-linecap="round" stroke-linejoin="round">'])

    def text(self, x, y, value, size=14, color=INK, anchor="middle"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')

    def path(self, d, color=INK, arrow=False):
        name = "accent" if color == ACCENT else "gray" if color == GRAY else "ink"
        marker = f' marker-end="url(#arrow-{name})"' if arrow else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.15"{marker}/>')

    def rect(self, x, y, width, height, stroke=LIGHT, fill="white"):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" stroke="{stroke}" fill="{fill}" stroke-width="1"/>')

    def prefix(self, center, y, cell=32, gap=10):
        start = center - (3 * cell + 2 * gap) / 2
        for i, label in enumerate(["z¹", "…", "zᵐ"]):
            x = start + i * (cell + gap)
            if label != "…":
                self.rect(x, y, cell, 25, stroke=GRAY)
            self.text(x + cell / 2, y + 18, label, 15)

    def tensors(self, center, y, color=GRAY):
        # Two small matrices schematically depict keys and values, not dimensions.
        for offset in [-36, 7]:
            for row in range(3):
                for col in range(3):
                    self.rect(center + offset + col * 9, y + row * 7, 6, 4, stroke=color, fill="white")

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text("\n".join(self.parts + ['</g>', '</svg>']) + "\n")


def desktop():
    d = Drawing(620, 270)
    d.text(310, 22, "Same generated visual prefix z₁:ₘ", 15)
    d.prefix(310, 34)
    d.path("M310 64V76H166V93", INK, arrow=True)
    d.path("M310 76H466V93", INK, arrow=True)

    d.text(18, 112, "old prompt", 13, GRAY, anchor="start")
    d.path("M85 107H96", GRAY, arrow=True)
    d.text(166, 112, "text cross-attention", 14)
    d.text(317, 112, "new prompt", 13, ACCENT, anchor="start")
    d.path("M390 107H400", ACCENT, arrow=True)
    d.text(466, 112, "text cross-attention", 14)

    for center, color in [(166, GRAY), (466, ACCENT)]:
        d.path(f"M{center} 123V143", color, arrow=True)
        d.text(center, 160, "later video self-attn", 14)
        d.path(f"M{center} 171V191", color, arrow=True)
    d.text(166, 208, "K, V (old)", 16, GRAY)
    d.text(466, 208, "K, V (new)", 16, ACCENT)
    d.tensors(166, 218, GRAY)
    d.tensors(466, 218, ACCENT)
    d.text(166, 261, "stale after prompt switch", 13, GRAY)
    d.text(466, 261, "recompute and use", 13, ACCENT)
    d.save("conditional-cache.svg")


def mobile():
    d = Drawing(360, 310)
    d.text(180, 22, "Same visual prefix z₁:ₘ", 15)
    d.prefix(180, 35, cell=30, gap=8)
    # Prefix entries approach from the inner sides; prompts enter from above.
    d.path("M180 66V75H149V132H139", INK, arrow=True)
    d.path("M180 75H211V132H221", INK, arrow=True)
    d.text(88, 96, "old prompt", 14, GRAY)
    d.text(272, 96, "new prompt", 14, ACCENT)
    d.path("M88 104V114", GRAY, arrow=True)
    d.path("M272 104V114", ACCENT, arrow=True)
    for center, color in [(88, GRAY), (272, ACCENT)]:
        d.text(center, 137, "text cross-attn", 14)
        d.path(f"M{center} 148V166", color, arrow=True)
        d.text(center, 184, "later self-attn", 14)
        d.path(f"M{center} 195V214", color, arrow=True)
    d.text(88, 237, "K, V (old)", 15, GRAY)
    d.text(272, 237, "K, V (new)", 15, ACCENT)
    d.tensors(88, 249, GRAY)
    d.tensors(272, 249, ACCENT)
    d.text(88, 296, "stale at switch", 13, GRAY)
    d.text(272, 296, "recompute and use", 13, ACCENT)
    d.save("conditional-cache-mobile.svg")


if __name__ == "__main__":
    desktop()
    mobile()
    print("Wrote LongLive mechanism SVGs (620 × 270; 360 × 310).")
