"""Draw the exact MiniMax support example with stdlib only.

Run: python scripts/figures/minimax.py
The indexer scores all visible tokens; this figure shows the main-attention
support after selection, not the index branch's work or attention weights.
"""
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
INK, GRAY, LINE, ACCENT = '#282828', '#777', '#d8d8d8', '#526b78'


def render(mobile=False):
    width, height = (360, 300) if mobile else (620, 260)
    start, cell = (84, 32) if mobile else (132, 56)
    rows = [112, 156] if mobile else [90, 134]
    row_height = 30
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             '<title id="title">Group-specific selected token support</title>',
             '<desc id="desc">Eight visible tokens form four blocks. Group one selects tokens one, two, seven, and eight. Group two selects tokens three, four, seven, and eight. Block four is local and uses one of the two available slots. Heads in each group share this support but have separate attention weights. The indexer scores all visible tokens before selection.</desc>',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             '<g font-family="Lato,Arial,sans-serif">']

    def text(x, y, value, size=14, color=INK, anchor='start', weight='normal'):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(value)}</text>')

    def line(x1, y1, x2, y2):
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" stroke-width="1"/>')

    text(16 if mobile else 20, 22, 'Main-attention support at query token 8', size=14, weight='bold')
    block_y = 64 if mobile else 48
    token_y = 96 if mobile else 74
    text(16 if mobile else 20, block_y, 'Block', size=13, color=GRAY)
    text(16 if mobile else 20, token_y, 'Token', size=13, color=GRAY)
    for b in range(4):
        x = start + (2*b+1)*cell
        text(x, block_y, f'{b+1} · local' if b == 3 else str(b+1), size=13, color=GRAY, anchor='middle')
        line(start+2*b*cell+3, block_y+7, start+(2*b+2)*cell-3, block_y+7)
    for token in range(8):
        text(start+(token+.5)*cell, token_y, str(token+1), size=14, anchor='middle')
    for group, y in enumerate(rows):
        text(16 if mobile else 20, y+20, f'Group {group+1}', size=14)
        selected = {group, 3}
        for block in range(4):
            bx = start+2*block*cell
            if block in selected:
                fill = LINE if block == 3 else ACCENT
                opacity = '0.55' if block == 3 else '0.15'
                parts.append(f'<rect x="{bx}" y="{y}" width="{2*cell}" height="{row_height}" fill="{fill}" fill-opacity="{opacity}"/>')
            for j in range(2):
                cx = bx+(j+.5)*cell
                if block in selected:
                    fill = GRAY if block == 3 else ACCENT
                    parts.append(f'<circle cx="{cx}" cy="{y+15}" r="4" fill="{fill}"/>')
                else:
                    text(cx,y+20,'–',size=14,color=LINE,anchor='middle')
        line(start,y+row_height,start+8*cell,y+row_height)
        for b in range(1,4):
            line(start+2*b*cell,y,start+2*b*cell,y+row_height)
    legend_y = 211 if mobile else 190
    for x,label,color,opacity in [(16 if mobile else 20,'Selected nonlocal',ACCENT,'.15'),(188 if mobile else 220,'Reserved local',LINE,'.55')]:
        parts.append(f'<rect x="{x}" y="{legend_y-10}" width="12" height="12" fill="{color}" fill-opacity="{opacity}"/>')
        text(x+19,legend_y,label,size=13,color=GRAY)
    if mobile:
        text(16,240,'Budget: 1 nonlocal + 1 local = 2 blocks',size=13)
        text(16,263,'Heads in a group share support, not weights.',size=13)
        text(16,286,'Indexer scores all visible tokens first.',size=13,color=GRAY)
    else:
        text(20,220,'Each group uses 2 blocks: one selected nonlocal block and one local block.',size=13)
        text(20,245,'Heads share support, not weights. The indexer scores all visible tokens first.',size=13,color=GRAY)
    parts.extend(['</g>','</svg>'])
    target = ROOT/'public/images/blog/minimax-group-block-selection'/('selection-mobile.svg' if mobile else 'selection.svg')
    target.write_text('\n'.join(parts)+'\n')


if __name__ == '__main__':
    render()
    render(mobile=True)
