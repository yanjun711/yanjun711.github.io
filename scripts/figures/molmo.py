"""Draw MolmoAct2's exact four-cell prefix example using stdlib only.

Run: python scripts/figures/molmo.py
Values are pedagogical code IDs, not metric depths or a model evaluation.
Cached IDs are replayed as inputs under the current context; old KV states
are not copied into the new observation's prefix.
"""
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
INK, GRAY, LINE, ACCENT = '#282828', '#777', '#d8d8d8', '#526b78'
OLD, MASK, NEW = [12,33,7,80], [0,1,0,1], [12,35,7,81]
assert all(NEW[i] == OLD[i] for i in range(4) if MASK[i] == 0)


def render(mobile=False):
    width,height=(360,312) if mobile else (620,270)
    start,cell=(98,59) if mobile else (150,108)
    positions=[start+(i+.5)*cell for i in range(4)]
    head_y, old_y, mask_y, prefix_y = (68,100,134,186) if mobile else (62,96,128,176)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">Replay and predict within the current depth prefix</title>',
           '<desc id="desc">In causal raster order, cached codes 12, 33, 7, 80 and update mask 0, 1, 0, 1 produce current prefix 12, 35, 7, 81. Codes 12 and 7 are replayed as inputs in the current context. Codes 35 and 81 are predicted conditioned on the preceding filled prefix. All four codes are consumed; two are newly predicted.</desc>',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           '<g font-family="Lato,Arial,sans-serif">']

    def text(x,y,value,size=14,color=INK,anchor='start',weight='normal'):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>')

    def line(x1,y1,x2,y2):
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" stroke-width="1"/>')

    text(16 if mobile else 20,22,'Fill the current depth prefix',size=14,weight='bold')
    text(16 if mobile else 20,43,'Causal raster order: read from left to right',size=13,color=GRAY)
    text(16 if mobile else 20,head_y,'Cell',size=13,color=GRAY)
    for i,x in enumerate(positions):
        text(x,head_y,i+1,size=13,color=GRAY,anchor='middle')
    text(16 if mobile else 20,old_y,'Cached' if mobile else 'Cached codes',size=14)
    text(16 if mobile else 20,mask_y,'Mask' if mobile else 'Update mask',size=14)
    text(16 if mobile else 20,prefix_y,'Prefix' if mobile else 'Current prefix',size=14,weight='bold')
    for i,x in enumerate(positions):
        text(x,old_y,OLD[i],size=17,color=GRAY,anchor='middle')
        text(x,mask_y,MASK[i],size=14,color=ACCENT if MASK[i] else GRAY,anchor='middle')
        fill,opacity=(ACCENT,'.15') if MASK[i] else (LINE,'.55')
        parts.append(f'<rect x="{start+i*cell+3}" y="{prefix_y-24}" width="{cell-6}" height="36" fill="{fill}" fill-opacity="{opacity}"/>')
        text(x,prefix_y,NEW[i],size=18,color=ACCENT if MASK[i] else INK,anchor='middle')
        text(x,prefix_y+37,'predict' if MASK[i] else 'replay',size=13,color=ACCENT if MASK[i] else GRAY,anchor='middle')
    line(start,prefix_y+14,start+4*cell,prefix_y+14)
    if mobile:
        text(16,251,'All 4 codes consumed; only 2 newly predicted.',size=13)
        text(16,275,'Replay uses code IDs in the current context.',size=13,color=GRAY)
        text(16,299,'Prediction uses the preceding filled codes.',size=13,color=GRAY)
    else:
        text(20,235,'All 4 codes are consumed; only 2 require a new prediction.',size=13)
        text(20,254,'Replay processes cached code IDs under the current observation and prefix.',size=13,color=GRAY)
    parts.extend(['</g>','</svg>'])
    target=ROOT/'public/images/blog/molmoact2-selective-depth'/('replay-mobile.svg' if mobile else 'replay.svg')
    target.write_text('\n'.join(parts)+'\n')


if __name__ == '__main__':
    render()
    render(mobile=True)
