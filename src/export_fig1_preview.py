"""One-off: render the TikZ design figure (page 3, top region) from the built PDF
to results/arxiv/fig1_design_preview.png for viewing. The figure itself lives in
arxiv/main.tex (TikZ source), so no image file is uploaded for it."""
import os
import sys
from pathlib import Path

import pymupdf

pdf = Path(os.environ["TEMP"]) / "arxiv_build" / "main.pdf"
out = Path(__file__).resolve().parent.parent / "results" / "arxiv" / "fig1_design_preview.png"
doc = pymupdf.open(pdf)
page_no = int(sys.argv[1]) if len(sys.argv) > 1 else 3
page = doc[page_no - 1]
r = page.rect
clip = pymupdf.Rect(r.x0, r.y0 + 0.05 * r.height, r.x1, r.y0 + 0.30 * r.height)
page.get_pixmap(dpi=220, clip=clip).save(out)
print(out)
