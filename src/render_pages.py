"""Render selected pages of a PDF to PNG for visual checks.
usage: python render_pages.py <pdf> <out_prefix> <page> [<page> ...]  (1-based pages)
"""
import sys
import fitz  # pymupdf

pdf, prefix, *pages = sys.argv[1:]
doc = fitz.open(pdf)
for p in pages:
    pix = doc[int(p) - 1].get_pixmap(dpi=80)
    out = f"{prefix}_p{p}.png"
    pix.save(out)
    print(out)
