"""One-off: render PDF pages to PNG for visual checks. usage: render_pdf.py file.pdf 3 4"""
import sys
from pathlib import Path

try:
    import fitz  # pymupdf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pymupdf"])
    import fitz

pdf = Path(sys.argv[1])
doc = fitz.open(pdf)
for p in sys.argv[2:]:
    page = doc[int(p) - 1]
    out = pdf.with_name(f"{pdf.stem}_p{p}.png")
    page.get_pixmap(dpi=110).save(out)
    print(out)
