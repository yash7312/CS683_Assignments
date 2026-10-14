#!/usr/bin/env python3
"""Print report/tasks23.html to PDF with a headless Chromium browser (Edge or
Chrome) and prepend the Task 1 report to produce Summary.pdf.

    python scripts/build_pdf.py <task1.pdf> report/tasks23.html Summary.pdf

Needs: pypdf, and msedge / chrome on the PATH or at the usual Windows location.
"""
import os
import shutil
import subprocess
import sys

from pypdf import PdfReader, PdfWriter

task1, html, out = sys.argv[1], sys.argv[2], sys.argv[3]

candidates = [shutil.which("msedge"), shutil.which("chrome"), shutil.which("google-chrome"), shutil.which("chromium"),
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
browser = next(c for c in candidates if c and os.path.exists(c))

html_abs = os.path.abspath(html)
part2 = os.path.splitext(html_abs)[0] + ".pdf"
url = "file:///" + html_abs.replace(os.sep, "/")
subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf-no-header",
                "--print-to-pdf=" + part2, url], check=True, timeout=180)

w = PdfWriter()
for src in (task1, part2):
    for page in PdfReader(src).pages:
        w.add_page(page)
with open(out, "wb") as fh:
    w.write(fh)
print("wrote %s (%d pages)" % (out, len(w.pages)))
