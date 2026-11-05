"""Pixel-image route (IMG_PLAN.md): draw any 12-lead ECG onto a blank copy of the hospital printout page, and rasterise
hospital PDFs and drawn ECGs identically.

* The blank page: the black non-trace drawings of a hospital PDF (calibration pulses, column ticks), re-drawn on an
  empty page of the same size. The pink grid is omitted because the grid filter removes it. No text, no patient trace.
* Traces: the hospital layout (ecg_ef.signals): rows at y = 707 / 970 / 1233 pt, columns at x = 75 / 588 / 1102 /
  1615 pt, 205.2 pt per second, 82 pt per mV; 10-s lead II rhythm strip at y = 1496. Black, 2 pt wide.
* Raster: the original project's waveform clip (ecg_ef.config.WAVEFORM_CLIP_PT) with ALL text removed, then the
  original grid-removal filter (ecg_ef.preprocess.filter_trace) -> white trace on black, uint8 (H, W).
Label-free. Hospital PDFs are read only for development ECGs (callers pass development paths).
"""
from __future__ import annotations

import numpy as np
import pymupdf

from ecg_ef.config import PAGE_SIZE_PT, WAVEFORM_CLIP_PT
from ecg_ef.preprocess import filter_trace
from ecg_ef.signals import COL_X0, GRID, LEADS, PT_PER_MV, PT_PER_S, ROW_BASE

FS = 500
TRACE_W = 2.0
_TEMPLATE = None


def template_from_pdf(pdf_path):
    """List of non-trace drawings (lines / pulses) of one hospital page: (points, colour, width)."""
    out = []
    with pymupdf.open(str(pdf_path)) as doc:
        for d in doc[0].get_drawings():
            c = d.get("color")
            if not c:
                continue
            if max(c) <= 0.3 and len(d["items"]) > 20:            # a patient trace -> not part of the template
                continue
            if max(c) > 0.3:                                       # pink grid: removed by the grid filter anyway
                continue
            segs = [(it[1], it[2]) for it in d["items"] if it[0] == "l"]
            if segs:
                out.append(([((a.x, a.y), (b.x, b.y)) for a, b in segs], tuple(c), float(d.get("width") or 1.0)))
    return out


def set_template(pdf_path):
    global _TEMPLATE
    _TEMPLATE = template_from_pdf(pdf_path)
    return len(_TEMPLATE)


def _page_with_template():
    doc = pymupdf.open(); page = doc.new_page(width=PAGE_SIZE_PT[0], height=PAGE_SIZE_PT[1])
    sh = page.new_shape()
    for segs, col, w in _TEMPLATE:
        for a, b in segs:
            sh.draw_line(a, b)
        sh.finish(color=col, width=w, closePath=False)
    sh.commit()
    return doc, page


def draw_ecg(leads, rhythm, fs=FS):
    """leads (12, 2.5 s) and rhythm (10 s) in mV at fs -> PDF page with the traces drawn in the hospital layout."""
    assert _TEMPLATE is not None, "call set_template(<a development PDF>) first"
    doc, page = _page_with_template()
    sh = page.new_shape()
    t = np.arange(leads.shape[1]) / fs
    for r in range(3):
        for c in range(4):
            v = np.asarray(leads[LEADS.index(GRID[r][c])], float)
            sh.draw_polyline(list(zip(COL_X0[c] + t * PT_PER_S, ROW_BASE[r] - v * PT_PER_MV)))
    tr = np.arange(len(rhythm)) / fs
    sh.draw_polyline(list(zip(COL_X0[0] + tr * PT_PER_S, ROW_BASE[3] - np.asarray(rhythm, float) * PT_PER_MV)))
    sh.finish(color=(0, 0, 0), width=TRACE_W, closePath=False)
    sh.commit()
    return doc, page


def raster(page, out_hw=(320, 640), px_per_pt=0.6):
    """Waveform clip of a page, text removed, rasterised and grid-filtered -> uint8 (H, W)."""
    for w in page.get_text("words"):
        page.add_redact_annot(pymupdf.Rect(w[:4]), fill=(1, 1, 1))
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(px_per_pt, px_per_pt), clip=pymupdf.Rect(*WAVEFORM_CLIP_PT), alpha=False,
                          colorspace=pymupdf.csRGB)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    return filter_trace(rgb, out_hw)


def hospital_image(pdf_path, out_hw=(320, 640)):
    with pymupdf.open(str(pdf_path)) as doc:
        return raster(doc[0], out_hw)


def drawn_image(leads, rhythm, fs=FS, out_hw=(320, 640), px_per_pt=0.6):
    doc, page = draw_ecg(leads, rhythm, fs)                    # drawn page has no text: no redaction needed
    try:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(px_per_pt, px_per_pt), clip=pymupdf.Rect(*WAVEFORM_CLIP_PT),
                              alpha=False, colorspace=pymupdf.csRGB)
        return filter_trace(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3), out_hw)
    finally:
        doc.close()
