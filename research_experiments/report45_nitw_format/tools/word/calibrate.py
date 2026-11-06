"""Calibrate the Word layout against the calibrated LaTeX PDF: for every anchor, the gap to the line above (or the
absolute baseline) in the LibreOffice render of the docx is compared with the same gap in the LaTeX PDF, and the
corresponding cal.json spacing is corrected.  python3 -I calibrate.py [iterations]
"""
import json
import re
import subprocess
import sys

import pymupdf

W = "/tmp/claude-0/r45docx"
SK = "/root/.claude/skills/synced/3d8b842e-3a17-4394-9600-13c33974a57f_79e7ea49-56bd-4110-a41e-74be3384494f/docx"
LATEX = pymupdf.open("/tmp/claude-0/r45tex/main.pdf")


def build():
    subprocess.run(f"cd {W} && node build.js >/dev/null && python3 post.py report45_raw.docx report45.docx >/dev/null && rm -rf render && mkdir render && "
                   f"cd render && timeout 300 python3 {SK}/scripts/office/soffice.py --headless --convert-to pdf ../report45.docx >/dev/null 2>&1",
                   shell=True, check=True)
    return pymupdf.open(f"{W}/render/report45.pdf")


def lines(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] == 1:
            out.append((b["bbox"][1], b["bbox"][0], 0, "<IMG>", b["bbox"][3]))
            continue
        for l in b.get("lines", []):
            sp = [s for s in l["spans"] if s["text"].strip()]
            if sp:
                out.append((sp[0]["origin"][1], l["bbox"][0], sp[0]["size"], "".join(s["text"] for s in l["spans"]).strip(), None))
    out.sort()
    merged = []
    for r in out:                                   # one entry per baseline
        if merged and abs(r[0] - merged[-1][0]) < 1.5 and r[3] != "<IMG>" and merged[-1][3] != "<IMG>":
            merged[-1] = (merged[-1][0], min(merged[-1][1], r[1]), max(merged[-1][2], r[2]), merged[-1][3] + " " + r[3], None)
        else:
            merged.append(r)
    return merged


def rules(page):
    return sorted(d["rect"].y0 for d in page.get_drawings() if d["rect"].height < 2 and d["rect"].width > 60)


def find_page(doc, key, last=False):
    hits = [i for i, p in enumerate(doc) if key in p.get_text().replace("\n", " ")]
    return hits[-1] if last else hits[0]


def measure(doc, spec):
    page_key, last, anchor, kind = spec
    pg = doc[find_page(doc, page_key, last)]
    L = lines(pg)
    if anchor == "RULE":
        r = [y for y in rules(pg) if y > 300]
        y = r[0]
        above = [l[0] for l in L if l[0] < y]
        return y - above[-1]
    if anchor == "IMGTOP":
        im = [l for l in L if l[3] == "<IMG>"][0]
        above = [l[0] for l in L if l[0] < im[0] and l[3] != "<IMG>"]
        return im[0] - above[-1]
    for k, l in enumerate(L):
        if l[3].startswith(anchor):
            if kind == "abs":
                return l[0]
            prev = L[k - 1]
            return l[0] - (prev[4] if prev[3] == "<IMG>" else prev[0])
    raise KeyError(anchor)


# key: (page key, last occurrence?, anchor at line start, "abs" | "gap")
SPEC = {
    "tp_title": ("Phase I Project Progress Report submitted", False, "FOUR-CLASS", "abs"),
    "tp_A": ("Phase I Project Progress Report submitted", False, "Phase I Project", "gap"),
    "tp_B": ("Phase I Project Progress Report submitted", False, "of", "gap"),
    "tp_C": ("Phase I Project Progress Report submitted", False, "Bachelor of Technology", "gap"),
    "tp_E": ("Phase I Project Progress Report submitted", False, "by", "gap"),
    "tp_F": ("Phase I Project Progress Report submitted", False, "Sai Chaithr", "gap"),
    "tp_H": ("Phase I Project Progress Report submitted", False, "Under the guidance", "gap"),
    "tp_I": ("Phase I Project Progress Report submitted", False, "[Guide", "gap"),
    "tp_K": ("Phase I Project Progress Report submitted", False, "IMGTOP", "gap"),
    "tp_bottom": ("Phase I Project Progress Report submitted", False, "DEPARTMENT OF", "gap"),
    "ap_head": ("APPROVAL OF THE PROJECT", False, "APPROVAL", "abs"),
    "ap_date": ("APPROVAL OF THE PROJECT", False, "DD/MM", "gap"),
    "ap_text": ("APPROVAL OF THE PROJECT", False, "Certified that", "gap"),
    "ap_B": ("APPROVAL OF THE PROJECT", False, "(Member of the Board)", "gap"),
    "ap_C": ("APPROVAL OF THE PROJECT", False, "(Supervisor)", "gap"),
    "ap_D": ("APPROVAL OF THE PROJECT", False, "(Head of the Department)", "gap"),
    "ce_note": ("Departmental Letter-Head", False, "(The certificate", "abs"),
    "ce_head": ("Departmental Letter-Head", False, "CERTIFICATE", "gap"),
    "ce_text": ("Departmental Letter-Head", False, "This is to certify", "gap"),
    "ce_sig": ("Departmental Letter-Head", False, "RULE", "gap"),
    "ce_date": ("Departmental Letter-Head", False, "Date:", "gap"),
    "de_head": ("We certify that", False, "DECLARATION", "abs"),
    "de_we": ("We certify that", False, "We certify that", "gap"),
    "de_first": ("We certify that", False, "a.", "gap"),
    "de_item": ("We certify that", False, "c.", "gap"),
    "de_sig": ("We certify that", False, "RULE", "gap"),
    "de_cap": ("We certify that", False, "Signatures of the Students", "gap"),
    "frontBefore": ("We thank our supervisor", False, "Acknowledgements", "abs"),
    "frontAfter": ("We thank our supervisor", False, "We thank our supervisor", "gap"),
    "starBefore": ("Certificate of Approval", True, "Contents", "abs"),
    "starAfter": ("Certificate of Approval", True, "Certificate of Approval", "gap"),
    "chapBefore": ("Background and motivation", True, "Chapter 1", "abs"),
    "chapGap": ("Background and motivation", True, "Introduction", "gap"),
    "secAfterTitle": ("Background and motivation", True, "1.1", "gap"),
    "textAfterSec": ("Background and motivation", True, "Heart failure affects", "gap"),
    "headerDist": ("Chapter 1. Introduction", False, "Chapter 1. Introduction", "abs"),
}


def footer_baseline(doc, latex):
    pg = doc[find_page(doc, "Background and motivation", True)] if not latex else LATEX[find_page(LATEX, "Background and motivation", True)]
    return max(l[0] for l in lines(pg) if l[0] > 760)


iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3
target = {k: measure(LATEX, v) for k, v in SPEC.items()}
target["footerDist"] = footer_baseline(None, True)
for it in range(iters):
    doc = build()
    cal = json.load(open(f"{W}/cal.json"))
    worst = 0
    for k, v in SPEC.items():
        try:
            got = measure(doc, v)
        except Exception as e:
            print("  ", k, "not measurable:", e); continue
        err = got - target[k]
        worst = max(worst, abs(err))
        if k == "headerDist":
            cal[k] = round(cal[k] - err, 2)
        elif k in ("tp_title", "tp_bottom", "ap_head", "ce_note", "de_head", "frontBefore", "starBefore", "chapBefore"):
            cal[k] = round(cal[k] - err, 2)
        else:
            cal[k] = round(cal[k] - err, 2)
        if abs(err) > 0.3:
            print(f"  {k:14s} got {got:7.2f} target {target[k]:7.2f} err {err:+.2f} -> {cal[k]}")
    fb = footer_baseline(doc, False)
    err = fb - target["footerDist"]
    cal["footerDist"] = round(cal["footerDist"] + err, 2)     # footer distance is measured from the page bottom
    print(f"  footer baseline got {fb:.2f} target {target['footerDist']:.2f}")
    json.dump(cal, open(f"{W}/cal.json", "w"), indent=1)
    print("iteration", it, "worst", round(worst, 2), "pages", len(doc))
    if worst < 0.3 and abs(err) < 0.3:
        break
