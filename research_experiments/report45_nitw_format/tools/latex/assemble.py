"""Crop the figures from report_45.pdf as vector PDFs, extract the references, and assemble the final chapter files
(placeholders -> tables, figures, equations).  python3 -I assemble.py <report45.pdf> <outdir>
"""
import json
import re
import sys

import pymupdf

sys.path.insert(0, sys.argv[2] + "/tools")
SRC, OUT = sys.argv[1], sys.argv[2]
doc = pymupdf.open(SRC)
caps = json.load(open(OUT + "/tools/captions.json"))
regions = json.load(open(OUT + "/tools/regions.json"))
ORIG_TEXTWIDTH = 455.0           # report 45 text width in bp (85 -> 540)

# ---------------------------------------------------------------- figures: vector crops
figwidth = {}
FIG_TRIM = {"6.7": 0.05, "6.8": 0.05}   # 5 % narrower: keeps the last line of Chapter 6 on its page
for num, r in regions["fig_regions"].items():
    if num == "3.1":
        continue                                  # redrawn in TikZ (figures/fig3_1_system.tex)
    rect = pymupdf.Rect(r["rect"])
    out = pymupdf.open()
    pg = out.new_page(width=rect.width, height=rect.height)
    pg.show_pdf_page(pg.rect, doc, r["page"], clip=rect)
    out.save(f"{OUT}/figures/fig_{num.replace('.', '_')}.pdf")
    figwidth[num] = min(1.0, rect.width / ORIG_TEXTWIDTH)
    figwidth[num] -= FIG_TRIM.get(num, 0.0)

# ---------------------------------------------------------------- references
import convert as C  # noqa: E402

items, cur = [], None
for pno in range(42, 45):
    page = doc[pno]
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            sp = [s for s in l["spans"] if s["text"] != ""]
            raw = "".join(s["text"] for s in sp).strip()
            if not raw or raw == "References" or (l["bbox"][1] > 790 and re.fullmatch(r"\d+", raw)):
                continue
            m = re.match(r"^\[(\d+)\]", raw)
            if m:
                cur = dict(n=int(m.group(1)), lines=[])
                items.append(cur)
                sp = [dict(s) for s in sp]
                for s in sp:
                    if re.match(r"^\s*\[\d+\]", s["text"]):
                        s["text"] = re.sub(r"^\s*\[\d+\]\s*", "", s["text"])
                        break
            if cur is not None:
                cur["lines"].append(C.line_tex([s for s in sp if s["text"] != ""], 10.9).strip())
bib = [r"\begin{thebibliography}{99}"]
for it in items:
    t = C.join_lines(it["lines"])
    t = re.sub(r"(https?://\S+?)([.,]?)(\s|$)", lambda m: r"\url{" + m.group(1).replace(r"\_", "_").replace(r"\%", "%") + "}" + m.group(2) + m.group(3), t)
    bib.append(r"\bibitem{ref" + str(it["n"]) + "} " + t)
bib.append(r"\end{thebibliography}")
open(f"{OUT}/references.tex", "w").write("\n\n".join(bib) + "\n")

# ---------------------------------------------------------------- chapters
EQ = {"3.1": r"\EqThreeOne", "3.2": r"\EqThreeTwo", "3.3": r"\EqThreeThree", "3.4": r"\EqThreeFour"}
for c in range(1, 8):
    s = open(f"{OUT}/chapters/ch{c}.tex").read()
    s = re.sub(r"\[\[TABLECONT p\d+\]\]\n*", "", s)
    s = re.sub(r"\[\[TABLE (\d+\.\d+)\]\]", lambda m: r"\input{tables/tab_" + m.group(1).replace(".", "_") + "}", s)

    def fig(m):
        num = m.group(1)
        cap = caps.get("F" + num, "")
        if num == "3.1":
            body = r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{\input{figures/fig3_1_system.tex}}"
        else:
            body = r"\includegraphics[width=" + f"{figwidth.get(num, 1.0):.2f}" + r"\linewidth,height=0.80\textheight,keepaspectratio]{figures/fig_" + num.replace(".", "_") + "}"
        return (r"\begin{figure}[htbp]" + "\n" + r"\centering" + "\n" + body + "\n" + r"\caption{" + cap + r"}\label{fig:" + num + "}\n"
                + r"\end{figure}")
    s = re.sub(r"\[\[FIGURE (\d+\.\d+)\]\]", fig, s)
    s = re.sub(r"\n*\[\[EQ (\d+\.\d+)\]\]\n*", lambda m: "\n" + EQ[m.group(1)] + "\n", s)
    open(f"{OUT}/body/ch{c}.tex", "w").write(s)
print("figures cropped:", sorted(figwidth), "| references:", len(items))
