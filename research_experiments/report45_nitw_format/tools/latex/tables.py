"""Rebuild the 17 tables of report_45.pdf as booktabs LaTeX in the template style (small type, bold header).
Cell text comes from the PDF spans (no retyping); rows are recovered from line gaps (wrapped lines are ~2 pt closer
than rows); rules from the PDF's horizontal lines.  python3 -I tables.py <report45.pdf> <outdir>
"""
import json
import math
import re
import statistics
import sys

import pymupdf

sys.path.insert(0, sys.argv[2] + "/tools")
import convert as C  # noqa: E402  (re-uses the span -> LaTeX conversion and post-processing)

SRC, OUT = sys.argv[1], sys.argv[2]
doc = pymupdf.open(SRC)
caps = json.load(open(OUT + "/tools/captions.json"))
regions = json.load(open(OUT + "/tools/regions.json"))


def span_text(s):
    """Rebuild the span text from its characters, inserting spaces where the PDF only leaves a positional gap."""
    out, prev = [], None
    for ch in s["chars"]:
        c = ch["c"]
        if prev is not None and c != " " and prev["c"] != " " and ch["bbox"][0] - prev["bbox"][2] > 0.18 * s["size"]:
            out.append(" ")
        out.append(c)
        prev = ch
    return "".join(out)


def spans_in(page, bbox):
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                s = dict(s)
                s["text"] = span_text(s)
                if not s["text"].strip():
                    continue
                x0, y0, x1, y1 = s["bbox"]
                if x0 >= bbox[0] - 1 and x1 <= bbox[2] + 1 and y0 >= bbox[1] - 1 and y1 <= bbox[3] + 1:
                    out.append(s)
    return out


class ManualTab:
    """A table given by its bounding box; column starts are taken from the header line's span positions."""
    def __init__(self, page, bbox):
        self.bbox = bbox
        sp = sorted(spans_in(page, bbox), key=lambda s: (round(s["origin"][1]), s["origin"][0]))
        y_head = sp[0]["origin"][1]
        xs = sorted(s["bbox"][0] for s in sp if abs(s["origin"][1] - y_head) < 2.6)
        starts = []
        for x in xs:
            if not starts or x - starts[-1] > 12:
                starts.append(x)
        self.edges = starts
        self.col_count = len(starts)


def build(page, tab):
    bbox = tab.bbox
    if isinstance(tab, ManualTab):
        ncol, edges = tab.col_count, list(tab.edges)
    else:
        # column left edges from the detected cells
        ncol = tab.col_count
        lefts = [[] for _ in range(ncol)]
        for row in tab.rows:
            for j, c in enumerate(row.cells):
                if c is not None:
                    lefts[j].append(c[0])
        edges = [statistics.median(v) if v else None for v in lefts]
        for j in range(ncol):
            if edges[j] is None:
                edges[j] = edges[j - 1] + 20
        # the detector's box starts at the header's top edge (sometimes a fraction of a point too low):
        # extend it up to the booktabs top rule so the header row is never cut
        bbox = list(bbox)
        hrules = sorted(d["rect"].y0 for d in page.get_drawings()
                        if d["rect"].height < 1.6 and d["rect"].width > 0.5 * (bbox[2] - bbox[0]))
        above = [y for y in hrules if bbox[1] - 15 <= y <= bbox[1] + 1]
        if above:
            bbox[1] = min(above) + 0.3
        below = [y for y in hrules if bbox[3] - 1 <= y <= bbox[3] + 25]       # ... and down to the rule under it
        if below:
            bbox[3] = min(below) - 0.3
        # columns must agree with the header line just above the mid rule: drop edges no header word starts in,
        # split a column holding two separate header words
        mid = next((y for y in hrules if y > bbox[1] + 3), None)
        hsp = [s for s in spans_in(page, bbox) if mid is not None and s["origin"][1] < mid]
        if hsp:
            ylast = max(s["origin"][1] for s in hsp)
            hsp = sorted((s for s in hsp if abs(s["origin"][1] - ylast) < 2.6), key=lambda s: s["bbox"][0])
            groups = []
            for s in hsp:
                if groups and s["bbox"][0] - groups[-1][1] < 6:
                    groups[-1][1] = s["bbox"][2]
                else:
                    groups.append([s["bbox"][0], s["bbox"][2]])
            starts = [g[0] for g in groups]
            new = []
            for j, e in enumerate(edges):
                nxt = edges[j + 1] if j + 1 < len(edges) else bbox[2]
                inside = [x for x in starts if e - 4 <= x < nxt - 4]
                if j == 0 or inside:
                    new.append(e)
                    new += [x - 4 for x in inside[1:]]
            edges, ncol = new, len(new)
    rights = edges[1:] + [bbox[2]]
    spans = spans_in(page, bbox)
    # lines by baseline
    spans.sort(key=lambda s: (round(s["origin"][1], 1), s["origin"][0]))
    lines = []
    for s in spans:
        y = s["origin"][1]
        if lines and abs(y - lines[-1]["y"]) < 2.6:
            lines[-1]["spans"].append(s)
        else:
            lines.append(dict(y=y, spans=[s]))
    body_size = statistics.median(s["size"] for s in spans)
    # per line: cells
    for ln in lines:
        cells = [[] for _ in range(ncol)]
        for s in sorted(ln["spans"], key=lambda s: s["origin"][0]):
            x = s["bbox"][0]
            j = max([k for k in range(ncol) if edges[k] <= x + 2] or [0])
            if cells[j]:
                p = cells[j][-1]
                if s["bbox"][0] - p["bbox"][2] > 0.18 * s["size"] and not p["text"].endswith(" ") and not s["text"].startswith(" "):
                    p["text"] = p["text"] + " "
            cells[j].append(s)
        ln["cells"] = cells
    # rules (booktabs lines) inside the table
    rules = sorted(set(round(d["rect"].y0, 1) for d in page.get_drawings()
                       if d["rect"].height < 1.6 and d["rect"].width > 0.5 * (bbox[2] - bbox[0])
                       and bbox[1] - 3 <= d["rect"].y0 <= bbox[3] + 3))
    # row grouping by gap
    gaps = [b["y"] - a["y"] for a, b in zip(lines, lines[1:])
            if not any(a["y"] < r < b["y"] for r in rules)]
    small = min(gaps) if gaps else 0
    usual = sorted(g for g in gaps if g < small + 4)              # ignore section breaks
    two_levels = bool(usual) and (usual[-1] - usual[0]) > 1.2       # wrapped lines vs rows
    thr = (usual[0] + usual[-1]) / 2 if two_levels else -1
    rows = []
    for k, ln in enumerate(lines):
        rule_between = k > 0 and any(lines[k - 1]["y"] < r < ln["y"] for r in rules)
        cont = (not two_levels) and k > 0 and not rule_between and not ln["cells"][0]   # uniform gaps: empty first cell = wrapped line
        if k == 0 or rule_between or ((ln["y"] - lines[k - 1]["y"]) > thr and not cont):
            rows.append(dict(lines=[ln], rule_above=rule_between))
        else:
            rows[-1]["lines"].append(ln)
    # cell text
    out_rows = []
    for r in rows:
        cells = []
        for j in range(ncol):
            parts = [C.line_tex(ln["cells"][j], body_size).strip() for ln in r["lines"] if ln["cells"][j]]
            cells.append(C.postproc(C.join_lines(parts)))
        out_rows.append(dict(cells=cells, rule_above=r["rule_above"]))
    widths = [rights[j] - edges[j] for j in range(ncol)]
    return out_rows, widths, rules


# ---- column widths: start from the original proportions, but never narrower than the longest unbreakable word
#      (measured with the Latin Modern metrics the report is set in, \small = 10.95 pt)
LINEWIDTH, TABCOLSEP, SMALL = 433.27, 6.0, 10.95
LM = {False: pymupdf.Font(fontfile="/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"),
      True: pymupdf.Font(fontfile="/usr/share/texmf/fonts/opentype/public/lm/lmroman10-bold.otf")}


def plain(tex):
    """Rough printed form of a cell's LaTeX, for measuring."""
    t = re.sub(r"\\cite\{[^}]*\}", "[00]", tex)
    t = re.sub(r"\\(textbf|emph|textit|mathrm|text|bm)\{", "", t)
    t = t.replace(r"\%", "%").replace("---", "—").replace("--", "–").replace("~", " ")
    t = re.sub(r"\\(le|ge|times|pm|approx|rightarrow|sigma|Delta|kappa|circ)\b", "x", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    return re.sub(r"[{}$^_\\]", "", t)


def text_w(tex, bold=False):
    return LM[bold].text_length(plain(tex), fontsize=SMALL) * 1.03


def min_w(tex, bold=False, header=False):
    """Widest piece TeX cannot break: words split after hyphens/slashes; long plain words may hyphenate."""
    best = 0.0
    for tok in plain(tex).split():
        for part in re.split(r"(?<=[-/–])", tok):
            w = LM[bold].text_length(part, fontsize=SMALL) * 1.03
            if re.fullmatch(r"[a-z]{9,}[.,;:)]?", part) or (header and re.fullmatch(r"[A-Z][a-z]{7,}[.,;:)]?", part)):
                w *= 0.62          # can be hyphenated (template: "Parame-ters"); body names like "Poterucha" cannot
            best = max(best, w)
    return best


def fit_fracs(widths, rows, nhead):
    n = len(widths)
    need = [0.0] * n
    for i, r in enumerate(rows):
        for j, c in enumerate(r["cells"]):
            if c and not (i >= nhead and j == 0 and r.get("span", 1) > 1):
                need[j] = max(need[j], min_w(c, bold=i < nhead or c.startswith(r"\textbf"), header=i < nhead) + 1.0)
    content = [w / sum(widths) * LINEWIDTH - 2 * TABCOLSEP for w in widths]
    fixed = set()
    for _ in range(n):
        short = [j for j in range(n) if j not in fixed and content[j] < need[j]]
        if not short:
            break
        fixed |= set(short)
        for j in short:
            content[j] = need[j]
        free = [j for j in range(n) if j not in fixed]
        room = LINEWIDTH - 2 * TABCOLSEP * n - sum(content[j] for j in fixed)
        base = sum(widths[j] for j in free)
        if not free or room <= 0:
            break
        for j in free:
            content[j] = room * widths[j] / base
    # then move width to wherever it removes the most wrapped lines (Latin Modern is wider than the original
    # Times, so the original proportions over-wrap the long columns); line counts simulate word wrapping
    space = LM[False].text_length(" ", fontsize=SMALL)
    toks = []
    for i, r in enumerate(rows):
        row = []
        for j, c in enumerate(r["cells"]):
            if not c or (i >= nhead and j == 0 and r.get("span", 1) > 1):
                row.append(None)
                continue
            bold = i < nhead or c.startswith(r"\textbf")
            row.append([LM[bold].text_length(t, fontsize=SMALL) * 1.03 for t in plain(c).split()])
        toks.append(row)
    cache = {}

    def cell_lines(i, j, width):
        key = (i, j, round(width, 1))
        if key not in cache:
            n_l, x = 1, 0.0
            for w in toks[i][j]:
                if x == 0.0:
                    x = w
                elif x + space + w <= width:
                    x += space + w
                else:
                    n_l, x = n_l + 1, w
            cache[key] = n_l
        return cache[key]

    def nlines(cont):
        return sum(max([cell_lines(i, j, cont[j]) for j in range(n) if toks[i][j]] or [1]) for i in range(len(toks)))

    # global start: for a target of T lines per row, give every column the narrowest width that fits all its cells
    # in T lines (header rows: max(T, 3)); keep the best feasible T, spreading spare width like the original
    avail = LINEWIDTH - 2 * TABCOLSEP * n

    def width_for(j, T):
        lo, hi = need[j], avail
        if all(cell_lines(i, j, lo) <= (max(T, 3) if i < nhead else T) for i in range(len(toks)) if toks[i][j]):
            return lo
        for _ in range(30):
            mid = (lo + hi) / 2
            if all(cell_lines(i, j, mid) <= (max(T, 3) if i < nhead else T) for i in range(len(toks)) if toks[i][j]):
                hi = mid
            else:
                lo = mid
        return hi + 0.5

    cands = [content]
    for T in range(1, 13):
        base_w = [width_for(j, T) for j in range(n)]
        spare = avail - sum(base_w)
        if spare < 0:
            continue
        cands.append([base_w[j] + spare * widths[j] / sum(widths) for j in range(n)])
    content = min(cands, key=nlines)
    step = 0.01 * LINEWIDTH
    cur = nlines(content)
    for _ in range(200):
        best = None
        for a in range(n):
            for b in range(n):
                if b == a:
                    continue
                for k in range(1, 21):
                    if content[a] - k * step < need[a]:
                        break
                    trial = list(content)
                    trial[a] -= k * step
                    trial[b] += k * step
                    h = nlines(trial)
                    if h < cur and (best is None or (cur - h) / k > best[0]):
                        best = ((cur - h) / k, h, trial)
        if best is None:
            break
        _, cur, content = best
    tot = sum(content) + 2 * TABCOLSEP * n
    return [(c + 2 * TABCOLSEP) / tot for c in content], cur


def pcol(frac):
    return r">{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr " + f"{math.floor(frac * 1000) / 1000:.3f}" + r"\linewidth-2\tabcolsep\relax}"


def to_latex(num, rows, widths, longtab=False):
    cap = caps.get("T" + num, "")
    # header = rows before the first rule inside the table
    first_rule = next((i for i, r in enumerate(rows) if r["rule_above"]), 1)
    head, body = rows[:first_rule], rows[first_rule:]
    # a body row whose text sits in column 1 with the next cells empty and far too long for it spans those cells
    for r in body:
        c = r["cells"]
        k = 1
        while k < len(c) and not c[k]:
            k += 1
        r["span"] = k if k > 1 and c[0] and text_w(c[0]) > 1.6 * (widths[0] / sum(widths) * LINEWIDTH) else 1
    global SMALL, TABCOLSEP
    SMALL, TABCOLSEP = 10.95, 6.0
    fracs, nl = fit_fracs(widths, rows, len(head))
    size_cmd = r"\small"
    est_h = nl * 13.55 + len(rows) * 1.62 + 30 + (text_w(cap) / LINEWIDTH + 1) * 13.6   # rows, rules, caption (pt)
    if (len(widths) >= 7 and nl > 1.6 * len(rows)) or (est_h > 0.97 * 698 and not longtab):   # very wide, or a float taller than a page
        SMALL, TABCOLSEP = 10.0, 4.0
        fracs, nl2 = fit_fracs(widths, rows, len(head))
        size_cmd = r"\footnotesize\setlength{\tabcolsep}{4pt}"
        COMPACT.append((num, nl, nl2, len(rows)))
    SMALL, TABCOLSEP = 10.95, 6.0
    spec = "".join(pcol(f) for f in fracs)

    def fmt(r, bold=False):
        cells = [(r"\textbf{" + c + "}" if bold and c and not c.startswith(r"\textbf") else c) for c in r["cells"]]
        k = r.get("span", 1)
        if k > 1:
            cells = [r"\multicolumn{" + str(k) + "}{" + pcol(sum(fracs[:k])) + "}{" + cells[0] + "}"] + cells[k:]
            SPANNED.append((num, plain(r["cells"][0])[:50], k))
        return " & ".join(cells) + r" \\"
    hdr = "\n".join(fmt(r, True) for r in head)
    lines = []
    for r in body:
        if r["rule_above"] and lines:
            lines.append(r"\midrule")
        lines.append(fmt(r))
    bodytex = "\n".join(lines)
    label = r"\label{tab:" + num + "}"
    if longtab:
        n = len(widths)
        return ("{" + size_cmd + "\n" + r"\begin{longtable}{" + spec + "}\n"
                + r"\caption{" + cap + "}" + label + r"\\" + "\n" + r"\toprule" + "\n" + hdr + "\n" + r"\midrule" + "\n" + r"\endfirsthead"
                + "\n" + r"\multicolumn{" + str(n) + r"}{l}{\small\emph{Table~\thetable\ (continued)}}\\" + "\n" + r"\toprule" + "\n" + hdr + "\n"
                + r"\midrule" + "\n" + r"\endhead" + "\n" + r"\bottomrule" + "\n" + r"\endlastfoot" + "\n" + bodytex + "\n"
                + r"\end{longtable}" + "\n}\n")
    return (r"\begin{table}[htbp]" + "\n" + r"\caption{" + cap + "}" + label + "\n" + size_cmd + "\n"
            + r"\begin{tabular}{" + spec + "}\n" + r"\toprule" + "\n" + hdr + "\n" + r"\midrule" + "\n" + bodytex + "\n"
            + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n")


# map tables to captions by position: each caption owns the table between it and the next caption on the page
tcaps = regions["table_caps"]
report = []
HAND = {"5.1"}
SPANNED = []
COMPACT = []
bypage = {}
for num, (pno, y0, y1) in tcaps.items():
    bypage.setdefault(pno, []).append((y0, y1, num))
def region_tables(pno, rect):
    tmp = pymupdf.open()
    pg = tmp.new_page(width=rect.width, height=rect.height)
    pg.show_pdf_page(pg.rect, doc, pno, clip=rect)
    return tmp, pg, pg.find_tables().tables


for pno in sorted(bypage):
    cl = sorted(bypage[pno])
    for k, (y0, y1, num) in enumerate(cl):
        bottom = cl[k + 1][0] - 1 if k + 1 < len(cl) else 805
        MANUAL = {"4.7": (80, 140, 545, 431), "4.8": (80, 528, 545, 748),   # page 28: merged by the detector
                  "6.1": (80, 388, 545, 535)}                            # justified cells split into extra columns
        if num in MANUAL:
            rows, widths, rules = build(doc[pno], ManualTab(doc[pno], MANUAL[num]))
        else:
            tmp, pg, tabs = region_tables(pno, pymupdf.Rect(60, y1 + 1, 550, bottom))
            tabs = sorted(tabs, key=lambda t: t.bbox[1])
            if not tabs:
                print("NO TABLE FOUND for", num); continue
            rows, widths, rules = build(pg, tabs[0])
        if num == "2.1":                                        # continues on the next page: append its body rows
            tmp2, pg2, t2 = region_tables(pno + 1, pymupdf.Rect(60, 60, 550, 805))
            r2, _, _ = build(pg2, sorted(t2, key=lambda t: t.bbox[1])[0])
            fr = next((i for i, r in enumerate(r2) if r["rule_above"]), 1)
            rows += r2[fr:]
        if num in HAND:                                         # hand-set (grouped header with partial rules)
            open(f"{OUT}/tables/tab_{num.replace('.', '_')}.tex", "w").write(open(f"{OUT}/tools/hand/tab_{num.replace('.', '_')}.tex").read())
            report.append((num, len(rows), len(widths), "hand", [c[:28] for c in rows[1]["cells"]])); continue
        big = num == "2.1"
        open(f"{OUT}/tables/tab_{num.replace('.', '_')}.tex", "w").write(to_latex(num, rows, widths, longtab=big))
        report.append((num, len(rows), len(widths), "longtable" if big else "float", [c[:28] for c in rows[0]["cells"]]))
print("spanning rows:", SPANNED)
print("footnotesize tables (lines small, lines footnotesize, rows):", COMPACT)
for r in sorted(report, key=lambda r: tuple(int(x) for x in r[0].split("."))):
    print(r)
