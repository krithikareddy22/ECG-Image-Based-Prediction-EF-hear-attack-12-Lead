"""Convert the body of report_45.pdf (PDF pages 11-42) into LaTeX chapter files for the NITW template.
Text is taken programmatically from the PDF (no retyping): headings, paragraphs, lists, inline bold/italic, inline
maths, super/subscripts, hyphenation, citations -> \\cite, Figure/Table/Chapter/Section -> \\ref.
Tables, figures and display equations become placeholders ([[TABLE x.y]], [[FIGURE x.y]], [[EQ x.y]]) filled later.
    python3 -I convert.py <report45.pdf> <outdir>
"""
import collections
import json
import re
import sys
import unicodedata

import pymupdf

SRC, OUT = sys.argv[1], sys.argv[2]
doc = pymupdf.open(SRC)
BODY = range(10, 42)                       # 0-based PDF pages 11..42 = body pages 1..32
MATHFONTS = ("NewTXMI", "txmiaX", "txsys", "txexs", "NewTXMI7")
EQ_RANGES = {}                             # filled from eq_ranges.json (page -> list of [y0, y1, label])
try:
    EQ_RANGES = {int(k): v for k, v in json.load(open(OUT + "/tools/eq_ranges.json")).items()}
except FileNotFoundError:
    pass


def fname(s):
    return s["font"].split("+")[-1]


# ---------------------------------------------------------------- vocabulary for hyphenation decisions
ALLTEXT = " ".join(doc[p].get_text() for p in range(len(doc)))
VOCAB = collections.Counter(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]+[A-Za-z]", ALLTEXT))


def join_hyphen(left, right):
    """left ends with '-' at a line end; decide whether it is a hyphenation break or a real hyphen."""
    lw = re.search(r"([A-Za-z]+)-$", left)
    rw = re.match(r"([A-Za-z]+)", right)
    if not lw or not rw:
        return left + right
    a, b = lw.group(1), rw.group(1)
    whole, hyph = (a + b).lower(), (a + "-" + b).lower()
    if VOCAB[hyph] > 0 and VOCAB[whole] == 0:
        return left + right                                  # real compound: keep the hyphen
    if VOCAB[whole] > 0 and VOCAB[hyph] == 0:
        return left[:-1] + right                             # word seen unbroken elsewhere: hyphenation break
    if VOCAB[hyph] > 0 and VOCAB[whole] > 0:
        return left + right if VOCAB[hyph] >= VOCAB[whole] else left[:-1] + right
    if b[0].isupper():
        return left + right                                  # e.g. "ECG-\nFM"
    return left[:-1] + right                                 # hyphenation break


# ---------------------------------------------------------------- unicode -> LaTeX
TEXTMAP = {"\u201c": "``", "\u201d": "''", "\u2018": "`", "\u2019": "'", "\u2013": "--", "\u2014": "---",
           "\u2026": r"\ldots{}", "\u00b0": r"\textdegree{}", "\u00b5": r"\textmu{}", "\u00a0": "~", "\u2009": r"\,",
           "\u202f": r"\,", "%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "$": r"\$", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}", "\ufb01": "fi", "\ufb02": "fl",
           "\u2032": "'", "\u00d7": r"$\times$", "\u2212": r"$-$", "\u2264": r"$\le$", "\u2265": r"$\ge$", "\u2248": r"$\approx$",
           "\u00b1": r"$\pm$", "\u2192": r"$\rightarrow$", "\u2190": r"$\leftarrow$", "\u2194": r"$\leftrightarrow$",
           "\u2208": r"$\in$", "\u00b7": r"$\cdot$", "\u223c": r"$\sim$", "\u2260": r"$\neq$", "\u221e": r"$\infty$",
           "\u03c3": r"$\sigma$", "\u03bc": r"$\mu$", "\u03a6": r"$\Phi$", "\u03b1": r"$\alpha$", "\u03b2": r"$\beta$",
           "\u03ba": r"$\kappa$", "\u03c1": r"$\rho$", "\u0394": r"$\Delta$", "\u2206": r"$\Delta$", "\u2211": r"$\sum$",
           "\u2022": r"\textbullet{}", "\u25e6": r"$^\circ$", "\u2010": "-", "\u2011": "-", "\u00e9": r"\'e", "\u00e8": r"\`e", "\u00fc": r'\"u',
           "\u00f6": r'\"o', "\u00e4": r'\"a', "\u00e1": r"\'a", "\u00ed": r"\'i", "\u00f3": r"\'o", "\u00e7": r"\c{c}",
           "\u00df": r"\ss{}", "\u00c9": r"\'E", "\u00d6": r'\"O', "\u2002": " ", "\u2003": " ", "\u2007": "~",
           "\u2215": "/", "\u2044": "/"}
MATHSYM = {"\u2212": "-", "\u2264": r"\le ", "\u2265": r"\ge ", "\u2248": r"\approx ", "\u00d7": r"\times ",
           "\u00b1": r"\pm ", "\u2192": r"\rightarrow ", "\u2208": r"\in ", "\u00b7": r"\cdot ", "\u2211": r"\sum ",
           "\u223c": r"\sim ", "\u2260": r"\neq ", "\u2217": "*", "\u2032": "'", "\u221a": r"\sqrt ", "\u00ac": r"\neg ",
           "\u2223": "|", "\u2225": r"\| ", "\u22c5": r"\cdot ", "\u2026": r"\ldots ", "\u0302": r"\hat ", "\u2016": r"\| ",
           "{": r"\{", "}": r"\}", "%": r"\%", "\u2061": "", "\u2062": "", "\u25e6": r"\circ "}


def char_math(c):
    if c in MATHSYM:
        return MATHSYM[c]
    try:
        n = unicodedata.name(c)
    except ValueError:
        return c
    m = re.match(r"MATHEMATICAL (BOLD |ITALIC |BOLD ITALIC )?(SMALL|CAPITAL) (.+)$", n)
    if m:
        bold = "BOLD" in (m.group(1) or "")
        name = m.group(3)
        if len(name) == 1:
            ch = name.lower() if m.group(2) == "SMALL" else name
        else:
            greek = name.lower().replace("final ", "")
            ch = "\\" + (greek if m.group(2) == "SMALL" else greek.capitalize()) + " "
        return r"\mathbf{" + ch + "}" if bold else ch
    if c in TEXTMAP and TEXTMAP[c].startswith("$"):
        return TEXTMAP[c][1:-1] + " "
    return c


def text_tex(s):
    return "".join(TEXTMAP.get(c, c) for c in s)


def math_tex(s):
    return "".join(char_math(c) for c in s)


# ---------------------------------------------------------------- spans -> LaTeX for one line
def line_tex(spans, body_size):
    base = max(spans, key=lambda s: s["size"])["origin"][1]
    out, mode = [], None                     # mode: None (text) or "math"
    runs = []                                # (kind, style, text): kind text|math|sup|sub
    for s in spans:
        t = s["text"]
        if not t:
            continue
        small = s["size"] < 0.82 * body_size
        oy = s["origin"][1] - base
        f = fname(s)
        if small:
            kind = "sup" if oy < -1.0 else "sub"
            runs.append((kind, f, t))
        elif f.startswith(MATHFONTS) and t.strip(" .,;:") == "":
            runs.append(("text", "", t))
        elif f.startswith(MATHFONTS):
            runs.append(("math", f, t))
        else:
            style = ("b" if s["flags"] & 16 else "") + ("i" if s["flags"] & 2 else "")
            runs.append(("text", style, t))
    # merge: math runs absorb following sup/sub; build output
    res = []
    i = 0
    while i < len(runs):
        kind, st, t = runs[i]
        if kind in ("math", "sup", "sub"):
            m = ""
            while i < len(runs) and (runs[i][0] in ("math", "sup", "sub") or (runs[i][0] == "text" and runs[i][2].strip() == "")):
                k, f, tt = runs[i]
                if k == "math":
                    m += math_tex(tt)
                elif k in ("sup", "sub"):
                    pieces = []
                    while i < len(runs) and runs[i][0] == k:
                        f2, t2 = runs[i][1], runs[i][2]
                        if f2.startswith(MATHFONTS) or any(ord(c) > 0x2000 for c in t2) or re.fullmatch(r"[0-9\-+.,]+", t2):
                            pieces.append(math_tex(t2))
                        else:
                            pieces.append(r"\text{" + text_tex(t2) + "}")
                        i += 1
                    m += ("^{" if k == "sup" else "_{") + "".join(pieces).strip() + "}"
                    continue
                else:
                    m += " "
                i += 1
            lead = " " if m.startswith(" ") else ""
            trail = " " if m.endswith(" ") else ""
            res.append(lead + "$" + m.strip() + "$" + trail)
            continue
        # merge consecutive text runs of the same style
        j = i + 1
        while j < len(runs) and runs[j][0] == "text" and (runs[j][1] == st or runs[j][2].strip() == ""):
            t += runs[j][2]
            j += 1
        lead = t[: len(t) - len(t.lstrip())]
        trail = t[len(t.rstrip()):]
        core = text_tex(t.strip())
        if core and st == "b":
            core = r"\textbf{" + core + "}"
        elif core and st == "i":
            core = r"\emph{" + core + "}"
        elif core and st == "bi":
            core = r"\textbf{\emph{" + core + "}}"
        res.append((" " if lead else "") + core + (" " if trail else ""))
        i = j
    s = "".join(res)
    return s


# ---------------------------------------------------------------- page structure
def page_lines(pno):
    page = doc[pno]
    L = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            sp = [s for s in l["spans"] if s["text"] != ""]
            if not any(s["text"].strip() for s in sp):
                continue
            main = max(sp, key=lambda s: s["size"])
            L.append(dict(y0=l["bbox"][1], y1=l["bbox"][3], x0=l["bbox"][0], x1=l["bbox"][2], size=round(main["size"], 1),
                          font=fname(main), bold=bool(main["flags"] & 16), spans=sp,
                          raw="".join(s["text"] for s in sp)))
    L.sort(key=lambda d: (round(d["y0"]), d["x0"]))
    return L


def in_rect(l, r, pad=1.5):
    return l["x0"] >= r[0] - pad and l["x1"] <= r[2] + pad and l["y0"] >= r[1] - pad and l["y1"] <= r[3] + pad


elements = []        # dicts: type heading/para/item/enum/table/figure/eq, page, y
fig_regions = {}
table_index = []
for pno in BODY:
    page = doc[pno]
    L = page_lines(pno)
    hr = sorted(dr["rect"].y0 for dr in page.get_drawings() if dr["rect"].height < 1.6 and dr["rect"].width > 200)

    def grow(b):
        b = list(b)
        up = [y for y in hr if b[1] - 15 <= y <= b[1] + 1]
        dn = [y for y in hr if b[3] - 1 <= y <= b[3] + 25]
        if up:
            b[1] = min(up) - 0.5
        if dn:
            b[3] = min(dn) + 0.5
        return pymupdf.Rect(b)
    tabs = [grow(t.bbox) for t in page.find_tables().tables]
    # captions
    caps = []
    i = 0
    while i < len(L):
        l = L[i]
        m = re.match(r"^(Table|Figure) (\d+\.\d+)", l["raw"].strip())
        if m and l["bold"] and 10.5 <= l["size"] <= 11.2:
            j, text = i, [l]
            while j + 1 < len(L) and 10.5 <= L[j + 1]["size"] <= 11.2 and L[j + 1]["y0"] - L[j]["y1"] < 6 \
                    and not re.match(r"^(Table|Figure) \d", L[j + 1]["raw"].strip()) and not any(in_rect(L[j + 1], t) for t in tabs):
                j += 1
                text.append(L[j])
            caps.append(dict(kind=m.group(1), num=m.group(2), lines=text, y0=l["y0"], y1=text[-1]["y1"]))
            i = j + 1
        else:
            i += 1
    cap_ids = {id(x) for c in caps for x in c["lines"]}
    # figure regions: above each figure caption, below the last body-font text line
    body_like = [l for l in L if id(l) not in cap_ids and l["font"].startswith(("TeXGyreTermesX",) + MATHFONTS)
                 and l["size"] >= 11.5 and not any(in_rect(l, t) for t in tabs) and l["y1"] < 800]
    for c in caps:
        if c["kind"] != "Figure":
            continue
        above = [l["y1"] for l in body_like if l["y1"] < c["y0"] - 1]
        top = max(above) + 2 if above else 60
        # previous caption/table above on the same page bounds the figure too
        for c2 in caps:
            if c2["y1"] < c["y0"] and c2["y1"] > top - 2:
                top = c2["y1"] + 2
        for t in tabs:
            if t[3] < c["y0"] and t[3] > top - 2:
                top = t[3] + 2
        reg = [70, top, 545, c["y0"] - 1.5]
        dr = [d["rect"] for d in page.get_drawings() if d["rect"].y0 >= reg[1] - 1 and d["rect"].y1 <= reg[3] + 1]
        im = [pymupdf.Rect(x["bbox"]) for x in page.get_image_info() if x["bbox"][1] >= reg[1] - 1 and x["bbox"][3] <= reg[3] + 1]
        txt = [pymupdf.Rect(l["x0"], l["y0"], l["x1"], l["y1"]) for l in L if l["y0"] >= reg[1] - 1 and l["y1"] <= reg[3] + 1]
        allr = dr + im + txt
        if allr:
            reg = [min(r.x0 for r in allr) - 2, min(r.y0 for r in allr) - 2, max(r.x1 for r in allr) + 2, max(r.y1 for r in allr) + 2]
        fig_regions[c["num"]] = dict(page=pno, rect=reg)
        elements.append(dict(type="figure", num=c["num"], page=pno, y=c["y0"],
                             caption=" ".join(line_tex(x["spans"], x["size"]) for x in c["lines"])))
    for c in caps:
        if c["kind"] == "Table":
            elements.append(dict(type="table", num=c["num"], page=pno, y=c["y0"],
                                 caption=" ".join(line_tex(x["spans"], x["size"]) for x in c["lines"])))
    for t in tabs:
        table_index.append(dict(page=pno, bbox=list(t)))
        if not any(c["kind"] == "Table" and c["y1"] < t[1] + 2 and c["y1"] > t[1] - 40 for c in caps):
            elements.append(dict(type="table_cont", page=pno, y=t[1], bbox=list(t)))
    figrects = [v["rect"] for v in fig_regions.values() if v["page"] == pno]
    eqr = EQ_RANGES.get(pno + 1, [])
    for y0, y1, lab in eqr:
        elements.append(dict(type="eq", num=lab, page=pno, y=y0))
    # text lines
    for l in L:
        if id(l) in cap_ids or any(in_rect(l, t) for t in tabs) or any(in_rect(l, r, 3) for r in figrects):
            continue
        if l["y0"] > 790 and re.fullmatch(r"\s*\d+\s*", l["raw"]):
            continue
        if any(y0 - 1 <= l["y0"] and l["y1"] <= y1 + 1 for y0, y1, _ in eqr):
            continue
        raw = l["raw"].strip()
        if l["bold"] and l["size"] >= 13.5:
            lvl = {17.9: 0, 15.9: 1, 13.9: 2}.get(l["size"], 1 if l["size"] > 15 else 2)
            elements.append(dict(type="heading", level=lvl, page=pno, y=l["y0"], raw=raw, x0=l["x0"]))
            continue
        sp = l["spans"]
        if raw.startswith("\u2022"):
            sp = [dict(x) for x in sp]
            for x in sp:
                if "\u2022" in x["text"]:
                    x["text"] = x["text"].replace("\u2022", "", 1)
                    break
        elements.append(dict(type="line", page=pno, y=l["y0"], y1=l["y1"], x0=l["x0"], x1=l["x1"], size=l["size"], raw=raw,
                             tex=line_tex(sp, l["size"]).strip()))

elements.sort(key=lambda e: (e["page"], e["y"]))
table_caps = {}
for pno in BODY:
    for l in page_lines(pno):
        m = re.match(r"^Table (\d+\.\d+)", l["raw"].strip())
        if m and l["bold"] and 10.5 <= l["size"] <= 11.2:
            table_caps[m.group(1)] = [pno, l["y0"], l["y1"]]
json.dump(dict(fig_regions=fig_regions, tables=table_index, table_caps=table_caps), open(OUT + "/tools/regions.json", "w"), indent=1)

# ---------------------------------------------------------------- assemble headings / paragraphs / lists
blocks = []
cur = None
prev = None


def flush():
    global cur
    if cur:
        blocks.append(cur)
    cur = None


i = 0
while i < len(elements):
    e = elements[i]
    if e["type"] == "heading":
        flush()
        # merge consecutive heading lines of the same level (wrapped titles)
        texts = [e["raw"]]
        while i + 1 < len(elements) and elements[i + 1]["type"] == "heading" and elements[i + 1]["level"] == e["level"] \
                and elements[i + 1]["page"] == e["page"] and elements[i + 1]["y"] - e["y"] < 80:
            nxt = elements[i + 1]
            if e["level"] == 0 and re.match(r"^Chapter \d+$", texts[0]) and len(texts) == 1:
                texts.append(nxt["raw"])
            elif re.match(r"^\d+(\.\d+)+$", texts[-1]) or (nxt["x0"] > 100 and abs(nxt["y"] - e["y"]) < 3):
                texts.append(nxt["raw"])
            elif e["level"] > 0 and nxt["y"] - e["y"] < 30:
                texts.append(nxt["raw"])
            else:
                break
            e = dict(e, y=nxt["y"])
            i += 1
        blocks.append(dict(type="heading", level=elements[i]["level"] if e["level"] == elements[i]["level"] else e["level"],
                           text=" ".join(texts), page=e["page"]))
        prev = None
        i += 1
        continue
    if e["type"] in ("table", "figure", "eq", "table_cont"):
        if e["type"] == "eq":
            # equations sit inside paragraphs/items: keep the current block open
            if cur:
                cur.setdefault("parts", []).append(dict(eq=e["num"]))
                prev = e
                i += 1
                continue
        else:
            # floats do not break a paragraph that continues around them
            blocks.append(e)
            i += 1
            continue
    if e["type"] == "line":
        raw = e["raw"]
        is_bullet = raw.startswith("\u2022")
        is_enum = re.match(r"^(\d+)\.\s+\S", raw) and e["x0"] < 110 and (not prev or prev.get("type") != "line" or
                                                                          prev["raw"].endswith((".", ":")) or cur and cur["type"] == "enum")
        gap = None if prev is None or prev.get("type") != "line" else (e["y"] - prev["y1"] if e["page"] == prev["page"] else None)
        new_para = cur is None or prev is None or prev.get("type") != "line"
        if gap is not None and gap > 8.5:
            new_para = True
        if gap is None and prev is not None and prev.get("type") == "line" and e["page"] != prev["page"]:
            new_para = bool(re.search(r"[.:!?]['\")\]]*$", prev["raw"])) and not raw[:1].islower()
        if is_bullet or is_enum:
            flush()
            cur = dict(type="item" if is_bullet else "enum", lines=[], parts=[])
            tex = re.sub(r"^\\textbullet\{\}\s*", "", e["tex"])
            tex = re.sub(r"^\d+\.\s+", "", tex) if is_enum else tex
            cur["lines"].append(tex)
        elif new_para and not (cur and cur["type"] in ("item", "enum") and e["x0"] > 95 and (gap is None or gap < 8.5)):
            flush()
            cur = dict(type="para", lines=[e["tex"]], parts=[], indent=e["x0"] > 95)
        else:
            cur["lines"].append(e["tex"])
        prev = e
    i += 1
flush()


def join_lines(lines):
    s = ""
    for ln in lines:
        if not s:
            s = ln
        elif s.endswith("-") and not s.endswith("--"):
            s = join_hyphen(s, ln)
        elif s.endswith("--") and ln[:1].isdigit():                 # a range broken after its dash: 18--147
            s = s + ln
        else:
            s = s + " " + ln
    return s


# ---------------------------------------------------------------- citations and cross-references
NREF = 0
refs_txt = "".join(doc[p].get_text() for p in range(42, 45))
NREF = max(int(x) for x in re.findall(r"^\[(\d+)\]", refs_txt, flags=re.M))
CITES = []


def cite_sub(m):
    body = m.group(1)
    nums = []
    for part in re.split(r"\s*,\s*", body):
        r = re.split(r"\s*(?:--|-)\s*", part)
        if len(r) == 2:
            nums += list(range(int(r[0]), int(r[1]) + 1))
        else:
            nums.append(int(r[0]))
    pre = m.string[max(0, m.start() - 3):m.start()]
    interval = re.search(r"(\d|\\%)\s*$", pre) and re.fullmatch(r"\s*\d+\s*,\s*\d+\s*", body)   # "90 % [86, 94]" is a CI
    if any(n < 1 or n > NREF for n in nums) or interval:
        return m.group(0)
    CITES.append(m.group(0))
    return r"\cite{" + ",".join(f"ref{n}" for n in nums) + "}"


def postproc(s):
    s = re.sub(r"\[(\d+(?:\s*(?:--|-)\s*\d+)?(?:\s*,\s*\d+(?:\s*(?:--|-)\s*\d+)?)*)\]", cite_sub, s)
    s = re.sub(r"\bFigure (\d+\.\d+)", r"Figure~\\ref{fig:\1}", s)
    s = re.sub(r"\bTable (\d+\.\d+)", r"Table~\\ref{tab:\1}", s)
    s = re.sub(r"\bChapter (\d+)\b(?!\.\d)", r"Chapter~\\ref{ch:\1}", s)
    s = re.sub(r"\bSection (\d+\.\d+)", r"Section~\\ref{sec:\1}", s)
    s = re.sub(r"\$([^$]*?(?:=|\\approx|\\le|\\ge|<|>|\\pm|\\times|\+|-)\s*)\$\s?([0-9]+(?:\.[0-9]+)?)", r"$\1\2$", s)
    s = re.sub(r"(\$[^$]+\$)(?=[A-Za-z])", r"\1 ", s)                 # space after inline maths before a word
    s = re.sub(r"(?<=[A-Za-z])(\$(?:[A-Za-z]|\\[a-zA-Z]))", r" \1", s)  # space before inline maths after a word
    s = re.sub(r" {2,}", " ", s)
    # "a × b" as printed in the original (binary spacing); "(×5" keeps its tight form
    s = re.sub(r"(\d)\s*\$\\times\s*([\d.,]+)\$", r"\1 $\\times$ \2", s)
    s = re.sub(r"(?<!\()\$\\times\s*([\d.,]+)\$", r"$\\times$ \1", s)
    s = re.sub(r"(\d)\$\\times\$(?=\d)", r"\1 $\\times$ ", s)
    return s


# ---------------------------------------------------------------- write chapters
chap, files = 0, {}
outbuf = []
LISTOPEN = None


def emit(s):
    outbuf.append(s)


def close_list():
    global LISTOPEN
    if LISTOPEN:
        emit(r"\end{" + LISTOPEN + "}\n")
        LISTOPEN = None


def render_block_text(b):
    """join lines; equations embedded as placeholders."""
    return postproc(join_lines(b["lines"])) + "".join(f"\n\n[[EQ {p['eq']}]]\n" for p in b.get("parts", []) if "eq" in p)


for b in blocks:
    if b["type"] == "heading":
        close_list()
        t = b["text"]
        if b["level"] == 0:
            if outbuf:
                files[chap] = "".join(outbuf)
            outbuf = []
            m = re.match(r"^Chapter (\d+)\s+(.*)$", t)
            chap = int(m.group(1)) if m else chap + 1
            title = m.group(2) if m else t
            emit(r"\chapter{" + text_tex(title) + r"}\label{ch:" + str(chap) + "}\n\n")
        else:
            m = re.match(r"^(\d+(?:\.\d+)+)\s+(.*)$", t)
            num, title = (m.group(1), m.group(2)) if m else ("", t)
            cmd = r"\section" if b["level"] == 1 else r"\subsection"
            emit(cmd + "{" + postproc(text_tex(title)) + "}" + (r"\label{sec:" + num + "}" if num else "") + "\n\n")
        continue
    if b["type"] in ("table", "figure", "table_cont"):
        close_list()
        if b["type"] == "table_cont":
            emit(f"[[TABLECONT p{b['page']+1}]]\n\n")
        else:
            emit(f"[[{b['type'].upper()} {b['num']}]]\n\n")
        continue
    if b["type"] == "para":
        close_list()
        emit(render_block_text(b) + "\n\n")
    elif b["type"] in ("item", "enum"):
        env = "itemize" if b["type"] == "item" else "enumerate"
        if LISTOPEN != env:
            close_list()
            emit(r"\begin{" + env + "}\n")
            LISTOPEN = env
        emit(r"  \item " + render_block_text(b) + "\n")
close_list()
files[chap] = "".join(outbuf)

LABEL_RE = re.compile(r"^\\textbf\{(?:Table|Figure)~\\ref\{[^}]*\}\}\s*")   # LaTeX prints the label itself
captions = {b["type"][0].upper() + b["num"]: LABEL_RE.sub("", postproc(b["caption"])) for b in blocks if b["type"] in ("table", "figure")}
json.dump(captions, open(OUT + "/tools/captions.json", "w"), indent=1)
for c, s in files.items():
    if c == 0:
        continue
    open(f"{OUT}/chapters/ch{c}.tex", "w").write(s)
print("chapters:", sorted(k for k in files if k), "| citations converted:", len(CITES), "| NREF", NREF)
print("figures:", sorted(k for k in fig_regions), "| tables:", sorted(k for k, v in captions.items() if k in [b.get("num") for b in blocks if b["type"] == "table"]))
