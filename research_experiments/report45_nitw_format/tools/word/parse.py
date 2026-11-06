"""Parse the LaTeX sources of the rebuilt report 45 (/tmp/claude-0/r45tex) into doc.json for the Word build.

Blocks: chapter, section, para, list, table, figure, equation; runs carry text + bold/italic/sup/sub/link/colour.
python3 -I parse.py <texdir> <outdir>
"""
import json
import re
import sys

import pymupdf

TEX, OUT = sys.argv[1], sys.argv[2]

# ---------------------------------------------------------------- inline text -> runs
ACCENT = {'"o': "ö", '"u': "ü", '"a': "ä", "'e": "é", "`e": "è", "'a": "á"}
MATHSYM = {r"\times": "×", r"\rightarrow": "→", r"\le": "≤", r"\ge": "≥", r"\approx": "≈", r"\pm": "±", r"\|": "‖",
           r"\mu": "μ", r"\sigma": "σ", r"\Phi": "Φ", r"\circ": "°", r"\cdot": "·", r"\in": "∈"}
RELS = set("=<>≤≥≈")
BINS = set("+−×±")


def cite_runs(keys):
    nums = sorted(int(k.strip().replace("ref", "")) for k in keys.split(","))
    groups, cur = [], [nums[0]]
    for n in nums[1:]:
        if n == cur[-1] + 1:
            cur.append(n)
        else:
            groups.append(cur); cur = [n]
    groups.append(cur)
    runs = [dict(t="[")]
    for gi, g in enumerate(groups):
        if gi:
            runs.append(dict(t=", "))
        if len(g) >= 3:
            runs += [dict(t=str(g[0]), link=f"ref_{g[0]}"), dict(t="–"), dict(t=str(g[-1]), link=f"ref_{g[-1]}")]
        else:
            for k, n in enumerate(g):
                if k:
                    runs.append(dict(t=", "))
                runs.append(dict(t=str(n), link=f"ref_{n}"))
    runs.append(dict(t="]"))
    return runs


def math_runs(m, bold=False):
    """Simple inline maths -> styled text (letters italic, sub/superscripts, relation spacing)."""
    m = m.strip()
    out = []
    i = 0

    def add(t, **kw):
        out.append(dict(t=t, b=bold or kw.pop("b", False), **kw))
    while i < len(m):
        c = m[i]
        if m.startswith(r"\text{", i) or m.startswith(r"\mathrm{", i):
            j = m.index("}", i); add(m[m.index("{", i) + 1:j]); i = j + 1; continue
        if m.startswith(r"\mathbf{", i):
            j = m.index("}", i); add(m[i + 8:j], b=True); i = j + 1; continue
        if m.startswith(r"\bm{", i):
            j = m.index("}", i); out += math_runs(m[i + 4:j], bold=True); i = j + 1; continue
        if m.startswith(r"\hat{", i):
            j = m.index("}", i); inner = math_runs(m[i + 5:j], bold)
            inner[-1]["t"] += "\u0302"; out += inner; i = j + 1; continue
        if c in "^_":
            if m[i + 1] == "{":
                j = i + 2; depth = 1
                while depth:
                    depth += {"{": 1, "}": -1}.get(m[j], 0); j += 1
                inner = m[i + 2:j - 1]
            else:
                j = i + 2; inner = m[i + 1]
            for r in math_runs(inner, bold):
                r["sup" if c == "^" else "sub"] = True
                out.append(r)
            i = j; continue
        if c == "\\":
            mm = re.match(r"\\[a-zA-Z]+|\\\|", m[i:])
            if mm is None:                                   # \, \; \  \{ \}
                nxt = m[i + 1]
                add({",": "\u2009", ";": " ", " ": " ", "{": "{", "}": "}"}.get(nxt, nxt))
                i += 2; continue
            cmd = mm.group(0)
            sym = MATHSYM.get(cmd, cmd)
            add(sym, i=cmd in (r"\mu", r"\sigma"))
            i += len(cmd); continue
        if c.isalpha():
            add(c, i=True); i += 1; continue
        if c == "-":
            add("−"); i += 1; continue
        if c in "{}":
            i += 1; continue
        if c == " ":
            i += 1; continue
        add(c); i += 1
    # spacing around relations / binary operators that sit between operands
    res = []
    for k, r in enumerate(out):
        t = r["t"]
        if t in RELS or (t in BINS and k > 0):
            if k > 0 and res and not r.get("sup") and not r.get("sub"):
                res.append(dict(t=" ", b=bold))
            res.append(r)
            if k + 1 < len(out) and not r.get("sup") and not r.get("sub"):
                res.append(dict(t=" ", b=bold))
        else:
            res.append(r)
    return res


def text_runs(s, bold=False, italic=False):
    """LaTeX inline text -> list of runs."""
    runs = []
    buf = []

    def flush():
        if buf:
            runs.append(dict(t="".join(buf), b=bold, i=italic)); buf.clear()
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            if s.startswith(r"\textbf{", i) or s.startswith(r"\emph{", i) or s.startswith(r"\textit{", i) or s.startswith(r"\url{", i):
                name = s[i + 1:s.index("{", i)]
                j = s.index("{", i) + 1; depth = 1; k = j
                while depth:
                    depth += {"{": 1, "}": -1}.get(s[k], 0)
                    if s[k] == "\\":
                        k += 1
                    k += 1
                inner = s[j:k - 1]
                flush()
                if name == "url":
                    runs.append(dict(t=inner, url=inner))
                else:
                    runs += text_runs(inner, bold or name == "textbf", italic or name in ("emph", "textit"))
                i = k; continue
            if s.startswith(r"\cite{", i):
                j = s.index("}", i); flush()
                for r in cite_runs(s[i + 6:j]):
                    r.setdefault("b", bold); runs.append(r)
                i = j + 1; continue
            if s.startswith(r"\ref{", i):
                j = s.index("}", i); key = s[i + 5:j]
                kind, num = key.split(":")
                flush(); runs.append(dict(t=num, b=bold, i=italic, link=f"{kind}_{num.replace('.', '_')}"))
                i = j + 1; continue
            two = s[i:i + 3]
            if two[1:3] in ACCENT:
                buf.append(ACCENT[two[1:3]]); i += 3; continue
            if s[i + 1] in "%&_#${}":
                buf.append(s[i + 1]); i += 2; continue
            if s[i + 1] == ",":
                buf.append("\u2009"); i += 2; continue
            if s[i + 1] == " ":
                buf.append(" "); i += 2; continue
            mm = re.match(r"\\[a-zA-Z]+", s[i:])
            if mm is None:
                raise ValueError("unhandled sequence " + repr(s[max(0, i - 40):i + 40]))
            cmd = mm.group(0)
            if cmd in MATHSYM:
                buf.append(MATHSYM[cmd]); i += len(cmd); continue
            raise ValueError("unhandled command " + cmd + " in: " + s[max(0, i - 40):i + 40])
        if c == "$":
            j = s.index("$", i + 1); flush()
            for r in math_runs(s[i + 1:j], bold):
                if italic:
                    r["i"] = True
                runs.append(r)
            i = j + 1; continue
        if s.startswith("---", i):
            buf.append("—"); i += 3; continue
        if s.startswith("--", i):
            buf.append("–"); i += 2; continue
        if s.startswith("``", i):
            buf.append("“"); i += 2; continue
        if s.startswith("''", i):
            buf.append("”"); i += 2; continue
        if c == "`":
            buf.append("‘"); i += 1; continue
        if c == "'":
            buf.append("’"); i += 1; continue
        if c == "~":
            buf.append("\u00a0"); i += 1; continue
        if c in "{}":
            i += 1; continue
        buf.append(c); i += 1
    flush()
    # merge neighbours with identical style
    merged = []
    for r in runs:
        r = {k: v for k, v in r.items() if v or k == "t"}
        if merged and {k: v for k, v in merged[-1].items() if k != "t"} == {k: v for k, v in r.items() if k != "t"}:
            merged[-1]["t"] += r["t"]
        else:
            merged.append(r)
    return merged


# ---------------------------------------------------------------- tables
def split_top(s, sep):
    """Split on sep at brace depth 0."""
    parts, depth, cur, i = [], 0, [], 0
    while i < len(s):
        if depth == 0 and s.startswith(sep, i):
            parts.append("".join(cur)); cur = []; i += len(sep); continue
        if s[i] == "\\" and i + 1 < len(s):
            cur.append(s[i:i + 2]); i += 2; continue
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        cur.append(s[i]); i += 1
    parts.append("".join(cur))
    return parts


def parse_table(path):
    s = open(path).read()
    num = re.search(r"\\label\{tab:([\d.]+)\}", s).group(1)
    cap = re.search(r"\\caption\{(.*?)\}\\label", s, re.S).group(1)
    small = r"\footnotesize" not in s
    long = r"\begin{longtable}" in s
    fracs = [float(x) for x in re.findall(r"p\{\\dimexpr ([\d.]+)\\linewidth", s.split("\\toprule")[0].split("\\caption")[0] if long else s.split("\\toprule")[0])]
    if long:
        body = s[s.index(r"\endlastfoot") + len(r"\endlastfoot"):s.index(r"\end{longtable}")]
        head = s[s.index(r"\toprule") + len(r"\toprule"):s.index(r"\endfirsthead")]
        head = head.replace(r"\midrule", "")
        src = r"\toprule" + head + r"\midrule" + body
    else:
        src = s[s.index(r"\toprule"):s.index(r"\end{tabular}")]
    rows, rule_above, partial, in_head = [], "top", [], True
    for chunk in split_top(src, r"\\"):
        chunk = chunk.strip()
        while True:
            m = re.match(r"^\\(toprule|midrule|bottomrule)\s*", chunk)
            m2 = re.match(r"^\\cmidrule\(lr\)\{(\d+)-(\d+)\}\s*", chunk)
            if m:
                if m.group(1) == "midrule":
                    rule_above = "mid"
                chunk = chunk[m.end():]
            elif m2:
                partial.append((int(m2.group(1)), int(m2.group(2))))
                chunk = chunk[m2.end():]
            else:
                break
        if not chunk:
            continue
        cells = []
        for c in split_top(chunk, "&"):
            c = c.strip()
            mm = re.match(r"^\\multicolumn\{(\d+)\}\{.*?\}\{(.*)\}$", c, re.S)
            if mm:
                # spec may contain braces: take the last {...} group as the content
                k = int(mm.group(1))
                inner = c[c.rindex("}{", 0, len(c) - 1) + 2:-1] if False else None
                depth, j, groups, start = 0, len(r"\multicolumn"), [], None
                while j < len(c):
                    if c[j] == "{":
                        if depth == 0:
                            start = j + 1
                        depth += 1
                    elif c[j] == "}":
                        depth -= 1
                        if depth == 0:
                            groups.append(c[start:j])
                    j += 1
                cells.append(dict(span=k, runs=text_runs(groups[2])))
            else:
                cells.append(dict(span=1, runs=text_runs(c)))
        rows.append(dict(cells=cells, rule=rule_above, cmid_below=[]))
        if partial:
            rows[-1]["cmid_below"] = partial; partial = []
        rule_above = None
    # header = rows before the first mid rule
    first_mid = next((k for k, r in enumerate(rows) if r["rule"] == "mid"), 1)
    for k, r in enumerate(rows):
        r["header"] = k < first_mid
    return dict(type="table", num=num, caption=text_runs(cap), fracs=fracs, small=small, long=long, rows=rows)


# ---------------------------------------------------------------- figures (PNG at 300 dpi)
def figure_png(name, outpng):
    doc = pymupdf.open(f"{TEX}/figures/{name}.pdf")
    pix = doc[0].get_pixmap(dpi=300)
    pix.save(outpng)
    return pix.width, pix.height


def fig31_png(outpng):
    """Crop the TikZ Figure 3.1 from the compiled PDF (vector -> 300 dpi PNG)."""
    d = pymupdf.open(f"{TEX}/main.pdf")
    for p in d:
        if "System architecture of the Phase I pipeline" in p.get_text() and "Chapter" in p.get_text():
            cap = [b for b in p.get_text("dict")["blocks"] if "System architecture" in "".join(s["text"] for l in b.get("lines", []) for s in l["spans"])][0]
            ys = [dr["rect"] for dr in p.get_drawings() if dr["rect"].y1 < cap["bbox"][1] and dr["rect"].y0 > 150]
            r = pymupdf.Rect(min(x.x0 for x in ys) - 2, min(x.y0 for x in ys) - 2, max(x.x1 for x in ys) + 2, max(x.y1 for x in ys) + 2)
            pix = p.get_pixmap(dpi=300, clip=r)
            pix.save(outpng)
            return pix.width, pix.height, r.width


# ---------------------------------------------------------------- chapters
blocks = []
for c in range(1, 8):
    lines = open(f"{TEX}/body/ch{c}.tex").read().split("\n")
    para, after_heading, continuation, i = [], False, False, 0

    def end_para():
        global para, after_heading, continuation
        if para:
            blocks.append(dict(type="para", runs=text_runs(" ".join(para)), indent=not (after_heading or continuation)))
            para = []; after_heading = False; continuation = False
    while i < len(lines):
        ln = lines[i].strip()
        m = re.match(r"^\\chapter\{(.*)\}\\label\{ch:(\d+)\}$", ln)
        ms = re.match(r"^\\(sub)?section\{(.*)\}\\label\{sec:([\d.]+)\}$", ln)
        if m:
            end_para(); blocks.append(dict(type="chapter", num=m.group(2), title=m.group(1))); after_heading = True
        elif ms:
            end_para(); blocks.append(dict(type="section", level=2 if ms.group(1) else 1, num=ms.group(3), title_runs=text_runs(ms.group(2)), title=ms.group(2))); after_heading = True
        elif ln == "":
            end_para(); continuation = False
        elif ln.startswith(r"\input{tables/"):
            end_para()
            blocks.append(parse_table(f"{TEX}/" + ln[len(r"\input{"):-1] + ".tex"))
        elif ln.startswith(r"\begin{figure}"):
            end_para()
            j = i
            while not lines[j].strip().startswith(r"\end{figure}"):
                j += 1
            fig = "\n".join(lines[i:j + 1])
            num = re.search(r"\\label\{fig:([\d.]+)\}", fig).group(1)
            cap = re.search(r"\\caption\{(.*)\}\\label", fig, re.S).group(1)
            png = f"fig_{num.replace('.', '_')}.png"
            if num == "3.1":
                w, h, wpt = fig31_png(f"{OUT}/media/{png}"); frac = min(1.0, wpt / 433.27)
            else:
                name = re.search(r"\{figures/(fig_[\d_]+)\}", fig).group(1)
                frac = float(re.search(r"width=([\d.]+)\\linewidth", fig).group(1))
                w, h = figure_png(name, f"{OUT}/media/{png}")
            blocks.append(dict(type="figure", num=num, caption=text_runs(cap), png=png, px=[w, h], frac=frac))
            i = j
        elif ln.startswith(r"\begin{itemize}") or ln.startswith(r"\begin{enumerate}"):
            end_para()
            kind = "bullet" if "itemize" in ln else "number"
            items = []
            j = i + 1
            eqs = {}
            while not lines[j].strip().startswith(r"\end{"):
                t = lines[j].strip()
                me = re.match(r"^\\EqThree(One|Two|Three|Four)$", t)
                if me:
                    eqs[len(items) - 1] = {"One": "3.1", "Two": "3.2", "Three": "3.3", "Four": "3.4"}[me.group(1)]
                elif t.startswith(r"\item"):
                    items.append(t[len(r"\item"):].strip())
                elif t:
                    items[-1] += " " + t
                j += 1
            blocks.append(dict(type="list", kind=kind, items=[dict(runs=text_runs(t), eq=eqs.get(k)) for k, t in enumerate(items)]))
            i = j; after_heading = False; continuation = True
        elif re.match(r"^\\EqThree(One|Two|Three|Four)$", ln):
            end_para()
            num = {"One": "3.1", "Two": "3.2", "Three": "3.3", "Four": "3.4"}[ln[len(r"\EqThree"):]]
            blocks.append(dict(type="equation", num=num)); continuation = True
        else:
            para.append(ln)
        i += 1
    end_para()

# ---------------------------------------------------------------- references
refs = []
for m in re.finditer(r"\\bibitem\{ref(\d+)\}\s*(.*?)(?=\n\n\\bibitem|\n\n\\end\{thebibliography\})", open(f"{TEX}/references.tex").read(), re.S):
    refs.append(dict(n=int(m.group(1)), runs=text_runs(m.group(2).strip())))

# ---------------------------------------------------------------- front matter text from frontmatter.tex
fm = open(f"{TEX}/frontmatter.tex").read()
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", fm))


def expand(t):
    for k, v in macros.items():
        t = t.replace("\\" + k + "\\ ", v + " ").replace("\\" + k, v)
    return t
ack = fm[fm.index(r"\frontheading{Acknowledgements}") + len(r"\frontheading{Acknowledgements}"):fm.index(r"\vspace{24pt}", fm.index(r"\frontheading{Acknowledgements}"))].strip()
abs_ = fm[fm.index(r"\frontheading{Abstract}") + len(r"\frontheading{Abstract}"):fm.index(r"\clearpage", fm.index(r"\frontheading{Abstract}"))].strip()
abstract = [text_runs(expand(p.strip())) for p in abs_.split("\n\n") if p.strip()]
decl = re.findall(r"\\item (.*)", fm[fm.index(r"\begin{enumerate}"):fm.index(r"\end{enumerate}")])


def abbr_rows(block):
    rows = []
    for ln in block.strip().split("\n"):
        ln = ln.strip().rstrip("\\").strip()
        if "&" in ln:
            a, b = ln.split("&", 1)
            rows.append([text_runs(a.strip()), text_runs(b.strip())])
    return rows
tabs = re.findall(r"\\abbrtable\{%\n(.*?)\n\}", fm, re.S)
front = dict(macros=macros, ack=text_runs(expand(ack)), abstract=abstract, declaration=[text_runs(expand(x)) for x in decl],
             abbreviations=abbr_rows(tabs[0]), notation=abbr_rows(tabs[1]))

json.dump(dict(blocks=blocks, refs=refs, front=front), open(f"{OUT}/doc.json", "w"), ensure_ascii=False, indent=0)
kinds = {}
for b in blocks:
    kinds[b["type"]] = kinds.get(b["type"], 0) + 1
print(kinds, "| refs", len(refs), "| abbreviations", len(front["abbreviations"]), "| notation", len(front["notation"]), "| abstract paras", len(abstract), "| decl", len(decl))

# ---------------------------------------------------------------- caption fits on one line? (template: then centred)
LMB = pymupdf.Font(fontfile="/usr/share/texmf/fonts/opentype/public/lm/lmroman10-bold.otf")
LMR = pymupdf.Font(fontfile="/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf")
d = json.load(open(f"{OUT}/doc.json"))
for b in d["blocks"]:
    if b["type"] in ("table", "figure"):
        label = ("Table " if b["type"] == "table" else "Figure ") + b["num"] + " "
        w = LMB.text_length(label, fontsize=11) + sum(LMR.text_length(r["t"], fontsize=11) for r in b["caption"])
        b["cap_single"] = w <= 433.0
json.dump(d, open(f"{OUT}/doc.json", "w"), ensure_ascii=False, indent=0)
print("single-line captions:", [b["type"][0] + b["num"] for b in d["blocks"] if b.get("cap_single")])

# ---------------------------------------------------------------- approval / certificate wording (same text as the LaTeX version)
appr = fm[fm.index("\\vspace{\\apA}") + len("\\vspace{\\apA}"):fm.index("\\vspace{\\apB}")].strip()
cert = fm[fm.index("\\vspace{\\ceB}") + len("\\vspace{\\ceB}"):fm.index("\\vspace{\\ceC}")].strip()
cert = cert.replace("\\noindent ", "")
d["front"]["approval"] = text_runs(expand(appr))
d["front"]["certificate"] = text_runs(expand(cert))
json.dump(d, open(f"{OUT}/doc.json", "w"), ensure_ascii=False, indent=0)
print("approval:", "".join(r["t"] for r in d["front"]["approval"])[:120])
print("certificate:", "".join(r["t"] for r in d["front"]["certificate"])[:120])

# ---------------------------------------------------------------- title lines exactly as broken in the template version
tl = []
for b in pymupdf.open(f"{TEX}/main.pdf")[0].get_text("dict")["blocks"]:
    for l in b.get("lines", []):
        s0 = l["spans"][0]
        if s0["size"] > 16 and s0["origin"][1] < 300:
            tl.append((s0["origin"][1], "".join(x["text"] for x in l["spans"]).strip()))
d["front"]["title_lines"] = [t for _, t in sorted(tl)]
json.dump(d, open(f"{OUT}/doc.json", "w"), ensure_ascii=False, indent=0)
print("title lines:", d["front"]["title_lines"])
