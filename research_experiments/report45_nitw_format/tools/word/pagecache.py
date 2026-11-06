"""Cached results for the PAGEREF fields: the page of every contents/list target in the rendered layout,
formatted as in the document (roman front matter, arabic from Chapter 1).  Word recomputes them on open."""
import json, re, zipfile
import pymupdf

def roman(n):
    out = ""
    for v, s in [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]:
        while n >= v:
            out += s; n -= v
    return out

D = json.load(open("doc.json"))
pdf = pymupdf.open("render/report45.pdf")
txt = [re.sub(r"[ \t ]+", " ", p.get_text()) for p in pdf]
ids = re.findall(r"PAGEREF (\w+)", zipfile.ZipFile("report45.docx").read("word/document.xml").decode())
first = lambda pred, start=0: next(i for i in range(start, len(txt)) if pred(txt[i]))
toc = first(lambda t: t.lstrip().startswith("Contents\n") and "Certificate of Approval" in t)
ch1 = first(lambda t: t.lstrip().startswith("Chapter 1\n"), toc + 1)
lof = first(lambda t: t.lstrip().startswith("List of Figures\n"), toc + 1)
lot = first(lambda t: t.lstrip().startswith("List of Tables\n"), lof + 1)
abbr = first(lambda t: "List of Abbreviations\nAUROC" in t, lot + 1)
pages = {"fm_approval": first(lambda t: "APPROVAL OF THE PROJECT EVALUATION BOARD" in t),
         "fm_cert": first(lambda t: "printed on the Departmental Letter-Head" in t),
         "fm_decl": first(lambda t: t.lstrip().startswith("DECLARATION")),
         "fm_ack": first(lambda t: "We thank our supervisor" in t),
         "fm_abs": first(lambda t: "Heart failure with reduced left ventricular" in t),
         "fm_toc": toc, "fm_lof": lof, "fm_lot": lot, "fm_abbr": abbr,
         "fm_refs": first(lambda t: t.lstrip().startswith("References\n"), ch1 + 1)}
for b in D["blocks"]:
    if b["type"] == "chapter":
        pages[f"ch_{b['num']}"] = first(lambda t, n=b["num"]: t.lstrip().startswith(f"Chapter {n}\n"), ch1)
    elif b["type"] == "section":
        head = re.escape(b["num"]) + r"\s+" + re.escape(b["title"][:18].replace("--", "–"))
        pages[f"sec_{b['num'].replace('.', '_')}"] = first(lambda t, h=head: re.search(r"(^|\n)" + h, t), ch1)
    elif b["type"] in ("figure", "table"):
        lab = ("Figure " if b["type"] == "figure" else "Table ") + b["num"]
        cap = "".join(r["t"] for r in b["caption"])[:14]
        pat = re.escape(lab) + r"\s+" + re.escape(cap).replace(r"\ ", r"\s+")
        pages[f"{b['type'][:3]}_{b['num'].replace('.', '_')}"] = first(lambda t, p=pat: re.search(p, t), ch1)
cache = {k: (roman(v + 1) if k.startswith("fm_") and k != "fm_refs" else str(v - ch1 + 1)) for k, v in pages.items()}
missing = [i for i in ids if i not in cache]
json.dump(cache, open("pagecache.json", "w"), indent=0)
print("fields", len(ids), "| cached", len(ids) - len(missing), "| missing", missing)
print({k: cache[k] for k in ["fm_approval", "fm_abs", "fm_lof", "fm_abbr", "ch_1", "ch_7", "fm_refs", "sec_3_5_1", "fig_6_8", "tab_2_1"]})
